"""
解纠缠损失函数模块

实现完整的解纠缠损失，包括：
1. 独特损失 L_private: 确保独特信息任务相关且与共享信息独立
2. 共享损失 L_shared: 确保共享信息对齐、任务相关且多样化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mutual_info_estimators import InfoNCE, CLUB, ConditionalCLUB, HSIC, BarlowTwins


class DisentangleLoss(nn.Module):
    """
    解纠缠损失函数
    
    损失组成：
    
    L_private = -λ1*(I(U1;Y) + I(U2;Y))                          # 最大化任务相关性
              + λ2*(I(U1;Z1) + I(U2;Z2))                         # 独特-共享独立（同模态）
              + λ6*I(U1;U2)                                      # 方案B：独特信息差异性（跨模态）
              + λ7*(I(H1;H2|Z1) + I(H1;H2|Z2))                   # 方案A：条件独立性
    
    L_shared = λ3*d(Z1,Z2)                     # 共享表征对齐
             - λ4*(I(Z1;Y) + I(Z2;Y))          # 最大化任务相关性
             + λ5*R_diversity(Z1,Z2)           # 多样性正则（防坍缩）
    
    Args:
        config: 配置对象，包含以下属性：
            - lambda1: I(Ui;Y) 任务相关性权重
            - lambda2: I(Ui;Zi) 独立性权重
            - lambda3: d(Z1,Z2) 对齐权重
            - lambda4: I(Zi;Y) 任务相关性权重
            - lambda5: R_diversity 多样性权重
            - lambda6: I(U1;U2) 独特信息差异性权重 [方案B]
            - lambda7: I(H1;H2|Z) 条件独立性权重 [方案A]
            - use_method_a: 是否使用方案A
            - use_method_b: 是否使用方案B
            - temperature: InfoNCE温度参数
            - d_shared: 共享表征维度
            - d_private: 独特表征维度
    """
    def __init__(self, config):
        super(DisentangleLoss, self).__init__()
        
        # 保存配置
        self.config = config
        self.lambda1 = config.lambda1
        self.lambda2 = config.lambda2
        self.lambda3 = config.lambda3
        self.lambda4 = config.lambda4
        self.lambda5 = config.lambda5
        self.lambda6 = config.lambda6
        self.lambda7 = config.lambda7
        self.use_method_a = config.use_method_a
        self.use_method_b = config.use_method_b
        
        # 初始化互信息估计器
        # I(U;Y) 和 I(Z;Y): 使用InfoNCE（最大化任务相关性）
        self.info_nce_u = InfoNCE(temperature=config.temperature)
        self.info_nce_z = InfoNCE(temperature=config.temperature)
        
        # I(U;Z): 使用CLUB（最小化独立性）
        # 需要为每个模态创建独立的CLUB估计器
        self.club_emg = CLUB(
            x_dim=config.d_private,  # U1维度
            y_dim=config.d_shared,   # Z1维度
            hidden_dim=128
        )
        self.club_imu = CLUB(
            x_dim=config.d_private,  # U2维度
            y_dim=config.d_shared,   # Z2维度
            hidden_dim=128
        )
        
        # I(U1;U2): 使用CLUB（方案B：最小化跨模态独特信息的互信息，确保差异性）
        if self.use_method_b:
            self.club_cross_private = CLUB(
                x_dim=config.d_private,  # U1维度
                y_dim=config.d_private,  # U2维度
                hidden_dim=128
            )
        
        # I(H1;H2|Z): 使用ConditionalCLUB（方案A：条件独立性）
        if self.use_method_a:
            # H1维度: 32*12=384, H2维度: 32*36=1152, Z维度: 128
            self.cond_club_z1 = ConditionalCLUB(
                x_dim=32*12,       # H1维度
                y_dim=32*36,       # H2维度
                z_dim=config.d_shared,  # Z1维度
                hidden_dim=256
            )
            self.cond_club_z2 = ConditionalCLUB(
                x_dim=32*12,       # H1维度
                y_dim=32*36,       # H2维度
                z_dim=config.d_shared,  # Z2维度
                hidden_dim=256
            )
        
        # R_diversity: 使用Barlow Twins（去冗余）
        self.barlow_twins = BarlowTwins(lambda_off_diag=0.005)
    
    def compute_private_loss(self, U1, U2, Z1, Z2, H1, H2, labels):
        """
        计算独特损失
        
        L_private = -λ1*(I(U1;Y) + I(U2;Y)) + λ2*(I(U1;Z1) + I(U2;Z2)) 
                  + λ6*I(U1;U2) [方案B] + λ7*(I(H1;H2|Z1) + I(H1;H2|Z2)) [方案A]
        
        Args:
            U1: [batch_size, d_private] EMG独特表征
            U2: [batch_size, d_private] IMU独特表征
            Z1: [batch_size, d_shared] EMG共享表征
            Z2: [batch_size, d_shared] IMU共享表征
            H1: [batch_size, 384] EMG原始特征
            H2: [batch_size, 1152] IMU原始特征
            labels: [batch_size] 标签
        
        Returns:
            loss: 标量，独特损失
            loss_dict: 字典，包含各项损失的详细信息
        """
        # 1. I(U1;Y) + I(U2;Y): 最大化独特信息的任务相关性
        # InfoNCE返回的是负的互信息下界，所以直接加上（相当于最大化MI）
        mi_u1_y = self.info_nce_u(U1, labels)
        mi_u2_y = self.info_nce_u(U2, labels)
        task_relevance_loss = mi_u1_y + mi_u2_y
        
        # 2. I(U1;Z1) + I(U2;Z2): 最小化独特-共享的互信息（独立性约束）
        # CLUB返回的是互信息上界，直接最小化
        mi_u1_z1 = self.club_emg(U1, Z1)
        mi_u2_z2 = self.club_imu(U2, Z2)
        independence_loss = mi_u1_z1 + mi_u2_z2
        
        # 3. 方案B：I(U1;U2) 最小化跨模态独特信息的互信息（确保U1和U2是不同的）
        if self.use_method_b:
            mi_u1_u2 = self.club_cross_private(U1, U2)
            diversity_b_loss = self.lambda6 * mi_u1_u2
        else:
            mi_u1_u2 = torch.tensor(0.0, device=U1.device)
            diversity_b_loss = torch.tensor(0.0, device=U1.device)
        
        # 4. 方案A：I(H1;H2|Z1) + I(H1;H2|Z2) 条件独立性（给定共享后，原始特征独立）
        if self.use_method_a:
            mi_h1_h2_given_z1 = self.cond_club_z1(H1, H2, Z1)
            mi_h1_h2_given_z2 = self.cond_club_z2(H1, H2, Z2)
            conditional_independence_loss = mi_h1_h2_given_z1 + mi_h1_h2_given_z2
            diversity_a_loss = self.lambda7 * conditional_independence_loss
        else:
            mi_h1_h2_given_z1 = torch.tensor(0.0, device=U1.device)
            mi_h1_h2_given_z2 = torch.tensor(0.0, device=U1.device)
            conditional_independence_loss = torch.tensor(0.0, device=U1.device)
            diversity_a_loss = torch.tensor(0.0, device=U1.device)
        
        # 总的独特损失
        loss = (-self.lambda1 * task_relevance_loss + 
                self.lambda2 * independence_loss + 
                diversity_b_loss + 
                diversity_a_loss)
        
        # 详细信息（用于监控）
        loss_dict = {
            'private_total': loss.item(),
            'private_task_relevance': task_relevance_loss.item(),
            'private_mi_u1_y': mi_u1_y.item(),
            'private_mi_u2_y': mi_u2_y.item(),
            'private_independence': independence_loss.item(),
            'private_mi_u1_z1': mi_u1_z1.item(),
            'private_mi_u2_z2': mi_u2_z2.item(),
        }
        
        # 添加方案B的监控指标
        if self.use_method_b:
            loss_dict.update({
                'private_diversity_b': mi_u1_u2.item(),
                'private_mi_u1_u2': mi_u1_u2.item()
            })
        
        # 添加方案A的监控指标
        if self.use_method_a:
            loss_dict.update({
                'private_diversity_a': conditional_independence_loss.item(),
                'private_mi_h1_h2_given_z1': mi_h1_h2_given_z1.item(),
                'private_mi_h1_h2_given_z2': mi_h1_h2_given_z2.item()
            })
        
        return loss, loss_dict
    
    def compute_shared_loss(self, Z1, Z2, labels):
        """
        计算共享损失
        
        L_shared = λ3*d(Z1,Z2) - λ4*(I(Z1;Y) + I(Z2;Y)) + λ5*R_diversity(Z1,Z2)
        
        Args:
            Z1: [batch_size, d_shared] EMG共享表征
            Z2: [batch_size, d_shared] IMU共享表征
            labels: [batch_size] 标签
        
        Returns:
            loss: 标量，共享损失
            loss_dict: 字典，包含各项损失的详细信息
        """
        # 1. d(Z1, Z2): 对齐共享表征（最小化欧式距离）
        # 使用MSE损失
        alignment_loss = F.mse_loss(Z1, Z2)
        
        # 2. I(Z1;Y) + I(Z2;Y): 最大化共享信息的任务相关性
        mi_z1_y = self.info_nce_z(Z1, labels)
        mi_z2_y = self.info_nce_z(Z2, labels)
        task_relevance_loss = mi_z1_y + mi_z2_y
        
        # 3. R_diversity(Z1, Z2): 多样性正则（防止表征坍缩）
        diversity_loss = self.barlow_twins(Z1, Z2)
        
        # 总的共享损失
        loss = (self.lambda3 * alignment_loss - 
                self.lambda4 * task_relevance_loss + 
                self.lambda5 * diversity_loss)
        
        # 详细信息（用于监控）
        loss_dict = {
            'shared_total': loss.item(),
            'shared_alignment': alignment_loss.item(),
            'shared_task_relevance': task_relevance_loss.item(),
            'shared_mi_z1_y': mi_z1_y.item(),
            'shared_mi_z2_y': mi_z2_y.item(),
            'shared_diversity': diversity_loss.item()
        }
        
        return loss, loss_dict
    
    def forward(self, disentangle_outputs, labels):
        """
        计算完整的解纠缠损失
        
        Args:
            disentangle_outputs: 字典，包含：
                - 'H1': [batch_size, 384] EMG原始特征
                - 'H2': [batch_size, 1152] IMU原始特征
                - 'Z1': [batch_size, d_shared] EMG共享表征
                - 'U1': [batch_size, d_private] EMG独特表征
                - 'Z2': [batch_size, d_shared] IMU共享表征
                - 'U2': [batch_size, d_private] IMU独特表征
            labels: [batch_size] 标签
        
        Returns:
            L_private: 标量，独特损失
            L_shared: 标量，共享损失
            loss_dict: 字典，包含所有损失项的详细信息
        """
        # 提取表征
        H1 = disentangle_outputs['H1']
        H2 = disentangle_outputs['H2']
        Z1 = disentangle_outputs['Z1']
        U1 = disentangle_outputs['U1']
        Z2 = disentangle_outputs['Z2']
        U2 = disentangle_outputs['U2']
        
        # 计算独特损失（包含H1和H2用于方案A）
        L_private, private_dict = self.compute_private_loss(U1, U2, Z1, Z2, H1, H2, labels)
        
        # 计算共享损失
        L_shared, shared_dict = self.compute_shared_loss(Z1, Z2, labels)
        
        # 合并所有损失信息
        loss_dict = {**private_dict, **shared_dict}
        
        return L_private, L_shared, loss_dict


class SimplifiedDisentangleLoss(nn.Module):
    """
    简化版解纠缠损失（用于快速实验和消融研究）
    
    只包含最核心的约束：
    - 对齐: d(Z1, Z2)
    - 独立: I(U;Z)
    
    Args:
        lambda_align: 对齐权重
        lambda_indep: 独立性权重
        d_shared: 共享表征维度
        d_private: 独特表征维度
    """
    def __init__(self, lambda_align=1.0, lambda_indep=0.5, 
                 d_shared=128, d_private=64):
        super(SimplifiedDisentangleLoss, self).__init__()
        
        self.lambda_align = lambda_align
        self.lambda_indep = lambda_indep
        
        # 只使用CLUB估计器
        self.club_emg = CLUB(x_dim=d_private, y_dim=d_shared, hidden_dim=128)
        self.club_imu = CLUB(x_dim=d_private, y_dim=d_shared, hidden_dim=128)
    
    def forward(self, disentangle_outputs, labels):
        """
        计算简化的解纠缠损失
        
        Returns:
            loss: 标量，总损失
            loss_dict: 字典，损失详情
        """
        Z1 = disentangle_outputs['Z1']
        U1 = disentangle_outputs['U1']
        Z2 = disentangle_outputs['Z2']
        U2 = disentangle_outputs['U2']
        
        # 对齐损失
        align_loss = F.mse_loss(Z1, Z2)
        
        # 独立性损失
        indep_loss = self.club_emg(U1, Z1) + self.club_imu(U2, Z2)
        
        # 总损失
        total_loss = self.lambda_align * align_loss + self.lambda_indep * indep_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'alignment': align_loss.item(),
            'independence': indep_loss.item()
        }
        
        return total_loss, loss_dict

