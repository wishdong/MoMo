"""
自适应融合模块（创新点2）
实现方案B：四专家单层路由（Four-Expert Single-Layer Routing）

核心思想：
1. 将Z1, U1, Z2, U2投影到统一维度
2. 使用路由器网络动态生成4个融合权重
3. 基于梯度重要性估计的对齐损失
4. 权重平衡损失防止退化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ProjectionLayer(nn.Module):
    """
    投影层：将不同维度的表征投影到统一维度
    
    作用：
    - Z1 [128维] → 统一维度 [unified_dim]
    - U1 [64维]  → 统一维度 [unified_dim]
    - Z2 [128维] → 统一维度 [unified_dim]
    - U2 [64维]  → 统一维度 [unified_dim]
    """
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            proj: [batch_size, output_dim]
        """
        return self.projection(x)


class RouterNetwork(nn.Module):
    """
    路由器网络：为4个专家（Z1, U1, Z2, U2）生成动态权重
    
    输入：4个表征的拼接
    输出：4个归一化的权重（和为1）
    """
    def __init__(self, unified_dim, hidden_dim=256, dropout=0.1, temperature=1.0):
        super(RouterNetwork, self).__init__()
        
        self.temperature = temperature
        
        # 路由器网络：输入4个表征的拼接 → 输出4个权重
        self.router = nn.Sequential(
            nn.Linear(unified_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 4)  # 输出4个logits
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, representations):
        """
        Args:
            representations: [batch_size, unified_dim * 4] 拼接后的4个表征
        
        Returns:
            weights: [batch_size, 4] Softmax归一化的权重
        """
        # 路由器前向传播
        logits = self.router(representations)  # [batch_size, 4]
        
        # Softmax归一化（使用温度参数控制分布锐利度）
        weights = F.softmax(logits / self.temperature, dim=1)  # [batch_size, 4]
        
        return weights


class AdaptiveFusionModule(nn.Module):
    """
    自适应融合模块（完整版）
    
    功能：
    1. 投影4个表征到统一维度
    2. 路由器生成动态融合权重
    3. 加权融合
    
    Args:
        d_shared: 共享表征维度（Z1, Z2的维度）
        d_private: 独特表征维度（U1, U2的维度）
        config: AdaptiveFusionConfigs配置对象
    """
    def __init__(self, d_shared=128, d_private=64, config=None):
        super(AdaptiveFusionModule, self).__init__()
        
        # 默认配置
        if config is None:
            from config import AdaptiveFusionConfigs
            config = AdaptiveFusionConfigs()
        
        self.config = config
        self.unified_dim = config.unified_dim
        self.use_gradient_importance = config.use_gradient_importance
        
        # ========== 投影层 ==========
        # Z1: 128 → unified_dim
        self.proj_z1 = ProjectionLayer(d_shared, config.unified_dim)
        # U1: 64 → unified_dim
        self.proj_u1 = ProjectionLayer(d_private, config.unified_dim)
        # Z2: 128 → unified_dim
        self.proj_z2 = ProjectionLayer(d_shared, config.unified_dim)
        # U2: 64 → unified_dim
        self.proj_u2 = ProjectionLayer(d_private, config.unified_dim)
        
        # ========== 路由器网络 ==========
        self.router = RouterNetwork(
            unified_dim=config.unified_dim,
            hidden_dim=config.router_hidden_dim,
            dropout=config.router_dropout,
            temperature=config.temperature
        )
        
        # ========== 用于梯度重要性计算的缓存 ==========
        self.cached_representations = None
        self.cached_weights = None
    
    def forward(self, Z1, U1, Z2, U2, return_weights=False):
        """
        前向传播
        
        Args:
            Z1: [batch_size, d_shared] EMG共享表征
            U1: [batch_size, d_private] EMG独特表征
            Z2: [batch_size, d_shared] IMU共享表征
            U2: [batch_size, d_private] IMU独特表征
            return_weights: 是否返回路由权重（用于可视化）
        
        Returns:
            fusion_feature: [batch_size, unified_dim] 融合后的特征
            weights: [batch_size, 4] 路由权重（如果return_weights=True）
        """
        batch_size = Z1.shape[0]
        
        # ========== 1. 投影到统一维度 ==========
        Z1_proj = self.proj_z1(Z1)  # [batch_size, unified_dim]
        U1_proj = self.proj_u1(U1)  # [batch_size, unified_dim]
        Z2_proj = self.proj_z2(Z2)  # [batch_size, unified_dim]
        U2_proj = self.proj_u2(U2)  # [batch_size, unified_dim]
        
        # ========== 2. 拼接所有表征 ==========
        all_representations = torch.cat([Z1_proj, U1_proj, Z2_proj, U2_proj], dim=1)
        # [batch_size, unified_dim * 4]
        
        # ========== 3. 路由器生成权重 ==========
        weights = self.router(all_representations)  # [batch_size, 4]
        
        # ========== 4. 加权融合 ==========
        # 方式1：加权求和
        # fusion = w1*Z1 + w2*U1 + w3*Z2 + w4*U2
        fusion_feature = (
            weights[:, 0:1] * Z1_proj +
            weights[:, 1:2] * U1_proj +
            weights[:, 2:3] * Z2_proj +
            weights[:, 3:4] * U2_proj
        )  # [batch_size, unified_dim]
        
        # ========== 5. 缓存（用于计算重要性和对齐损失）==========
        if self.training:
            # 缓存投影后的表征和权重
            self.cached_representations = {
                'Z1_proj': Z1_proj,
                'U1_proj': U1_proj,
                'Z2_proj': Z2_proj,
                'U2_proj': U2_proj
            }
            self.cached_weights = weights
        
        if return_weights:
            return fusion_feature, weights
        else:
            return fusion_feature


class AdaptiveFusionLoss(nn.Module):
    """
    自适应融合损失函数
    
    包含两部分：
    1. L_align: 权重-重要性对齐损失（基于梯度重要性）
    2. L_balance: 权重平衡损失（防止权重退化）
    
    Args:
        config: AdaptiveFusionConfigs配置对象
    """
    def __init__(self, config):
        super(AdaptiveFusionLoss, self).__init__()
        
        self.config = config
        self.lambda_align = config.lambda_align
        self.lambda_balance = config.lambda_balance
        self.balance_type = config.balance_type
        self.min_weight = config.min_weight
    
    def compute_activation_importance(self, representations):
        """
        计算激活重要性：基于每个表征的激活强度（L2范数）
        
        改进策略：使用激活强度而非梯度，更简单、更稳定
        直觉：激活越强的表征对任务越重要
        
        Args:
            representations: 字典，包含4个表征的Tensor
                - 'Z1_proj': [batch_size, unified_dim]
                - 'U1_proj': [batch_size, unified_dim]
                - 'Z2_proj': [batch_size, unified_dim]
                - 'U2_proj': [batch_size, unified_dim]
        
        Returns:
            importance: [batch_size, 4] 归一化的重要性分数
        """
        # 计算每个表征的L2范数（激活强度）
        activation_norms = []
        
        for name in ['Z1_proj', 'U1_proj', 'Z2_proj', 'U2_proj']:
            rep = representations[name]
            # 计算每个样本的L2范数
            norm = torch.norm(rep, p=2, dim=1)  # [batch_size]
            activation_norms.append(norm)
        
        # 拼接成 [batch_size, 4]
        importance = torch.stack(activation_norms, dim=1)  # [batch_size, 4]
        
        # Softmax归一化（转换为概率分布）
        importance = F.softmax(importance, dim=1)
        
        return importance
    
    def compute_alignment_loss(self, weights, importance):
        """
        计算权重-重要性对齐损失（双向KL散度）
        
        Args:
            weights: [batch_size, 4] 路由权重
            importance: [batch_size, 4] 梯度重要性
        
        Returns:
            loss: 标量，对齐损失
        """
        # 双向KL散度: KL(W||I) + KL(I||W)
        # KL(P||Q) = Σ P * log(P/Q)
        
        # 添加eps防止log(0)
        eps = 1e-8
        weights = weights + eps
        importance = importance + eps
        
        # KL(W||I)
        kl_w_i = torch.sum(weights * torch.log(weights / importance), dim=1)
        
        # KL(I||W)
        kl_i_w = torch.sum(importance * torch.log(importance / weights), dim=1)
        
        # 平均
        loss = (kl_w_i + kl_i_w).mean()
        
        return loss
    
    def compute_balance_loss(self, weights):
        """
        计算权重平衡损失（防止权重过度集中）
        
        Args:
            weights: [batch_size, 4] 路由权重
        
        Returns:
            loss: 标量，平衡损失
        """
        if self.balance_type == 'entropy':
            # 负熵损失：鼓励均匀分布
            # H(W) = -Σ W * log(W)
            eps = 1e-8
            entropy = -torch.sum(weights * torch.log(weights + eps), dim=1)  # [batch_size]
            # 负熵：熵越大越好，所以损失是-entropy
            loss = -entropy.mean()
            
        elif self.balance_type == 'variance':
            # 方差损失：惩罚权重方差
            # 理想均匀分布：[0.25, 0.25, 0.25, 0.25]，方差=0
            mean_weight = weights.mean(dim=1, keepdim=True)  # [batch_size, 1]
            variance = ((weights - mean_weight) ** 2).mean(dim=1)  # [batch_size]
            loss = variance.mean()
        
        else:
            raise ValueError(f"Unknown balance_type: {self.balance_type}")
        
        return loss
    
    def forward(self, fusion_module):
        """
        计算完整的自适应融合损失
        
        Args:
            fusion_module: AdaptiveFusionModule实例（包含缓存的表征、权重和融合输出）
        
        Returns:
            total_loss: 标量，总损失
            loss_dict: 字典，包含各项损失的详细信息
        """
        # ========== 1. 获取缓存的表征、权重和融合输出 ==========
        if not hasattr(fusion_module, 'cached_representations') or \
           fusion_module.cached_representations is None:
            # 如果没有缓存，返回零损失
            device = next(fusion_module.parameters()).device
            return torch.tensor(0.0, device=device), {
                'adaptive_fusion_total': 0.0,
                'adaptive_fusion_align': 0.0,
                'adaptive_fusion_balance': 0.0
            }
        
        representations = fusion_module.cached_representations
        weights = fusion_module.cached_weights
        
        # ========== 2. 计算激活重要性 ==========
        # 基于表征的激活强度估计其对任务的重要性
        importance = self.compute_activation_importance(representations)
        
        # ========== 3. 对齐损失 ==========
        L_align = self.compute_alignment_loss(weights, importance)
        
        # ========== 4. 平衡损失 ==========
        L_balance = self.compute_balance_loss(weights)
        
        # ========== 5. 总损失 ==========
        total_loss = self.lambda_align * L_align + self.lambda_balance * L_balance
        
        # ========== 6. 详细信息（用于监控）==========
        loss_dict = {
            'adaptive_fusion_total': total_loss.item(),
            'adaptive_fusion_align': L_align.item(),
            'adaptive_fusion_balance': L_balance.item(),
            # 额外监控：平均权重分布
            'weight_Z1_mean': weights[:, 0].mean().item(),
            'weight_U1_mean': weights[:, 1].mean().item(),
            'weight_Z2_mean': weights[:, 2].mean().item(),
            'weight_U2_mean': weights[:, 3].mean().item(),
            # 权重熵（越大越均匀）
            'weight_entropy': -torch.sum(weights * torch.log(weights + 1e-8), dim=1).mean().item()
        }
        
        return total_loss, loss_dict


# ==================== 辅助函数 ====================

def visualize_routing_weights(weights, labels=None, save_path=None):
    """
    可视化路由权重分布
    
    Args:
        weights: [batch_size, 4] 或 [num_samples, 4] 路由权重
        labels: [batch_size] 标签（可选，用于按类别分析）
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    weights_np = weights.detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 子图1: 平均权重分布（柱状图）
    ax = axes[0, 0]
    mean_weights = weights_np.mean(axis=0)
    ax.bar(['Z1(EMG共享)', 'U1(EMG独特)', 'Z2(IMU共享)', 'U2(IMU独特)'], mean_weights)
    ax.set_ylabel('平均权重')
    ax.set_title('平均路由权重分布')
    ax.set_ylim(0, 1)
    
    # 子图2: 权重分布（箱线图）
    ax = axes[0, 1]
    ax.boxplot([weights_np[:, i] for i in range(4)],
               labels=['Z1', 'U1', 'Z2', 'U2'])
    ax.set_ylabel('权重值')
    ax.set_title('权重分布（箱线图）')
    ax.grid(True, alpha=0.3)
    
    # 子图3: 权重相关性热图
    ax = axes[1, 0]
    corr = np.corrcoef(weights_np.T)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                xticklabels=['Z1', 'U1', 'Z2', 'U2'],
                yticklabels=['Z1', 'U1', 'Z2', 'U2'],
                ax=ax)
    ax.set_title('权重相关性矩阵')
    
    # 子图4: 按类别的权重分布（如果提供标签）
    ax = axes[1, 1]
    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        unique_labels = np.unique(labels_np)
        
        # 计算每个类别的平均权重
        class_weights = np.zeros((len(unique_labels), 4))
        for i, label in enumerate(unique_labels):
            mask = labels_np == label
            class_weights[i] = weights_np[mask].mean(axis=0)
        
        # 绘制堆叠条形图
        x = np.arange(min(10, len(unique_labels)))  # 只显示前10个类别
        width = 0.6
        
        ax.bar(x, class_weights[:10, 0], width, label='Z1')
        ax.bar(x, class_weights[:10, 1], width, bottom=class_weights[:10, 0], label='U1')
        ax.bar(x, class_weights[:10, 2], width, 
               bottom=class_weights[:10, 0] + class_weights[:10, 1], label='Z2')
        ax.bar(x, class_weights[:10, 3], width,
               bottom=class_weights[:10, 0] + class_weights[:10, 1] + class_weights[:10, 2],
               label='U2')
        
        ax.set_xlabel('手势类别')
        ax.set_ylabel('权重占比')
        ax.set_title('不同手势类别的权重分布')
        ax.legend()
    else:
        ax.text(0.5, 0.5, '未提供标签\n无法按类别分析', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"权重可视化已保存至: {save_path}")
    
    plt.show()