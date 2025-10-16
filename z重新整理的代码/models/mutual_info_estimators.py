"""
互信息估计器模块

实现多种互信息估计方法：
1. InfoNCE：基于对比学习的互信息下界估计（用于最大化）
2. CLUB：基于变分上界的互信息估计（用于最小化）
3. HSIC：基于核方法的独立性度量
4. Barlow Twins：协方差矩阵去相关（防止表征坍缩）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InfoNCE(nn.Module):
    """
    InfoNCE估计器：用于最大化互信息 I(X;Y)
    
    基于对比学习，通过拉近正样本、推远负样本来最大化互信息下界
    
    适用场景：
    - I(U;Y): 最大化独特信息与标签的互信息
    - I(Z;Y): 最大化共享信息与标签的互信息
    
    Args:
        temperature: 温度参数，控制对比学习的难度 (默认0.07)
    """
    def __init__(self, temperature=0.07):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        计算InfoNCE损失（返回负值，因为要最大化MI）
        
        Args:
            features: [batch_size, feature_dim] 特征表征
            labels: [batch_size] 标签
        
        Returns:
            loss: 标量，InfoNCE损失（负的互信息下界）
        """
        batch_size = features.shape[0]
        
        # 归一化特征
        features = F.normalize(features, p=2, dim=1)
        
        # 计算相似度矩阵: [batch_size, batch_size]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 构建正样本mask: 同类别为1，不同类别为0
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # 去除对角线（自己和自己）
        mask = mask - torch.eye(batch_size, device=features.device)
        
        # 计算每个样本的正样本数量
        positive_counts = mask.sum(dim=1)
        
        # 对于没有正样本的样本，避免除零
        positive_counts = torch.clamp(positive_counts, min=1.0)
        
        # 计算InfoNCE损失
        # exp_sim: [batch_size, batch_size]
        exp_sim = torch.exp(similarity_matrix)
        
        # 分子：正样本的相似度之和
        numerator = (exp_sim * mask).sum(dim=1)
        
        # 分母：所有样本的相似度之和（除了自己）
        denominator = exp_sim.sum(dim=1) - torch.diag(exp_sim)
        
        # 避免除零
        denominator = torch.clamp(denominator, min=1e-8)
        
        # 平均每个正样本的损失
        loss = -torch.log(numerator / denominator / positive_counts + 1e-8).mean()
        
        return loss


class ConditionalCLUB(nn.Module):
    """
    条件CLUB估计器：用于估计条件互信息 I(X;Y|Z) 的上界
    
    适用场景：
    - I(H1;H2|Z): 给定共享表征Z，最小化原始特征H1和H2之间的条件互信息
    
    原理：
        I(X;Y|Z) ≤ E[log q(y|x,z)] - E[log q(y|x',z)]
        其中 q(y|x,z) 是条件分布，建模为高斯分布
    
    Args:
        x_dim: X的维度
        y_dim: Y的维度
        z_dim: 条件变量Z的维度
        hidden_dim: 隐藏层维度
    """
    def __init__(self, x_dim, y_dim, z_dim, hidden_dim=256):
        super(ConditionalCLUB, self).__init__()
        
        # 变分分布 q(y|x,z) 建模为高斯分布
        # 输入: [x, z] 拼接，输出: 均值和对数方差
        self.q_network = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim * 2)  # 输出均值和log_var
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, y, z):
        """
        计算条件CLUB上界 I(X;Y|Z)
        
        Args:
            x: [batch_size, x_dim]
            y: [batch_size, y_dim]
            z: [batch_size, z_dim] 条件变量
        
        Returns:
            mi_upper_bound: 标量，条件互信息上界
        """
        batch_size = x.shape[0]
        
        # 拼接 x 和 z 作为输入
        xz = torch.cat([x, z], dim=1)  # [batch_size, x_dim+z_dim]
        
        # 通过网络预测 y 的条件分布参数 q(y|x,z)
        params = self.q_network(xz)  # [batch_size, y_dim*2]
        y_dim = y.shape[1]
        
        mu = params[:, :y_dim]  # 均值
        log_var = params[:, y_dim:]  # 对数方差
        
        # 限制log_var范围，避免数值不稳定
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        # 计算负对数似然 -log q(y|x,z) - 正样本对
        # 注意：这里计算的是负对数似然（正值）
        positive_nll = 0.5 * (
            ((y - mu) ** 2) / torch.exp(log_var) + 
            log_var + 
            math.log(2 * math.pi)
        ).sum(dim=1).mean()
        
        # 计算负对数似然 -log q(y|x',z) - 负样本对
        # 通过随机打乱 x 来构造负样本（保持z不变）
        x_shuffle = x[torch.randperm(batch_size)]
        xz_shuffle = torch.cat([x_shuffle, z], dim=1)
        
        params_shuffle = self.q_network(xz_shuffle)
        mu_shuffle = params_shuffle[:, :y_dim]
        log_var_shuffle = params_shuffle[:, y_dim:]
        log_var_shuffle = torch.clamp(log_var_shuffle, min=-10, max=10)
        
        negative_nll = 0.5 * (
            ((y - mu_shuffle) ** 2) / torch.exp(log_var_shuffle) + 
            log_var_shuffle + 
            math.log(2 * math.pi)
        ).sum(dim=1).mean()
        
        # CLUB上界: E[-log q(y|x,z)] - E[-log q(y|x',z)]
        # 正确的互信息上界（应该是正值）
        mi_upper_bound = negative_nll - positive_nll
        
        # 确保返回非负值（互信息上界应该≥0）
        mi_upper_bound = torch.clamp(mi_upper_bound, min=0.0)
        
        return mi_upper_bound


class CLUB(nn.Module):
    """
    CLUB (Contrastive Log-ratio Upper Bound) 估计器
    用于最小化互信息 I(X;Y) 的变分上界
    
    适用场景：
    - I(U;Z): 最小化独特信息与共享信息的互信息（独立性约束）
    
    原理：
        I(X;Y) ≤ E[log q(y|x)] - E[log q(y|x')]
        其中 q(y|x) 是一个变分分布，x' 是从边缘分布采样的负样本
    
    Args:
        x_dim: X的维度
        y_dim: Y的维度
        hidden_dim: 隐藏层维度
    """
    def __init__(self, x_dim, y_dim, hidden_dim=256):
        super(CLUB, self).__init__()
        
        # 变分分布 q(y|x) 建模为高斯分布
        # 网络输出均值和对数方差
        self.q_network = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim * 2)  # 输出均值和log_var
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, y):
        """
        计算CLUB上界
        
        Args:
            x: [batch_size, x_dim]
            y: [batch_size, y_dim]
        
        Returns:
            mi_upper_bound: 标量，互信息上界
        """
        batch_size = x.shape[0]
        
        # 通过网络预测 y 的分布参数
        params = self.q_network(x)  # [batch_size, y_dim*2]
        y_dim = y.shape[1]
        
        mu = params[:, :y_dim]  # 均值
        log_var = params[:, y_dim:]  # 对数方差
        
        # 限制log_var范围，避免数值不稳定
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        # 计算负对数似然 -log q(y|x) - 正样本对
        # 注意：这里计算的是负对数似然（正值）
        positive_nll = 0.5 * (
            ((y - mu) ** 2) / torch.exp(log_var) + 
            log_var + 
            math.log(2 * math.pi)
        ).sum(dim=1).mean()
        
        # 计算负对数似然 -log q(y|x') - 负样本对
        # 通过随机打乱y来构造负样本
        y_shuffle = y[torch.randperm(batch_size)]
        
        negative_nll = 0.5 * (
            ((y_shuffle - mu) ** 2) / torch.exp(log_var) + 
            log_var + 
            math.log(2 * math.pi)
        ).sum(dim=1).mean()
        
        # CLUB上界: E[-log q(y|x)] - E[-log q(y|x')]
        # 正确的互信息上界（应该是正值）
        mi_upper_bound = negative_nll - positive_nll
        
        # 确保返回非负值（互信息上界应该≥0）
        mi_upper_bound = torch.clamp(mi_upper_bound, min=0.0)
        
        return mi_upper_bound


class HSIC(nn.Module):
    """
    HSIC (Hilbert-Schmidt Independence Criterion) 独立性度量
    基于核方法的独立性检验，用于最小化两个变量之间的依赖关系
    
    适用场景：
    - I(U1;U2): 最小化不同模态独特信息之间的依赖
    
    原理：
        HSIC(X,Y) = (1/(n-1)^2) * tr(KHLH)
        其中 K, L 是核矩阵，H 是中心化矩阵
    
    Args:
        kernel_type: 核函数类型 ('rbf' 或 'linear')
        sigma: RBF核的带宽参数（如果使用RBF核）
    """
    def __init__(self, kernel_type='rbf', sigma=None):
        super(HSIC, self).__init__()
        self.kernel_type = kernel_type
        self.sigma = sigma
    
    def _compute_kernel(self, x):
        """
        计算核矩阵
        
        Args:
            x: [batch_size, dim]
        
        Returns:
            K: [batch_size, batch_size] 核矩阵
        """
        if self.kernel_type == 'linear':
            # 线性核: K = XX^T
            K = torch.matmul(x, x.T)
        elif self.kernel_type == 'rbf':
            # RBF核: K_ij = exp(-||x_i - x_j||^2 / (2*sigma^2))
            # 计算成对距离
            x_norm = (x ** 2).sum(dim=1).view(-1, 1)
            dist_sq = x_norm + x_norm.T - 2 * torch.matmul(x, x.T)
            
            # 自动设置sigma（如果未指定）
            if self.sigma is None:
                sigma = torch.median(torch.sqrt(dist_sq + 1e-8))
            else:
                sigma = self.sigma
            
            K = torch.exp(-dist_sq / (2 * sigma ** 2))
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        return K
    
    def forward(self, x, y):
        """
        计算HSIC独立性度量
        
        Args:
            x: [batch_size, x_dim]
            y: [batch_size, y_dim]
        
        Returns:
            hsic: 标量，HSIC值（越小越独立）
        """
        batch_size = x.shape[0]
        
        # 计算核矩阵
        K = self._compute_kernel(x)  # [batch_size, batch_size]
        L = self._compute_kernel(y)  # [batch_size, batch_size]
        
        # 中心化矩阵 H = I - (1/n)*11^T
        H = torch.eye(batch_size, device=x.device) - torch.ones(batch_size, batch_size, device=x.device) / batch_size
        
        # HSIC = (1/(n-1)^2) * tr(KHLH)
        KH = torch.matmul(K, H)
        LH = torch.matmul(L, H)
        hsic = torch.trace(torch.matmul(KH, LH)) / ((batch_size - 1) ** 2)
        
        return hsic


class BarlowTwins(nn.Module):
    """
    Barlow Twins 去冗余损失
    通过最小化跨模态特征的交叉相关矩阵与单位矩阵的差异来去除冗余
    
    适用场景：
    - R_diversity(Z1, Z2): 确保共享表征的多样性，防止坍缩
    
    原理：
        L = Σ_i (1 - C_ii)^2 + λ * Σ_i Σ_{j≠i} C_ij^2
        其中 C 是归一化的交叉相关矩阵
    
    Args:
        lambda_off_diag: 非对角项的权重（默认0.005）
    """
    def __init__(self, lambda_off_diag=0.005):
        super(BarlowTwins, self).__init__()
        self.lambda_off_diag = lambda_off_diag
    
    def forward(self, z1, z2):
        """
        计算Barlow Twins损失
        
        Args:
            z1: [batch_size, dim] 模态1的表征
            z2: [batch_size, dim] 模态2的表征
        
        Returns:
            loss: 标量，Barlow Twins损失
        """
        batch_size = z1.shape[0]
        feature_dim = z1.shape[1]
        
        # 沿batch维度标准化（零均值，单位方差）
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-8)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-8)
        
        # 计算交叉相关矩阵 C = (1/N) * Z1^T * Z2
        # C: [feature_dim, feature_dim]
        cross_corr = torch.matmul(z1_norm.T, z2_norm) / batch_size
        
        # 对角项损失：希望对角元素接近1（完全相关）
        on_diag = torch.diagonal(cross_corr)
        on_diag_loss = ((1 - on_diag) ** 2).sum()
        
        # 非对角项损失：希望非对角元素接近0（不相关）
        off_diag_mask = ~torch.eye(feature_dim, dtype=torch.bool, device=z1.device)
        off_diag = cross_corr[off_diag_mask]
        off_diag_loss = (off_diag ** 2).sum()
        
        # 总损失
        loss = on_diag_loss + self.lambda_off_diag * off_diag_loss
        
        return loss


class CovarianceRegularizer(nn.Module):
    """
    协方差正则化器：简化版的去冗余方法
    直接惩罚特征维度之间的协方差（除了方差）
    
    适用场景：
    - 作为Barlow Twins的轻量级替代
    
    Args:
        lambda_cov: 协方差惩罚权重
    """
    def __init__(self, lambda_cov=0.01):
        super(CovarianceRegularizer, self).__init__()
        self.lambda_cov = lambda_cov
    
    def forward(self, z):
        """
        计算协方差正则损失
        
        Args:
            z: [batch_size, dim]
        
        Returns:
            loss: 标量，协方差正则损失
        """
        batch_size = z.shape[0]
        
        # 中心化
        z_centered = z - z.mean(dim=0, keepdim=True)
        
        # 计算协方差矩阵
        cov = torch.matmul(z_centered.T, z_centered) / (batch_size - 1)
        
        # 惩罚非对角元素
        feature_dim = z.shape[1]
        off_diag_mask = ~torch.eye(feature_dim, dtype=torch.bool, device=z.device)
        off_diag_cov = cov[off_diag_mask]
        
        loss = self.lambda_cov * (off_diag_cov ** 2).sum()
        
        return loss

