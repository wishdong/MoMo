"""
解纠缠模块：共享编码器和独特编码器

用于将模态特征H分解为：
- 共享表征Z：跨模态共享的语义信息
- 独特表征U：模态特有的独特信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedEncoder(nn.Module):
    """
    共享编码器：提取跨模态共享的语义信息
    
    架构：
        输入 → Linear(256) → ReLU → Dropout → Linear(output_dim) → L2 Normalize
    
    Args:
        input_dim: 输入特征维度 (例如 32*12=384 for EMG, 32*36=1152 for IMU)
        output_dim: 输出共享表征维度 (默认128)
        hidden_dim: 隐藏层维度 (默认256)
        dropout: Dropout率 (默认0.3)
    """
    def __init__(self, input_dim, output_dim=128, hidden_dim=256, dropout=0.3):
        super(SharedEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
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
            z: [batch_size, output_dim] L2归一化的共享表征
        """
        z = self.encoder(x)
        # L2归一化：有助于对比学习和距离度量
        z = F.normalize(z, p=2, dim=1)
        return z


class PrivateEncoder(nn.Module):
    """
    独特编码器：提取模态特有的独特信息
    
    架构：
        输入 → Linear(128) → ReLU → Dropout → Linear(output_dim) → L2 Normalize
    
    Args:
        input_dim: 输入特征维度
        output_dim: 输出独特表征维度 (默认64)
        hidden_dim: 隐藏层维度 (默认128)
        dropout: Dropout率 (默认0.3)
    """
    def __init__(self, input_dim, output_dim=64, hidden_dim=128, dropout=0.3):
        super(PrivateEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
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
            u: [batch_size, output_dim] L2归一化的独特表征
        """
        u = self.encoder(x)
        # L2归一化
        u = F.normalize(u, p=2, dim=1)
        return u


class ProjectionHead(nn.Module):
    """
    投影头：用于对比学习的额外投影层（可选）
    
    参考SimCLR/MoCo，投影头可以提升对比学习效果
    
    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度 (默认128)
        output_dim: 输出维度 (默认64)
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super(ProjectionHead, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
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
            proj: [batch_size, output_dim] 投影后的表征
        """
        proj = self.projection(x)
        proj = F.normalize(proj, p=2, dim=1)
        return proj

