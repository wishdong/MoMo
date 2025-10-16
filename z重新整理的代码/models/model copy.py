"""
模型定义：EMG-IMU多模态手势识别网络
架构：三路并行（IMU单模态 + EMG单模态 + 多模态融合）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

initializer = nn.init.xavier_uniform_


# ==================== 基础层定义 ====================

class ChebConv(nn.Module):
    """切比雪夫图卷积层"""
    def __init__(self, in_channels, out_channels, K):
        super(ChebConv, self).__init__()
        self.K = K  # 多项式阶数
        self.theta = nn.Parameter(torch.Tensor(K+1, in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)

    def forward(self, x, L_tilde):
        T_0 = x
        out = torch.matmul(T_0, self.theta[0])
        if self.K > 0:
            T_1 = torch.matmul(L_tilde, x)
            out = out + torch.matmul(T_1, self.theta[1])
        for k in range(2, self.K + 1):
            T_2 = 2 * torch.matmul(L_tilde, T_1) - T_0
            out = out + torch.matmul(T_2, self.theta[k])
            T_0, T_1 = T_1, T_2
        return out


class GNN(nn.Module):
    """图神经网络层（自适应邻接矩阵 + 切比雪夫GCN）"""
    def __init__(self, input_feat, output_feat, indicator):
        super(GNN, self).__init__()
        self.W_gnn = nn.Parameter(initializer(torch.randn(output_feat, output_feat)))
        self.B_gnn = nn.Parameter(torch.randn(output_feat))
        self.cheb = ChebConv(output_feat, output_feat, K=1)
        self.MHA = nn.MultiheadAttention(embed_dim=output_feat, num_heads=4, batch_first=True)
        self.indicator = indicator
        self.output_feat = output_feat

    def forward(self, x):
        B, T, C, F = x.shape  # [batch, time, channel, features]
        x = torch.transpose(x, 1, 2)
        x = x.reshape(B, C, -1)  # [batch, channel, time*features]
        
        # 使用多头注意力机制生成自适应邻接矩阵
        a, b = self.MHA(x, x, x)  # b是自适应邻接矩阵
        x = self.cheb(x, b)  # 切比雪夫GCN
        
        x = x.reshape(B, -1, C, 1)  # [batch, length, channel, 1]
        return torch.nn.functional.relu(x), b


class Temporal_layer(nn.Module):
    """时序卷积层（GLU门控机制）"""
    def __init__(self, in_dim, out_dim):
        super(Temporal_layer, self).__init__()
        self.WT_input = nn.Parameter(initializer(torch.randn(out_dim, in_dim, 1, 1)))
        self.WT_glu = nn.Parameter(initializer(torch.randn(out_dim*2, in_dim, 1, 1)))
        self.B_input = nn.Parameter(torch.FloatTensor(out_dim))
        self.B_glu = nn.Parameter(torch.FloatTensor(out_dim*2))
        self.out_dim = out_dim
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_input = F.conv2d(x, self.WT_input)
        x_glu = F.conv2d(x, self.WT_glu)
        return (x_glu[:, 0:self.out_dim, :, :] + x_input) * self.sigmoid(x_glu[:, -self.out_dim:, :, :])


class SEBlock(nn.Module):
    """通道注意力机制（Squeeze-and-Excitation）"""
    def __init__(self, channel, reduction=1):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()  # [N, C, L]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Spatial_layer(nn.Module):
    """空间卷积层（SE注意力 + GNN）"""
    def __init__(self, in_dim, out_dim, indicator, cc):
        super(Spatial_layer, self).__init__()
        self.WS_input = nn.Parameter(initializer(torch.randn(out_dim, in_dim, 1, 1)))
        self.B_input = nn.Parameter(torch.FloatTensor(out_dim))
        self.out_dim = out_dim
        self.gnn = GNN(in_dim, out_dim, indicator)
        self.se_block = SEBlock(channel=cc)

    def forward(self, x):
        batch, L, channel, _ = x.shape
        
        # SE注意力
        x = x.reshape(batch, channel, L)
        x_attention = self.se_block(x).reshape(batch, L, channel, 1)
        
        # 卷积
        x_input = F.conv2d(x_attention, self.WS_input)
        
        # GNN
        x_gnn, b = self.gnn(x_input)
        
        # 残差连接 + ReLU
        return F.relu(x_input + x_gnn), b


# ==================== 编码器定义 ====================

class Encoder(nn.Module):
    """单模态编码器（时空卷积 + GNN）"""
    def __init__(self, indicator, channel, channels_config):
        super(Encoder, self).__init__()
        first, second, third, fourth = channels_config  # 从配置读取
        self.Temp1 = Temporal_layer(first, first) 
        self.batch1 = nn.BatchNorm2d(first)
        self.Spat1 = Spatial_layer(first, second, indicator, channel)
        self.Temp2 = Temporal_layer(second, third)
        self.batch2 = nn.BatchNorm2d(third)
        self.Temp3 = Temporal_layer(third, fourth)
        self.batch3 = nn.BatchNorm2d(fourth)
        
    def forward(self, x):
        # 时间1
        x = self.Temp1(x)
        x = self.batch1(x)
        x_recon = x
        
        # 空间1
        x, attention_map1 = self.Spat1(x)
        
        # 时间2
        x = self.Temp2(x)
        x = self.batch2(x)
        
        # 时间3
        x = self.Temp3(x)
        x = self.batch3(x)
        
        return x, x_recon, attention_map1


# ==================== 融合模块定义 ====================

class ConcatFusion(nn.Module):
    """简单拼接融合模块（EMG与IMU直接拼接）"""
    def __init__(self, indicator=2, channel=48, channels_config=None, dropout=0.0):
        super(ConcatFusion, self).__init__()
        if channels_config is None:
            first, second, third, fourth = 400, 128, 64, 32
        else:
            first, second, third, fourth = channels_config
        self.fourth = fourth
        self.Temp = Temporal_layer(fourth, fourth)
        self.batch = nn.BatchNorm2d(fourth)
        # self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, imu, emg):
        # EMG与IMU直接拼接（不再重复3倍）
        emg = torch.squeeze(emg, dim=-1)  # [N, fourth, 12]
        imu = torch.squeeze(imu, dim=-1)  # [N, fourth, 36]
        
        # 直接concat
        feature = torch.cat((emg, imu), dim=2)  # [N, fourth, 48]
        x = torch.unsqueeze(feature, dim=-1)  # [N, fourth, 48, 1]
        
        # 时间层
        x = self.Temp(x)
        x = self.batch(x)
        # x = self.dropout(x)
        
        return x  # [bs, fourth, 48, 1]


class FusionEncoder(nn.Module):
    """融合编码器（简化版 - 直接使用拼接融合）"""
    def __init__(self, indicator=2, channels_config=None):
        super(FusionEncoder, self).__init__() 
        self.concat_fusion = ConcatFusion(channels_config=channels_config)
        
    def forward(self, imu, emg):
        # 直接返回拼接融合的输出
        x = self.concat_fusion(imu, emg)  # [bs, fourth, 48, 1]
        return x


# ==================== 分类器定义 ====================

class Classifier(nn.Module):
    """MLP分类器（简化为2层）"""
    def __init__(self, configs, enc_in):
        super(Classifier, self).__init__()
        first, second, third, fourth = configs.channels
        # 简化：输入 → 128 → 50
        self.MLP1 = nn.Linear(fourth * enc_in * 1, 128)
        self.MLP2 = nn.Linear(128, 50)
        self.drop1 = nn.Dropout(p=configs.dropout)
        
    def forward(self, x):
        bs = x.shape[0]
        x = x.reshape(bs, -1)
        
        x = self.MLP1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.MLP2(x)
        y = F.log_softmax(x, dim=1)
        return y


# ==================== 完整模型 ====================

class MultimodalGestureNet(nn.Module):
    """
    EMG-IMU多模态手势识别网络
    三路并行架构：
        1. IMU单模态分支
        2. EMG单模态分支  
        3. 多模态融合分支
    """
    def __init__(self, imu_configs, emg_configs):
        super(MultimodalGestureNet, self).__init__()
        
        # 编码器（传入channels配置）
        self.imu_encoder = Encoder(
            indicator=imu_configs.indicator, 
            channel=imu_configs.enc_in,
            channels_config=imu_configs.channels
        )
        self.emg_encoder = Encoder(
            indicator=emg_configs.indicator, 
            channel=emg_configs.enc_in,
            channels_config=emg_configs.channels
        )
        # 融合编码器使用统一的配置（两者的channels都一样）
        self.fusion_encoder = FusionEncoder(channels_config=imu_configs.channels)
        
        # 分类器
        self.imu_classifier = Classifier(imu_configs, enc_in=36)
        self.emg_classifier = Classifier(emg_configs, enc_in=12)
        # 融合分类器
        self.fusion_classifier = Classifier(imu_configs, enc_in=48)  # 36+12

    def forward(self, emg, imu):
        """
        Args:
            emg: [batch, 400, 12, 1]
            imu: [batch, 400, 36, 1]
            
        Returns:
            final_score: 三路分数加权求和 [batch, 50]
        """
        # 特征提取
        imu_feature, _, _ = self.imu_encoder(imu)  # [bs, 32, 36, 1]
        emg_feature, _, _ = self.emg_encoder(emg)  # [bs, 32, 12, 1]
        fusion_feature = self.fusion_encoder(imu_feature, emg_feature)  # [bs, 32, 48, 1]
        
        # 分类
        imu_score = self.imu_classifier(imu_feature)      # [bs, 50]
        emg_score = self.emg_classifier(emg_feature)      # [bs, 50]
        fusion_score = self.fusion_classifier(fusion_feature)  # [bs, 50]
        
        # 三路加权求和
        final_score = imu_score + emg_score + fusion_score
        
        return final_score