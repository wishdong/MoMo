"""
RMSCM: 归多尺度卷积模块
用于EMG和IMU的多模态手势识别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    基础卷积块
    包含：Conv2d -> BatchNorm -> ReLU -> Dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dropout=0.5):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=kernel_size, 
                             padding=padding, 
                             bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MultiScaleConv(nn.Module):
    """
    多尺度卷积（MSC）模块
    使用4个不同尺度的卷积核提取短期时间特征
    """
    def __init__(self, in_channels, out_channels=64, dropout=0.5):
        super(MultiScaleConv, self).__init__()
        
        # 4个不同尺度的卷积块
        # 卷积核尺寸：1×1, 5×5, 7×7, 9×9
        self.conv_block1 = ConvBlock(in_channels, out_channels, 
                                     kernel_size=(1, 1), padding=0, dropout=dropout)
        self.conv_block2 = ConvBlock(in_channels, out_channels, 
                                     kernel_size=(5, 1), padding=(2, 0), dropout=dropout)
        self.conv_block3 = ConvBlock(in_channels, out_channels, 
                                     kernel_size=(7, 1), padding=(3, 0), dropout=dropout)
        self.conv_block4 = ConvBlock(in_channels, out_channels, 
                                     kernel_size=(9, 1), padding=(4, 0), dropout=dropout)
        
    def forward(self, x):
        # 提取不同尺度的短期特征
        st_b1 = self.conv_block1(x)  # (N, 64, T, V)
        st_b2 = self.conv_block2(x)  # (N, 64, T, V)
        st_b3 = self.conv_block3(x)  # (N, 64, T, V)
        st_b4 = self.conv_block4(x)  # (N, 64, T, V)
        
        return st_b1, st_b2, st_b3, st_b4


class CNNBiLSTM(nn.Module):
    """
    CNN-BiLSTM模块
    用于提取长期时间依赖特征
    结构：ConvBlock5 -> BiLSTM -> ConvBlock6 -> ConvBlock7
    """
    def __init__(self, in_channels, feature_dim=64, hidden_dim=64, dropout=0.5):
        super(CNNBiLSTM, self).__init__()
        
        self.feature_dim = feature_dim
        
        # ConvBlock5: 浅层特征压缩
        self.conv_block5 = ConvBlock(in_channels, feature_dim, 
                                     kernel_size=(3, 1), padding=(1, 0), dropout=dropout)
        
        # BiLSTM: 双向LSTM提取长期依赖
        self.bilstm = nn.LSTM(feature_dim, hidden_dim, 
                             num_layers=1, 
                             batch_first=True, 
                             bidirectional=True,
                             dropout=0 if dropout == 0 else dropout)
        
        # ConvBlock6-7: 调整特征维度
        self.conv_block6 = ConvBlock(hidden_dim * 2, feature_dim, 
                                     kernel_size=(3, 1), padding=(1, 0), dropout=dropout)
        self.conv_block7 = ConvBlock(feature_dim, feature_dim, 
                                     kernel_size=(3, 1), padding=(1, 0), dropout=dropout)
        
    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        
        # ConvBlock5: 特征压缩
        x = self.conv_block5(x)  # (N, 64, T, V)
        
        # 准备BiLSTM输入：(N, V, T, 64) -> (N*V, T, 64)
        x = x.permute(0, 3, 2, 1).contiguous()  # (N, V, T, 64)
        x = x.view(N * V, T, self.feature_dim)  # (N*V, T, 64)
        
        # BiLSTM
        x, _ = self.bilstm(x)  # (N*V, T, 128)
        
        # 恢复形状：(N*V, T, 128) -> (N, 128, T, V)
        x = x.view(N, V, T, -1)  # (N, V, T, 128)
        x = x.permute(0, 3, 2, 1).contiguous()  # (N, 128, T, V)
        
        # ConvBlock6-7: 调整维度
        x = self.conv_block6(x)  # (N, 64, T, V)
        x = self.conv_block7(x)  # (N, 64, T, V)
        
        return x


class RMSCM(nn.Module):
    """
    归多尺度卷积模块（Recurrent Multi-Scale Convolutional Module）
    结合多尺度卷积（MSC）和CNN-BiLSTM提取短-长时时间特征
    """
    def __init__(self, in_channels, feature_dim=64, hidden_dim=64, dropout=0.5):
        super(RMSCM, self).__init__()
        
        # 多尺度卷积：提取短期特征
        self.msc = MultiScaleConv(in_channels, feature_dim, dropout)
        
        # CNN-BiLSTM：提取长期特征
        self.cnn_bilstm = CNNBiLSTM(in_channels, feature_dim, hidden_dim, dropout)
        
    def forward(self, x):
        # x: (N, C, T, V)
        
        # 多尺度卷积：短期特征
        st_b1, st_b2, st_b3, st_b4 = self.msc(x)
        
        # CNN-BiLSTM：长期特征
        lt = self.cnn_bilstm(x)
        
        # 特征融合：拼接短期和长期特征
        # LST = Concat(ST_b1, ST_b2, ST_b3, LT)
        # 注意：这里使用st_b1, st_b2, st_b3（不包含st_b4）
        lst = torch.cat([st_b1, st_b2, st_b3, lt], dim=1)  # (N, 4*64=256, T, V)
        
        return lst


class Classifier(nn.Module):
    """
    分类器
    用于将特征映射到类别logits
    """
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super(Classifier, self).__init__()
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x: (N, C, T, V)
        x = self.gap(x)  # (N, C, 1, 1)
        x = x.view(x.size(0), -1)  # (N, C)
        x = self.fc(x)  # (N, num_classes)
        return x


class MultiModalRMSCM(nn.Module):
    """
    多模态RMSCM网络
    
    架构：
    1. 各模态特征提取：两个独立的RMSCM（EMG和IMU）
    2. 多模态特征融合：Concat
    3. 多阶段分类器：
       - 单模态分类器1（EMG）
       - 单模态分类器2（IMU）
       - 多模态分类器（融合特征）
    4. 决策级融合：logit1 + logit2 + logit3
    """
    def __init__(self, emg_channels=12, imu_channels=36, num_classes=50, 
                 feature_dim=64, hidden_dim=64, dropout=0.5):
        """
        参数:
            emg_channels: EMG通道数（默认12）
            imu_channels: IMU通道数（默认36）
            num_classes: 类别数（默认50）
            feature_dim: 特征维度（默认64）
            hidden_dim: LSTM隐藏层维度（默认64）
            dropout: Dropout比例（默认0.5）
        """
        super(MultiModalRMSCM, self).__init__()
        
        # 1. 各模态特征提取
        # 输入需要reshape为 (N, 1, T, C) 格式用于Conv2d
        self.rmscm_emg = RMSCM(in_channels=1, feature_dim=feature_dim, 
                               hidden_dim=hidden_dim, dropout=dropout)
        self.rmscm_imu = RMSCM(in_channels=1, feature_dim=feature_dim, 
                               hidden_dim=hidden_dim, dropout=dropout)
        
        # RMSCM输出特征维度：4 * feature_dim
        rmscm_out_dim = 4 * feature_dim
        
        # 2. 单模态分类器
        self.classifier_emg = Classifier(rmscm_out_dim, num_classes, dropout)
        self.classifier_imu = Classifier(rmscm_out_dim, num_classes, dropout)
        
        # 3. 多模态分类器
        # 融合特征维度：2 * rmscm_out_dim
        self.classifier_fusion = Classifier(2 * rmscm_out_dim, num_classes, dropout)
        
    def forward(self, emg, imu):
        """
        前向传播
        
        参数:
            emg: EMG数据 (N, T, C_emg) - (batch_size, 400, 12)
            imu: IMU数据 (N, T, C_imu) - (batch_size, 400, 36)
            
        返回:
            logit_emg: EMG分类器输出 (N, num_classes)
            logit_imu: IMU分类器输出 (N, num_classes)
            logit_fusion: 融合分类器输出 (N, num_classes)
            logit_final: 决策级融合输出 (N, num_classes)
        """
        # 调整输入形状为Conv2d格式：(N, T, C) -> (N, 1, T, C)
        emg = emg.unsqueeze(1)  # (N, 1, T, C_emg)
        imu = imu.unsqueeze(1)  # (N, 1, T, C_imu)
        
        # 1. 各模态特征提取
        feat_emg = self.rmscm_emg(emg)  # (N, 256, T, C_emg)
        feat_imu = self.rmscm_imu(imu)  # (N, 256, T, C_imu)
        
        # 2. 单模态分类
        logit_emg = self.classifier_emg(feat_emg)  # (N, num_classes)
        logit_imu = self.classifier_imu(feat_imu)  # (N, num_classes)
        
        # 3. 多模态特征融合
        # 先对时间和通道维度进行全局平均池化，然后concat
        feat_emg_pooled = F.adaptive_avg_pool2d(feat_emg, (1, 1))  # (N, 256, 1, 1)
        feat_imu_pooled = F.adaptive_avg_pool2d(feat_imu, (1, 1))  # (N, 256, 1, 1)
        
        # 拼接特征
        feat_fusion = torch.cat([feat_emg_pooled, feat_imu_pooled], dim=1)  # (N, 512, 1, 1)
        
        # 4. 多模态分类
        logit_fusion = self.classifier_fusion(feat_fusion)  # (N, num_classes)
        
        # 5. 决策级融合
        logit_final = logit_emg + logit_imu + logit_fusion  # (N, num_classes)
        
        return logit_emg, logit_imu, logit_fusion, logit_final


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    
    结合三个分类器的损失：
    Loss = α * Loss_emg + β * Loss_imu + γ * Loss_fusion
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        """
        参数:
            alpha: EMG分类器损失权重
            beta: IMU分类器损失权重
            gamma: 融合分类器损失权重
        """
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, logit_emg, logit_imu, logit_fusion, labels):
        """
        计算多任务损失
        
        参数:
            logit_emg: EMG分类器输出 (N, num_classes)
            logit_imu: IMU分类器输出 (N, num_classes)
            logit_fusion: 融合分类器输出 (N, num_classes)
            labels: 真实标签 (N,)
            
        返回:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 计算各分类器的交叉熵损失
        loss_emg = self.ce_loss(logit_emg, labels)
        loss_imu = self.ce_loss(logit_imu, labels)
        loss_fusion = self.ce_loss(logit_fusion, labels)
        
        # 加权求和
        total_loss = (self.alpha * loss_emg + 
                     self.beta * loss_imu + 
                     self.gamma * loss_fusion)
        
        # 返回总损失和各项损失
        loss_dict = {
            'total': total_loss.item(),
            'emg': loss_emg.item(),
            'imu': loss_imu.item(),
            'fusion': loss_fusion.item()
        }
        
        return total_loss, loss_dict


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model, emg_channels=12, imu_channels=36, window_size=400):
    """
    获取模型信息
    
    参数:
        model: 模型实例
        emg_channels: EMG通道数
        imu_channels: IMU通道数
        window_size: 窗口大小
    """
    total_params = count_parameters(model)
    
    # 获取模型所在的设备
    device = next(model.parameters()).device
    
    # 创建示例输入并移动到模型设备
    emg_dummy = torch.randn(1, window_size, emg_channels).to(device)
    imu_dummy = torch.randn(1, window_size, imu_channels).to(device)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        logit_emg, logit_imu, logit_fusion, logit_final = model(emg_dummy, imu_dummy)
    
    info = {
        'total_parameters': total_params,
        'total_parameters_M': total_params / 1e6,
        'input_shape_emg': tuple(emg_dummy.shape),
        'input_shape_imu': tuple(imu_dummy.shape),
        'output_shape_emg': tuple(logit_emg.shape),
        'output_shape_imu': tuple(logit_imu.shape),
        'output_shape_fusion': tuple(logit_fusion.shape),
        'output_shape_final': tuple(logit_final.shape)
    }
    
    return info
