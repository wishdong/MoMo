"""
配置文件：定义EMG和IMU两种模态的模型配置
支持多数据集配置（DB2, DB3, DB5, DB7）
"""

# ==================== 数据集配置 ====================
DATASET_CONFIGS = {
    'DB2': {
        'name': 'NinaPro DB2',
        'num_class': 50,
        'emg_channels': 12,
        'imu_channels': 36,
        'num_subjects': 40,
        'available_subjects': list(range(1, 41)),  # S1-S40
        'sampling_rate': 2000,
        'description': '40 intact subjects'
    },
    'DB3': {
        'name': 'NinaPro DB3',
        'num_class': 50,
        'emg_channels': 12,
        'imu_channels': 36,
        'num_subjects': 6,
        'available_subjects': [2, 4, 5, 6, 9, 11],  # 实际可用的受试者
        'sampling_rate': 2000,
        'description': '6 amputated subjects (processed)'
    },
    'DB5': {
        'name': 'NinaPro DB5',
        'num_class': 53,
        'emg_channels': 16,
        'imu_channels': 3,  # 只有加速度计，没有陀螺仪
        'num_subjects': 10,
        'available_subjects': list(range(1, 11)),  # S1-S10
        'sampling_rate': 2000,
        'description': '10 intact subjects, 16 EMG + 3 IMU channels'
    },
    'DB7': {
        'name': 'NinaPro DB7',
        'num_class': 41,
        'emg_channels': 12,
        'imu_channels': 36,
        'num_subjects': 11,
        'available_subjects': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # 实际可用的受试者
        'sampling_rate': 2000,
        'description': '11 subjects (processed)'
    }
}


def get_dataset_config(dataset_name):
    """
    获取数据集配置
    
    Args:
        dataset_name: 数据集名称 ('DB2', 'DB3', 'DB5', 'DB7')
    
    Returns:
        dataset_config: 数据集配置字典
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"未知数据集: {dataset_name}. 支持的数据集: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]


class EMG_Configs:
    """EMG信号配置（动态，根据数据集调整）"""
    def __init__(self, dataset='DB2'):
        dataset_config = get_dataset_config(dataset)
        self.dataset = dataset
        self.num_class = dataset_config['num_class']
        self.channels = [400, 128, 64, 32]  # 各层通道数
        self.enc_in = dataset_config['emg_channels']
        self.indicator = 2               # GNN类型（2表示使用自适应注意力机制）
        self.dropout = 0.5              # Dropout率
        self.modality = 'emg'           # 模态类型


class IMU_Configs:
    """IMU信号配置（动态，根据数据集调整）"""
    def __init__(self, dataset='DB2'):
        dataset_config = get_dataset_config(dataset)
        self.dataset = dataset
        self.num_class = dataset_config['num_class']
        self.channels = [400, 128, 64, 32]  # 各层通道数
        self.enc_in = dataset_config['imu_channels']
        self.indicator = 2               # GNN类型
        self.dropout = 0.5              # Dropout率
        self.modality = 'imu'           # 模态类型


class DisentangleConfigs:
    """解纠缠模块配置"""
    # ========== 编码器维度 ==========
    d_shared = 128              # 共享表征维度
    d_private = 64              # 独特表征维度
    dropout = 0.3               # 解纠缠编码器的Dropout率
    
    # ========== 损失权重配置方式 ==========
    # 提供两种配置方式，注释掉不用的即可：
    
    # ===== 配置方式1：简化版（推荐用于论文，只需调2个超参数） =====
    # 固定所有细粒度权重为1.0，减少超参数空间
    # 论文表述：We set λ1-λ4=1.0 for simplicity, only tune α and β
    # lambda1 = 1.0               # I(Ui;Y) 任务相关性权重 [固定]
    # lambda2 = 1.0               # I(Ui;Zi) 独立性权重 [固定]
    # lambda3 = 1.0               # d(Z1,Z2) 对齐权重 [固定]
    # lambda4 = 1.0               # I(Zi;Y) 任务相关性权重 [固定]
    # lambda5 = 0.05             # R_diversity 多样性权重 [固定，引用Barlow Twins]
    # lambda6 = 0.3               # I(U1;U2) 独特信息差异性权重 [方案B]
    # lambda7 = 0.3               # I(H1;H2|Z) 条件独立性权重 [方案A]
    # alpha = 0.3                 # L_private 总权重 [可调]
    # beta = 0.3                  # L_shared 总权重 [可调]
    # use_method_a = False        # 方案A：条件独立
    # use_method_b = True         # 方案B：直接约束
    
    # ===== 配置方式2：完整版（用于精细调优和消融实验） =====
    # 如需使用，取消下面的注释，并注释掉上面的配置方式1
    lambda1 = 0.1               # I(Ui;Y) 任务相关性权重
    lambda2 = 0.5               # I(Ui;Zi) 独立性权重
    lambda3 = 1.0               # d(Z1,Z2) 对齐权重
    lambda4 = 0.1               # I(Zi;Y) 任务相关性权重
    lambda5 = 0.05              # R_diversity 多样性权重
    lambda6 = 0.3               # I(U1;U2) 独特信息差异性权重 [方案B]
    lambda7 = 0.3               # I(H1;H2|Z) 条件独立性权重 [方案A]
    alpha = 0.5                 # L_private 总权重
    beta = 0.5                  # L_shared 总权重
    
    # ========== 独特信息差异性方案选择 ==========
    # use_method_a: 使用方案A (条件独立: I(H1;H2|Z1) + I(H1;H2|Z2))
    # use_method_b: 使用方案B (直接约束: I(U1;U2))
    # 可以同时启用两个方案进行对比
    use_method_a = False        # 方案A：条件独立（理论优雅，计算复杂）
    use_method_b = True         # 方案B：直接约束（简单直接）
    
    # ===== 配置方式3：验证 =====
    # 如需使用，取消下面的注释，并注释掉上面的配置方式1
    # lambda1 = 0.1               # I(Ui;Y) 任务相关性权重
    # lambda2 = 0.5               # I(Ui;Zi) 独立性权重
    # lambda3 = 1.0               # d(Z1,Z2) 对齐权重
     # lambda4 = 0.1               # I(Zi;Y) 任务相关性权重
     # lambda5 = 0.05              # R_diversity 多样性权重
     # lambda6 = 0.3               # I(U1;U2) 独特信息差异性权重 [方案B]
     # lambda7 = 0.3               # I(H1;H2|Z) 条件独立性权重 [方案A]
     # alpha = 0.5                 # L_private 总权重
     # beta = 0.5                  # L_shared 总权重
     # use_method_a = False        # 方案A：条件独立
     # use_method_b = True         # 方案B：直接约束
    
    # ========== InfoNCE参数 ==========
    temperature = 0.07          # 温度参数（控制对比学习难度，引用SimCLR/MoCo）
    
    # ========== 训练策略 ==========
    use_disentangle = True      # 是否启用解纠缠（默认True）
    warmup_epochs = 0           # 预热轮数（前N轮不使用解纠缠损失）


class AdaptiveFusionConfigs:
    """自适应融合模块配置（创新点2）"""
    # ========== 路由器配置 ==========
    use_adaptive_fusion = True   # 是否启用自适应融合
    router_hidden_dim = 256      # 路由器隐藏层维度
    router_dropout = 0.1         # 路由器Dropout率
    temperature = 1.0            # Softmax温度参数（控制权重分布的锐利度）
    
    # ========== 表征统一维度 ==========
    unified_dim = 128            # 统一投影维度（Z1, U1, Z2, U2都投影到此维度）
    
    # ========== 损失权重 ==========
    lambda_align = 0.1           # 权重-重要性对齐损失权重
    lambda_balance = 0.05        # 权重平衡损失权重
    
    # ========== 平衡策略 ==========
    balance_type = 'entropy'     # 平衡损失类型：'entropy'（负熵）或 'variance'（方差）
    min_weight = 0.05            # 最小权重阈值（防止某个表征完全被忽略）
    
    # ========== 训练策略 ==========
    warmup_epochs = 0            # 自适应融合预热轮数（前N轮不启用自适应融合损失）
    use_gradient_importance = True  # （已弃用）实际使用激活强度估计重要性，更稳定


