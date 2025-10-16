"""
主训练脚本
使用方法:
    # 默认：使用三分支损失
    python train.py --s 10 --gpu 0
    
    # 消融实验：只使用融合分支损失
    python train.py --s 10 --gpu 0 --no-branch-loss
    
训练策略: AdamW统一优化器 + 端到端联合训练
"""

import os
import random
import argparse
import numpy as np
import torch

from config import EMG_Configs, IMU_Configs, DisentangleConfigs, AdaptiveFusionConfigs
from models import (
    MultimodalGestureNet, 
    MultimodalGestureNetWithDisentangle,
    MultimodalGestureNetWithAdaptiveFusion
)
from data_loader import load_dataloader_for_both
from trainer import train_model


def set_seed(seed):
    """设置随机种子以保证可复现性"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    """主训练流程"""
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据
    print("=" * 50)
    print(f"加载受试者 {args.s} 的数据...")
    
    train_loader, val_loader, class_counts = load_dataloader_for_both(
        data_dir=args.data_dir,
        subject=args.s,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        add_test_to_train_ratio=args.ratio
    )
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    print(f"数据加载完成")
    
    # 构建模型
    print("=" * 50)
    print("构建模型...")
    
    # 初始化配置
    disentangle_config = None
    adaptive_fusion_config = None
    
    if args.use_adaptive_fusion:
        # 使用完整模型：解纠缠 + 自适应融合
        adaptive_fusion_config = AdaptiveFusionConfigs()
        
        # 可以通过命令行参数覆盖配置
        if hasattr(args, 'unified_dim') and args.unified_dim is not None:
            adaptive_fusion_config.unified_dim = args.unified_dim
        if hasattr(args, 'lambda_align') and args.lambda_align is not None:
            adaptive_fusion_config.lambda_align = args.lambda_align
        if hasattr(args, 'lambda_balance') and args.lambda_balance is not None:
            adaptive_fusion_config.lambda_balance = args.lambda_balance
        
        model = MultimodalGestureNetWithAdaptiveFusion(
            IMU_Configs, EMG_Configs,
            d_shared=args.d_shared,
            d_private=args.d_private,
            dropout=DisentangleConfigs.dropout,
            adaptive_fusion_config=adaptive_fusion_config
        )
        print(f"✓ 使用完整模型（解纠缠 + 自适应融合）")
        print(f"  - 共享表征维度: {args.d_shared}")
        print(f"  - 独特表征维度: {args.d_private}")
        print(f"  - 统一融合维度: {adaptive_fusion_config.unified_dim}")
        print(f"  - 路由器隐藏层: {adaptive_fusion_config.router_hidden_dim}")
        
        # 配置解纠缠参数（自适应融合模型也需要解纠缠损失）
        disentangle_config = DisentangleConfigs()
        if hasattr(args, 'alpha') and args.alpha is not None:
            disentangle_config.alpha = args.alpha
        if hasattr(args, 'beta') and args.beta is not None:
            disentangle_config.beta = args.beta
        disentangle_config.d_shared = args.d_shared
        disentangle_config.d_private = args.d_private
        
    elif args.use_disentangle:
        # 只使用解纠缠模型
        model = MultimodalGestureNetWithDisentangle(
            IMU_Configs, EMG_Configs,
            d_shared=args.d_shared,
            d_private=args.d_private,
            dropout=DisentangleConfigs.dropout
        )
        print(f"✓ 使用解纠缠模型")
        print(f"  - 共享表征维度: {args.d_shared}")
        print(f"  - 独特表征维度: {args.d_private}")
        
        # 配置解纠缠参数
        disentangle_config = DisentangleConfigs()
        if hasattr(args, 'alpha') and args.alpha is not None:
            disentangle_config.alpha = args.alpha
        if hasattr(args, 'beta') and args.beta is not None:
            disentangle_config.beta = args.beta
        disentangle_config.d_shared = args.d_shared
        disentangle_config.d_private = args.d_private
    else:
        # 使用原有模型
        model = MultimodalGestureNet(IMU_Configs, EMG_Configs)
        print(f"✓ 使用基础模型（不使用解纠缠）")
    
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 开始训练
    print("=" * 50)
    print("开始训练...")
    print(f"损失策略: {'使用三分支损失' if args.use_branch_loss else '仅使用融合分支损失（消融实验）'}")
    if args.use_adaptive_fusion:
        print(f"解纠缠策略: 启用（α={disentangle_config.alpha}, β={disentangle_config.beta}）")
        print(f"自适应融合策略: 启用（λ_align={adaptive_fusion_config.lambda_align}, λ_balance={adaptive_fusion_config.lambda_balance}）")
    elif args.use_disentangle:
        print(f"解纠缠策略: 启用（α={disentangle_config.alpha}, β={disentangle_config.beta}）")
    
    best_weights = train_model(
        model=model,
        dataloaders=dataloaders,
        num_epochs=args.num_epochs,
        precision=args.precision,
        device=device,
        use_swanlab=args.use_swanlab,
        swanlab_project=args.swanlab_project,
        subject=args.s,
        add_test_ratio=args.ratio,
        use_branch_loss=args.use_branch_loss,
        use_disentangle=args.use_disentangle or args.use_adaptive_fusion,  # 自适应融合也需要解纠缠
        disentangle_config=disentangle_config,
        use_adaptive_fusion=args.use_adaptive_fusion,
        adaptive_fusion_config=adaptive_fusion_config
    )
    
    # 保存模型
    print("=" * 50)
    print("保存模型...")
    
    # 创建目录结构: weights/subject{id}/
    if args.use_adaptive_fusion:
        model_type = "adaptive_fusion_model"
    elif args.use_disentangle:
        model_type = "disentangle_model"
    else:
        model_type = "base_model"
        
    subject_dir = os.path.join(args.save_dir, f'subject{args.s}')
    os.makedirs(subject_dir, exist_ok=True)
    
    # 保存路径: weights/subject{id}/{model_type}_best.pt
    save_path = os.path.join(subject_dir, f'{model_type}_best.pt')
    torch.save(best_weights, save_path)
    print(f"模型已保存至: {save_path}")
    
    print("=" * 50)
    print("训练完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EMG-IMU多模态手势识别训练')
    
    # 数据参数
    parser.add_argument('--s', type=int, default=10, 
                        help='受试者编号 (10, 23, 36)')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/mlsnrs/data/wrj/MoMo/Momo/processed_data',
                        help='数据目录（包含S{subject}_train.h5和S{subject}_test.h5）')
    parser.add_argument('--ratio', type=float, default=0.4,
                        help='从测试集中取多少比例加入训练集 (0-1, 默认0.4即40%%)')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=20, 
                        help='最大训练轮数')
    parser.add_argument('--precision', type=float, default=1e-8, 
                        help='早停精度')
    parser.add_argument('--seed', type=int, default=0, 
                        help='随机种子')
    
    # 设备参数
    parser.add_argument('--gpu', type=int, default=6, 
                        help='GPU编号')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./weights', 
                        help='模型保存目录')
    
    # SwanLab监控参数
    parser.add_argument('--use_swanlab', action='store_true', default=True,
                        help='是否使用SwanLab监控')
    parser.add_argument('--swanlab_project', type=str, default='Gesture-Recognition',
                        help='SwanLab项目名称')
    
    # 训练策略参数
    parser.add_argument('--use_branch_loss', action='store_true', default=True,
                        help='是否使用单模态分支损失（默认True，使用--no-branch-loss禁用）')
    parser.add_argument('--no-branch-loss', dest='use_branch_loss', action='store_false',
                        help='禁用单模态分支损失（消融实验）')
    
    # 解纠缠参数
    parser.add_argument('--use-disentangle', action='store_true', default=False,
                        help='是否使用解纠缠损失（默认False）')
    parser.add_argument('--d-shared', type=int, default=128,
                        help='共享表征维度（默认128）')
    parser.add_argument('--d-private', type=int, default=64,
                        help='独特表征维度（默认64）')
    parser.add_argument('--alpha', type=float, default=None,
                        help='L_private权重（默认使用config中的值）')
    parser.add_argument('--beta', type=float, default=None,
                        help='L_shared权重（默认使用config中的值）')
    
    # 自适应融合参数（创新点2）
    parser.add_argument('--use-adaptive-fusion', action='store_true', default=False,
                        help='是否使用自适应融合（默认False，启用时自动包含解纠缠）')
    parser.add_argument('--unified-dim', type=int, default=None,
                        help='统一融合维度（默认使用config中的128）')
    parser.add_argument('--lambda-align', type=float, default=None,
                        help='权重-重要性对齐损失权重（默认使用config中的值）')
    parser.add_argument('--lambda-balance', type=float, default=None,
                        help='权重平衡损失权重（默认使用config中的值）')
    
    args = parser.parse_args()
    
    main(args)

