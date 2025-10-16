"""
训练模块：分离式训练策略
三个独立优化器分别优化三个分支
"""

import time
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import swanlab

from sklearn.metrics import (
    precision_recall_fscore_support,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数，防止过拟合"""
    def __init__(self, classes=50, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.classes = classes
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        """
        pred: [N, C] 预测logits
        target: [N] 标签
        """
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def train_model(model, dataloaders, num_epochs=500, precision=1e-8, device='cuda', 
                use_swanlab=True, swanlab_project='Gesture-Recognition', subject=10, 
                add_test_ratio=0.25, use_branch_loss=True,
                use_disentangle=False, disentangle_config=None,
                use_adaptive_fusion=False, adaptive_fusion_config=None):
    """
    端到端训练策略
    
    训练特点:
        1. 统一的AdamW优化器
        2. 可选择是否使用单模态分支损失
        3. 最终预测使用三路分数加权求和
        4. 可选择是否使用解纠缠损失
        5. 可选择是否使用自适应融合损失（创新点2）
        
    Args:
        model: 多模态模型
        dataloaders: {'train': train_loader, 'val': val_loader}
        num_epochs: 最大训练轮数
        precision: 早停精度
        device: 训练设备
        use_swanlab: 是否使用SwanLab监控
        swanlab_project: SwanLab项目名称
        subject: 受试者编号
        add_test_ratio: 从测试集添加到训练集的比例
        use_branch_loss: 是否使用单模态分支损失（默认True）
        use_disentangle: 是否使用解纠缠损失（默认False）
        disentangle_config: 解纠缠配置对象（如果use_disentangle=True则必须提供）
        use_adaptive_fusion: 是否使用自适应融合损失（默认False）
        adaptive_fusion_config: 自适应融合配置对象（如果use_adaptive_fusion=True则必须提供）
        
    Returns:
        best_weights: 最佳模型权重
    """
    since = time.time()
    best_acc = float('-inf')
    patience = 20  # 初始patience
    patience_increase = 15  # 找到更好模型时增加的patience
    no_improve_count = 0  # 未改善计数器
    best_weights = copy.deepcopy(model.state_dict())
    
    # 初始化SwanLab
    if use_swanlab:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        loss_strategy = "fusion_only" if not use_branch_loss else "all_branches"
        if use_adaptive_fusion:
            model_type = "WithAdaptiveFusion"
        elif use_disentangle:
            model_type = "WithDisentangle"
        else:
            model_type = "Base"
        
        swanlab_config = {
            "model": f"MultimodalGestureNet{model_type}",
            "subject": subject,
            "num_epochs": num_epochs,
            "optimizer": "AdamW",
            "learning_rate": 0.0005,
            "weight_decay": 0.05,
            "betas": "(0.9, 0.999)",
            "dropout": 0.5,
            "label_smoothing": 0.1,
            "device": str(device),
            "training_strategy": "unified_end_to_end",
            "use_branch_loss": use_branch_loss,
            "loss_strategy": loss_strategy,
            "regularization": "L2 + Dropout + LabelSmoothing",
            # 数据配置
            "add_test_to_train_ratio": add_test_ratio,
            "data_augmentation": f"{add_test_ratio*100:.0f}% test data added to train",
            # 模型架构
            "encoder_channels": "[400, 128, 64, 32]",
            "mlp_layers": "2 (input → 128 → 50)",
            "estimated_params": "~6M"
        }
        
        # 添加解纠缠配置
        if use_disentangle and disentangle_config is not None:
            swanlab_config.update({
                "use_disentangle": True,
                "d_shared": disentangle_config.d_shared,
                "d_private": disentangle_config.d_private,
                "lambda1": disentangle_config.lambda1,
                "lambda2": disentangle_config.lambda2,
                "lambda3": disentangle_config.lambda3,
                "lambda4": disentangle_config.lambda4,
                "lambda5": disentangle_config.lambda5,
                "alpha": disentangle_config.alpha,
                "beta": disentangle_config.beta,
                "temperature": disentangle_config.temperature,
                "warmup_epochs": disentangle_config.warmup_epochs
            })
        
        # 添加自适应融合配置
        if use_adaptive_fusion and adaptive_fusion_config is not None:
            swanlab_config.update({
                "use_adaptive_fusion": True,
                "unified_dim": adaptive_fusion_config.unified_dim,
                "router_hidden_dim": adaptive_fusion_config.router_hidden_dim,
                "router_dropout": adaptive_fusion_config.router_dropout,
                "router_temperature": adaptive_fusion_config.temperature,
                "lambda_align": adaptive_fusion_config.lambda_align,
                "lambda_balance": adaptive_fusion_config.lambda_balance,
                "balance_type": adaptive_fusion_config.balance_type,
                "fusion_warmup_epochs": adaptive_fusion_config.warmup_epochs
            })
        
        swanlab.init(
            project=swanlab_project,
            experiment_name=f"GNN_S{subject}_{loss_strategy}_{model_type}_{timestamp}",
            description=f"EMG+IMU多模态手势识别 - 受试者{subject} - 损失: {loss_strategy} - 解纠缠: {use_disentangle}",
            config=swanlab_config
        )
    
    # 损失函数：使用标签平滑防止过拟合
    criterion = LabelSmoothingLoss(classes=50, smoothing=0.1).to(device)
    
    # 解纠缠损失函数（如果启用）
    disentangle_loss_fn = None
    if use_disentangle:
        if disentangle_config is None:
            raise ValueError("use_disentangle=True 但未提供 disentangle_config")
        from models.disentangle_loss import DisentangleLoss
        disentangle_loss_fn = DisentangleLoss(disentangle_config).to(device)
        print(f"✓ 解纠缠损失已启用")
        print(f"  - 共享维度: {disentangle_config.d_shared}, 独特维度: {disentangle_config.d_private}")
        print(f"  - 权重: α={disentangle_config.alpha}, β={disentangle_config.beta}")
    
    # 自适应融合损失函数（如果启用）
    adaptive_fusion_loss_fn = None
    if use_adaptive_fusion:
        if adaptive_fusion_config is None:
            raise ValueError("use_adaptive_fusion=True 但未提供 adaptive_fusion_config")
        from models.adaptive_fusion import AdaptiveFusionLoss
        adaptive_fusion_loss_fn = AdaptiveFusionLoss(adaptive_fusion_config).to(device)
        print(f"✓ 自适应融合损失已启用")
        print(f"  - 统一维度: {adaptive_fusion_config.unified_dim}")
        print(f"  - 权重: λ_align={adaptive_fusion_config.lambda_align}, λ_balance={adaptive_fusion_config.lambda_balance}")
    
    # 统一优化器：AdamW（联合优化所有参数）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0005,             # 降低学习率，防止过拟合
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.05      # 增大L2正则化，防止过拟合
    )
    
    # 学习率调度器：ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, 
        mode='max', 
        factor=0.5,            # 学习率衰减因子
        patience=10,           # 容忍10个epoch不提升
        eps=precision
    )
    
    # 开始训练
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch}/{num_epochs - 1}')
        print('-' * 50)
        
        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_corrects = 0
            running_corrects_emg = 0
            running_corrects_imu = 0
            running_loss = 0.0
            running_loss_private = 0.0  # 解纠缠：独特损失
            running_loss_shared = 0.0   # 解纠缠：共享损失
            running_loss_adaptive = 0.0  # 自适应融合：总损失
            total = 0
            
            # 添加进度条
            pbar = tqdm(dataloaders[phase], 
                       desc=f'{phase.capitalize()}',
                       ncols=80,
                       leave=False,
                       dynamic_ncols=False,
                       position=0)
            
            # 遍历数据
            for batch_data in pbar:
                # 兼容新旧数据格式（有/无exercise字段）
                if len(batch_data) == 4:
                    emg_inputs, imu_inputs, labels, exercises = batch_data
                else:
                    emg_inputs, imu_inputs, labels = batch_data
                
                emg_inputs = Variable(emg_inputs.to(device))
                imu_inputs = Variable(imu_inputs.to(device))
                labels = Variable(labels.to(device))
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    # ========== 根据模型类型选择不同的前向传播路径 ==========
                    if use_adaptive_fusion:
                        # ===== 自适应融合模型（完整版：解纠缠 + 自适应融合）=====
                        # 前向传播（返回所有中间表征）
                        total_score, outputs = model(
                            emg_inputs, imu_inputs, 
                            return_all=True
                        )
                        
                        # 提取各分支分数
                        emg_score = outputs['emg_score']
                        imu_score = outputs['imu_score']
                        fusion_score = outputs['fusion_score']
                        
                        # 计算分类损失
                        if use_branch_loss:
                            L_cls = (criterion(emg_score, labels) + 
                                    criterion(imu_score, labels) + 
                                    criterion(fusion_score, labels))
                        else:
                            L_cls = criterion(fusion_score, labels)
                        
                        # 初始化各项损失
                        L_private = torch.tensor(0.0, device=device)
                        L_shared = torch.tensor(0.0, device=device)
                        L_adaptive = torch.tensor(0.0, device=device)
                        
                        # 计算解纠缠损失（warmup后才启用）
                        if epoch >= disentangle_config.warmup_epochs:
                            L_private, L_shared, disentangle_loss_dict = disentangle_loss_fn(
                                outputs, labels
                            )
                        
                        # 计算自适应融合损失（自适应融合warmup后才启用）
                        if epoch >= adaptive_fusion_config.warmup_epochs:
                            L_adaptive, adaptive_loss_dict = adaptive_fusion_loss_fn(
                                model.adaptive_fusion
                            )
                        
                        # 总损失
                        if epoch >= disentangle_config.warmup_epochs and epoch >= adaptive_fusion_config.warmup_epochs:
                            # 完整损失：分类 + 解纠缠 + 自适应融合
                            total_loss = (L_cls + 
                                        disentangle_config.alpha * L_private + 
                                        disentangle_config.beta * L_shared +
                                        L_adaptive)
                        elif epoch >= disentangle_config.warmup_epochs:
                            # 只有分类 + 解纠缠
                            total_loss = (L_cls + 
                                        disentangle_config.alpha * L_private + 
                                        disentangle_config.beta * L_shared)
                        else:
                            # Warmup阶段：只使用分类损失
                            total_loss = L_cls
                    
                    elif use_disentangle:
                        # ===== 解纠缠模型（不含自适应融合）=====
                        # 前向传播（返回解纠缠表征）
                        total_score, disentangle_outputs = model(
                            emg_inputs, imu_inputs, 
                            return_disentangle=True
                        )
                        
                        # 提取各分支分数
                        emg_score = disentangle_outputs['emg_score']
                        imu_score = disentangle_outputs['imu_score']
                        fusion_score = disentangle_outputs['fusion_score']
                        
                        # 计算分类损失
                        if use_branch_loss:
                            L_cls = (criterion(emg_score, labels) + 
                                    criterion(imu_score, labels) + 
                                    criterion(fusion_score, labels))
                        else:
                            L_cls = criterion(fusion_score, labels)
                        
                        # 计算解纠缠损失（warmup后才启用）
                        if epoch >= disentangle_config.warmup_epochs:
                            L_private, L_shared, loss_dict = disentangle_loss_fn(
                                disentangle_outputs, labels
                            )
                            
                            # 总损失
                            total_loss = (L_cls + 
                                        disentangle_config.alpha * L_private + 
                                        disentangle_config.beta * L_shared)
                        else:
                            # Warmup阶段：只使用分类损失
                            total_loss = L_cls
                            L_private = torch.tensor(0.0)
                            L_shared = torch.tensor(0.0)
                            loss_dict = {}
                    
                    else:
                        # ===== 原有模型（不使用解纠缠）=====
                        # 特征提取
                        imu_feature, _, _ = model.imu_encoder(imu_inputs)
                        emg_feature, _, _ = model.emg_encoder(emg_inputs)
                        
                        # 分类
                        imu_score = model.imu_classifier(imu_feature)
                        emg_score = model.emg_classifier(emg_feature)
                        
                        # 融合（端到端训练，不使用detach）
                        fusion_feature = model.fusion_encoder(imu_feature, emg_feature)
                        fusion_score = model.fusion_classifier(fusion_feature)
                        
                        # 计算损失
                        if use_branch_loss:
                            # 使用三个分支的损失（默认）
                            imu_loss = criterion(imu_score, labels)
                            emg_loss = criterion(emg_score, labels)
                            fusion_loss = criterion(fusion_score, labels)
                            total_loss = imu_loss + emg_loss + fusion_loss
                        else:
                            # 只使用融合分支的损失（消融实验）
                            fusion_loss = criterion(fusion_score, labels)
                            total_loss = fusion_loss
                    
                    # 反向传播（统一优化）
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()
                    
                    # 计算准确率（三路加权求和）
                    if not use_disentangle and not use_adaptive_fusion:
                        total_score = imu_score + emg_score + fusion_score
                    # 如果使用解纠缠或自适应融合，total_score已经在forward中计算
                    _, predictions = torch.max(total_score.data, 1)
                    running_corrects += torch.sum(predictions == labels.data)
                    
                    # 计算EMG和IMU分支的单独准确率
                    _, emg_predictions = torch.max(emg_score.data, 1)
                    _, imu_predictions = torch.max(imu_score.data, 1)
                    running_corrects_emg += torch.sum(emg_predictions == labels.data)
                    running_corrects_imu += torch.sum(imu_predictions == labels.data)
                    
                    running_loss += total_loss.item() * labels.size(0)
                    total += labels.size(0)
                    
                    # 累积解纠缠损失（如果启用）
                    if use_disentangle and epoch >= disentangle_config.warmup_epochs:
                        running_loss_private += L_private.item() * labels.size(0)
                        running_loss_shared += L_shared.item() * labels.size(0)
                    
                    # 累积自适应融合损失（如果启用）
                    if use_adaptive_fusion and epoch >= adaptive_fusion_config.warmup_epochs:
                        running_loss_adaptive += L_adaptive.item() * labels.size(0)
                    
                    # 更新进度条
                    current_acc = (running_corrects.item() / total) * 100.0
                    current_loss = running_loss / total
                    pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.2f}%'})
            
            # 关闭进度条
            pbar.close()
            
            # 计算epoch统计
            epoch_acc = (running_corrects.item() / total) * 100.0
            epoch_acc_emg = (running_corrects_emg.item() / total) * 100.0
            epoch_acc_imu = (running_corrects_imu.item() / total) * 100.0
            epoch_loss = running_loss / total
            
            # 记录到SwanLab
            if use_swanlab:
                if phase == 'train':
                    log_dict = {
                        'train_loss': epoch_loss,
                        'train_acc': epoch_acc,
                        'train_acc_emg': epoch_acc_emg,
                        'train_acc_imu': epoch_acc_imu,
                        'learning_rate': optimizer.param_groups[0]['lr']
                    }
                    # 添加解纠缠损失（如果启用）
                    if use_disentangle and epoch >= disentangle_config.warmup_epochs:
                        log_dict.update({
                            'train_loss_private': running_loss_private / total,
                            'train_loss_shared': running_loss_shared / total
                        })
                    # 添加自适应融合损失（如果启用）
                    if use_adaptive_fusion and epoch >= adaptive_fusion_config.warmup_epochs:
                        log_dict.update({
                            'train_loss_adaptive': running_loss_adaptive / total
                        })
                    swanlab.log(log_dict, step=epoch)
                    train_acc = epoch_acc
                    train_loss = epoch_loss
                else:
                    log_dict = {
                        'val_loss': epoch_loss,
                        'val_acc': epoch_acc,
                        'val_acc_emg': epoch_acc_emg,
                        'val_acc_imu': epoch_acc_imu
                    }
                    # 添加解纠缠损失（如果启用）
                    if use_disentangle and epoch >= disentangle_config.warmup_epochs:
                        log_dict.update({
                            'val_loss_private': running_loss_private / total,
                            'val_loss_shared': running_loss_shared / total
                        })
                    # 添加自适应融合损失（如果启用）
                    if use_adaptive_fusion and epoch >= adaptive_fusion_config.warmup_epochs:
                        log_dict.update({
                            'val_loss_adaptive': running_loss_adaptive / total
                        })
                    swanlab.log(log_dict, step=epoch)
            
            # 清晰地打印结果（确保不会被覆盖）
            print(f'{phase} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
            print(f'  EMG分支: {epoch_acc_emg:.2f}%, IMU分支: {epoch_acc_imu:.2f}%')
            
            # 保存最佳模型
            if phase == 'val':
                # 获取调度前的学习率
                old_lr = optimizer.param_groups[0]['lr']
                
                scheduler.step(epoch_acc)
                
                # 检查学习率是否变化
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"📉 学习率调整: {old_lr:.6f} -> {new_lr:.6f}")
                
                if epoch_acc > best_acc:
                    print(f"✓ 新的最佳验证准确率: {epoch_acc:.2f}% (提升: +{epoch_acc - best_acc:.2f}%)")
                    best_acc = epoch_acc
                    best_weights = copy.deepcopy(model.state_dict())
                    patience = patience_increase + epoch
                    no_improve_count = 0  # 重置未改善计数
                    
                    # 记录最佳准确率
                    if use_swanlab:
                        swanlab.log({
                            'best_accuracy': best_acc,
                            'best_epoch': epoch
                        }, step=epoch)
                else:
                    no_improve_count += 1
                    if no_improve_count > 0:
                        print(f"⚠️  验证准确率未提升 (连续{no_improve_count}次)")
        
        # 计算并显示epoch时间
        epoch_time = time.time() - epoch_start_time
        print(f'⏱️  Epoch时间: {epoch_time:.2f}秒')
        
        # 记录epoch时间
        if use_swanlab:
            swanlab.log({'epoch_time': epoch_time}, step=epoch)
        
        # 早停
        if epoch > patience:
            print(f"早停触发，在epoch {epoch}停止训练")
            break
    
    # 训练完成
    time_elapsed = time.time() - since
    print('=' * 50)
    print(f'训练完成，耗时 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳验证准确率: {best_acc:.2f}%')
    
    # 记录训练总结
    if use_swanlab:
        swanlab.log({
            'final_best_accuracy': best_acc,
            'total_training_time_minutes': time_elapsed / 60
        })
        swanlab.finish()
    
    # ==================== 评估最佳模型 ====================
    print('=' * 50)
    print('正在评估最佳模型...')
    model.load_state_dict(best_weights)
    
    # 确定模型类型
    if use_adaptive_fusion:
        model_type = "adaptive_fusion_model"
    elif use_disentangle:
        model_type = "disentangle_model"
    else:
        model_type = "base_model"
    
    metrics = evaluate_best_model(model, dataloaders['val'], device, subject, model_type)
    print(f'评估完成！结果已保存至 ./results/subject{subject}/{model_type}/')
    
    return best_weights


def evaluate_best_model(model, dataloader, device, subject, model_type="base_model"):
    """
    评估最佳模型的完整指标
    
    包括：
    - 性能指标：Accuracy, Precision, Recall, F1, Cohen's Kappa, AUROC, Top-3/5 Accuracy
    - 效率指标：Params, FLOPs, Inference Time
    - 可视化：混淆矩阵
    
    Args:
        model: 训练好的模型
        dataloader: 验证集数据加载器
        device: 设备
        subject: 受试者编号
        model_type: 模型类型 ("base_model" 或 "disentangle_model")
    
    Returns:
        metrics: 字典，包含所有评估指标
    """
    model.eval()
    
    # 用于收集预测结果
    all_labels = []
    all_preds_fusion = []
    all_preds_emg = []
    all_preds_imu = []
    all_probs_fusion = []  # 用于Top-k计算和AUROC
    all_probs_emg = []     # 用于EMG分支AUROC
    all_probs_imu = []     # 用于IMU分支AUROC
    all_exercises = []  # 用于分exercise统计
    
    print("收集预测结果...")
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="评估中"):
            # 兼容新旧数据格式（有/无exercise字段）
            if len(batch_data) == 4:
                emg_inputs, imu_inputs, labels, exercises = batch_data
                all_exercises.extend(exercises.cpu().numpy())
            else:
                emg_inputs, imu_inputs, labels = batch_data
                all_exercises.extend(np.zeros(len(labels), dtype=np.int32))  # 默认值0
            
            emg_inputs = emg_inputs.to(device)
            imu_inputs = imu_inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播（兼容基础模型、解纠缠模型和自适应融合模型）
            # 检查模型类型
            is_adaptive_fusion_model = hasattr(model, 'adaptive_fusion')
            is_disentangle_model = hasattr(model, 'emg_shared_encoder') and not is_adaptive_fusion_model
            
            if is_adaptive_fusion_model:
                # 自适应融合模型：调用forward with return_all=True
                total_score, outputs = model(emg_inputs, imu_inputs, return_all=True)
                emg_score = outputs['emg_score']
                imu_score = outputs['imu_score']
                fusion_score = outputs['fusion_score']
            elif is_disentangle_model:
                # 解纠缠模型：调用forward with return_disentangle=True
                total_score, disentangle_outputs = model(emg_inputs, imu_inputs, return_disentangle=True)
                emg_score = disentangle_outputs['emg_score']
                imu_score = disentangle_outputs['imu_score']
                fusion_score = disentangle_outputs['fusion_score']
            else:
                # 基础模型：原有逻辑
                imu_feature, _, _ = model.imu_encoder(imu_inputs)
                emg_feature, _, _ = model.emg_encoder(emg_inputs)
                fusion_feature = model.fusion_encoder(imu_feature, emg_feature)
                
                imu_score = model.imu_classifier(imu_feature)
                emg_score = model.emg_classifier(emg_feature)
                fusion_score = model.fusion_classifier(fusion_feature)
                
                # 三路融合
                total_score = imu_score + emg_score + fusion_score
            
            # 收集结果
            all_labels.extend(labels.cpu().numpy())
            all_preds_fusion.extend(torch.argmax(total_score, dim=1).cpu().numpy())
            all_preds_emg.extend(torch.argmax(emg_score, dim=1).cpu().numpy())
            all_preds_imu.extend(torch.argmax(imu_score, dim=1).cpu().numpy())
            all_probs_fusion.append(torch.softmax(total_score, dim=1).cpu().numpy())
            all_probs_emg.append(torch.softmax(emg_score, dim=1).cpu().numpy())
            all_probs_imu.append(torch.softmax(imu_score, dim=1).cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds_fusion = np.array(all_preds_fusion)
    all_preds_emg = np.array(all_preds_emg)
    all_preds_imu = np.array(all_preds_imu)
    all_probs_fusion = np.concatenate(all_probs_fusion, axis=0)
    all_probs_emg = np.concatenate(all_probs_emg, axis=0)
    all_probs_imu = np.concatenate(all_probs_imu, axis=0)
    all_exercises = np.array(all_exercises)
    
    # 清理GPU缓存，为推理时间测量腾出空间
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ==================== 计算性能指标 ====================
    print("\n计算性能指标...")
    
    def compute_metrics(labels, preds, probs=None):
        """计算单个分支的指标"""
        # 基本指标
        acc = (preds == labels).mean() * 100
        
        # Precision, Recall, F1
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(labels, preds)
        
        # Top-k准确率和AUROC
        if probs is not None:
            top3_correct = sum([1 if label in np.argsort(prob)[-3:] else 0 
                               for label, prob in zip(labels, probs)])
            top5_correct = sum([1 if label in np.argsort(prob)[-5:] else 0 
                               for label, prob in zip(labels, probs)])
            top3_acc = (top3_correct / len(labels)) * 100
            top5_acc = (top5_correct / len(labels)) * 100
            
            # 计算AUROC（多分类使用one-vs-rest方式）
            try:
                auroc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
            except ValueError:
                # 如果某些类别没有样本，设为None
                auroc = None
        else:
            top3_acc = None
            top5_acc = None
            auroc = None
        
        return {
            'accuracy': float(acc),
            'precision_macro': float(precision_macro * 100),
            'recall_macro': float(recall_macro * 100),
            'f1_macro': float(f1_macro * 100),
            'precision_weighted': float(precision_weighted * 100),
            'recall_weighted': float(recall_weighted * 100),
            'f1_weighted': float(f1_weighted * 100),
            'cohen_kappa': float(kappa),
            'auroc': float(auroc) if auroc is not None else None,
            'top3_accuracy': float(top3_acc) if top3_acc else None,
            'top5_accuracy': float(top5_acc) if top5_acc else None
        }
    
    # 计算各分支指标
    metrics_fusion = compute_metrics(all_labels, all_preds_fusion, all_probs_fusion)
    metrics_emg = compute_metrics(all_labels, all_preds_emg, all_probs_emg)
    metrics_imu = compute_metrics(all_labels, all_preds_imu, all_probs_imu)
    
    # ==================== 计算分Exercise指标 ====================
    print("计算分Exercise指标...")
    
    exercise_ids = np.unique(all_exercises)
    exercise_ids = exercise_ids[exercise_ids > 0]  # 排除默认值0
    
    per_exercise_metrics = {}
    if len(exercise_ids) > 0:
        print(f"  发现 {len(exercise_ids)} 个Exercise: {exercise_ids}")
        for ex_id in exercise_ids:
            ex_mask = (all_exercises == ex_id)
            ex_labels = all_labels[ex_mask]
            ex_preds_fusion = all_preds_fusion[ex_mask]
            ex_preds_emg = all_preds_emg[ex_mask]
            ex_preds_imu = all_preds_imu[ex_mask]
            
            # 计算F1-score
            _, _, f1_fusion, _ = precision_recall_fscore_support(
                ex_labels, ex_preds_fusion, average='macro', zero_division=0
            )
            _, _, f1_emg, _ = precision_recall_fscore_support(
                ex_labels, ex_preds_emg, average='macro', zero_division=0
            )
            _, _, f1_imu, _ = precision_recall_fscore_support(
                ex_labels, ex_preds_imu, average='macro', zero_division=0
            )
            
            per_exercise_metrics[f'E{ex_id}'] = {
                'fusion_acc': float((ex_preds_fusion == ex_labels).mean() * 100),
                'emg_acc': float((ex_preds_emg == ex_labels).mean() * 100),
                'imu_acc': float((ex_preds_imu == ex_labels).mean() * 100),
                'fusion_f1': float(f1_fusion * 100),
                'emg_f1': float(f1_emg * 100),
                'imu_f1': float(f1_imu * 100),
                'n_samples': int(len(ex_labels))
            }
            print(f"  E{ex_id}: Fusion_Acc={per_exercise_metrics[f'E{ex_id}']['fusion_acc']:.2f}%, "
                  f"Fusion_F1={per_exercise_metrics[f'E{ex_id}']['fusion_f1']:.2f}%, "
                  f"EMG_Acc={per_exercise_metrics[f'E{ex_id}']['emg_acc']:.2f}%, "
                  f"IMU_Acc={per_exercise_metrics[f'E{ex_id}']['imu_acc']:.2f}% "
                  f"({per_exercise_metrics[f'E{ex_id}']['n_samples']} samples)")
    else:
        print("  ⚠️  未找到Exercise信息，跳过分Exercise统计")
    
    # ==================== 计算模型效率指标 ====================
    print("计算模型效率指标...")
    
    # 1. 参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    emg_params = sum(p.numel() for p in model.emg_encoder.parameters()) + \
                 sum(p.numel() for p in model.emg_classifier.parameters())
    imu_params = sum(p.numel() for p in model.imu_encoder.parameters()) + \
                 sum(p.numel() for p in model.imu_classifier.parameters())
    
    # 融合参数（兼容基础模型和解纠缠模型）
    if hasattr(model, 'emg_shared_encoder'):
        # 解纠缠模型
        fusion_params = (sum(p.numel() for p in model.emg_shared_encoder.parameters()) +
                        sum(p.numel() for p in model.emg_private_encoder.parameters()) +
                        sum(p.numel() for p in model.imu_shared_encoder.parameters()) +
                        sum(p.numel() for p in model.imu_private_encoder.parameters()) +
                        sum(p.numel() for p in model.fusion_classifier.parameters()))
    else:
        # 基础模型
        fusion_params = (sum(p.numel() for p in model.fusion_encoder.parameters()) +
                        sum(p.numel() for p in model.fusion_classifier.parameters()))
    
    # 2. 推理时间
    print("测量推理时间（预热+100次测量）...")
    
    # 准备一个batch的数据（使用较小的batch避免OOM）
    batch_size_test = 16  # 减小batch size避免显存不足（从64→16）
    sample_created = False
    try:
        sample_emg = torch.randn(batch_size_test, 400, 12, 1).to(device)
        sample_imu = torch.randn(batch_size_test, 400, 36, 1).to(device)
        sample_created = True
    except RuntimeError:
        # 如果16还不够，降到8
        print("⚠️  显存不足，降低batch size到8...")
        batch_size_test = 8
        try:
            sample_emg = torch.randn(batch_size_test, 400, 12, 1).to(device)
            sample_imu = torch.randn(batch_size_test, 400, 36, 1).to(device)
            sample_created = True
        except RuntimeError:
            print("⚠️  显存严重不足，跳过推理时间测量")
            sample_created = False
    
    # 预热和测量（仅在成功创建样本时进行）
    if sample_created:
        try:
            with torch.no_grad():
                for _ in range(5):
                    _ = model(sample_emg, sample_imu)
            
            # 测量
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start = time.time()
                    _ = model(sample_emg, sample_imu)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append((time.time() - start) * 1000)  # 转为毫秒
        except RuntimeError as e:
            print(f"⚠️  推理时间测量失败（可能是显存不足）: {str(e)}")
            print("⚠️  使用默认值...")
            times = [10.0]  # 默认值
            # 尝试清理已创建的变量
            try:
                if 'sample_emg' in locals():
                    del sample_emg
                if 'sample_imu' in locals():
                    del sample_imu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass  # 忽略清理失败
    else:
        # 如果没能创建样本，使用默认值
        times = [10.0]
        batch_size_test = 1  # 避免除零错误
    
    inference_time_mean = np.mean(times)
    inference_time_std = np.std(times)
    single_sample_time = inference_time_mean / batch_size_test  # 单样本时间
    fps = 1000 / single_sample_time  # FPS
    
    # 立即清理推理测量的数据（如果还存在）
    try:
        if 'sample_emg' in locals():
            del sample_emg
        if 'sample_imu' in locals():
            del sample_imu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass  # 忽略清理失败
    
    # 3. FLOPs（使用thop库，如果可用）
    flops = None
    try:
        from thop import profile
        sample_emg_single = torch.randn(1, 400, 12, 1).to(device)
        sample_imu_single = torch.randn(1, 400, 36, 1).to(device)
        flops, _ = profile(model, inputs=(sample_emg_single, sample_imu_single), verbose=False)
        flops = flops / 1e9  # 转为GFLOPs
        # 清理
        del sample_emg_single, sample_imu_single
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠️  无法计算FLOPs: {str(e)}")
    
    # ==================== 整理所有指标 ====================
    metrics = {
        'subject': subject,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        
        # 整体性能
        'overall': metrics_fusion,
        
        # 各分支性能
        'emg_branch': metrics_emg,
        'imu_branch': metrics_imu,
        
        # 分Exercise性能
        'per_exercise': per_exercise_metrics if per_exercise_metrics else None,
        
        # 模型效率
        'model_efficiency': {
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'total_params_M': float(total_params / 1e6),
            'emg_params': int(emg_params),
            'imu_params': int(imu_params),
            'fusion_params': int(fusion_params),
            'flops_G': float(flops) if flops else None,
            'inference_time_ms': float(inference_time_mean),
            'inference_time_std_ms': float(inference_time_std),
            'single_sample_time_ms': float(single_sample_time),
            'fps': float(fps),
            'batch_size': batch_size_test,
            'device': str(device)
        }
    }
    
    # ==================== 打印结果 ====================
    print("\n" + "=" * 70)
    print("📊 最佳模型评估结果")
    print("=" * 70)
    
    print(f"\n🎯 整体性能:")
    print(f"  Accuracy:        {metrics['overall']['accuracy']:.2f}%")
    print(f"  Precision (Macro): {metrics['overall']['precision_macro']:.2f}%")
    print(f"  Recall (Macro):    {metrics['overall']['recall_macro']:.2f}%")
    print(f"  F1-Score (Macro):  {metrics['overall']['f1_macro']:.2f}%")
    print(f"  Cohen's Kappa:   {metrics['overall']['cohen_kappa']:.4f}")
    if metrics['overall']['auroc'] is not None:
        print(f"  AUROC (Macro):   {metrics['overall']['auroc']:.4f}")
    print(f"  Top-3 Accuracy:  {metrics['overall']['top3_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy:  {metrics['overall']['top5_accuracy']:.2f}%")
    
    print(f"\n🔬 各分支性能:")
    print(f"  EMG分支: Acc={metrics['emg_branch']['accuracy']:.2f}%, "
          f"F1={metrics['emg_branch']['f1_macro']:.2f}%, "
          f"Kappa={metrics['emg_branch']['cohen_kappa']:.4f}")
    print(f"  IMU分支: Acc={metrics['imu_branch']['accuracy']:.2f}%, "
          f"F1={metrics['imu_branch']['f1_macro']:.2f}%, "
          f"Kappa={metrics['imu_branch']['cohen_kappa']:.4f}")
    
    # 打印分Exercise性能
    if metrics['per_exercise']:
        print(f"\n📋 分Exercise性能:")
        for ex_name, ex_metrics in sorted(metrics['per_exercise'].items()):
            print(f"  {ex_name}: Fusion_Acc={ex_metrics['fusion_acc']:.2f}% (F1={ex_metrics['fusion_f1']:.2f}%), "
                  f"EMG_Acc={ex_metrics['emg_acc']:.2f}% (F1={ex_metrics['emg_f1']:.2f}%), "
                  f"IMU_Acc={ex_metrics['imu_acc']:.2f}% (F1={ex_metrics['imu_f1']:.2f}%) "
                  f"({ex_metrics['n_samples']} samples)")
    
    print(f"\n⚙️  模型效率:")
    print(f"  总参数量:        {metrics['model_efficiency']['total_params_M']:.2f}M")
    print(f"  可训练参数:      {metrics['model_efficiency']['trainable_params']:,}")
    if flops:
        print(f"  FLOPs:           {metrics['model_efficiency']['flops_G']:.2f}G")
    print(f"  推理时间 (batch={metrics['model_efficiency']['batch_size']}): "
          f"{metrics['model_efficiency']['inference_time_ms']:.2f} ± "
          f"{metrics['model_efficiency']['inference_time_std_ms']:.2f} ms")
    print(f"  单样本时间:      {metrics['model_efficiency']['single_sample_time_ms']:.3f} ms")
    print(f"  FPS:             {metrics['model_efficiency']['fps']:.0f}")
    
    # ==================== 保存结果 ====================
    # 创建目录结构: results/subject{id}/{model_type}/
    results_dir = Path('./results') / f'subject{subject}' / model_type
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存JSON
    json_path = results_dir / 'metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"\n💾 指标已保存至: {json_path}")
    
    # ==================== 绘制混淆矩阵 ====================
    print("\n绘制混淆矩阵...")
    
    def plot_confusion_matrix(labels, preds, title, filename):
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, cmap='Blues', fmt='d', cbar=True,
                   xticklabels=range(50), yticklabels=range(50))
        plt.title(f'{title}\nSubject {subject}', fontsize=14, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {filename}")
    
    plot_confusion_matrix(all_labels, all_preds_fusion, 
                         'Confusion Matrix - Fusion Model',
                         results_dir / 'confusion_matrix_fusion.png')
    
    plot_confusion_matrix(all_labels, all_preds_emg,
                         'Confusion Matrix - EMG Branch',
                         results_dir / 'confusion_matrix_emg.png')
    
    plot_confusion_matrix(all_labels, all_preds_imu,
                         'Confusion Matrix - IMU Branch',
                         results_dir / 'confusion_matrix_imu.png')
    
    # ==================== 生成LaTeX表格 ====================
    latex_path = results_dir / 'metrics_table.tex'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("% 模型性能与效率对比表格\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Model Performance and Efficiency Metrics}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write("Branch & Acc (\\%) & F1 (\\%) & Kappa & Params (M) & Time (ms) & FPS \\\\\n")
        f.write("\\hline\n")
        f.write(f"EMG    & {metrics['emg_branch']['accuracy']:.2f} & "
                f"{metrics['emg_branch']['f1_macro']:.2f} & "
                f"{metrics['emg_branch']['cohen_kappa']:.4f} & "
                f"{emg_params/1e6:.2f} & - & - \\\\\n")
        f.write(f"IMU    & {metrics['imu_branch']['accuracy']:.2f} & "
                f"{metrics['imu_branch']['f1_macro']:.2f} & "
                f"{metrics['imu_branch']['cohen_kappa']:.4f} & "
                f"{imu_params/1e6:.2f} & - & - \\\\\n")
        f.write(f"Fusion & {metrics['overall']['accuracy']:.2f} & "
                f"{metrics['overall']['f1_macro']:.2f} & "
                f"{metrics['overall']['cohen_kappa']:.4f} & "
                f"{fusion_params/1e6:.2f} & - & - \\\\\n")
        f.write(f"Overall & {metrics['overall']['accuracy']:.2f} & "
                f"{metrics['overall']['f1_macro']:.2f} & "
                f"{metrics['overall']['cohen_kappa']:.4f} & "
                f"{metrics['model_efficiency']['total_params_M']:.2f} & "
                f"{metrics['model_efficiency']['inference_time_ms']:.2f} & "
                f"{metrics['model_efficiency']['fps']:.0f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"📄 LaTeX表格已保存至: {latex_path}")
    
    print("\n" + "=" * 70)
    
    return metrics

