"""
RMSCM多模态手势识别模型训练脚本
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from model.rmscm_model import MultiModalRMSCM, MultiTaskLoss, get_model_info
from dataset import create_dataloaders


class Trainer:
    """训练器类"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # 创建保存目录
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # 初始化数据加载器
        self._init_dataloaders()
        
        # 初始化模型
        self._init_model()
        
        # 初始化优化器和学习率调度器
        self._init_optimizer()
        
        # 记录最佳准确率和Early Stopping
        self.best_acc = 0.0
        self.start_epoch = 0
        self.patience_counter = 0  # Early Stopping计数器
        
        # 如果指定了恢复训练的checkpoint
        if args.resume:
            self._load_checkpoint(args.resume)
    
    def _init_dataloaders(self):
        """初始化数据加载器"""
        print(f"\n{'='*60}")
        print("初始化数据加载器")
        print(f"{'='*60}")
        
        self.train_loader, self.test_loader = create_dataloaders(
            data_dir=self.args.data_dir,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            mode='both',
            load_to_memory=self.args.load_to_memory,
            subject=self.args.subject
        )
        
        print(f"数据加载完成")
        print(f"训练批次数: {len(self.train_loader)}")
        print(f"测试批次数: {len(self.test_loader)}")
    
    def _init_model(self):
        """初始化模型"""
        print(f"\n{'='*60}")
        print("初始化模型")
        print(f"{'='*60}")
        
        self.model = MultiModalRMSCM(
            emg_channels=self.args.emg_channels,
            imu_channels=self.args.imu_channels,
            num_classes=self.args.num_classes,
            feature_dim=self.args.feature_dim,
            hidden_dim=self.args.hidden_dim,
            dropout=self.args.dropout
        ).to(self.device)
        
        # 打印模型信息
        model_info = get_model_info(self.model, self.args.emg_channels, 
                                    self.args.imu_channels, self.args.window_size)
        print(f"模型参数量: {model_info['total_parameters_M']:.2f}M")
        print(f"输入形状 - EMG: {model_info['input_shape_emg']}")
        print(f"输入形状 - IMU: {model_info['input_shape_imu']}")
        print(f"输出形状: {model_info['output_shape_final']}")
        
        # 初始化损失函数
        self.criterion = MultiTaskLoss(
            alpha=self.args.alpha,
            beta=self.args.beta,
            gamma=self.args.gamma
        )
        
        print(f"损失函数权重: α={self.args.alpha}, β={self.args.beta}, γ={self.args.gamma}")
    
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.args.optimizer}")
        
        # 学习率调度器
        if self.args.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.args.step_size,
                gamma=self.args.gamma_lr
            )
        elif self.args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs
            )
        else:
            self.scheduler = None
        
        print(f"优化器: {self.args.optimizer}, 初始学习率: {self.args.lr}")
        if self.scheduler:
            print(f"学习率调度器: {self.args.scheduler}")
    
    def _save_checkpoint(self, epoch, acc, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'args': self.args
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型: {best_path} (准确率: {acc:.4f})")
        
        # 定期保存checkpoint
        if (epoch + 1) % self.args.save_freq == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, epoch_path)
    
    def _load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        print(f"\n从 {checkpoint_path} 恢复训练")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"从epoch {self.start_epoch} 继续训练, 最佳准确率: {self.best_acc:.4f}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        loss_emg_total = 0.0
        loss_imu_total = 0.0
        loss_fusion_total = 0.0
        
        correct_emg = 0
        correct_imu = 0
        correct_fusion = 0
        correct_final = 0
        total_samples = 0
        
        start_time = time.time()
        
        for batch_idx, (emg, imu, labels) in enumerate(self.train_loader):
            # 数据移到设备
            emg = emg.to(self.device)
            imu = imu.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            logit_emg, logit_imu, logit_fusion, logit_final = self.model(emg, imu)
            
            # 计算损失
            loss, loss_dict = self.criterion(logit_emg, logit_imu, logit_fusion, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss_dict['total']
            loss_emg_total += loss_dict['emg']
            loss_imu_total += loss_dict['imu']
            loss_fusion_total += loss_dict['fusion']
            
            # 计算准确率
            _, pred_emg = torch.max(logit_emg, 1)
            _, pred_imu = torch.max(logit_imu, 1)
            _, pred_fusion = torch.max(logit_fusion, 1)
            _, pred_final = torch.max(logit_final, 1)
            
            correct_emg += (pred_emg == labels).sum().item()
            correct_imu += (pred_imu == labels).sum().item()
            correct_fusion += (pred_fusion == labels).sum().item()
            correct_final += (pred_final == labels).sum().item()
            total_samples += labels.size(0)
            
            # 打印进度
            if (batch_idx + 1) % self.args.print_freq == 0:
                avg_loss = total_loss / (batch_idx + 1)
                acc_final = 100.0 * correct_final / total_samples
                print(f"Epoch [{epoch+1}/{self.args.epochs}] "
                      f"Batch [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {avg_loss:.4f} "
                      f"Acc: {acc_final:.2f}%")
        
        # Epoch统计
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(self.train_loader)
        acc_emg = 100.0 * correct_emg / total_samples
        acc_imu = 100.0 * correct_imu / total_samples
        acc_fusion = 100.0 * correct_fusion / total_samples
        acc_final = 100.0 * correct_final / total_samples
        
        # TensorBoard记录
        self.writer.add_scalar('Train/Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/Acc_EMG', acc_emg, epoch)
        self.writer.add_scalar('Train/Acc_IMU', acc_imu, epoch)
        self.writer.add_scalar('Train/Acc_Fusion', acc_fusion, epoch)
        self.writer.add_scalar('Train/Acc_Final', acc_final, epoch)
        self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], epoch)
        
        return {
            'loss': avg_loss,
            'acc_emg': acc_emg,
            'acc_imu': acc_imu,
            'acc_fusion': acc_fusion,
            'acc_final': acc_final,
            'time': epoch_time
        }
    
    def evaluate(self, epoch):
        """评估模型"""
        self.model.eval()
        
        total_loss = 0.0
        correct_emg = 0
        correct_imu = 0
        correct_fusion = 0
        correct_final = 0
        total_samples = 0
        
        with torch.no_grad():
            for emg, imu, labels in self.test_loader:
                # 数据移到设备
                emg = emg.to(self.device)
                imu = imu.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                logit_emg, logit_imu, logit_fusion, logit_final = self.model(emg, imu)
                
                # 计算损失
                loss, loss_dict = self.criterion(logit_emg, logit_imu, logit_fusion, labels)
                total_loss += loss_dict['total']
                
                # 计算准确率
                _, pred_emg = torch.max(logit_emg, 1)
                _, pred_imu = torch.max(logit_imu, 1)
                _, pred_fusion = torch.max(logit_fusion, 1)
                _, pred_final = torch.max(logit_final, 1)
                
                correct_emg += (pred_emg == labels).sum().item()
                correct_imu += (pred_imu == labels).sum().item()
                correct_fusion += (pred_fusion == labels).sum().item()
                correct_final += (pred_final == labels).sum().item()
                total_samples += labels.size(0)
        
        # 统计
        avg_loss = total_loss / len(self.test_loader)
        acc_emg = 100.0 * correct_emg / total_samples
        acc_imu = 100.0 * correct_imu / total_samples
        acc_fusion = 100.0 * correct_fusion / total_samples
        acc_final = 100.0 * correct_final / total_samples
        
        # TensorBoard记录
        self.writer.add_scalar('Test/Loss', avg_loss, epoch)
        self.writer.add_scalar('Test/Acc_EMG', acc_emg, epoch)
        self.writer.add_scalar('Test/Acc_IMU', acc_imu, epoch)
        self.writer.add_scalar('Test/Acc_Fusion', acc_fusion, epoch)
        self.writer.add_scalar('Test/Acc_Final', acc_final, epoch)
        
        return {
            'loss': avg_loss,
            'acc_emg': acc_emg,
            'acc_imu': acc_imu,
            'acc_fusion': acc_fusion,
            'acc_final': acc_final
        }
    
    def train(self):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print("开始训练")
        print(f"{'='*60}")
        print(f"设备: {self.device}")
        print(f"训练轮数: {self.args.epochs}")
        print(f"批次大小: {self.args.batch_size}")
        if self.args.early_stopping:
            print(f"Early Stopping: 开启 (patience={self.args.patience})")
        if self.args.grad_clip > 0:
            print(f"梯度裁剪: {self.args.grad_clip}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            print(f"\n{'='*60}")
            print(f"Epoch [{epoch+1}/{self.args.epochs}]")
            print(f"{'='*60}")
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            print(f"\n训练结果:")
            print(f"  损失: {train_metrics['loss']:.4f}")
            print(f"  EMG准确率: {train_metrics['acc_emg']:.2f}%")
            print(f"  IMU准确率: {train_metrics['acc_imu']:.2f}%")
            print(f"  融合准确率: {train_metrics['acc_fusion']:.2f}%")
            print(f"  最终准确率: {train_metrics['acc_final']:.2f}%")
            print(f"  用时: {train_metrics['time']:.2f}s")
            
            # 评估（每隔eval_freq个epoch评估一次）
            should_evaluate = (epoch + 1) % self.args.eval_freq == 0 or (epoch + 1) == self.args.epochs
            
            if should_evaluate:
                test_metrics = self.evaluate(epoch)
                
                print(f"\n测试结果:")
                print(f"  损失: {test_metrics['loss']:.4f}")
                print(f"  EMG准确率: {test_metrics['acc_emg']:.2f}%")
                print(f"  IMU准确率: {test_metrics['acc_imu']:.2f}%")
                print(f"  融合准确率: {test_metrics['acc_fusion']:.2f}%")
                print(f"  最终准确率: {test_metrics['acc_final']:.2f}%")
                
                # 保存checkpoint
                is_best = test_metrics['acc_final'] > self.best_acc
                if is_best:
                    self.best_acc = test_metrics['acc_final']
                    self.patience_counter = 0  # 重置计数器
                    print(f"  ✓ 新的最佳模型！准确率: {self.best_acc:.2f}%")
                else:
                    self.patience_counter += 1
                    print(f"  当前最佳准确率: {self.best_acc:.2f}% "
                          f"(patience: {self.patience_counter}/{self.args.patience})")
                
                self._save_checkpoint(epoch, test_metrics['acc_final'], is_best)
                
                # Early Stopping检查
                if self.args.early_stopping and self.patience_counter >= self.args.patience:
                    print(f"\n{'='*60}")
                    print(f"Early Stopping触发！")
                    print(f"测试准确率连续{self.args.patience}个评估周期未提升")
                    print(f"{'='*60}")
                    break
            else:
                # 不评估时也保存checkpoint
                self._save_checkpoint(epoch, self.best_acc, is_best=False)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
        
        print(f"\n{'='*60}")
        print("训练完成")
        print(f"{'='*60}")
        print(f"最佳测试准确率: {self.best_acc:.2f}%")
        
        self.writer.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RMSCM多模态手势识别训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='processed_dat',
                       help='数据目录')
    parser.add_argument('--subject', type=int, default=10,
                       help='受试者编号 (10, 23, 36), None表示使用合并数据')
    parser.add_argument('--emg_channels', type=int, default=12,
                       help='EMG通道数')
    parser.add_argument('--imu_channels', type=int, default=36,
                       help='IMU通道数')
    parser.add_argument('--num_classes', type=int, default=50,
                       help='类别数')
    parser.add_argument('--window_size', type=int, default=400,
                       help='窗口大小')
    
    # 模型参数
    parser.add_argument('--feature_dim', type=int, default=64,
                       help='特征维度')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='LSTM隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout比例')
    
    # 损失函数权重
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='EMG分类器损失权重')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='IMU分类器损失权重')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='融合分类器损失权重')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'], help='优化器')
    parser.add_argument('--scheduler', type=str, default='step',
                       choices=['step', 'cosine', 'none'], help='学习率调度器')
    parser.add_argument('--step_size', type=int, default=30,
                       help='StepLR的step_size')
    parser.add_argument('--gamma_lr', type=float, default=0.1,
                       help='StepLR的gamma')
    
    # 数据加载参数
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--load_to_memory', action='store_true',
                       help='是否将数据加载到内存')
    
    # 防止过拟合参数
    parser.add_argument('--early_stopping', action='store_true',
                       help='是否启用Early Stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early Stopping的patience（连续多少个评估周期不提升就停止）')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='梯度裁剪阈值（0表示不裁剪）')
    parser.add_argument('--eval_freq', type=int, default=1,
                       help='评估频率（每隔几个epoch评估一次）')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='checkpoint保存目录')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='日志保存目录')
    parser.add_argument('--print_freq', type=int, default=10,
                       help='打印频率')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='保存checkpoint频率')
    parser.add_argument('--resume', type=str, default='',
                       help='恢复训练的checkpoint路径')
    
    args = parser.parse_args()
    return args


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建训练器
    trainer = Trainer(args)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()

