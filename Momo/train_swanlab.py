"""
RMSCM多模态手势识别模型训练脚本 - SwanLab集成版本
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
import swanlab

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from model.rmscm_model import MultiModalRMSCM, MultiTaskLoss, get_model_info
from dataset import create_dataloaders


class Trainer:
    """训练器类 - 集成SwanLab监控"""
    
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device(args)
        
        # 创建保存目录
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        if args.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None
        
        # 初始化数据加载器
        self._init_dataloaders()
        
        # 初始化模型
        self._init_model()
        
        # 初始化优化器和学习率调度器
        self._init_optimizer()
        
        # SwanLab初始化（在模型创建之后）
        if args.use_swanlab:
            self._init_swanlab()
        
        # 记录最佳准确率和Early Stopping
        self.best_acc = 0.0
        self.start_epoch = 0
        self.patience_counter = 0  # Early Stopping计数器
        
        # 如果指定了恢复训练的checkpoint
        if args.resume:
            self._load_checkpoint(args.resume)
    
    def _setup_device(self, args):
        """设置计算设备（支持GPU选择）"""
        if not torch.cuda.is_available():
            print("⚠️ CUDA不可用，使用CPU")
            return torch.device('cpu')
        
        # 显示可用GPU信息
        gpu_count = torch.cuda.device_count()
        print(f"\n{'='*60}")
        print("GPU设备信息")
        print(f"{'='*60}")
        print(f"可用GPU数量: {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1024**3:.1f} GB)")
        
        # 处理设备选择
        if args.device == 'cpu':
            print("✅ 使用CPU进行训练")
            return torch.device('cpu')
        elif args.device == 'cuda' or args.device.startswith('cuda:'):
            if args.device == 'cuda':
                # 使用默认GPU（GPU 0）
                device_id = 0
                device = torch.device('cuda:0')
            else:
                # 解析指定的GPU ID
                try:
                    device_id = int(args.device.split(':')[1])
                    if device_id >= gpu_count:
                        print(f"❌ GPU {device_id} 不存在，使用GPU 0")
                        device_id = 0
                    device = torch.device(f'cuda:{device_id}')
                except (ValueError, IndexError):
                    print(f"❌ 无效的设备格式: {args.device}，使用GPU 0")
                    device_id = 0
                    device = torch.device('cuda:0')
            
            # 设置当前GPU
            torch.cuda.set_device(device_id)
            
            # 显示GPU使用信息
            print(f"✅ 使用GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
            
            # 显示GPU内存信息
            memory_allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device_id) / 1024**3  
            memory_total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            print(f"GPU内存: {memory_allocated:.2f} GB / {memory_total:.1f} GB (已分配)")
            print(f"GPU内存: {memory_reserved:.2f} GB / {memory_total:.1f} GB (已预留)")
            
            return device
        else:
            print(f"❌ 未知设备: {args.device}，使用GPU 0")
            torch.cuda.set_device(0)
            return torch.device('cuda:0')
    
    def _init_swanlab(self):
        """初始化SwanLab"""
        print(f"\n{'='*60}")
        print("初始化SwanLab监控")
        print(f"{'='*60}")
        
        # 获取模型信息
        try:
            model_info = get_model_info(self.model, self.args.emg_channels, 
                                      self.args.imu_channels, self.args.window_size)
        except Exception as e:
            print(f"⚠️ 无法获取模型信息: {e}")
            model_info = None
        
        # 实验配置
        experiment_config = {
            # 模型参数
            "model_name": "MultiModalRMSCM",
            "emg_channels": self.args.emg_channels,
            "imu_channels": self.args.imu_channels, 
            "num_classes": self.args.num_classes,
            "feature_dim": self.args.feature_dim,
            "hidden_dim": self.args.hidden_dim,
            "dropout": self.args.dropout,
            
            # 训练参数
            "epochs": self.args.epochs,
            "batch_size": self.args.batch_size,
            "learning_rate": self.args.lr,
            "weight_decay": self.args.weight_decay,
            "optimizer": self.args.optimizer,
            "scheduler": self.args.scheduler,
            "device": str(self.device),
            
            # 数据参数
            "subject": self.args.subject,
            "window_size": self.args.window_size,
            
            # 损失函数参数
            "alpha": self.args.alpha,
            "beta": self.args.beta, 
            "gamma": self.args.gamma,
        }
        
        # 添加模型信息到配置（如果可用）
        if model_info:
            experiment_config.update({
                "model_parameters_M": model_info['total_parameters_M'],
                "emg_input_shape": str(model_info['input_shape_emg']),
                "imu_input_shape": str(model_info['input_shape_imu']),
                "emg_output_shape": str(model_info['output_shape_emg']),
                "imu_output_shape": str(model_info['output_shape_imu']),
                "fusion_output_shape": str(model_info['output_shape_fusion']),
                "final_output_shape": str(model_info['output_shape_final'])
            })
        
        # 初始化SwanLab实验
        swanlab.init(
            project=self.args.swanlab_project,
            experiment_name=self._get_experiment_name(),
            description=f"EMG+IMU多模态手势识别 - 受试者{self.args.subject}",
            config=experiment_config,
            logdir=self.args.swanlab_logdir if self.args.swanlab_logdir else None
        )
        
        print("SwanLab初始化完成")
    
    def _get_experiment_name(self):
        """生成实验名称"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"RMSCM_S{self.args.subject}_{self.args.optimizer}_lr{self.args.lr}_{timestamp}"
    
    def _init_dataloaders(self):
        """初始化数据加载器"""
        print(f"\n{'='*60}")
        print("初始化数据加载器")
        print(f"{'='*60}")
        
        # 强制使用内存加载（解决HDF5读取缓慢问题）
        self.train_loader, self.test_loader = create_dataloaders(
            data_dir=self.args.data_dir,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            mode='both',
            load_to_memory=True,  # 强制加载到内存，解决I/O瓶颈
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
        
        # 模型信息已经在SwanLab配置中记录，这里不需要额外记录
        
        # 初始化损失函数
        self.criterion = MultiTaskLoss(
            alpha=self.args.alpha,
            beta=self.args.beta, 
            gamma=self.args.gamma
        )
    
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        print(f"\n{'='*60}")
        print("初始化优化器")
        print(f"{'='*60}")
        
        # 优化器
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
        
        print(f"优化器: {self.args.optimizer}")
        print(f"学习率: {self.args.lr}")
        print(f"权重衰减: {self.args.weight_decay}")
        
        # 学习率调度器
        self.scheduler = None
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
        
        # 时间测量变量
        epoch_start_time = time.time()
        step_start_time = time.time()
        step_times = []  # 存储每个step的时间
        
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
            
            # 打印进度（增强版，包含详细时间信息）
            if (batch_idx + 1) % self.args.print_freq == 0:
                # 计算当前step的时间
                current_time = time.time()
                step_time = current_time - step_start_time
                step_times.append(step_time)
                
                # 计算平均批次时间和step时间
                avg_batch_time = step_time / self.args.print_freq
                avg_step_time = sum(step_times) / len(step_times)
                
                # 预估剩余时间
                remaining_batches = len(self.train_loader) - (batch_idx + 1)
                remaining_time_epoch = remaining_batches * avg_batch_time
                
                # 预估总剩余时间（包括剩余epochs）
                remaining_epochs = self.args.epochs - epoch
                if len(step_times) > 1:  # 有足够数据进行预估
                    estimated_epoch_time = len(self.train_loader) * avg_batch_time
                    total_remaining_time = remaining_time_epoch + (remaining_epochs - 1) * estimated_epoch_time
                else:
                    total_remaining_time = 0
                
                # 格式化时间显示
                def format_time(seconds):
                    if seconds < 60:
                        return f"{seconds:.1f}s"
                    elif seconds < 3600:
                        return f"{seconds/60:.1f}m"
                    else:
                        return f"{seconds/3600:.1f}h"
                
                # 计算当前指标
                avg_loss = total_loss / (batch_idx + 1)
                acc_final = 100.0 * correct_final / total_samples
                progress_percent = (batch_idx + 1) / len(self.train_loader) * 100
                
                # 打印详细进度信息
                print(f"Epoch [{epoch+1}/{self.args.epochs}] "
                      f"Step [{(batch_idx+1)//self.args.print_freq}] "
                      f"Batch [{batch_idx+1}/{len(self.train_loader)}] ({progress_percent:.1f}%)")
                print(f"  Loss: {avg_loss:.4f} | Acc: {acc_final:.2f}%")
                print(f"  ⏱️  Step时间: {format_time(step_time)} "
                      f"| 平均batch: {avg_batch_time*1000:.1f}ms "
                      f"| 平均step: {format_time(avg_step_time)}")
                print(f"  🕒 本轮剩余: {format_time(remaining_time_epoch)} "
                      f"| 训练剩余: {format_time(total_remaining_time)}")
                
                # GPU内存使用情况
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated(self.device) / 1024**3
                    gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                    gpu_utilization = gpu_memory_used / gpu_memory_total * 100
                    print(f"  📊 GPU内存: {gpu_memory_used:.1f}GB/{gpu_memory_total:.1f}GB ({gpu_utilization:.1f}%)")
                
                print("-" * 80)
                
                # 重置step计时器
                step_start_time = current_time
        
        # Epoch统计
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(self.train_loader)
        avg_loss_emg = loss_emg_total / len(self.train_loader)
        avg_loss_imu = loss_imu_total / len(self.train_loader)
        avg_loss_fusion = loss_fusion_total / len(self.train_loader)
        
        acc_emg = 100.0 * correct_emg / total_samples
        acc_imu = 100.0 * correct_imu / total_samples
        acc_fusion = 100.0 * correct_fusion / total_samples
        acc_final = 100.0 * correct_final / total_samples
        
        # 计算时间相关指标
        avg_batch_time = epoch_time / len(self.train_loader)
        avg_step_time = sum(step_times) / len(step_times) if step_times else epoch_time / (len(self.train_loader) // self.args.print_freq)
        samples_per_second = total_samples / epoch_time
        
        # 记录到监控系统
        train_metrics = {
            'train_loss': avg_loss,
            'train_loss_emg': avg_loss_emg,
            'train_loss_imu': avg_loss_imu,
            'train_loss_fusion': avg_loss_fusion,
            'train_acc_emg': acc_emg,
            'train_acc_imu': acc_imu,
            'train_acc_fusion': acc_fusion,
            'train_acc_final': acc_final,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time,
            'avg_batch_time_ms': avg_batch_time * 1000,  # 毫秒
            'avg_step_time': avg_step_time,
            'samples_per_second': samples_per_second,
            'gpu_memory_used_gb': torch.cuda.memory_allocated(self.device) / 1024**3 if torch.cuda.is_available() else 0
        }
        
        # TensorBoard记录
        if self.writer:
            self.writer.add_scalar('Train/Loss', avg_loss, epoch)
            self.writer.add_scalar('Train/Loss_EMG', avg_loss_emg, epoch)
            self.writer.add_scalar('Train/Loss_IMU', avg_loss_imu, epoch)
            self.writer.add_scalar('Train/Loss_Fusion', avg_loss_fusion, epoch)
            self.writer.add_scalar('Train/Acc_EMG', acc_emg, epoch)
            self.writer.add_scalar('Train/Acc_IMU', acc_imu, epoch)
            self.writer.add_scalar('Train/Acc_Fusion', acc_fusion, epoch)
            self.writer.add_scalar('Train/Acc_Final', acc_final, epoch)
            self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], epoch)
            # 时间相关指标
            self.writer.add_scalar('Performance/Epoch_Time', epoch_time, epoch)
            self.writer.add_scalar('Performance/Avg_Batch_Time_ms', avg_batch_time * 1000, epoch)
            self.writer.add_scalar('Performance/Samples_Per_Second', samples_per_second, epoch)
            if torch.cuda.is_available():
                self.writer.add_scalar('Performance/GPU_Memory_GB', torch.cuda.memory_allocated(self.device) / 1024**3, epoch)
        
        # SwanLab记录
        if self.args.use_swanlab:
            swanlab.log(train_metrics, step=epoch)
        
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
        loss_emg_total = 0.0
        loss_imu_total = 0.0
        loss_fusion_total = 0.0
        
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
        
        # 统计
        avg_loss = total_loss / len(self.test_loader)
        avg_loss_emg = loss_emg_total / len(self.test_loader)
        avg_loss_imu = loss_imu_total / len(self.test_loader)
        avg_loss_fusion = loss_fusion_total / len(self.test_loader)
        
        acc_emg = 100.0 * correct_emg / total_samples
        acc_imu = 100.0 * correct_imu / total_samples
        acc_fusion = 100.0 * correct_fusion / total_samples
        acc_final = 100.0 * correct_final / total_samples
        
        # 记录到监控系统
        test_metrics = {
            'test_loss': avg_loss,
            'test_loss_emg': avg_loss_emg,
            'test_loss_imu': avg_loss_imu,
            'test_loss_fusion': avg_loss_fusion,
            'test_acc_emg': acc_emg,
            'test_acc_imu': acc_imu,
            'test_acc_fusion': acc_fusion,
            'test_acc_final': acc_final
        }
        
        # TensorBoard记录
        if self.writer:
            self.writer.add_scalar('Test/Loss', avg_loss, epoch)
            self.writer.add_scalar('Test/Loss_EMG', avg_loss_emg, epoch)
            self.writer.add_scalar('Test/Loss_IMU', avg_loss_imu, epoch)
            self.writer.add_scalar('Test/Loss_Fusion', avg_loss_fusion, epoch)
            self.writer.add_scalar('Test/Acc_EMG', acc_emg, epoch)
            self.writer.add_scalar('Test/Acc_IMU', acc_imu, epoch)
            self.writer.add_scalar('Test/Acc_Fusion', acc_fusion, epoch)
            self.writer.add_scalar('Test/Acc_Final', acc_final, epoch)
        
        # SwanLab记录
        if self.args.use_swanlab:
            swanlab.log(test_metrics, step=epoch)
        
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
        
        total_start_time = time.time()
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # 训练
            train_results = self.train_epoch(epoch)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 评估
            if (epoch + 1) % self.args.eval_freq == 0:
                test_results = self.evaluate(epoch)
                
                print(f"\nEpoch [{epoch+1}/{self.args.epochs}]")
                print(f"训练 - Loss: {train_results['loss']:.4f}, "
                      f"Acc(EMG/IMU/Fusion/Final): "
                      f"{train_results['acc_emg']:.2f}%/"
                      f"{train_results['acc_imu']:.2f}%/"
                      f"{train_results['acc_fusion']:.2f}%/"
                      f"{train_results['acc_final']:.2f}%")
                print(f"测试 - Loss: {test_results['loss']:.4f}, "
                      f"Acc(EMG/IMU/Fusion/Final): "
                      f"{test_results['acc_emg']:.2f}%/"
                      f"{test_results['acc_imu']:.2f}%/"
                      f"{test_results['acc_fusion']:.2f}%/"
                      f"{test_results['acc_final']:.2f}%")
                print(f"时间: {train_results['time']:.2f}s")
                
                # 保存最佳模型
                is_best = test_results['acc_final'] > self.best_acc
                if is_best:
                    self.best_acc = test_results['acc_final']
                    self.patience_counter = 0  # 重置patience计数器
                    
                    # 记录最佳准确率到SwanLab
                    if self.args.use_swanlab:
                        swanlab.log({
                            'best_accuracy': self.best_acc,
                            'best_epoch': epoch + 1
                        }, step=epoch)
                else:
                    self.patience_counter += 1
                
                # 保存checkpoint
                self._save_checkpoint(epoch, test_results['acc_final'], is_best)
                
                # Early Stopping检查
                if self.args.early_stopping and self.patience_counter >= self.args.patience:
                    print(f"\nEarly Stopping触发！连续{self.args.patience}个评估周期未提升")
                    print(f"最佳准确率: {self.best_acc:.4f}")
                    break
        
        # 训练完成
        total_time = time.time() - total_start_time
        print(f"\n{'='*60}")
        print("训练完成")
        print(f"{'='*60}")
        print(f"总时间: {total_time:.2f}s ({total_time/3600:.2f}h)")
        print(f"最佳准确率: {self.best_acc:.4f}")
        
        # 记录训练总结到SwanLab
        if self.args.use_swanlab:
            swanlab.log({
                'final_best_accuracy': self.best_acc,
                'total_training_time_hours': total_time / 3600,
                'training_completed': True
            })
            
            # 完成实验
            swanlab.finish()
        
        # 关闭TensorBoard
        if self.writer:
            self.writer.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RMSCM多模态手势识别训练 - SwanLab集成版本')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='processed_data',
                       help='预处理数据目录')
    parser.add_argument('--subject', type=int, default=10,
                       help='受试者编号 (10, 23, 36)')
    
    # 模型参数
    parser.add_argument('--emg_channels', type=int, default=12,
                       help='EMG通道数')
    parser.add_argument('--imu_channels', type=int, default=36,
                       help='IMU通道数')
    parser.add_argument('--num_classes', type=int, default=50,
                       help='类别数')
    parser.add_argument('--window_size', type=int, default=400,
                       help='窗口大小（样本点数）')
    parser.add_argument('--feature_dim', type=int, default=64,
                       help='特征维度')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout比率')
    
    # 损失函数参数
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='EMG损失权重')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='IMU损失权重')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Fusion损失权重')
    
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
    parser.add_argument('--grad_clip', type=float, default=0.0,
                       help='梯度裁剪阈值（0表示不裁剪）')
    parser.add_argument('--eval_freq', type=int, default=1,
                       help='评估频率（每隔几个epoch评估一次）')
    
    # 监控参数
    parser.add_argument('--use_tensorboard', action='store_true', default=False,
                       help='是否使用TensorBoard监控')
    parser.add_argument('--use_swanlab', action='store_true', default=True,
                       help='是否使用SwanLab监控')
    parser.add_argument('--swanlab_project', type=str, default='Momo-Gesture-Recognition',
                       help='SwanLab项目名称')
    parser.add_argument('--swanlab_logdir', type=str, default=None,
                       help='SwanLab日志目录（可选）')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备选择: cpu, cuda, cuda:0, cuda:1, ..., cuda:7 (默认: cuda:0)')
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
