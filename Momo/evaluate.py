"""
RMSCM多模态手势识别模型评估脚本
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))

from model.rmscm_model import MultiModalRMSCM
from dataset import create_dataloaders


class Evaluator:
    """评估器类"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据加载器
        self._init_dataloaders()
        
        # 初始化模型
        self._init_model()
        
        # 加载checkpoint
        self._load_checkpoint()
    
    def _init_dataloaders(self):
        """初始化数据加载器"""
        print(f"\n{'='*60}")
        print("初始化数据加载器")
        print(f"{'='*60}")
        
        _, self.test_loader = create_dataloaders(
            data_dir=self.args.data_dir,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            mode='both',
            load_to_memory=False,
            subject=self.args.subject
        )
        
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
            dropout=0.0  # 评估时不使用dropout
        ).to(self.device)
    
    def _load_checkpoint(self):
        """加载checkpoint"""
        print(f"\n加载模型: {self.args.checkpoint}")
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已加载 (Epoch: {checkpoint['epoch']}, Best Acc: {checkpoint['best_acc']:.4f})")
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        
        all_labels = []
        all_pred_emg = []
        all_pred_imu = []
        all_pred_fusion = []
        all_pred_final = []
        
        print(f"\n{'='*60}")
        print("开始评估")
        print(f"{'='*60}")
        
        with torch.no_grad():
            for batch_idx, (emg, imu, labels) in enumerate(self.test_loader):
                # 数据移到设备
                emg = emg.to(self.device)
                imu = imu.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                logit_emg, logit_imu, logit_fusion, logit_final = self.model(emg, imu)
                
                # 预测
                _, pred_emg = torch.max(logit_emg, 1)
                _, pred_imu = torch.max(logit_imu, 1)
                _, pred_fusion = torch.max(logit_fusion, 1)
                _, pred_final = torch.max(logit_final, 1)
                
                # 保存结果
                all_labels.extend(labels.cpu().numpy())
                all_pred_emg.extend(pred_emg.cpu().numpy())
                all_pred_imu.extend(pred_imu.cpu().numpy())
                all_pred_fusion.extend(pred_fusion.cpu().numpy())
                all_pred_final.extend(pred_final.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"处理批次 [{batch_idx+1}/{len(self.test_loader)}]")
        
        # 转换为numpy数组
        all_labels = np.array(all_labels)
        all_pred_emg = np.array(all_pred_emg)
        all_pred_imu = np.array(all_pred_imu)
        all_pred_fusion = np.array(all_pred_fusion)
        all_pred_final = np.array(all_pred_final)
        
        # 计算准确率
        acc_emg = accuracy_score(all_labels, all_pred_emg) * 100
        acc_imu = accuracy_score(all_labels, all_pred_imu) * 100
        acc_fusion = accuracy_score(all_labels, all_pred_fusion) * 100
        acc_final = accuracy_score(all_labels, all_pred_final) * 100
        
        print(f"\n{'='*60}")
        print("评估结果")
        print(f"{'='*60}")
        print(f"EMG分类器准确率: {acc_emg:.2f}%")
        print(f"IMU分类器准确率: {acc_imu:.2f}%")
        print(f"融合分类器准确率: {acc_fusion:.2f}%")
        print(f"最终准确率: {acc_final:.2f}%")
        
        results = {
            'labels': all_labels,
            'pred_emg': all_pred_emg,
            'pred_imu': all_pred_imu,
            'pred_fusion': all_pred_fusion,
            'pred_final': all_pred_final,
            'acc_emg': acc_emg,
            'acc_imu': acc_imu,
            'acc_fusion': acc_fusion,
            'acc_final': acc_final
        }
        
        return results
    
    def plot_confusion_matrix(self, labels, predictions, title, filename):
        """绘制混淆矩阵"""
        cm = confusion_matrix(labels, predictions)
        
        # 如果类别太多，只显示前20个类别
        if self.args.num_classes > 20:
            top_classes = np.argsort(np.bincount(labels))[-20:]
            mask = np.isin(labels, top_classes)
            cm = confusion_matrix(labels[mask], predictions[mask], labels=top_classes)
            class_labels = [f"C{i}" for i in top_classes]
        else:
            class_labels = [f"C{i}" for i in range(self.args.num_classes)]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title(title)
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"混淆矩阵已保存: {filename}")
    
    def plot_per_class_accuracy(self, labels, predictions, title, filename):
        """绘制每类准确率"""
        unique_classes = np.unique(labels)
        per_class_acc = []
        
        for cls in unique_classes:
            mask = labels == cls
            if np.sum(mask) > 0:
                acc = np.mean(predictions[mask] == cls) * 100
                per_class_acc.append(acc)
            else:
                per_class_acc.append(0)
        
        plt.figure(figsize=(15, 6))
        plt.bar(range(len(unique_classes)), per_class_acc, color='steelblue')
        plt.xlabel('类别')
        plt.ylabel('准确率 (%)')
        plt.title(title)
        plt.ylim([0, 105])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"每类准确率图已保存: {filename}")
    
    def save_classification_report(self, results):
        """保存分类报告"""
        report_file = self.output_dir / 'classification_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("分类报告\n")
            f.write("="*60 + "\n\n")
            
            for name, preds, acc in [
                ('EMG分类器', results['pred_emg'], results['acc_emg']),
                ('IMU分类器', results['pred_imu'], results['acc_imu']),
                ('融合分类器', results['pred_fusion'], results['acc_fusion']),
                ('最终预测', results['pred_final'], results['acc_final'])
            ]:
                f.write(f"\n{name} (准确率: {acc:.2f}%)\n")
                f.write("-"*60 + "\n")
                report = classification_report(results['labels'], preds, 
                                             target_names=[f"Class_{i}" for i in range(self.args.num_classes)],
                                             zero_division=0)
                f.write(report)
                f.write("\n")
        
        print(f"分类报告已保存: {report_file}")
    
    def save_results(self, results):
        """保存评估结果"""
        # 保存numpy数组
        np.savez(self.output_dir / 'predictions.npz',
                labels=results['labels'],
                pred_emg=results['pred_emg'],
                pred_imu=results['pred_imu'],
                pred_fusion=results['pred_fusion'],
                pred_final=results['pred_final'])
        
        print(f"预测结果已保存: predictions.npz")
    
    def run(self):
        """运行完整评估流程"""
        # 评估
        results = self.evaluate()
        
        # 保存结果
        self.save_results(results)
        
        # 生成混淆矩阵
        print(f"\n{'='*60}")
        print("生成可视化")
        print(f"{'='*60}")
        
        self.plot_confusion_matrix(results['labels'], results['pred_emg'],
                                   'EMG分类器混淆矩阵', 'confusion_matrix_emg.png')
        self.plot_confusion_matrix(results['labels'], results['pred_imu'],
                                   'IMU分类器混淆矩阵', 'confusion_matrix_imu.png')
        self.plot_confusion_matrix(results['labels'], results['pred_fusion'],
                                   '融合分类器混淆矩阵', 'confusion_matrix_fusion.png')
        self.plot_confusion_matrix(results['labels'], results['pred_final'],
                                   '最终预测混淆矩阵', 'confusion_matrix_final.png')
        
        # 生成每类准确率图
        self.plot_per_class_accuracy(results['labels'], results['pred_emg'],
                                     'EMG分类器每类准确率', 'per_class_acc_emg.png')
        self.plot_per_class_accuracy(results['labels'], results['pred_imu'],
                                     'IMU分类器每类准确率', 'per_class_acc_imu.png')
        self.plot_per_class_accuracy(results['labels'], results['pred_fusion'],
                                     '融合分类器每类准确率', 'per_class_acc_fusion.png')
        self.plot_per_class_accuracy(results['labels'], results['pred_final'],
                                     '最终预测每类准确率', 'per_class_acc_final.png')
        
        # 保存分类报告
        self.save_classification_report(results)
        
        print(f"\n{'='*60}")
        print("评估完成")
        print(f"{'='*60}")
        print(f"结果保存在: {self.output_dir}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RMSCM多模态手势识别评估')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='processed_data',
                       help='数据目录')
    parser.add_argument('--subject', type=int, default=10,
                       help='受试者编号')
    parser.add_argument('--emg_channels', type=int, default=12,
                       help='EMG通道数')
    parser.add_argument('--imu_channels', type=int, default=36,
                       help='IMU通道数')
    parser.add_argument('--num_classes', type=int, default=50,
                       help='类别数')
    
    # 模型参数
    parser.add_argument('--feature_dim', type=int, default=64,
                       help='特征维度')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='LSTM隐藏层维度')
    
    # 评估参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='结果保存目录')
    
    args = parser.parse_args()
    return args


def main():
    """主函数"""
    args = parse_args()
    
    # 创建评估器
    evaluator = Evaluator(args)
    
    # 运行评估
    evaluator.run()


if __name__ == '__main__':
    main()

