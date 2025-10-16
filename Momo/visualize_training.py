"""
训练过程可视化工具
从TensorBoard日志中读取数据并生成可视化图表
"""

import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_data(log_dir):
    """从TensorBoard日志中加载数据"""
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    data = {}
    
    # 获取所有标量数据
    tags = ea.Tags()['scalars']
    
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data


def plot_losses(data, output_dir):
    """绘制损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'Train/Loss' in data:
        ax.plot(data['Train/Loss']['steps'], 
               data['Train/Loss']['values'], 
               label='训练损失', linewidth=2)
    
    if 'Test/Loss' in data:
        ax.plot(data['Test/Loss']['steps'], 
               data['Test/Loss']['values'], 
               label='测试损失', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('损失', fontsize=12)
    ax.set_title('训练和测试损失曲线', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("损失曲线已保存: loss_curve.png")


def plot_accuracies(data, output_dir):
    """绘制准确率曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 训练准确率
    for key, label in [
        ('Train/Acc_EMG', 'EMG'),
        ('Train/Acc_IMU', 'IMU'),
        ('Train/Acc_Fusion', '融合'),
        ('Train/Acc_Final', '最终')
    ]:
        if key in data:
            ax1.plot(data[key]['steps'], data[key]['values'], 
                    label=label, linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('准确率 (%)', fontsize=12)
    ax1.set_title('训练准确率', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # 测试准确率
    for key, label in [
        ('Test/Acc_EMG', 'EMG'),
        ('Test/Acc_IMU', 'IMU'),
        ('Test/Acc_Fusion', '融合'),
        ('Test/Acc_Final', '最终')
    ]:
        if key in data:
            ax2.plot(data[key]['steps'], data[key]['values'], 
                    label=label, linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.set_title('测试准确率', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("准确率曲线已保存: accuracy_curves.png")


def plot_learning_rate(data, output_dir):
    """绘制学习率变化曲线"""
    if 'Train/LR' not in data:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(data['Train/LR']['steps'], 
           data['Train/LR']['values'], 
           linewidth=2, color='orange')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('学习率', fontsize=12)
    ax.set_title('学习率变化', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("学习率曲线已保存: learning_rate.png")


def plot_comparison(data, output_dir):
    """绘制训练vs测试对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 最终准确率对比
    if 'Train/Acc_Final' in data and 'Test/Acc_Final' in data:
        train_steps = data['Train/Acc_Final']['steps']
        train_acc = data['Train/Acc_Final']['values']
        test_steps = data['Test/Acc_Final']['steps']
        test_acc = data['Test/Acc_Final']['values']
        
        ax.plot(train_steps, train_acc, label='训练准确率', linewidth=2)
        ax.plot(test_steps, test_acc, label='测试准确率', linewidth=2)
        
        # 计算过拟合程度
        if len(train_steps) == len(test_steps):
            gap = np.array(train_acc) - np.array(test_acc)
            ax2 = ax.twinx()
            ax2.plot(train_steps, gap, label='训练-测试差距', 
                    linewidth=2, color='red', alpha=0.5, linestyle='--')
            ax2.set_ylabel('准确率差距 (%)', fontsize=12, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('准确率 (%)', fontsize=12)
    ax.set_title('训练vs测试对比', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'train_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("训练测试对比图已保存: train_test_comparison.png")


def generate_summary(data, output_file):
    """生成训练总结"""
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("训练总结\n")
        f.write("="*60 + "\n\n")
        
        # 最终结果
        f.write("最终结果:\n")
        f.write("-"*60 + "\n")
        
        for key, name in [
            ('Train/Acc_Final', '训练最终准确率'),
            ('Test/Acc_Final', '测试最终准确率'),
            ('Test/Acc_EMG', '测试EMG准确率'),
            ('Test/Acc_IMU', '测试IMU准确率'),
            ('Test/Acc_Fusion', '测试融合准确率')
        ]:
            if key in data and len(data[key]['values']) > 0:
                final_value = data[key]['values'][-1]
                f.write(f"  {name}: {final_value:.2f}%\n")
        
        # 最佳结果
        f.write("\n最佳结果:\n")
        f.write("-"*60 + "\n")
        
        for key, name in [
            ('Test/Acc_Final', '测试最终准确率'),
            ('Test/Acc_EMG', '测试EMG准确率'),
            ('Test/Acc_IMU', '测试IMU准确率'),
            ('Test/Acc_Fusion', '测试融合准确率')
        ]:
            if key in data and len(data[key]['values']) > 0:
                best_value = max(data[key]['values'])
                best_epoch = data[key]['steps'][np.argmax(data[key]['values'])]
                f.write(f"  {name}: {best_value:.2f}% (Epoch {best_epoch})\n")
        
        # 训练稳定性
        if 'Test/Acc_Final' in data and len(data['Test/Acc_Final']['values']) > 0:
            test_acc = data['Test/Acc_Final']['values']
            f.write("\n训练稳定性:\n")
            f.write("-"*60 + "\n")
            f.write(f"  最后10个epoch平均准确率: {np.mean(test_acc[-10:]):.2f}%\n")
            f.write(f"  最后10个epoch标准差: {np.std(test_acc[-10:]):.2f}%\n")
    
    print(f"训练总结已保存: {output_file.name}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='可视化训练过程')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='TensorBoard日志目录')
    parser.add_argument('--output_dir', type=str, default='visualization_results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"\n{'='*60}")
    print("加载TensorBoard数据")
    print(f"{'='*60}")
    print(f"日志目录: {args.log_dir}")
    
    data = load_tensorboard_data(args.log_dir)
    print(f"加载了 {len(data)} 个指标")
    
    # 生成可视化
    print(f"\n{'='*60}")
    print("生成可视化图表")
    print(f"{'='*60}")
    
    plot_losses(data, output_dir)
    plot_accuracies(data, output_dir)
    plot_learning_rate(data, output_dir)
    plot_comparison(data, output_dir)
    
    # 生成总结
    print(f"\n{'='*60}")
    print("生成训练总结")
    print(f"{'='*60}")
    
    generate_summary(data, output_dir / 'training_summary.txt')
    
    print(f"\n{'='*60}")
    print("可视化完成")
    print(f"{'='*60}")
    print(f"结果保存在: {output_dir}")


if __name__ == '__main__':
    main()

