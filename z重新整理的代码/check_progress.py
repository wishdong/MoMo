"""
实验进度检查工具

功能：
1. 扫描results目录，统计各阶段完成情况
2. 显示进度条和百分比
3. 列出失败/缺失的实验
4. 估算剩余时间
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import argparse


# 实验配置（根据实际可用数据更新）
EXPERIMENT_PLAN = {
    'stage1_ablation': {
        'DB5': {'subjects': [1, 5, 10], 'experiments': ['M0_base', 'M1_disentangle', 'M3_full', 
                                                        'D1_private_only', 'D2_shared_only',
                                                        'FA1_no_constraint', 'FA2_align_only', 'FA3_balance_only']},
        'DB7': {'subjects': [3, 7, 11], 'experiments': ['M0_base', 'M1_disentangle', 'M3_full',
                                                        'D1_private_only', 'D2_shared_only',
                                                        'FA1_no_constraint', 'FA2_align_only', 'FA3_balance_only']}
    },
    'stage2_full_model': {
        'DB2': {'subjects': list(range(1, 41)), 'experiments': ['M3_full']},          # S1-S40 (40个)
        'DB3': {'subjects': [2, 4, 5, 6, 9, 11], 'experiments': ['M3_full']},       # 6个
        'DB5': {'subjects': list(range(1, 11)), 'experiments': ['M3_full']},         # S1-S10 (10个)
        'DB7': {'subjects': list(range(2, 13)), 'experiments': ['M3_full']}          # S2-S12 (11个)
    },
    'stage3_hyperparam_grid': {
        'DB3': {'subjects': [2, 6, 11], 'experiments': 'GRID'},
        'DB5': {'subjects': [1, 5, 10], 'experiments': 'GRID'}  # 改为DB3和DB5
    },
    'stage4_hyperparam_lambda': {
        'DB2': {'subjects': [10, 20, 30], 'experiments': 'LAMBDA'},
        'DB3': {'subjects': [2, 6, 11], 'experiments': 'LAMBDA'},
        'DB5': {'subjects': [1, 5, 10], 'experiments': 'LAMBDA'}  # 新增
    }
}

# 实际可用的受试者列表
AVAILABLE_SUBJECTS = {
    'DB2': list(range(1, 41)),              # S1-S40 (40个)
    'DB3': [2, 4, 5, 6, 9, 11],            # 6个
    'DB5': list(range(1, 11)),              # S1-S10 (10个)
    'DB7': list(range(2, 13))               # S2-S12 (11个)
}

# 生成超参数组合
def generate_hp_grid():
    """生成α和β的网格组合"""
    alpha_values = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    beta_values = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    
    experiments = []
    for alpha in alpha_values:
        for beta in beta_values:
            experiments.append(f'HP_a{alpha}_b{beta}')
    
    return experiments

def generate_lambda_experiments():
    """生成λ搜索的实验列表"""
    lambda_align_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    lambda_balance_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    experiments = []
    for val in lambda_align_values:
        experiments.append(f'HP_align{val}')
    for val in lambda_balance_values:
        experiments.append(f'HP_balance{val}')
    
    return experiments


def check_experiment_status(dataset, subject, experiment_id, results_dir='./results'):
    """
    检查单个实验的状态
    
    Returns:
        'completed': 完成
        'incomplete': 不完整（只有部分文件）
        'missing': 缺失
    """
    result_dir = Path(results_dir) / dataset / f'subject{subject}' / experiment_id
    
    metrics_file = result_dir / 'metrics.json'
    predictions_file = result_dir / 'predictions.pkl'
    
    if metrics_file.exists() and predictions_file.exists():
        return 'completed'
    elif metrics_file.exists() or predictions_file.exists():
        return 'incomplete'
    else:
        return 'missing'


def print_progress_bar(completed, total, prefix='', length=50):
    """打印进度条"""
    if total == 0:
        percent = 0
    else:
        percent = 100 * (completed / total)
    
    filled = int(length * completed / total) if total > 0 else 0
    bar = '█' * filled + '-' * (length - filled)
    
    print(f'\r{prefix} |{bar}| {completed}/{total} ({percent:.1f}%)', end='')


def check_stage_progress(stage_name, stage_config, results_dir='./results', verbose=False):
    """检查某个阶段的进度"""
    print(f"\n{'='*80}")
    print(f"📊 {stage_name}")
    print('='*80)
    
    total_tasks = 0
    completed_tasks = 0
    incomplete_tasks = 0
    missing_tasks = 0
    
    missing_list = []
    incomplete_list = []
    
    for dataset, config in stage_config.items():
        subjects = config['subjects']
        experiments = config['experiments']
        
        # 处理特殊类型
        if experiments == 'GRID':
            experiments = generate_hp_grid()
        elif experiments == 'LAMBDA':
            experiments = generate_lambda_experiments()
        
        dataset_completed = 0
        dataset_total = len(subjects) * len(experiments)
        
        for subject in subjects:
            for exp_id in experiments:
                total_tasks += 1
                status = check_experiment_status(dataset, subject, exp_id, results_dir)
                
                if status == 'completed':
                    completed_tasks += 1
                    dataset_completed += 1
                elif status == 'incomplete':
                    incomplete_tasks += 1
                    incomplete_list.append(f"{dataset}-S{subject}-{exp_id}")
                else:
                    missing_tasks += 1
                    missing_list.append(f"{dataset}-S{subject}-{exp_id}")
        
        # 打印每个数据集的进度
        percent = 100 * dataset_completed / dataset_total if dataset_total > 0 else 0
        print(f"  {dataset}: {dataset_completed}/{dataset_total} ({percent:.1f}%)")
    
    # 打印总体进度
    print(f"\n📈 总体进度:")
    print(f"  ✅ 已完成: {completed_tasks}")
    print(f"  ⚠️  不完整: {incomplete_tasks}")
    print(f"  ❌ 缺失:   {missing_tasks}")
    print(f"  📊 总计:   {total_tasks}")
    print_progress_bar(completed_tasks, total_tasks, '  进度')
    print()  # 换行
    
    # 详细列出问题
    if verbose:
        if incomplete_list:
            print(f"\n⚠️  不完整的实验（{len(incomplete_list)}个）:")
            for item in incomplete_list[:10]:  # 只显示前10个
                print(f"    - {item}")
            if len(incomplete_list) > 10:
                print(f"    ... 还有 {len(incomplete_list)-10} 个")
        
        if missing_list:
            print(f"\n❌ 缺失的实验（{len(missing_list)}个）:")
            for item in missing_list[:10]:  # 只显示前10个
                print(f"    - {item}")
            if len(missing_list) > 10:
                print(f"    ... 还有 {len(missing_list)-10} 个")
    
    return {
        'total': total_tasks,
        'completed': completed_tasks,
        'incomplete': incomplete_tasks,
        'missing': missing_tasks,
        'missing_list': missing_list,
        'incomplete_list': incomplete_list
    }


def estimate_remaining_time(missing_count, avg_time_per_exp=30):
    """
    估算剩余时间
    
    Args:
        missing_count: 缺失的实验数量
        avg_time_per_exp: 单个实验的平均时间（分钟）
    
    Returns:
        str: 格式化的时间字符串
    """
    total_minutes = missing_count * avg_time_per_exp
    hours = total_minutes / 60
    
    if hours < 1:
        return f"{total_minutes:.0f}分钟"
    elif hours < 24:
        return f"{hours:.1f}小时"
    else:
        days = hours / 24
        return f"{days:.1f}天"


def main():
    parser = argparse.ArgumentParser(description='检查实验进度')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='结果目录')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'stage1', 'stage2', 'stage3', 'stage4'],
                        help='检查哪个阶段')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细信息（列出缺失的实验）')
    parser.add_argument('--export', type=str, default=None,
                        help='导出缺失实验列表到文件')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 实验进度检查工具")
    print("=" * 80)
    
    all_stats = {}
    
    # 检查各阶段
    if args.stage == 'all' or args.stage == 'stage1':
        stats = check_stage_progress('阶段1：消融实验（DB5, DB7）', 
                                     EXPERIMENT_PLAN['stage1_ablation'],
                                     args.results_dir, args.verbose)
        all_stats['stage1'] = stats
    
    if args.stage == 'all' or args.stage == 'stage2':
        stats = check_stage_progress('阶段2：完整模型全受试者（所有数据集）',
                                     EXPERIMENT_PLAN['stage2_full_model'],
                                     args.results_dir, args.verbose)
        all_stats['stage2'] = stats
    
    if args.stage == 'all' or args.stage == 'stage3':
        stats = check_stage_progress('阶段3：超参数网格搜索（α和β）',
                                     EXPERIMENT_PLAN['stage3_hyperparam_grid'],
                                     args.results_dir, args.verbose)
        all_stats['stage3'] = stats
    
    if args.stage == 'all' or args.stage == 'stage4':
        stats = check_stage_progress('阶段4：超参数搜索（λ）',
                                     EXPERIMENT_PLAN['stage4_hyperparam_lambda'],
                                     args.results_dir, args.verbose)
        all_stats['stage4'] = stats
    
    # ============================================================
    # 总体统计
    # ============================================================
    if args.stage == 'all':
        print(f"\n{'='*80}")
        print("📊 总体统计")
        print('='*80)
        
        total_all = sum(s['total'] for s in all_stats.values())
        completed_all = sum(s['completed'] for s in all_stats.values())
        missing_all = sum(s['missing'] for s in all_stats.values())
        incomplete_all = sum(s['incomplete'] for s in all_stats.values())
        
        print(f"  ✅ 已完成: {completed_all} / {total_all} ({100*completed_all/total_all:.1f}%)")
        print(f"  ❌ 缺失:   {missing_all}")
        print(f"  ⚠️  不完整: {incomplete_all}")
        
        # 估算剩余时间
        if missing_all > 0:
            print(f"\n⏱️  估算剩余时间:")
            print(f"  单GPU串行: {estimate_remaining_time(missing_all, 30)}")
            print(f"  5个GPU并行: {estimate_remaining_time(missing_all/5, 30)}")
        
        print_progress_bar(completed_all, total_all, '  总进度')
        print()
    
    # ============================================================
    # 导出缺失列表
    # ============================================================
    if args.export and args.stage == 'all':
        all_missing = []
        for stage_name, stats in all_stats.items():
            all_missing.extend(stats['missing_list'])
        
        with open(args.export, 'w') as f:
            for item in all_missing:
                f.write(f"{item}\n")
        
        print(f"\n💾 缺失实验列表已导出至: {args.export}")
        print(f"   总计: {len(all_missing)} 个")


if __name__ == '__main__':
    main()

