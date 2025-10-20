"""
批量实验运行脚本

功能：
1. 读取experiments_config.json配置文件
2. 自动生成训练命令
3. 支持断点续训（检查已完成的实验）
4. 支持并行执行（多GPU）
5. 生成实验日志
"""

import json
import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import time


def check_experiment_completed(dataset, subject, experiment_id, results_dir='./results'):
    """
    检查实验是否已完成
    
    Args:
        dataset: 数据集名称
        subject: 受试者编号
        experiment_id: 实验ID
        results_dir: 结果目录
    
    Returns:
        bool: True表示已完成，False表示未完成
    """
    result_path = Path(results_dir) / dataset / f'subject{subject}' / experiment_id / 'metrics.json'
    pred_path = Path(results_dir) / dataset / f'subject{subject}' / experiment_id / 'predictions.pkl'
    
    # 两个文件都存在才算完成
    return result_path.exists() and pred_path.exists()


def get_all_subjects(dataset):
    """
    获取数据集的所有受试者编号（根据实际可用数据）
    
    Args:
        dataset: 数据集名称
    
    Returns:
        list: 受试者编号列表
    """
    dataset_subjects = {
        'DB2': list(range(1, 41)),      # S1-S40 (40个)
        'DB3': [2, 4, 5, 6, 9, 11],    # 6个
        'DB5': list(range(1, 11)),      # S1-S10 (10个)
        'DB7': list(range(2, 13))       # S2-S12 (11个)
    }
    return dataset_subjects.get(dataset, [])


def generate_command(dataset, subject, experiment_args, gpu=0, base_args=''):
    """
    生成训练命令
    
    Args:
        dataset: 数据集名称
        subject: 受试者编号
        experiment_args: 实验特定的参数
        gpu: GPU编号
        base_args: 基础参数（如batch_size, num_epochs等）
    
    Returns:
        str: 完整的训练命令
    """
    cmd = f"python train.py --dataset {dataset} --s {subject} --gpu {gpu} {base_args} {experiment_args}"
    return cmd


def run_experiment_group(config_section, config, args):
    """
    运行一组实验
    
    Args:
        config_section: 配置节名称
        config: 配置字典
        args: 命令行参数
    """
    print("=" * 80)
    print(f"📊 实验组: {config_section}")
    print(f"📝 描述: {config.get('description', 'N/A')}")
    print("=" * 80)
    
    datasets = config.get('datasets', [])
    experiments = config.get('experiments', [])
    
    if not datasets or not experiments:
        print(f"⚠️  跳过 {config_section}: 没有配置数据集或实验")
        return
    
    # 收集所有实验任务
    tasks = []
    
    for dataset in datasets:
        # 获取受试者列表
        if args.subjects == 'all':
            subjects = get_all_subjects(dataset)
        elif args.subjects == 'representative':
            # 使用代表性受试者
            rep_key = f"{dataset}_subjects"
            subjects = config.get(rep_key, get_all_subjects(dataset)[:3])
        else:
            # 解析指定的受试者
            subjects = []
            for part in args.subjects.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    subjects.extend(range(start, end + 1))
                else:
                    subjects.append(int(part))
        
        for subject in subjects:
            for exp in experiments:
                exp_id = exp['id']
                exp_args = exp['args']
                
                # 检查是否已完成
                if check_experiment_completed(dataset, subject, exp_id, args.results_dir):
                    if args.verbose:
                        print(f"✓ 已完成: {dataset} - S{subject} - {exp_id}")
                    continue
                
                # 生成命令
                cmd = generate_command(
                    dataset=dataset,
                    subject=subject,
                    experiment_args=exp_args,
                    gpu=args.gpu,
                    base_args=args.base_args
                )
                
                tasks.append({
                    'dataset': dataset,
                    'subject': subject,
                    'experiment_id': exp_id,
                    'command': cmd,
                    'description': exp.get('description', '')
                })
    
    # 显示任务统计
    print(f"\n📈 待执行任务: {len(tasks)} 个")
    
    if len(tasks) == 0:
        print("✅ 所有实验已完成！")
        return
    
    # 询问是否继续
    if not args.yes:
        response = input(f"\n是否开始执行这 {len(tasks)} 个任务？ (y/n): ")
        if response.lower() != 'y':
            print("❌ 用户取消")
            return
    
    # 执行任务
    success_count = 0
    fail_count = 0
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(tasks)}] {task['dataset']} - S{task['subject']} - {task['experiment_id']}")
        print(f"📝 {task['description']}")
        print(f"🖥️  命令: {task['command']}")
        print('='*80)
        
        if args.dry_run:
            print("💡 (--dry-run模式，不实际执行)")
            continue
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行命令
        try:
            result = subprocess.run(task['command'], shell=True, check=True)
            elapsed_time = time.time() - start_time
            
            print(f"\n✅ 完成！耗时: {elapsed_time/60:.1f} 分钟")
            success_count += 1
            
            # 记录到日志
            log_success(task, elapsed_time, args.log_file)
            
        except subprocess.CalledProcessError as e:
            elapsed_time = time.time() - start_time
            print(f"\n❌ 失败！错误码: {e.returncode}")
            fail_count += 1
            
            # 记录到日志
            log_failure(task, e, elapsed_time, args.log_file)
            
            # 如果设置了--stop-on-error，则停止
            if args.stop_on_error:
                print("⚠️  检测到错误，停止执行（--stop-on-error）")
                break
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 执行总结")
    print("=" * 80)
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {fail_count}")
    print(f"⏭️  跳过: {len(tasks) - success_count - fail_count}")
    print("=" * 80)


def run_hyperparameter_grid_search(args):
    """
    运行超参数网格搜索（alpha和beta）
    """
    print("=" * 80)
    print("🔬 超参数网格搜索：alpha 和 beta")
    print("=" * 80)
    
    # 读取配置
    with open('experiments_config.json', 'r', encoding='utf-8') as f:
        full_config = json.load(f)
    
    config = full_config.get('hyperparameter_search_alpha_beta', {})
    datasets = config.get('datasets', [])
    alpha_values = config.get('alpha_values', [])
    beta_values = config.get('beta_values', [])
    
    # 收集任务
    tasks = []
    
    for dataset in datasets:
        # 获取受试者
        if args.subjects == 'all':
            subjects = get_all_subjects(dataset)
        else:
            subjects = [int(s) for s in args.subjects.split(',')]
        
        for subject in subjects:
            for alpha in alpha_values:
                for beta in beta_values:
                    exp_id = f"HP_a{alpha}_b{beta}"
                    
                    # 检查是否已完成
                    if check_experiment_completed(dataset, subject, exp_id, args.results_dir):
                        if args.verbose:
                            print(f"✓ 已完成: {dataset} - S{subject} - {exp_id}")
                        continue
                    
                    # 生成命令
                    exp_args = f"--use-disentangle --alpha {alpha} --beta {beta} --save-predictions --experiment_id {exp_id}"
                    cmd = generate_command(dataset, subject, exp_args, args.gpu, args.base_args)
                    
                    tasks.append({
                        'dataset': dataset,
                        'subject': subject,
                        'experiment_id': exp_id,
                        'command': cmd,
                        'description': f'Alpha={alpha}, Beta={beta}'
                    })
    
    print(f"\n📈 待执行任务: {len(tasks)} 个（{len(alpha_values)}×{len(beta_values)} 网格）")
    
    # 后续执行逻辑同run_experiment_group
    # ...(省略，逻辑相同)


def log_success(task, elapsed_time, log_file):
    """记录成功的实验"""
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] ✅ SUCCESS | {task['dataset']} | S{task['subject']} | {task['experiment_id']} | {elapsed_time/60:.1f}min\n")


def log_failure(task, error, elapsed_time, log_file):
    """记录失败的实验"""
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] ❌ FAILED  | {task['dataset']} | S{task['subject']} | {task['experiment_id']} | Error: {error}\n")


def main():
    parser = argparse.ArgumentParser(description='批量实验运行脚本')
    
    # 实验组选择
    parser.add_argument('--group', type=str, default='all',
                        choices=['all', 'main_ablation', 'disentangle_ablation', 
                                'adaptive_fusion_ablation', 'full_model_all_datasets',
                                'hyperparameter_search'],
                        help='要运行的实验组')
    
    # 受试者选择
    parser.add_argument('--subjects', type=str, default='all',
                        help='受试者选择：all, representative, 或指定如"1,2,3"或"1-10"')
    
    # GPU设置
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU编号')
    
    # 基础参数
    parser.add_argument('--base-args', type=str, 
                        default='--batch_size 64 --num_epochs 20',
                        help='所有实验共用的基础参数')
    
    # 执行控制
    parser.add_argument('--dry-run', action='store_true',
                        help='只显示命令，不实际执行')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='不询问直接执行')
    parser.add_argument('--stop-on-error', action='store_true',
                        help='遇到错误立即停止')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细信息')
    
    # 路径设置
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='结果目录')
    parser.add_argument('--log-file', type=str, default='./experiment_log.txt',
                        help='日志文件')
    
    args = parser.parse_args()
    
    # 读取配置文件
    config_file = 'experiments_config.json'
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        full_config = json.load(f)
    
    # 执行实验组
    if args.group == 'all':
        # 依次执行所有组
        groups = ['main_ablation', 'disentangle_ablation', 
                 'adaptive_fusion_ablation', 'full_model_all_datasets']
        for group in groups:
            if group in full_config:
                run_experiment_group(group, full_config[group], args)
    elif args.group == 'hyperparameter_search':
        run_hyperparameter_grid_search(args)
    else:
        # 执行指定组
        if args.group in full_config:
            run_experiment_group(args.group, full_config[args.group], args)
        else:
            print(f"❌ 未找到实验组: {args.group}")


if __name__ == '__main__':
    main()

