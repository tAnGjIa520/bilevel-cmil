#!/usr/bin/env python3
"""
并行运行所有实验组合，汇总标签不一致率结果
"""

import subprocess
import argparse
import os
import time
import json
from pathlib import Path
import pandas as pd
import re

def parse_mismatch_stats(log_file):
    """从日志文件中解析标签不一致统计信息"""
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # 查找训练集标签不一致率
    train_pattern = r'训练集:.*?\((\d+\.\d+)%\).*?包标签不一致'
    train_match = re.search(train_pattern, content)
    train_mismatch_rate = float(train_match.group(1)) if train_match else None

    # 查找正包→负包
    pos_to_neg_pattern = r'正包→负包:\s*(\d+)'
    pos_to_neg_match = re.search(pos_to_neg_pattern, content)
    pos_to_neg = int(pos_to_neg_match.group(1)) if pos_to_neg_match else None

    # 查找负包→正包
    neg_to_pos_pattern = r'负包→正包:\s*(\d+)'
    neg_to_pos_match = re.search(neg_to_pos_pattern, content)
    neg_to_pos = int(neg_to_pos_match.group(1)) if neg_to_pos_match else None

    return {
        'train_mismatch_rate': train_mismatch_rate,
        'pos_to_neg': pos_to_neg,
        'neg_to_pos': neg_to_pos
    }

def run_experiment(task, method, gpu_id, log_dir, args):
    """运行单个实验"""
    exp_name = f"{task}_{method}"
    log_file = os.path.join(log_dir, f"{exp_name}.log")

    cmd = [
        'python', 'demo/attention_mil_harder_tasks111.py',
        '--task', task,
        '--selection_method', method,
        '--new_size', str(args.new_size),
        '--bag_size_min', str(args.bag_size_min),
        '--bag_size_max', str(args.bag_size_max),
        '--num_bags', str(args.num_bags),
        '--epochs1', str(args.epochs1),
        '--epochs2', str(args.epochs2),
        '--kibo_outer_it', str(args.kibo_outer_it),
        '--kibo_inner_it', str(args.kibo_inner_it),
        '--kibo_lr_weight', str(args.kibo_lr_weight),
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"[GPU {gpu_id}] 启动实验: {exp_name}")

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os.getcwd()
        )

    return {
        'task': task,
        'method': method,
        'gpu_id': gpu_id,
        'process': process,
        'log_file': log_file,
        'exp_name': exp_name
    }

def main():
    parser = argparse.ArgumentParser(description='并行运行所有实验组合')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7',
                        help='可用的GPU ID列表，逗号分隔')
    parser.add_argument('--log_dir', type=str, default='logs/all_experiments',
                        help='日志文件保存目录')
    parser.add_argument('--output_csv', type=str, default='mismatch_summary.csv',
                        help='结果汇总CSV文件')

    # 实验参数
    parser.add_argument('--new_size', type=int, default=10, help='新包大小')
    parser.add_argument('--bag_size_min', type=int, default=20, help='包大小最小值')
    parser.add_argument('--bag_size_max', type=int, default=50, help='包大小最大值')
    parser.add_argument('--num_bags', type=int, default=1000, help='训练集包数量')
    parser.add_argument('--epochs1', type=int, default=10, help='第一阶段轮数')
    parser.add_argument('--epochs2', type=int, default=10, help='第二阶段轮数')
    parser.add_argument('--kibo_outer_it', type=int, default=10, help='KIBO外层迭代')
    parser.add_argument('--kibo_inner_it', type=int, default=2, help='KIBO内层迭代')
    parser.add_argument('--kibo_lr_weight', type=float, default=0.1, help='KIBO权重学习率')

    args = parser.parse_args()

    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)

    # 解析GPU列表
    gpu_list = [int(g.strip()) for g in args.gpus.split(',')]
    print(f"可用GPU: {gpu_list}")

    # 定义所有任务和方法
    tasks = ['large_range', 'triplet', 'sum', 'increasing']
    methods = ['attention', 'kibo', 'random']

    # 生成所有实验组合
    all_experiments = [(task, method) for task in tasks for method in methods]
    print(f"\n总共 {len(all_experiments)} 个实验组合")
    print("="*80)

    # 运行实验
    running_jobs = []
    completed_jobs = []
    exp_queue = all_experiments.copy()
    gpu_available = {gpu: True for gpu in gpu_list}

    print("\n开始并行运行实验...")
    print("="*80)

    while exp_queue or running_jobs:
        # 启动新任务
        while exp_queue and any(gpu_available.values()):
            # 找到空闲的GPU
            free_gpu = None
            for gpu in gpu_list:
                if gpu_available[gpu]:
                    free_gpu = gpu
                    break

            if free_gpu is None:
                break

            # 从队列中取出一个实验
            task, method = exp_queue.pop(0)

            # 启动实验
            job = run_experiment(task, method, free_gpu, args.log_dir, args)
            running_jobs.append(job)
            gpu_available[free_gpu] = False

        # 检查已完成的任务
        time.sleep(2)
        for job in running_jobs[:]:
            if job['process'].poll() is not None:
                # 任务完成
                running_jobs.remove(job)
                completed_jobs.append(job)
                gpu_available[job['gpu_id']] = True

                return_code = job['process'].returncode
                if return_code == 0:
                    print(f"[GPU {job['gpu_id']}] ✓ 完成: {job['exp_name']}")
                else:
                    print(f"[GPU {job['gpu_id']}] ✗ 失败: {job['exp_name']} (返回码: {return_code})")

        # 显示进度
        if running_jobs:
            running_names = [j['exp_name'] for j in running_jobs]
            print(f"\r进度: {len(completed_jobs)}/{len(all_experiments)} | "
                  f"运行中: {', '.join(running_names)}", end='', flush=True)

    print("\n\n所有实验完成！")
    print("="*80)

    # 汇总结果
    print("\n解析结果并生成汇总表格...")
    results = []

    for job in completed_jobs:
        stats = parse_mismatch_stats(job['log_file'])
        if stats:
            results.append({
                'Task': job['task'],
                'Method': job['method'],
                'Mismatch Rate (%)': stats['train_mismatch_rate'],
                'Pos→Neg': stats['pos_to_neg'],
                'Neg→Pos': stats['neg_to_pos'],
                'Log File': job['log_file']
            })
        else:
            print(f"警告: 无法解析 {job['exp_name']} 的结果")

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 按任务和方法排序
    task_order = ['large_range', 'triplet', 'sum', 'increasing']
    method_order = ['attention', 'kibo', 'random']
    df['Task'] = pd.Categorical(df['Task'], categories=task_order, ordered=True)
    df['Method'] = pd.Categorical(df['Method'], categories=method_order, ordered=True)
    df = df.sort_values(['Task', 'Method'])

    # 保存为CSV
    csv_path = os.path.join(args.log_dir, args.output_csv)
    df.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"\n结果已保存到: {csv_path}")

    # 打印表格
    print("\n" + "="*80)
    print("标签不一致率汇总表")
    print("="*80)
    print(df.to_string(index=False))

    # 生成透视表（方法为列，任务为行）
    print("\n" + "="*80)
    print("透视表 - 标签不一致率 (%)")
    print("="*80)
    pivot = df.pivot(index='Task', columns='Method', values='Mismatch Rate (%)')
    print(pivot.to_string())

    # 保存透视表
    pivot_path = os.path.join(args.log_dir, 'mismatch_pivot.csv')
    pivot.to_csv(pivot_path, float_format='%.2f')
    print(f"\n透视表已保存到: {pivot_path}")

    # 统计摘要
    print("\n" + "="*80)
    print("统计摘要")
    print("="*80)
    print(f"平均不一致率: {df['Mismatch Rate (%)'].mean():.2f}%")
    print(f"最高不一致率: {df['Mismatch Rate (%)'].max():.2f}% "
          f"({df.loc[df['Mismatch Rate (%)'].idxmax(), 'Task']}-{df.loc[df['Mismatch Rate (%)'].idxmax(), 'Method']})")
    print(f"最低不一致率: {df['Mismatch Rate (%)'].min():.2f}% "
          f"({df.loc[df['Mismatch Rate (%)'].idxmin(), 'Task']}-{df.loc[df['Mismatch Rate (%)'].idxmin(), 'Method']})")

    print("\n按方法的平均不一致率:")
    method_avg = df.groupby('Method')['Mismatch Rate (%)'].mean()
    for method, avg in method_avg.items():
        print(f"  {method}: {avg:.2f}%")

    print("\n按任务的平均不一致率:")
    task_avg = df.groupby('Task')['Mismatch Rate (%)'].mean()
    for task, avg in task_avg.items():
        print(f"  {task}: {avg:.2f}%")

if __name__ == '__main__':
    main()
