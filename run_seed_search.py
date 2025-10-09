#!/usr/bin/env python
"""
超参数搜索脚本 - Seed搜索
支持多GPU并行运行，自动分配CUDA设备
"""

import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime
import re
import pandas as pd
import os


class GPUScheduler:
    """GPU任务调度器"""

    def __init__(self, gpu_ids: List[int], max_jobs_per_gpu: int = 1):
        """
        初始化GPU调度器

        Args:
            gpu_ids: 可用的GPU ID列表，例如 [0, 1, 2, 3]
            max_jobs_per_gpu: 每个GPU上最多并行运行的任务数
        """
        self.gpu_ids = gpu_ids
        self.max_jobs_per_gpu = max_jobs_per_gpu
        # 记录每个GPU上当前运行的任务数
        self.gpu_jobs = {gpu_id: 0 for gpu_id in gpu_ids}
        # 记录所有运行中的进程
        self.running_processes: List[Dict] = []
        # 记录所有完成的任务结果
        self.completed_jobs: List[Dict] = []

    def get_available_gpu(self) -> int:
        """获取一个可用的GPU ID"""
        # 找到任务数最少的GPU
        min_jobs = min(self.gpu_jobs.values())
        if min_jobs >= self.max_jobs_per_gpu:
            return None  # 所有GPU都满了

        # 返回第一个任务数最少的GPU
        for gpu_id in self.gpu_ids:
            if self.gpu_jobs[gpu_id] == min_jobs:
                return gpu_id
        return None

    def submit_job(self, gpu_id: int, cmd: List[str], job_name: str, seed: int, exp_name: str, log_dir: str):
        """
        在指定GPU上提交任务

        Args:
            gpu_id: GPU ID
            cmd: 命令列表
            job_name: 任务名称
            seed: 随机种子
            exp_name: 实验名称
            log_dir: 日志目录
        """
        # 设置环境变量指定GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # 启动进程
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 记录任务信息
        job_info = {
            'process': process,
            'gpu_id': gpu_id,
            'cmd': ' '.join(cmd),
            'job_name': job_name,
            'seed': seed,
            'exp_name': exp_name,
            'log_dir': log_dir,
            'start_time': time.time()
        }
        self.running_processes.append(job_info)
        self.gpu_jobs[gpu_id] += 1

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 启动任务: {job_name}")
        print(f"  - GPU: {gpu_id}")
        print(f"  - 命令:")
        print(f"    CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)}")
        print()

        return process

    def extract_results_from_logs(self, log_dir: str, exp_name: str) -> Optional[Dict]:
        """
        从日志文件中提取结果

        Args:
            log_dir: 日志根目录
            exp_name: 实验名称

        Returns:
            包含准确率的字典，如果提取失败则返回None
        """
        results = {}

        # 尝试读取 results.csv 文件
        results_csv = Path(log_dir) / exp_name / 'results.csv'
        if results_csv.exists():
            try:
                df = pd.read_csv(results_csv)
                # 提取所有 acc 列
                acc_cols = [col for col in df.columns if 'acc' in col.lower()]
                for col in acc_cols:
                    # 取最后一行（最终任务）的值
                    results[col] = df[col].iloc[-1] if len(df) > 0 else None

                # 计算平均准确率
                if acc_cols:
                    valid_accs = [results[col] for col in acc_cols if results[col] is not None]
                    if valid_accs:
                        results['avg_acc'] = sum(valid_accs) / len(valid_accs)

                return results
            except Exception as e:
                print(f"    警告: 无法读取 {results_csv}: {e}")
                return None

        return None

    def check_and_clean_finished(self):
        """检查并清理已完成的任务"""
        finished_jobs = []

        for job_info in self.running_processes:
            process = job_info['process']
            if process.poll() is not None:  # 进程已结束
                finished_jobs.append(job_info)

                # 释放GPU资源
                gpu_id = job_info['gpu_id']
                self.gpu_jobs[gpu_id] -= 1

                # 计算运行时间
                elapsed = time.time() - job_info['start_time']

                # 检查返回码
                success = process.returncode == 0

                if success:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ 任务完成: {job_info['job_name']}")

                    # 尝试提取结果
                    results = self.extract_results_from_logs(
                        job_info['log_dir'],
                        job_info['exp_name']
                    )

                    if results and 'avg_acc' in results:
                        print(f"  - 平均准确率: {results['avg_acc']:.4f}")
                    else:
                        print(f"  - 警告: 无法提取准确率")
                        results = {'error': 'Failed to extract results'}
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ 任务失败: {job_info['job_name']} (返回码: {process.returncode})")
                    results = {'error': f'Process failed with code {process.returncode}'}

                print(f"  - 运行时间: {elapsed/60:.1f} 分钟")
                print(f"  - GPU {gpu_id} 释放")
                print()

                # 保存完成的任务信息
                completed_info = {
                    'seed': job_info['seed'],
                    'exp_name': job_info['exp_name'],
                    'job_name': job_info['job_name'],
                    'gpu_id': gpu_id,
                    'elapsed_time': elapsed,
                    'success': success,
                    'results': results
                }
                self.completed_jobs.append(completed_info)

        # 从运行列表中移除已完成的任务
        for job in finished_jobs:
            self.running_processes.remove(job)

    def wait_for_slot(self, check_interval: float = 5.0):
        """等待直到有可用的GPU槽位"""
        while True:
            self.check_and_clean_finished()
            gpu_id = self.get_available_gpu()
            if gpu_id is not None:
                return gpu_id
            time.sleep(check_interval)

    def wait_all_jobs(self, check_interval: float = 5.0):
        """等待所有任务完成"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 等待所有任务完成...")
        while self.running_processes:
            self.check_and_clean_finished()
            time.sleep(check_interval)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 所有任务已完成！")

    def print_summary_table(self, log_dir: str = None):
        """打印汇总表格"""
        if not self.completed_jobs:
            print("没有完成的任务")
            return

        print("\n" + "=" * 100)
        print("实验结果汇总")
        print("=" * 100)

        # 准备表格数据
        table_data = []
        for job in sorted(self.completed_jobs, key=lambda x: x['seed']):
            row = {
                'Seed': job['seed'],
                'Status': '✓' if job['success'] else '✗',
                'Time(min)': f"{job['elapsed_time']/60:.1f}",
            }

            # 添加准确率信息
            if job['success'] and 'results' in job and job['results']:
                results = job['results']
                if 'error' not in results:
                    # 提取所有任务的准确率
                    for key, value in results.items():
                        if 'acc' in key.lower() and value is not None:
                            row[key] = f"{value:.4f}"
                else:
                    row['Error'] = results['error']
            else:
                row['Error'] = 'N/A'

            table_data.append(row)

        # 使用pandas打印表格
        if table_data:
            df = pd.DataFrame(table_data)
            print("\n" + df.to_string(index=False))

            # 计算统计信息
            print("\n" + "=" * 100)
            print("统计信息")
            print("=" * 100)

            successful_jobs = [j for j in self.completed_jobs if j['success']]
            print(f"成功任务数: {len(successful_jobs)}/{len(self.completed_jobs)}")

            if successful_jobs:
                # 计算平均准确率
                avg_accs = []
                for job in successful_jobs:
                    if 'results' in job and job['results'] and 'avg_acc' in job['results']:
                        avg_accs.append(job['results']['avg_acc'])

                if avg_accs:
                    print(f"\n平均准确率:")
                    print(f"  - Mean: {sum(avg_accs)/len(avg_accs):.4f}")
                    print(f"  - Std:  {pd.Series(avg_accs).std():.4f}")
                    print(f"  - Max:  {max(avg_accs):.4f}")
                    print(f"  - Min:  {min(avg_accs):.4f}")

                # 平均运行时间
                avg_time = sum(j['elapsed_time'] for j in successful_jobs) / len(successful_jobs)
                print(f"\n平均运行时间: {avg_time/60:.1f} 分钟")

        print("\n" + "=" * 100)

    def save_results(self, log_dir: str):
        """保存结果到CSV文件"""
        if not self.completed_jobs:
            return

        # 准备表格数据
        table_data = []
        for job in sorted(self.completed_jobs, key=lambda x: x['seed']):
            row = {
                'Seed': job['seed'],
                'Status': 'Success' if job['success'] else 'Failed',
                'Time(min)': f"{job['elapsed_time']/60:.1f}",
            }

            # 添加准确率信息
            if job['success'] and 'results' in job and job['results']:
                results = job['results']
                if 'error' not in results:
                    for key, value in results.items():
                        if 'acc' in key.lower() and value is not None:
                            row[key] = value  # 保持数值格式，方便后续分析
                else:
                    row['Error'] = results['error']
            else:
                row['Error'] = 'N/A'

            table_data.append(row)

        # 保存结果到CSV
        if table_data:
            os.makedirs(log_dir, exist_ok=True)
            results_file = os.path.join(log_dir, f"seed_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df = pd.DataFrame(table_data)
            df.to_csv(results_file, index=False)
            print(f"\n结果已保存到: {results_file}")
            print("=" * 100 + "\n")


def generate_commands(
    base_config: str,
    cl_method: str,
    buffer_size: int,
    exp_name_prefix: str,
    seeds: List[int],
    log_dir: str
) -> List[Dict]:
    """
    生成所有实验命令

    Returns:
        List[Dict]: 包含命令和任务名称的字典列表
    """
    commands = []

    for seed in seeds:
        exp_name = f"{exp_name_prefix}_seed{seed}"

        cmd = [
            "python", "main_cl.py",
            "--preset", base_config,
            "--cl_method", cl_method,
            "--buffer_size", str(buffer_size),
            "--exp_name", exp_name,
            "--seed", str(seed),
            "--log_dir", log_dir
        ]

        commands.append({
            'cmd': cmd,
            'job_name': f"seed={seed}",
            'seed': seed,
            'exp_name': exp_name
        })

    return commands


def main():
    parser = argparse.ArgumentParser(
        description='超参数搜索 - Seed搜索',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # GPU配置
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='可用的GPU ID，用逗号分隔，例如: 0,1,2,3')
    parser.add_argument('--max-jobs-per-gpu', type=int, default=1,
                        help='每个GPU上最多并行运行的任务数')

    # 实验配置
    parser.add_argument('--preset', type=str, default='configs/csc_clam_cl.yaml',
                        help='配置文件路径')
    parser.add_argument('--cl-method', type=str, default='prev',
                        help='持续学习方法')
    parser.add_argument('--buffer-size', type=int, default=42,
                        help='Buffer大小')
    parser.add_argument('--exp-name-prefix', type=str, default='csc_clam_cl_buf42_attn_logit',
                        help='实验名称前缀，会自动添加 _seedX')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='日志根目录')

    # Seed搜索配置
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5',
                        help='要搜索的seed列表，用逗号分隔，例如: 1,2,3,4,5')

    # 其他配置
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印命令，不实际运行')
    parser.add_argument('--check-interval', type=float, default=10.0,
                        help='检查任务完成的时间间隔（秒）')

    args = parser.parse_args()

    # 解析GPU ID
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    print(f"可用GPU: {gpu_ids}")
    print(f"每GPU最大任务数: {args.max_jobs_per_gpu}")
    print()

    # 解析seed列表
    seeds = [int(x.strip()) for x in args.seeds.split(',')]
    print(f"搜索的Seeds: {seeds}")
    print(f"总任务数: {len(seeds)}")
    print()

    # 生成所有命令
    commands = generate_commands(
        base_config=args.preset,
        cl_method=args.cl_method,
        buffer_size=args.buffer_size,
        exp_name_prefix=args.exp_name_prefix,
        seeds=seeds,
        log_dir=args.log_dir
    )

    # 确保日志目录存在
    os.makedirs(args.log_dir, exist_ok=True)

    # 保存实验配置到日志目录
    config_file = os.path.join(args.log_dir, f"seed_search_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(config_file, 'w') as f:
        json.dump({
            'gpus': gpu_ids,
            'max_jobs_per_gpu': args.max_jobs_per_gpu,
            'seeds': seeds,
            'base_config': args.preset,
            'cl_method': args.cl_method,
            'buffer_size': args.buffer_size,
            'exp_name_prefix': args.exp_name_prefix,
            'log_dir': args.log_dir,
            'total_jobs': len(commands)
        }, f, indent=2)
    print(f"实验配置已保存到: {config_file}\n")

    if args.dry_run:
        print("=" * 80)
        print("DRY RUN - 只显示命令，不实际执行")
        print("=" * 80)
        for i, cmd_info in enumerate(commands, 1):
            print(f"\n任务 {i}/{len(commands)}: {cmd_info['job_name']}")
            print(f"命令: {' '.join(cmd_info['cmd'])}")
        return

    # 创建GPU调度器
    scheduler = GPUScheduler(gpu_ids, args.max_jobs_per_gpu)

    print("=" * 80)
    print("开始执行任务")
    print("=" * 80)
    print()

    # 提交所有任务
    for cmd_info in commands:
        # 等待可用的GPU槽位
        gpu_id = scheduler.wait_for_slot(check_interval=args.check_interval)

        # 提交任务
        scheduler.submit_job(
            gpu_id=gpu_id,
            cmd=cmd_info['cmd'],
            job_name=cmd_info['job_name'],
            seed=cmd_info['seed'],
            exp_name=cmd_info['exp_name'],
            log_dir=args.log_dir
        )

    # 等待所有任务完成
    scheduler.wait_all_jobs(check_interval=args.check_interval)

    # 打印汇总表格并保存结果
    scheduler.print_summary_table(log_dir=args.log_dir)
    scheduler.save_results(log_dir=args.log_dir)


if __name__ == "__main__":
    main()
