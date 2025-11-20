#!/usr/bin/env python
"""
消融实验脚本
用于测试 bcsr_neumann_series_depth、buffer_slide_size、bcsr_max_inner_it 的影响
每个参数值运行多个seed，计算均值和标准差

使用示例:
    # 测试 Neumann 深度（使用默认值 [1,3,5,10] 和 seed [1,2,3]）
    python run_ablation_study.py --param bcsr_neumann_series_depth

    # 测试 buffer_slide_size（自定义值）
    python run_ablation_study.py --param buffer_slide_size --values "50;100;200" --seeds "1;2;3"

    # 运行完整消融实验（所有参数）
    python run_ablation_study.py --run-all
"""

import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime
import pandas as pd
import numpy as np
import os
import sys

# ==================== 消融实验配置 ====================

# 基准配置（其他参数保持不变）
BASELINE_CONFIG = {
    'bcsr_neumann_series_depth': 3,
    'buffer_slide_size': 100,
    'bcsr_max_inner_it': 1,
}

# 消融参数定义
ABLATION_PARAMS = {
    'bcsr_neumann_series_depth': {
        'values': [1, 3, 5, 10],
        'description': 'Neumann级数深度 Q'
    },
    'buffer_slide_size': {
        'values': [50, 100, 200, 300],
        'description': '每个伪包的instance数量'
    },
    'bcsr_max_inner_it': {
        'values': [1, 3, 5, 10],
        'description': '内层迭代次数 N'
    }
}

# 默认seed列表
DEFAULT_SEEDS = [1, 1997, 1231]


class AblationStudyScheduler:
    """消融实验GPU调度器"""

    def __init__(self, gpu_ids: List[int], max_jobs_per_gpu: int = 1):
        """初始化消融实验调度器"""
        self.gpu_ids = gpu_ids
        self.max_jobs_per_gpu = max_jobs_per_gpu
        self.gpu_jobs = {gpu_id: 0 for gpu_id in gpu_ids}
        self.running_processes: List[Dict] = []
        self.completed_jobs: List[Dict] = []

    def get_available_gpu(self) -> int:
        """获取一个可用的GPU ID"""
        min_jobs = min(self.gpu_jobs.values())
        if min_jobs >= self.max_jobs_per_gpu:
            return None

        for gpu_id in self.gpu_ids:
            if self.gpu_jobs[gpu_id] == min_jobs:
                return gpu_id
        return None

    def submit_job(self, gpu_id: int, cmd: List[str], job_name: str,
                   param_value: str, seed: int, exp_name: str, log_dir: str):
        """在指定GPU上提交任务"""
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        task_log_dir = Path(log_dir) / exp_name
        task_log_dir.mkdir(parents=True, exist_ok=True)

        log_file_path = task_log_dir / f"run.log"
        log_file = open(log_file_path, 'w', buffering=1)

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )

        job_info = {
            'process': process,
            'gpu_id': gpu_id,
            'cmd': ' '.join(cmd),
            'job_name': job_name,
            'param_value': param_value,
            'seed': seed,
            'exp_name': exp_name,
            'log_dir': log_dir,
            'log_file': log_file,
            'log_file_path': str(log_file_path),
            'start_time': time.time()
        }
        self.running_processes.append(job_info)
        self.gpu_jobs[gpu_id] += 1

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 启动任务: {job_name}")
        print(f"  - GPU: {gpu_id}")
        print(f"  - 日志文件: {log_file_path}")
        print()

        return process

    def extract_results_from_logs(self, log_dir: str, exp_name: str) -> Optional[Dict]:
        """从日志文件中提取结果"""
        results = {}

        # 尝试读取 cl_metrics.csv 文件
        cl_metrics_csv = Path(log_dir) / exp_name / 'cl_metrics.csv'
        if cl_metrics_csv.exists():
            try:
                df = pd.read_csv(cl_metrics_csv)
                if 'AACC_mean' in df.columns:
                    results['AACC'] = df['AACC_mean'].iloc[0] if len(df) > 0 else None
                if 'BWT_mean' in df.columns:
                    results['BWT'] = df['BWT_mean'].iloc[0] if len(df) > 0 else None
                if 'IM_mean' in df.columns:
                    results['IM'] = df['IM_mean'].iloc[0] if len(df) > 0 else None
                return results
            except Exception as e:
                print(f"    警告: 无法读取 {cl_metrics_csv}: {e}")

        # 尝试读取 results.csv 文件
        results_csv = Path(log_dir) / exp_name / 'results.csv'
        if results_csv.exists():
            try:
                df = pd.read_csv(results_csv)
                acc_cols = [col for col in df.columns if 'acc' in col.lower()]
                for col in acc_cols:
                    results[col] = df[col].iloc[-1] if len(df) > 0 else None

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
            if process.poll() is not None:
                finished_jobs.append(job_info)

                if 'log_file' in job_info and job_info['log_file']:
                    try:
                        job_info['log_file'].close()
                    except Exception as e:
                        print(f"警告: 关闭日志文件失败: {e}")

                gpu_id = job_info['gpu_id']
                self.gpu_jobs[gpu_id] -= 1

                elapsed = time.time() - job_info['start_time']
                success = process.returncode == 0

                if success:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ 任务完成: {job_info['job_name']}")

                    results = self.extract_results_from_logs(
                        job_info['log_dir'],
                        job_info['exp_name']
                    )

                    if results and 'AACC' in results:
                        print(f"  - AACC: {results['AACC']:.4f}")
                    else:
                        print(f"  - 警告: 无法提取AACC")
                        results = {'error': 'Failed to extract results'}
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ 任务失败: {job_info['job_name']} (返回码: {process.returncode})")
                    results = {'error': f'Process failed with code {process.returncode}'}

                print(f"  - 运行时间: {elapsed/60:.1f} 分钟")
                print(f"  - GPU {gpu_id} 释放")
                if 'log_file_path' in job_info:
                    print(f"  - 日志文件: {job_info['log_file_path']}")
                print()

                completed_info = {
                    'param_value': job_info['param_value'],
                    'seed': job_info['seed'],
                    'exp_name': job_info['exp_name'],
                    'job_name': job_info['job_name'],
                    'gpu_id': gpu_id,
                    'elapsed_time': elapsed,
                    'success': success,
                    'results': results
                }
                if 'log_file_path' in job_info:
                    completed_info['log_file_path'] = job_info['log_file_path']
                self.completed_jobs.append(completed_info)

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

    def print_ablation_summary(self, param_name: str, seeds: List[int]):
        """打印消融实验结果汇总（按参数值分组）"""
        if not self.completed_jobs:
            print("没有完成的任务")
            return

        # 按 param_value 分组
        grouped = {}
        for job in self.completed_jobs:
            if job['success'] and 'results' in job and job['results'] and 'AACC' in job['results']:
                pv = str(job['param_value'])
                seed = job['seed']
                aacc = job['results']['AACC']

                if pv not in grouped:
                    grouped[pv] = {}
                grouped[pv][seed] = aacc

        # 构建表格
        print("\n" + "=" * 100)
        print(f"消融实验结果 - {param_name}")
        print("=" * 100)
        print(f"参数描述: {ABLATION_PARAMS.get(param_name, {}).get('description', 'N/A')}")
        print(f"其他参数: ", end="")
        baseline_strs = []
        for key, value in BASELINE_CONFIG.items():
            if key != param_name:
                baseline_strs.append(f"{key}={value}")
        print(", ".join(baseline_strs))
        print()

        table_data = []
        for pv in sorted(grouped.keys(), key=lambda x: float(x)):
            row = {param_name: pv}

            seed_results = []
            for seed in seeds:
                col_name = f'Seed{seed}'
                if seed in grouped[pv]:
                    aacc_val = grouped[pv][seed]
                    row[col_name] = f"{aacc_val:.4f}"
                    seed_results.append(aacc_val)
                else:
                    row[col_name] = "N/A"

            # 计算均值和标准差
            if seed_results and len(seed_results) > 1:
                mean_val = np.mean(seed_results)
                std_val = np.std(seed_results)
                row['均值 ± 标准差'] = f"{mean_val:.4f} ± {std_val:.4f}"
            elif seed_results:
                row['均值 ± 标准差'] = f"{seed_results[0]:.4f}"
            else:
                row['均值 ± 标准差'] = "N/A"

            table_data.append(row)

        # 使用pandas打印表格
        if table_data:
            df = pd.DataFrame(table_data)
            print(df.to_string(index=False))
            print()

            # 找出最好的配置
            best_job = None
            best_aacc = -1
            for job in self.completed_jobs:
                if job['success'] and 'results' in job and job['results'] and 'AACC' in job['results']:
                    if job['results']['AACC'] > best_aacc:
                        best_aacc = job['results']['AACC']
                        best_job = job

            if best_job:
                print(f"最佳配置: {param_name}={best_job['param_value']}, seed={best_job['seed']}, AACC={best_aacc:.4f}")

        print("=" * 100 + "\n")

    def save_ablation_results(self, param_name: str, seeds: List[int], log_dir: str):
        """保存消融实验结果"""
        if not self.completed_jobs:
            return

        # 按 param_value 分组统计
        grouped = {}
        for job in self.completed_jobs:
            if job['success'] and 'results' in job and job['results'] and 'AACC' in job['results']:
                pv = str(job['param_value'])
                seed = job['seed']
                aacc = job['results']['AACC']

                if pv not in grouped:
                    grouped[pv] = {}
                grouped[pv][seed] = aacc

        # 构建汇总表格
        summary_data = []
        for pv in sorted(grouped.keys(), key=lambda x: float(x)):
            row = {param_name: pv}

            seed_results = []
            for seed in seeds:
                col_name = f'Seed{seed}'
                if seed in grouped[pv]:
                    row[col_name] = grouped[pv][seed]
                    seed_results.append(grouped[pv][seed])
                else:
                    row[col_name] = None

            if seed_results and len(seed_results) > 1:
                row['Mean'] = np.mean(seed_results)
                row['Std'] = np.std(seed_results)
            elif seed_results:
                row['Mean'] = seed_results[0]
                row['Std'] = 0.0
            else:
                row['Mean'] = None
                row['Std'] = None

            summary_data.append(row)

        # 保存为CSV
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(log_dir, f"ablation_{param_name}_summary_{timestamp}.csv")
        df = pd.DataFrame(summary_data)
        df.to_csv(summary_file, index=False)
        print(f"\n消融实验汇总已保存到: {summary_file}")

        # 保存详细结果
        detailed_file = os.path.join(log_dir, f"ablation_{param_name}_detailed_{timestamp}.csv")
        detailed_data = []
        for job in sorted(self.completed_jobs, key=lambda x: (x['param_value'], x['seed'])):
            row = {
                param_name: job['param_value'],
                'Seed': job['seed'],
                'Status': 'Success' if job['success'] else 'Failed',
                'Time(min)': job['elapsed_time']/60 if 'elapsed_time' in job else None,
            }
            if job['success'] and 'results' in job and job['results']:
                if 'AACC' in job['results']:
                    row['AACC'] = job['results']['AACC']
                if 'BWT' in job['results']:
                    row['BWT'] = job['results']['BWT']
                if 'IM' in job['results']:
                    row['IM'] = job['results']['IM']
            detailed_data.append(row)

        df_detailed = pd.DataFrame(detailed_data)
        df_detailed.to_csv(detailed_file, index=False)
        print(f"详细结果已保存到: {detailed_file}")


def generate_ablation_commands(
    base_config: str,
    param_name: str,
    param_values: List[str],
    seeds: List[int],
    exp_name_prefix: str,
    log_dir: str
) -> List[Dict]:
    """生成消融实验命令"""
    commands = []

    for param_value in param_values:
        for seed in seeds:
            # 实验名称
            safe_value = str(param_value).replace('[', '').replace(']', '').replace(',', '_').replace(' ', '')
            exp_name = f"{exp_name_prefix}_{param_name}_{safe_value}_seed{seed}"

            # 基础命令
            cmd = [
                "python", "main_cl.py",
                "--preset", base_config,
                "--exp_name", exp_name,
                "--log_dir", log_dir,
                "--seed", str(seed)
            ]

            # 添加当前测试的参数
            cmd.extend([f"--{param_name}", str(param_value)])

            # 添加其他baseline参数（保持不变）
            for key, value in BASELINE_CONFIG.items():
                if key != param_name:
                    cmd.extend([f"--{key}", str(value)])

            commands.append({
                'cmd': cmd,
                'job_name': f"{param_name}={param_value}, seed={seed}",
                'param_value': str(param_value),
                'seed': seed,
                'exp_name': exp_name
            })

    return commands


def main():
    parser = argparse.ArgumentParser(
        description='消融实验工具 - 测试单个超参数的影响',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # GPU配置
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='可用的GPU ID，用逗号分隔')
    parser.add_argument('--max-jobs-per-gpu', type=int, default=1,
                        help='每个GPU上最多并行运行的任务数')

    # 实验配置
    parser.add_argument('--preset', type=str, default='configs/csc_clam_cl.yaml',
                        help='配置文件路径')
    parser.add_argument('--log-dir', type=str, default='ablation_logs',
                        help='日志根目录')
    parser.add_argument('--exp-name-prefix', type=str, default='ablation',
                        help='实验名称前缀')

    # 消融参数
    parser.add_argument('--param', type=str, choices=['bcsr_neumann_series_depth', 'buffer_slide_size', 'bcsr_max_inner_it'],
                        help='要测试的参数名称')
    parser.add_argument('--values', type=str, default=None,
                        help='参数值列表（分号分隔），不指定则使用默认值')
    parser.add_argument('--seeds', type=str, default='1;2;3',
                        help='seed列表（分号分隔），默认 "1;2;3"')
    parser.add_argument('--run-all', action='store_true',
                        help='运行所有参数的完整消融实验')

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
    seeds = [int(x.strip()) for x in args.seeds.split(';')]
    print(f"Seeds: {seeds}")
    print()

    # 确定要测试的参数
    if args.run_all:
        params_to_test = list(ABLATION_PARAMS.keys())
    elif args.param:
        params_to_test = [args.param]
    else:
        parser.error("请指定 --param 或 --run-all")

    # 所有命令列表
    all_commands = []

    # 生成命令
    for param_name in params_to_test:
        if args.values and args.param == param_name:
            # 使用用户指定的参数值
            param_values = [x.strip() for x in args.values.split(';')]
        else:
            # 使用默认参数值
            param_values = [str(v) for v in ABLATION_PARAMS[param_name]['values']]

        print(f"参数: {param_name}")
        print(f"  - 描述: {ABLATION_PARAMS[param_name]['description']}")
        print(f"  - 值: {param_values}")
        print(f"  - 实验数: {len(param_values)} 个参数值 × {len(seeds)} 个seed = {len(param_values) * len(seeds)} 个实验")
        print()

        commands = generate_ablation_commands(
            base_config=args.preset,
            param_name=param_name,
            param_values=param_values,
            seeds=seeds,
            exp_name_prefix=args.exp_name_prefix,
            log_dir=args.log_dir
        )
        all_commands.extend(commands)

    total_jobs = len(all_commands)
    print(f"总任务数: {total_jobs}")
    print()

    # 确保日志目录存在
    os.makedirs(args.log_dir, exist_ok=True)

    # 保存实验配置
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_file = os.path.join(args.log_dir, f"ablation_config_{timestamp}.json")
    with open(config_file, 'w') as f:
        json.dump({
            'gpus': gpu_ids,
            'max_jobs_per_gpu': args.max_jobs_per_gpu,
            'params_to_test': params_to_test,
            'seeds': seeds,
            'baseline_config': BASELINE_CONFIG,
            'base_config': args.preset,
            'total_jobs': total_jobs
        }, f, indent=2)
    print(f"实验配置已保存到: {config_file}\n")

    if args.dry_run:
        print("=" * 80)
        print("DRY RUN - 只显示命令，不实际执行")
        print("=" * 80)
        for i, cmd_info in enumerate(all_commands, 1):
            print(f"\n任务 {i}/{total_jobs}: {cmd_info['job_name']}")
            print(f"命令: {' '.join(cmd_info['cmd'])}")
        return

    # 创建调度器
    scheduler = AblationStudyScheduler(gpu_ids, args.max_jobs_per_gpu)

    print("=" * 80)
    print("开始执行消融实验")
    print("=" * 80)
    print()

    # 提交所有任务
    for cmd_info in all_commands:
        gpu_id = scheduler.wait_for_slot(check_interval=args.check_interval)

        scheduler.submit_job(
            gpu_id=gpu_id,
            cmd=cmd_info['cmd'],
            job_name=cmd_info['job_name'],
            param_value=cmd_info['param_value'],
            seed=cmd_info['seed'],
            exp_name=cmd_info['exp_name'],
            log_dir=args.log_dir
        )

    # 等待所有任务完成
    scheduler.wait_all_jobs(check_interval=args.check_interval)

    # 按参数分组打印结果
    for param_name in params_to_test:
        # 过滤出该参数的任务
        param_jobs = [j for j in scheduler.completed_jobs if param_name in j['job_name']]
        if param_jobs:
            # 临时更新 completed_jobs，然后打印
            temp_completed_jobs = scheduler.completed_jobs
            scheduler.completed_jobs = param_jobs
            scheduler.print_ablation_summary(param_name=param_name, seeds=seeds)
            scheduler.save_ablation_results(param_name=param_name, seeds=seeds, log_dir=args.log_dir)
            scheduler.completed_jobs = temp_completed_jobs



if __name__ == "__main__":
    main()
