#!/usr/bin/env python
"""
BCSR 超参数收敛性搜索脚本

针对 main_cl_buffer_analysis.py 脚本，搜索以下BCSR相关超参数：
- bcsr_lr_proxy_model: 代理模型学习率
- bcsr_beta: beta参数
- bcsr_max_inner_it: 最大内层迭代次数
- bcsr_weight_lr: 权重学习率
- distall_lamda: 蒸馏损失系数
- topk_method: TopK选择方法 ('sigmoid', 'gumbel', 'ste')
- topk_temperature: TopK温度参数
- normalize_method: 权重归一化方法 ('l2', 'softmax', 'zscore', 'none')
- use_simplex_projection: 是否使用单纯形投影 ('True', 'False')
- distill_target: 知识蒸馏目标 ('logits', 'features', 'both')

注意: bcsr_max_outer_it 参数已固定，不再作为搜索超参数

目标：找到收敛最好的超参数组合

使用示例:
    # 使用预定义的Grid Search
    python run_bcsr_convergence_search.py --gpus 0,1,2,3

    # 自定义GPU和其他参数
    python run_bcsr_convergence_search.py --gpus 0,1 --max-jobs-per-gpu 1
"""

import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime
import pandas as pd
import os
import wandb
import itertools


class GPUScheduler:
    """GPU任务调度器 - 支持收敛指标提取"""

    def __init__(self, gpu_ids: List[int], max_jobs_per_gpu: int = 1,
                 wandb_project: str = "BCSR_Convergence", wandb_config: Dict = None,
                 exp_name_prefix: str = "bcsr_search", notes: str = None):
        """初始化GPU调度器"""
        self.gpu_ids = gpu_ids
        self.max_jobs_per_gpu = max_jobs_per_gpu
        self.gpu_jobs = {gpu_id: 0 for gpu_id in gpu_ids}
        self.running_processes: List[Dict] = []
        self.completed_jobs: List[Dict] = []

        # 登录 wandb
        wandb.login(key="bd9541d4de0608784f26f2a79c055900f364d762", relogin=True)

        # 初始化 wandb
        self.wandb_run = wandb.init(
            project=wandb_project,
            name=f"bcsr_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=wandb_config or {},
            notes=notes or "BCSR hyperparameter search for convergence optimization"
        )

    def get_available_gpu(self) -> Optional[int]:
        """获取一个可用的GPU ID"""
        min_jobs = min(self.gpu_jobs.values())
        if min_jobs >= self.max_jobs_per_gpu:
            return None

        for gpu_id in self.gpu_ids:
            if self.gpu_jobs[gpu_id] == min_jobs:
                return gpu_id
        return None

    def submit_job(self, gpu_id: int, cmd: List[str], job_name: str,
                   param_combo: Dict, exp_name: str, log_dir: str):
        """在指定GPU上提交任务"""
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        task_log_dir = Path(log_dir) / exp_name
        task_log_dir.mkdir(parents=True, exist_ok=True)

        log_file_path = task_log_dir / "run.log"
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
            'param_combo': param_combo,
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
        print(f"  - 超参数组合:")
        for k, v in param_combo.items():
            print(f"    {k}: {v}")
        print()

        return process

    def extract_convergence_metrics(self, log_dir: str, exp_name: str) -> Optional[Dict]:
        """
        从日志中提取收敛指标

        读取 convergence_metrics.json 文件
        """
        convergence_file = Path(log_dir) / exp_name / 'convergence_metrics.json'

        if convergence_file.exists():
            try:
                with open(convergence_file, 'r') as f:
                    metrics = json.load(f)
                return metrics
            except Exception as e:
                print(f"    警告: 无法读取 {convergence_file}: {e}")
                return None
        else:
            print(f"    警告: 收敛指标文件不存在: {convergence_file}")
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

                    # 提取收敛指标
                    convergence = self.extract_convergence_metrics(
                        job_info['log_dir'],
                        job_info['exp_name']
                    )

                    if convergence:
                        print(f"  - 收敛得分: {convergence['score']:.4f} ({convergence['status']})")
                        print(f"  - 初始损失: {convergence['details']['initial_loss']:.4f}")
                        print(f"  - 最终损失: {convergence['details']['final_loss']:.4f}")
                        print(f"  - 总下降量: {convergence['details']['total_decrease']:.4f}")
                        results = convergence
                    else:
                        print(f"  - 警告: 无法提取收敛指标")
                        results = {'error': 'Failed to extract convergence metrics'}
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ 任务失败: {job_info['job_name']} (返回码: {process.returncode})")
                    results = {'error': f'Process failed with code {process.returncode}'}

                print(f"  - 运行时间: {elapsed/60:.1f} 分钟")
                print(f"  - GPU {gpu_id} 释放")
                if 'log_file_path' in job_info:
                    print(f"  - 日志文件: {job_info['log_file_path']}")
                print()

                completed_info = {
                    'param_combo': job_info['param_combo'],
                    'exp_name': job_info['exp_name'],
                    'job_name': job_info['job_name'],
                    'gpu_id': gpu_id,
                    'elapsed_time': elapsed,
                    'success': success,
                    'convergence_metrics': results
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

    def print_summary_table(self):
        """打印收敛指标汇总表格"""
        if not self.completed_jobs:
            print("没有完成的任务")
            return

        print("\n" + "=" * 120)
        print("BCSR 超参数搜索 - 收敛性分析")
        print("=" * 120)

        # 准备表格数据
        table_data = []
        for job in sorted(self.completed_jobs,
                         key=lambda x: x['convergence_metrics'].get('score', -1) if x['success'] else -1,
                         reverse=True):
            row = {
                'Exp': job['exp_name'][:30],
                'Status': '✓' if job['success'] else '✗',
                'Time(min)': f"{job['elapsed_time']/60:.1f}",
            }

            # 添加超参数
            for param, value in job['param_combo'].items():
                row[param] = value

            # 添加收敛指标
            if job['success'] and 'convergence_metrics' in job:
                metrics = job['convergence_metrics']
                if 'error' not in metrics and 'score' in metrics:
                    row['Score'] = f"{metrics['score']:.4f}"
                    row['Status_Conv'] = metrics['status']
                    row['Init_Loss'] = f"{metrics['details']['initial_loss']:.4f}"
                    row['Final_Loss'] = f"{metrics['details']['final_loss']:.4f}"
                    row['Decrease'] = f"{metrics['details']['total_decrease']:.4f}"
                    row['Monotonicity'] = f"{metrics['details']['monotonicity']:.4f}"
                    row['Stability'] = f"{metrics['details']['stability']:.4f}"
                else:
                    row['Error'] = metrics.get('error', 'Unknown')
            else:
                row['Error'] = 'N/A'

            table_data.append(row)

        # 使用pandas打印表格
        if table_data:
            df = pd.DataFrame(table_data)
            print("\n" + df.to_string(index=False))

            # 统计信息
            print("\n" + "=" * 120)
            print("统计信息")
            print("=" * 120)

            successful_jobs = [j for j in self.completed_jobs if j['success']]
            print(f"成功任务数: {len(successful_jobs)}/{len(self.completed_jobs)}")

            if successful_jobs:
                scores = []
                for job in successful_jobs:
                    if 'convergence_metrics' in job and 'score' in job['convergence_metrics']:
                        scores.append(job['convergence_metrics']['score'])

                if scores:
                    print(f"\n收敛得分统计:")
                    print(f"  - Mean: {sum(scores)/len(scores):.4f}")
                    print(f"  - Std:  {pd.Series(scores).std():.4f}")
                    print(f"  - Max:  {max(scores):.4f}")
                    print(f"  - Min:  {min(scores):.4f}")

                    # 找出最佳超参数组合
                    best_job = max(successful_jobs,
                                  key=lambda x: x['convergence_metrics'].get('score', -1))
                    print(f"\n最佳超参数组合:")
                    print(f"  - 收敛得分: {best_job['convergence_metrics']['score']:.4f}")
                    print(f"  - 超参数:")
                    for param, value in best_job['param_combo'].items():
                        print(f"    {param}: {value}")

                avg_time = sum(j['elapsed_time'] for j in successful_jobs) / len(successful_jobs)
                print(f"\n平均运行时间: {avg_time/60:.1f} 分钟")

        print("\n" + "=" * 120)

    def save_results(self, log_dir: str):
        """保存结果到CSV并上传到wandb"""
        if not self.completed_jobs:
            return

        # 准备详细数据
        table_data = []
        for job in self.completed_jobs:
            row = {
                'exp_name': job['exp_name'],
                'status': 'Success' if job['success'] else 'Failed',
                'time_min': job['elapsed_time']/60,
            }

            # 添加超参数
            for param, value in job['param_combo'].items():
                row[param] = value

            # 添加收敛指标
            if job['success'] and 'convergence_metrics' in job:
                metrics = job['convergence_metrics']
                if 'error' not in metrics and 'score' in metrics:
                    row['convergence_score'] = metrics['score']
                    row['convergence_status'] = metrics['status']
                    row['initial_loss'] = metrics['details']['initial_loss']
                    row['final_loss'] = metrics['details']['final_loss']
                    row['total_decrease'] = metrics['details']['total_decrease']
                    row['decrease_score'] = metrics['details']['decrease']
                    row['monotonicity'] = metrics['details']['monotonicity']
                    row['stability'] = metrics['details']['stability']
                    row['oscillation'] = metrics['details']['oscillation']
                else:
                    row['error'] = metrics.get('error', 'Unknown')
            else:
                row['error'] = 'Failed or N/A'

            table_data.append(row)

        # 保存详细结果到CSV
        if table_data:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = os.path.join(log_dir, f"bcsr_convergence_search_{timestamp}.csv")
            df = pd.DataFrame(table_data)
            df.to_csv(results_file, index=False)
            print(f"\n详细结果已保存到: {results_file}")

            # 保存成功任务的汇总
            successful_jobs = [j for j in self.completed_jobs if j['success']]
            if successful_jobs:
                summary_data = []
                scores = []

                for job in successful_jobs:
                    if 'convergence_metrics' in job and 'score' in job['convergence_metrics']:
                        metrics = job['convergence_metrics']
                        row = {'exp_name': job['exp_name']}

                        # 添加所有超参数
                        for param, value in job['param_combo'].items():
                            row[param] = value

                        # 添加收敛指标
                        row['score'] = metrics['score']
                        row['status'] = metrics['status']
                        row['initial_loss'] = metrics['details']['initial_loss']
                        row['final_loss'] = metrics['details']['final_loss']
                        row['decrease'] = metrics['details']['total_decrease']

                        summary_data.append(row)
                        scores.append(metrics['score'])

                if summary_data:
                    # 保存汇总文件
                    summary_file = os.path.join(log_dir, f"bcsr_convergence_summary.txt")
                    with open(summary_file, 'w') as f:
                        f.write("=" * 100 + "\n")
                        f.write("BCSR 超参数搜索 - 收敛性汇总\n")
                        f.write("=" * 100 + "\n\n")

                        # 找到最佳组合
                        best_job = max(successful_jobs,
                                      key=lambda x: x['convergence_metrics'].get('score', -1))

                        f.write("最佳超参数组合:\n")
                        f.write("-" * 100 + "\n")
                        for param, value in best_job['param_combo'].items():
                            f.write(f"  {param:<25}: {value}\n")
                        f.write(f"\n  收敛得分                : {best_job['convergence_metrics']['score']:.4f}\n")
                        f.write(f"  收敛状态                : {best_job['convergence_metrics']['status']}\n")
                        f.write(f"  初始损失                : {best_job['convergence_metrics']['details']['initial_loss']:.4f}\n")
                        f.write(f"  最终损失                : {best_job['convergence_metrics']['details']['final_loss']:.4f}\n")
                        f.write(f"  总下降量                : {best_job['convergence_metrics']['details']['total_decrease']:.4f}\n")

                        f.write("\n" + "=" * 100 + "\n")
                        f.write("统计信息\n")
                        f.write("=" * 100 + "\n")
                        f.write(f"收敛得分: {sum(scores)/len(scores):.4f} ± {pd.Series(scores).std():.4f}\n")
                        f.write(f"  - Max: {max(scores):.4f}\n")
                        f.write(f"  - Min: {min(scores):.4f}\n")
                        f.write(f"\n成功任务数: {len(successful_jobs)}/{len(self.completed_jobs)}\n")
                        f.write("=" * 100 + "\n")

                    print(f"收敛性汇总已保存到: {summary_file}")

                    # 上传到 wandb - 创建表格
                    # 获取所有超参数列名
                    param_names = list(successful_jobs[0]['param_combo'].keys())

                    # 确定最大的loss_history长度
                    max_loss_len = 0
                    for job in successful_jobs:
                        if 'convergence_metrics' in job and 'details' in job['convergence_metrics']:
                            if 'loss_history' in job['convergence_metrics']['details']:
                                loss_len = len(job['convergence_metrics']['details']['loss_history'])
                                max_loss_len = max(max_loss_len, loss_len)

                    # 创建表格列：超参数 + 收敛指标 + loss_0, loss_1, ..., loss_N
                    loss_columns = [f'loss_{i}' for i in range(max_loss_len)]
                    columns = param_names + ['Score', 'Status', 'Init_Loss', 'Final_Loss', 'Decrease'] + loss_columns
                    wandb_table = wandb.Table(columns=columns)

                    for job in successful_jobs:
                        if 'convergence_metrics' not in job or 'score' not in job['convergence_metrics']:
                            continue

                        metrics = job['convergence_metrics']

                        # 超参数
                        row_data = [job['param_combo'][p] for p in param_names]

                        # 收敛指标
                        row_data.extend([
                            metrics['score'],
                            metrics['status'],
                            metrics['details']['initial_loss'],
                            metrics['details']['final_loss'],
                            metrics['details']['total_decrease']
                        ])

                        # Loss history
                        loss_history = metrics['details'].get('loss_history', [])
                        for i in range(max_loss_len):
                            if i < len(loss_history):
                                row_data.append(loss_history[i])
                            else:
                                row_data.append(None)  # 填充空值

                        wandb_table.add_data(*row_data)

                    wandb.log({"bcsr_convergence_results": wandb_table})

                    # 上传最佳超参数
                    best_params = best_job['param_combo'].copy()
                    best_params['best_convergence_score'] = best_job['convergence_metrics']['score']
                    wandb.log({"best_hyperparameters": best_params})

                    # 发送完成告警
                    best_job_metrics = best_job['convergence_metrics']
                    alert_text = f"BCSR 超参数搜索完成！\n"
                    alert_text += f"成功任务数: {len(successful_jobs)}/{len(self.completed_jobs)}\n\n"
                    alert_text += "最佳超参数组合:\n"
                    alert_text += "=" * 50 + "\n"
                    for param, value in best_job['param_combo'].items():
                        alert_text += f"{param}: {value}\n"
                    alert_text += f"\n收敛得分: {best_job_metrics['score']:.4f} ({best_job_metrics['status']})\n"
                    alert_text += f"初始损失: {best_job_metrics['details']['initial_loss']:.4f}\n"
                    alert_text += f"最终损失: {best_job_metrics['details']['final_loss']:.4f}\n"
                    alert_text += f"总下降量: {best_job_metrics['details']['total_decrease']:.4f}\n"
                    alert_text += "=" * 50 + "\n"
                    alert_text += f"\n收敛得分统计: {sum(scores)/len(scores):.4f} ± {pd.Series(scores).std():.4f}\n"

                    wandb.alert(
                        title="BCSR 超参数搜索完成",
                        text=alert_text
                    )
                    print("\n已发送 wandb 告警通知")

        print("=" * 120 + "\n")


def generate_grid_search_commands(
    base_config: str,
    param_grid: Dict[str, List],
    exp_name_prefix: str,
    log_dir: str,
    script_name: str = "main_cl_buffer_analysis.py"
) -> List[Dict]:
    """
    生成网格搜索的所有命令

    Args:
        base_config: 基础配置文件路径
        param_grid: 超参数网格 {param_name: [value1, value2, ...]}
        exp_name_prefix: 实验名称前缀
        log_dir: 日志目录
        script_name: 要运行的脚本名称

    Returns:
        List[Dict]: 包含命令和任务信息的字典列表
    """
    commands = []

    # 生成所有超参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for idx, combo in enumerate(itertools.product(*param_values)):
        param_combo = dict(zip(param_names, combo))

        # 构造实验名称
        exp_name = f"{exp_name_prefix}_{idx:03d}"

        # 基础命令
        cmd = [
            "python", script_name,
            "--preset", base_config,
            "--exp_name", exp_name,
            "--log_dir", log_dir
        ]

        # 添加所有超参数
        for param_name, param_value in param_combo.items():
            # 特殊处理布尔参数（use_simplex_projection）
            if param_name == 'use_simplex_projection':
                # 只有当值为True时才添加该标志
                if str(param_value).lower() == 'true':
                    cmd.append(f"--{param_name}")
                # False时不添加标志（使用默认值）
                continue

            # 智能类型转换：如果是整数值的浮点数（如5.0），转为整数
            if isinstance(param_value, float) and param_value == int(param_value):
                param_value = int(param_value)

            cmd.extend([f"--{param_name}", str(param_value)])

        # 构造任务名称
        job_name = f"Combo_{idx:03d}"

        commands.append({
            'cmd': cmd,
            'job_name': job_name,
            'param_combo': param_combo,
            'exp_name': exp_name
        })

    return commands


def main():
    parser = argparse.ArgumentParser(
        description='BCSR 超参数收敛性搜索',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # GPU配置
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='可用的GPU ID，用逗号分隔')
    parser.add_argument('--max-jobs-per-gpu', type=int, default=1,
                        help='每个GPU上最多并行运行的任务数')

    # 实验配置
    parser.add_argument('--preset', type=str,
                        default='configs/csc_clam_cl_debug_buffer_ana.yaml',
                        help='配置文件路径')
    parser.add_argument('--exp-name-prefix', type=str, default='bcsr_search',
                        help='实验名称前缀')
    parser.add_argument('--log-dir', type=str, default='bcsr_convergence_search_logs',
                        help='日志根目录')
    parser.add_argument('--script', type=str, default='main_cl_buffer_analysis.py',
                        help='要运行的脚本名称')

    # 超参数网格定义
    parser.add_argument('--lr-proxy', type=str, default='1,5,10',
                        help='bcsr_lr_proxy_model的值，逗号分隔')
    parser.add_argument('--beta', type=str, default='0.01,0.1,1.0',
                        help='bcsr_beta的值，逗号分隔')
    parser.add_argument('--max-inner-it', type=str, default='1,3',
                        help='bcsr_max_inner_it的值，逗号分隔')
    parser.add_argument('--weight-lr', type=str, default='0.01,0.05,0.1',
                        help='bcsr_weight_lr的值，逗号分隔')
    parser.add_argument('--distall-lamda', type=str, default='0.1,0.5',
                        help='distall_lamda的值，逗号分隔')
    parser.add_argument('--topk-method', type=str, default='sigmoid,gumbel,ste',
                        help='topk_method的值，逗号分隔')
    parser.add_argument('--topk-temperature', type=str, default='0.1,0.5',
                        help='topk_temperature的值，逗号分隔')
    parser.add_argument('--normalize-method', type=str, default='none,l2,softmax,zscore',
                        help='normalize_method的值，逗号分隔')
    parser.add_argument('--use-simplex-projection', type=str, default='False',
                        help='use_simplex_projection的值，逗号分隔 (True or False)')
    parser.add_argument('--distill-target', type=str, default='logits',
                        help='distill_target的值，逗号分隔 (logits, features, both)')

    # 其他配置
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印命令，不实际运行')
    parser.add_argument('--check-interval', type=float, default=10.0,
                        help='检查任务完成的时间间隔（秒）')
    parser.add_argument('--notes', type=str, default=None,
                        help='wandb 实验备注')

    args = parser.parse_args()

    # 解析GPU ID
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    print(f"可用GPU: {gpu_ids}")
    print(f"每GPU最大任务数: {args.max_jobs_per_gpu}")
    print()

    # 构建超参数网格
    param_grid = {
        'bcsr_lr_proxy_model': [float(x) for x in args.lr_proxy.split(',')],
        'bcsr_beta': [float(x) for x in args.beta.split(',')],
        'bcsr_max_inner_it': [int(x) for x in args.max_inner_it.split(',')],
        'bcsr_weight_lr': [float(x) for x in args.weight_lr.split(',')],
        'distall_lamda': [float(x) for x in args.distall_lamda.split(',')],
        'topk_method': [x.strip() for x in args.topk_method.split(',')],
        'topk_temperature': [float(x) for x in args.topk_temperature.split(',')],
        'normalize_method': [x.strip() for x in args.normalize_method.split(',')],
        'use_simplex_projection': [x.strip() for x in args.use_simplex_projection.split(',')],
        'distill_target': [x.strip() for x in args.distill_target.split(',')],
    }

    print("超参数搜索空间:")
    total_combinations = 1
    for param_name, values in param_grid.items():
        print(f"  {param_name}: {values}")
        total_combinations *= len(values)
    print(f"\n总组合数: {total_combinations}")
    print()

    # 生成所有命令
    commands = generate_grid_search_commands(
        base_config=args.preset,
        param_grid=param_grid,
        exp_name_prefix=args.exp_name_prefix,
        log_dir=args.log_dir,
        script_name=args.script
    )

    print(f"生成了 {len(commands)} 个实验任务\n")

    # 估算运行时间（假设每个任务平均2分钟）
    avg_time_per_task_minutes = 2.0
    total_time_minutes = len(commands) * avg_time_per_task_minutes
    parallel_time_minutes = total_time_minutes / (len(gpu_ids) * args.max_jobs_per_gpu)

    print("=" * 80)
    print("预计运行时间估算")
    print("=" * 80)
    print(f"总任务数: {len(commands)}")
    print(f"可用GPU数: {len(gpu_ids)} (每GPU最多 {args.max_jobs_per_gpu} 个并行任务)")
    print(f"总并行度: {len(gpu_ids) * args.max_jobs_per_gpu}")
    print(f"\n假设每个任务平均运行 {avg_time_per_task_minutes} 分钟:")
    print(f"  - 串行总时间: {total_time_minutes/60:.1f} 小时 ({total_time_minutes:.0f} 分钟)")
    print(f"  - 并行预计时间: {parallel_time_minutes/60:.1f} 小时 ({parallel_time_minutes:.0f} 分钟)")
    print("=" * 80)
    print()

    # 确保日志目录存在
    os.makedirs(args.log_dir, exist_ok=True)

    # 保存实验配置
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_file = os.path.join(args.log_dir, f"bcsr_search_config_{timestamp}.json")
    with open(config_file, 'w') as f:
        json.dump({
            'gpus': gpu_ids,
            'max_jobs_per_gpu': args.max_jobs_per_gpu,
            'param_grid': {k: [str(v) for v in vals] for k, vals in param_grid.items()},
            'base_config': args.preset,
            'exp_name_prefix': args.exp_name_prefix,
            'log_dir': args.log_dir,
            'script': args.script,
            'total_combinations': total_combinations
        }, f, indent=2)
    print(f"实验配置已保存到: {config_file}\n")

    if args.dry_run:
        print("=" * 80)
        print("DRY RUN - 只显示前5个命令示例")
        print("=" * 80)
        for i, cmd_info in enumerate(commands[:5], 1):
            print(f"\n任务 {i}/{len(commands)}: {cmd_info['job_name']}")
            print(f"超参数组合:")
            for k, v in cmd_info['param_combo'].items():
                print(f"  {k}: {v}")
            print(f"命令: {' '.join(cmd_info['cmd'])}")
        print(f"\n... 以及其他 {len(commands)-5} 个任务")
        return

    # ========== 用户确认提示 ==========
    print("\n" + "!" * 80)
    print("⚠️  警告：即将开始大规模超参数搜索")
    print("!" * 80)
    print(f"\n将要执行 {len(commands)} 个实验，预计需要 {parallel_time_minutes/60:.1f} 小时")
    print(f"日志将保存到: {args.log_dir}")
    print(f"\n请确认以下信息:")
    print(f"  - GPU 设备: {gpu_ids}")
    print(f"  - 并行任务数: {len(gpu_ids) * args.max_jobs_per_gpu}")
    print(f"  - 预计完成时间: {parallel_time_minutes/60:.1f} 小时")
    print(f"  - 配置文件: {args.preset}")
    print(f"\n超参数搜索空间:")
    for param_name, values in param_grid.items():
        print(f"  - {param_name}: {values}")

    print("\n" + "!" * 80)
    user_input = input("\n确认开始运行吗？(输入 'yes' 确认，其他任何输入取消): ")
    print("!" * 80 + "\n")

    if user_input.strip().lower() != 'yes':
        print("❌ 用户取消运行")
        print("提示: 如果只想查看命令，可以使用 --dry-run 参数")
        return

    print("✅ 用户确认，开始执行任务...\n")

    # 创建GPU调度器
    wandb_config = {
        'gpus': gpu_ids,
        'max_jobs_per_gpu': args.max_jobs_per_gpu,
        'param_grid': param_grid,
        'base_config': args.preset,
        'total_combinations': total_combinations
    }
    scheduler = GPUScheduler(
        gpu_ids,
        args.max_jobs_per_gpu,
        wandb_project="BCSR_Convergence",
        wandb_config=wandb_config,
        exp_name_prefix=args.exp_name_prefix,
        notes=args.notes
    )

    print("=" * 80)
    print("开始执行任务")
    print("=" * 80)
    print()

    # 提交所有任务
    for cmd_info in commands:
        gpu_id = scheduler.wait_for_slot(check_interval=args.check_interval)
        scheduler.submit_job(
            gpu_id=gpu_id,
            cmd=cmd_info['cmd'],
            job_name=cmd_info['job_name'],
            param_combo=cmd_info['param_combo'],
            exp_name=cmd_info['exp_name'],
            log_dir=args.log_dir
        )

    # 等待所有任务完成
    scheduler.wait_all_jobs(check_interval=args.check_interval)

    # 打印汇总表格并保存结果
    scheduler.print_summary_table()
    scheduler.save_results(log_dir=args.log_dir)

    # 关闭 wandb
    wandb.finish()
    print("wandb run 已完成")


if __name__ == "__main__":
    main()
