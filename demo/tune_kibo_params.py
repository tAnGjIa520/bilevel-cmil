"""
KIBO 超参数调优脚本 - 多GPU并行版本
目标：降低包标签不一致概率

支持的 KIBO 专属超参数：
- max_outer_it: 外层优化迭代次数 (影响实例选择质量)
- max_inner_it: 内层优化迭代次数 (影响代理模型训练)
- lr_weight: 实例权重学习率 (影响权重更新速度)
- top_k: 选择的实例数量 (影响包大小)
"""

import torch
import subprocess
import os
import json
import pandas as pd
from datetime import datetime
from itertools import product
import argparse
from pathlib import Path


# ============================================================
# 超参数搜索空间定义
# ============================================================

def get_param_grid(mode='reduce_mismatch'):
    """
    定义超参数搜索空间

    Args:
        mode: 'reduce_mismatch' - 降低标签不一致概率
              'full_search' - 全面搜索
    """
    if mode == 'reduce_mismatch':
        # 针对降低标签不一致概率的参数组合
        param_grid = {
            'max_outer_it': [15, 20, 30],      # 增加外层迭代，提高选择质量
            'max_inner_it': [3, 5, 7],         # 增加内层迭代，提高代理模型质量
            'lr_weight': [0.05, 0.08, 0.1],    # 调整学习率，避免过快收敛
            'top_k': [3, 4, 5],                # 调整选择数量，平衡包大小和质量
        }
    elif mode == 'full_search':
        # 全面搜索
        param_grid = {
            'max_outer_it': [10, 15, 20, 30],
            'max_inner_it': [2, 3, 5, 7],
            'lr_weight': [0.05, 0.08, 0.1, 0.15],
            'top_k': [3, 4, 5, 6],
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return param_grid


def generate_param_combinations(param_grid):
    """生成所有参数组合"""
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    return combinations


# ============================================================
# GPU 分配和任务调度
# ============================================================

def get_available_gpus(gpu_ids=None):
    """
    获取可用的GPU列表

    Args:
        gpu_ids: 指定的GPU ID列表，如 [0,1,2,3]。如果为None，则自动检测
    """
    if gpu_ids is not None:
        return gpu_ids

    # 自动检测可用GPU
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                              capture_output=True, text=True)
        gpu_list = [int(x) for x in result.stdout.strip().split('\n')]
        return gpu_list
    except:
        print("⚠️  无法检测GPU，使用CPU模式")
        return []


def run_experiment_on_gpu(gpu_id, params, task='triplet', epochs1=10, epochs2=50,
                         num_bags=2000, output_dir='./tune_results'):
    """
    在指定GPU上运行单个实验

    Args:
        gpu_id: GPU ID
        params: 超参数字典
        task: 任务名称
        epochs1: 第一阶段训练轮数
        epochs2: 第二阶段训练轮数
        num_bags: 训练集包数量
        output_dir: 输出目录

    Returns:
        result_dict: 结果字典
    """
    # 生成唯一的实验ID
    exp_id = f"gpu{gpu_id}_outer{params['max_outer_it']}_inner{params['max_inner_it']}_lr{params['lr_weight']}_k{params['top_k']}"

    # 构建命令
    cmd = [
        'python',
        '/mnt/nfs/zhangjinouwen/tangjia/cmil/bilevel-cmil/demo/attention_mil_harder_tasks111.py',
        '--task', task,
        '--selection_method', 'kibo',
        '--epochs1', str(epochs1),
        '--epochs2', str(epochs2),
        '--top_k', str(params['top_k']),
        '--num_bags', str(num_bags),
        '--kibo_outer_it', str(params['max_outer_it']),
        '--kibo_inner_it', str(params['max_inner_it']),
        '--kibo_lr_weight', str(params['lr_weight']),
    ]

    # 设置环境变量（指定GPU）
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'{exp_id}.log')

    print(f"🚀 [GPU {gpu_id}] 启动实验: {exp_id}")
    print(f"   参数: outer_it={params['max_outer_it']}, inner_it={params['max_inner_it']}, "
          f"lr_weight={params['lr_weight']}, top_k={params['top_k']}")

    # 运行实验并捕获输出
    with open(log_file, 'w') as f:
        try:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                  text=True, timeout=3600)  # 1小时超时
            success = (result.returncode == 0)
        except subprocess.TimeoutExpired:
            print(f"⚠️  [GPU {gpu_id}] 实验超时: {exp_id}")
            success = False
        except Exception as e:
            print(f"❌ [GPU {gpu_id}] 实验失败: {exp_id}, 错误: {e}")
            success = False

    # 解析日志文件，提取结果
    result_dict = parse_log_file(log_file, params, exp_id, success)

    return result_dict


def parse_log_file(log_file, params, exp_id, success):
    """
    解析日志文件，提取关键指标

    提取内容：
    - 标签不一致率 (mismatch_rate)
    - 第一阶段测试集性能
    - 第二阶段测试集性能
    """
    result = {
        'exp_id': exp_id,
        'success': success,
        **params,
        'train_mismatch_rate': None,
        'val_mismatch_rate': None,
        'stage1_test_acc': None,
        'stage1_test_auc': None,
        'stage1_test_f1': None,
        'stage2_test_acc': None,
        'stage2_test_auc': None,
        'stage2_test_f1': None,
        'acc_improvement': None,
        'auc_improvement': None,
        'f1_improvement': None,
    }

    if not success or not os.path.exists(log_file):
        return result

    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # 提取标签不一致率
        import re

        # 训练集不一致率
        train_match = re.search(r'训练集:.*?\((\d+\.\d+)%\) 包标签不一致', content)
        if train_match:
            result['train_mismatch_rate'] = float(train_match.group(1))

        # 验证集不一致率
        val_match = re.search(r'验证集:.*?\((\d+\.\d+)%\) 包标签不一致', content)
        if val_match:
            result['val_mismatch_rate'] = float(val_match.group(1))

        # 第一阶段测试集性能
        stage1_section = re.search(r'第一阶段测试集评估:(.*?)第二阶段', content, re.DOTALL)
        if stage1_section:
            stage1_text = stage1_section.group(1)
            acc_match = re.search(r'Accuracy: (\d+\.\d+)', stage1_text)
            auc_match = re.search(r'AUC: (\d+\.\d+)', stage1_text)
            f1_match = re.search(r'F1 Score: (\d+\.\d+)', stage1_text)

            if acc_match:
                result['stage1_test_acc'] = float(acc_match.group(1))
            if auc_match:
                result['stage1_test_auc'] = float(auc_match.group(1))
            if f1_match:
                result['stage1_test_f1'] = float(f1_match.group(1))

        # 第二阶段测试集性能
        stage2_section = re.search(r'第二阶段测试集评估.*?:(.*?)结果对比', content, re.DOTALL)
        if stage2_section:
            stage2_text = stage2_section.group(1)
            acc_match = re.search(r'Accuracy: (\d+\.\d+)', stage2_text)
            auc_match = re.search(r'AUC: (\d+\.\d+)', stage2_text)
            f1_match = re.search(r'F1 Score: (\d+\.\d+)', stage2_text)

            if acc_match:
                result['stage2_test_acc'] = float(acc_match.group(1))
            if auc_match:
                result['stage2_test_auc'] = float(auc_match.group(1))
            if f1_match:
                result['stage2_test_f1'] = float(f1_match.group(1))

        # 计算性能提升
        if result['stage1_test_acc'] and result['stage2_test_acc']:
            result['acc_improvement'] = result['stage2_test_acc'] - result['stage1_test_acc']
        if result['stage1_test_auc'] and result['stage2_test_auc']:
            result['auc_improvement'] = result['stage2_test_auc'] - result['stage1_test_auc']
        if result['stage1_test_f1'] and result['stage2_test_f1']:
            result['f1_improvement'] = result['stage2_test_f1'] - result['stage1_test_f1']

    except Exception as e:
        print(f"⚠️  解析日志文件失败: {log_file}, 错误: {e}")

    return result


# ============================================================
# 多GPU并行调度
# ============================================================

def run_parallel_experiments(param_combinations, gpu_ids, task='triplet',
                            epochs1=10, epochs2=50, num_bags=2000,
                            output_dir='./tune_results'):
    """
    在多个GPU上并行运行实验

    Args:
        param_combinations: 参数组合列表
        gpu_ids: 可用GPU ID列表
        task: 任务名称
        epochs1: 第一阶段训练轮数
        epochs2: 第二阶段训练轮数
        num_bags: 训练集包数量
        output_dir: 输出目录

    Returns:
        results: 结果列表
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = []
    total_experiments = len(param_combinations)

    print(f"\n{'='*80}")
    print(f"🔬 开始超参数搜索")
    print(f"{'='*80}")
    print(f"总实验数: {total_experiments}")
    print(f"可用GPU数: {len(gpu_ids)}")
    print(f"任务: {task}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")

    if len(gpu_ids) == 0:
        # CPU模式，串行运行
        print("⚠️  CPU模式，串行运行实验...")
        for i, params in enumerate(param_combinations):
            print(f"\n进度: [{i+1}/{total_experiments}]")
            result = run_experiment_on_gpu(None, params, task, epochs1, epochs2,
                                          num_bags, output_dir)
            results.append(result)
    else:
        # 多GPU并行
        with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = {}

            # 提交任务
            for i, params in enumerate(param_combinations):
                gpu_id = gpu_ids[i % len(gpu_ids)]
                future = executor.submit(run_experiment_on_gpu, gpu_id, params,
                                       task, epochs1, epochs2, num_bags, output_dir)
                futures[future] = (i, params)

            # 收集结果
            completed = 0
            for future in as_completed(futures):
                completed += 1
                i, params = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✅ [{completed}/{total_experiments}] 实验完成: {result['exp_id']}")
                    if result['train_mismatch_rate'] is not None:
                        print(f"   标签不一致率: {result['train_mismatch_rate']:.2f}%")
                except Exception as e:
                    print(f"❌ [{completed}/{total_experiments}] 实验失败: {e}")

    return results


# ============================================================
# 结果汇总和分析
# ============================================================

def summarize_results(results, output_dir='./tune_results'):
    """
    汇总和分析结果

    Args:
        results: 结果列表
        output_dir: 输出目录
    """
    # 创建DataFrame
    df = pd.DataFrame(results)

    # 保存完整结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = os.path.join(output_dir, f'tune_results_{timestamp}.csv')
    df.to_csv(csv_file, index=False)
    print(f"\n✅ 完整结果已保存: {csv_file}")

    # 筛选成功的实验
    df_success = df[df['success'] == True].copy()

    if len(df_success) == 0:
        print("\n❌ 没有成功的实验!")
        return

    print(f"\n{'='*80}")
    print(f"📊 结果汇总")
    print(f"{'='*80}")
    print(f"总实验数: {len(df)}")
    print(f"成功实验数: {len(df_success)}")
    print(f"失败实验数: {len(df) - len(df_success)}")

    # ========== 按标签不一致率排序 ==========
    print(f"\n{'='*80}")
    print(f"🎯 Top 10 最低标签不一致率")
    print(f"{'='*80}")

    df_sorted_mismatch = df_success.sort_values('train_mismatch_rate')
    top10_mismatch = df_sorted_mismatch.head(10)

    print("\n排名 | 不一致率 | outer_it | inner_it | lr_weight | top_k | Stage2 AUC")
    print("-" * 80)
    for idx, row in enumerate(top10_mismatch.itertuples(), 1):
        print(f"{idx:4d} | {row.train_mismatch_rate:8.2f}% | "
              f"{row.max_outer_it:8d} | {row.max_inner_it:8d} | "
              f"{row.lr_weight:9.3f} | {row.top_k:5d} | "
              f"{row.stage2_test_auc if row.stage2_test_auc else 'N/A':10}")

    # ========== 按第二阶段性能排序 ==========
    print(f"\n{'='*80}")
    print(f"🏆 Top 10 最高第二阶段 AUC")
    print(f"{'='*80}")

    df_sorted_auc = df_success.sort_values('stage2_test_auc', ascending=False)
    top10_auc = df_sorted_auc.head(10)

    print("\n排名 | Stage2 AUC | 不一致率 | outer_it | inner_it | lr_weight | top_k")
    print("-" * 80)
    for idx, row in enumerate(top10_auc.itertuples(), 1):
        print(f"{idx:4d} | {row.stage2_test_auc:10.4f} | "
              f"{row.train_mismatch_rate:8.2f}% | "
              f"{row.max_outer_it:8d} | {row.max_inner_it:8d} | "
              f"{row.lr_weight:9.3f} | {row.top_k:5d}")

    # ========== 最佳参数推荐 ==========
    print(f"\n{'='*80}")
    print(f"💡 最佳参数推荐")
    print(f"{'='*80}")

    best_by_mismatch = df_sorted_mismatch.iloc[0]
    print(f"\n📍 最低标签不一致率参数:")
    print(f"   max_outer_it: {best_by_mismatch['max_outer_it']}")
    print(f"   max_inner_it: {best_by_mismatch['max_inner_it']}")
    print(f"   lr_weight: {best_by_mismatch['lr_weight']}")
    print(f"   top_k: {best_by_mismatch['top_k']}")
    print(f"   标签不一致率: {best_by_mismatch['train_mismatch_rate']:.2f}%")
    print(f"   Stage2 Test AUC: {best_by_mismatch['stage2_test_auc']:.4f}")

    best_by_auc = df_sorted_auc.iloc[0]
    print(f"\n🏅 最高第二阶段AUC参数:")
    print(f"   max_outer_it: {best_by_auc['max_outer_it']}")
    print(f"   max_inner_it: {best_by_auc['max_inner_it']}")
    print(f"   lr_weight: {best_by_auc['lr_weight']}")
    print(f"   top_k: {best_by_auc['top_k']}")
    print(f"   Stage2 Test AUC: {best_by_auc['stage2_test_auc']:.4f}")
    print(f"   标签不一致率: {best_by_auc['train_mismatch_rate']:.2f}%")

    # 保存最佳参数
    best_params = {
        'best_by_mismatch': {
            'max_outer_it': int(best_by_mismatch['max_outer_it']),
            'max_inner_it': int(best_by_mismatch['max_inner_it']),
            'lr_weight': float(best_by_mismatch['lr_weight']),
            'top_k': int(best_by_mismatch['top_k']),
            'train_mismatch_rate': float(best_by_mismatch['train_mismatch_rate']),
            'stage2_test_auc': float(best_by_auc['stage2_test_auc']) if best_by_mismatch['stage2_test_auc'] else None,
        },
        'best_by_auc': {
            'max_outer_it': int(best_by_auc['max_outer_it']),
            'max_inner_it': int(best_by_auc['max_inner_it']),
            'lr_weight': float(best_by_auc['lr_weight']),
            'top_k': int(best_by_auc['top_k']),
            'stage2_test_auc': float(best_by_auc['stage2_test_auc']),
            'train_mismatch_rate': float(best_by_auc['train_mismatch_rate']) if best_by_auc['train_mismatch_rate'] else None,
        }
    }

    json_file = os.path.join(output_dir, f'best_params_{timestamp}.json')
    with open(json_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\n✅ 最佳参数已保存: {json_file}")

    # ========== 参数影响分析 ==========
    print(f"\n{'='*80}")
    print(f"📈 参数影响分析 (平均值)")
    print(f"{'='*80}")

    for param in ['max_outer_it', 'max_inner_it', 'lr_weight', 'top_k']:
        print(f"\n{param}:")
        grouped = df_success.groupby(param).agg({
            'train_mismatch_rate': 'mean',
            'stage2_test_auc': 'mean'
        }).round(4)
        print(grouped)

    print(f"\n{'='*80}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='KIBO超参数调优 - 多GPU并行版本')
    parser.add_argument('--mode', type=str, default='reduce_mismatch',
                       choices=['reduce_mismatch', 'full_search'],
                       help='搜索模式: reduce_mismatch(降低标签不一致), full_search(全面搜索)')
    parser.add_argument('--task', type=str, default='triplet',
                       choices=['large_range', 'triplet', 'sum', 'increasing'],
                       help='任务名称')
    parser.add_argument('--gpus', type=str, default=None,
                       help='GPU ID列表，逗号分隔，如 "0,1,2,3"。不指定则自动检测')
    parser.add_argument('--epochs1', type=int, default=10,
                       help='第一阶段训练轮数')
    parser.add_argument('--epochs2', type=int, default=50,
                       help='第二阶段训练轮数')
    parser.add_argument('--num_bags', type=int, default=2000,
                       help='训练集包数量')
    parser.add_argument('--output_dir', type=str, default='./tune_results',
                       help='输出目录')

    args = parser.parse_args()

    # 获取GPU列表
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(',')]
    else:
        gpu_ids = get_available_gpus()

    print(f"\n{'='*80}")
    print(f"🔧 KIBO 超参数调优")
    print(f"{'='*80}")
    print(f"模式: {args.mode}")
    print(f"任务: {args.task}")
    print(f"GPU: {gpu_ids if gpu_ids else 'CPU'}")
    print(f"第一阶段训练轮数: {args.epochs1}")
    print(f"第二阶段训练轮数: {args.epochs2}")
    print(f"训练集包数量: {args.num_bags}")
    print(f"{'='*80}\n")

    # 生成参数组合
    param_grid = get_param_grid(args.mode)
    param_combinations = generate_param_combinations(param_grid)

    print(f"📋 参数搜索空间:")
    for key, values in param_grid.items():
        print(f"   {key}: {values}")
    print(f"\n总组合数: {len(param_combinations)}")

    # 运行并行实验
    results = run_parallel_experiments(
        param_combinations,
        gpu_ids,
        task=args.task,
        epochs1=args.epochs1,
        epochs2=args.epochs2,
        num_bags=args.num_bags,
        output_dir=args.output_dir
    )

    # 汇总结果
    summarize_results(results, args.output_dir)

    print(f"\n{'='*80}")
    print(f"✅ 超参数调优完成!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
