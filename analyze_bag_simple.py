#!/usr/bin/env python3
"""
简单的 Bag Instance 统计脚本

直接读取特征文件，统计每个bag的instance数量并生成直方图

Usage:
    # CSC 数据集 (读取 .npy 文件)
    python analyze_bag_simple.py --data_dir ./data/embeddings --file_type npy

    # Camelyon 数据集 (读取 .pt 文件)
    python analyze_bag_simple.py --data_dir ./data/pt_files --file_type pt
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from glob import glob
import torch


def load_features(file_path, file_type):
    """
    加载特征文件并返回instance数量

    Args:
        file_path: 文件路径
        file_type: 文件类型 ('npy', 'pt', 'h5')

    Returns:
        n_instances: instance数量
    """
    try:
        if file_type == 'npy':
            features = np.load(file_path)
            return features.shape[0]
        elif file_type == 'pt':
            features = torch.load(file_path)
            if isinstance(features, torch.Tensor):
                return features.shape[0]
            elif isinstance(features, dict) and 'features' in features:
                return features['features'].shape[0]
            else:
                return len(features)
        elif file_type == 'h5':
            import h5py
            with h5py.File(file_path, 'r') as f:
                # 通常特征存储在 'features' 键下
                if 'features' in f:
                    return f['features'].shape[0]
                else:
                    # 取第一个dataset
                    key = list(f.keys())[0]
                    return f[key].shape[0]
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def collect_instance_counts(data_dir, file_type='npy', pattern='*'):
    """
    收集目录下所有文件的instance数量

    Args:
        data_dir: 数据目录
        file_type: 文件类型
        pattern: 文件名模式

    Returns:
        List of tuples: [(filename, instance_count), ...]
    """
    # 查找所有文件
    search_pattern = os.path.join(data_dir, f'{pattern}.{file_type}')
    files = glob(search_pattern)

    print(f"Found {len(files)} .{file_type} files in {data_dir}")

    if len(files) == 0:
        print(f"Warning: No files found matching pattern: {search_pattern}")
        return []

    instance_counts = []

    for file_path in tqdm(files, desc="Processing files"):
        filename = os.path.basename(file_path)
        n_instances = load_features(file_path, file_type)

        if n_instances is not None:
            instance_counts.append((filename, n_instances))

    return instance_counts


def compute_statistics(instance_counts):
    """
    计算统计量

    Args:
        instance_counts: List of (filename, count) tuples

    Returns:
        Dictionary of statistics
    """
    if len(instance_counts) == 0:
        return None

    counts = [count for _, count in instance_counts]

    stats = {
        'n_bags': len(counts),
        'total_instances': sum(counts),
        'mean': np.mean(counts),
        'median': np.median(counts),
        'std': np.std(counts),
        'min': np.min(counts),
        'max': np.max(counts),
        'q25': np.percentile(counts, 25),
        'q75': np.percentile(counts, 75)
    }

    return stats


def plot_histogram(instance_counts, stats, output_path):
    """
    绘制直方图

    Args:
        instance_counts: List of (filename, count) tuples
        stats: Statistics dictionary
        output_path: 输出路径
    """
    matplotlib.use('Agg')

    # 设置论文风格
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10

    counts = [count for _, count in instance_counts]

    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # 绘制直方图
    n_bins = min(50, max(10, len(set(counts)) // 2))
    ax.hist(counts, bins=n_bins, color='#2E86AB', alpha=0.7,
            edgecolor='black', linewidth=0.5)

    # 添加均值线
    ax.axvline(stats['mean'], color='#F18F01', linestyle='--',
               linewidth=2, label=f"Mean: {stats['mean']:.1f}")

    # 添加中位数线
    ax.axvline(stats['median'], color='#06A77D', linestyle='--',
               linewidth=2, label=f"Median: {stats['median']:.1f}")

    # 标签
    ax.set_xlabel('Number of Instances per Bag', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Instance Count Distribution', fontsize=14, fontweight='bold')

    # 网格
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # 图例
    ax.legend(loc='best', fontsize=11, framealpha=0.9)

    # 布局
    plt.tight_layout()

    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Histogram saved to: {output_path}")
    plt.close()


def save_statistics(stats, output_path):
    """
    保存统计数据到CSV

    Args:
        stats: Statistics dictionary
        output_path: 输出路径
    """
    df = pd.DataFrame([stats])
    df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"✅ Statistics saved to: {output_path}")


def save_instance_list(instance_counts, output_path):
    """
    保存详细列表到CSV

    Args:
        instance_counts: List of (filename, count) tuples
        output_path: 输出路径
    """
    df = pd.DataFrame(instance_counts, columns=['filename', 'n_instances'])
    df = df.sort_values('n_instances', ascending=False)
    df.to_csv(output_path, index=False)
    print(f"✅ Instance list saved to: {output_path}")


def print_statistics(stats, n_sample_bags=None):
    """
    打印统计信息到控制台

    Args:
        stats: Statistics dictionary
        n_sample_bags: 如果指定，计算随机采样N个bags的expected instances
    """
    print(f"\n{'='*60}")
    print(f"  Instance Count Statistics")
    print(f"{'='*60}")
    print(f"  Number of Bags:       {stats['n_bags']}")
    print(f"  Total Instances:      {stats['total_instances']}")
    print(f"  Mean:                 {stats['mean']:.2f}")
    print(f"  Median:               {stats['median']:.2f}")
    print(f"  Std Dev:              {stats['std']:.2f}")
    print(f"  Min:                  {stats['min']}")
    print(f"  Max:                  {stats['max']}")
    print(f"  25th Percentile:      {stats['q25']:.2f}")
    print(f"  75th Percentile:      {stats['q75']:.2f}")

    # 如果指定了采样数量，计算期望的instance数
    if n_sample_bags is not None and n_sample_bags > 0:
        print(f"{'='*60}")
        print(f"  Random Sampling Estimation")
        print(f"{'='*60}")

        # 期望值计算：均值 × 采样数量
        expected_instances = stats['mean'] * n_sample_bags

        # 最小/最大可能值（假设采样到最小/最大的bags）
        min_possible = stats['min'] * n_sample_bags
        max_possible = stats['max'] * n_sample_bags

        # 使用中位数的估计
        median_estimate = stats['median'] * n_sample_bags

        print(f"  If randomly sampling {n_sample_bags} bags:")
        print(f"  Expected Instances (mean):   {expected_instances:.0f}")
        print(f"  Expected Instances (median): {median_estimate:.0f}")
        print(f"  Theoretical Min:             {min_possible:.0f}")
        print(f"  Theoretical Max:             {max_possible:.0f}")
        print(f"  Percentage of total:         {expected_instances/stats['total_instances']*100:.2f}%")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Simple bag instance count analysis')

    # 必需参数
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing feature files')
    parser.add_argument('--file_type', type=str, default='npy',
                        choices=['npy', 'pt', 'h5'],
                        help='Type of feature files')

    # 可选参数
    parser.add_argument('--pattern', type=str, default='*',
                        help='File name pattern (e.g., "patient_*")')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                        help='Output directory for results')
    parser.add_argument('--output_prefix', type=str, default='bag_instances',
                        help='Prefix for output files')
    parser.add_argument('--sample_bags', type=int, default=42,
                        help='Number of bags to estimate for random sampling (default: 42)')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Bag Instance Analysis (Simple)")
    print(f"{'='*60}")
    print(f"  Data Directory:  {args.data_dir}")
    print(f"  File Type:       .{args.file_type}")
    print(f"  Pattern:         {args.pattern}.{args.file_type}")
    print(f"  Output Dir:      {args.output_dir}")
    print(f"{'='*60}\n")

    # 收集数据
    instance_counts = collect_instance_counts(args.data_dir, args.file_type, args.pattern)

    if len(instance_counts) == 0:
        print("❌ No data collected. Please check your data directory and file type.")
        return

    # 计算统计量
    stats = compute_statistics(instance_counts)

    # 打印统计信息（包含采样估计）
    print_statistics(stats, n_sample_bags=args.sample_bags)

    # 生成输出文件
    output_prefix = os.path.join(args.output_dir, args.output_prefix)

    plot_histogram(instance_counts, stats, f'{output_prefix}_histogram.png')
    save_statistics(stats, f'{output_prefix}_stats.csv')
    save_instance_list(instance_counts, f'{output_prefix}_list.csv')

    print(f"\n✅ Analysis complete! Results saved to: {args.output_dir}\n")


if __name__ == '__main__':
    main()
