#!/usr/bin/env python3
"""
Bag Instance Count Analysis Script

This script analyzes the number of instances (patches) in each bag (WSI)
from MIL datasets and generates statistics and visualizations.

Usage:
    # Basic usage for CSC dataset
    python analyze_bag_instances.py --dataset csc_cl --data_root ./data --n_tasks 2 --task 0

    # Analyze specific split
    python analyze_bag_instances.py --dataset csc_cl --split train --task 0

    # Custom output directory
    python analyze_bag_instances.py --dataset csc_cl --output_dir ./my_analysis

Required for some datasets:
    --csv_file: CSV file name (for Camelyon datasets)
    --split_dir: Directory containing split files
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import torch

# Import dataset loading function
from datasets.dataset import datamodule_gen


class Args:
    """Simple argument container to mimic argparse Namespace"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def collect_instance_counts(dataloader, split_name):
    """
    Collect instance counts from a dataloader.

    Args:
        dataloader: PyTorch DataLoader
        split_name: Name of the split (train/val/test)

    Returns:
        List of tuples: [(slide_id, instance_count, label), ...]
    """
    instance_counts = []

    print(f"Collecting instance counts from {split_name} split...")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {split_name}")):
        features = batch['features']
        slide_ids = batch['slide_id']
        labels = batch['label']

        # Handle batch
        if isinstance(slide_ids, list):
            for i, slide_id in enumerate(slide_ids):
                # In MIL, batch_size is usually 1, so features are from one bag
                if len(slide_ids) == 1:
                    n_instances = features.shape[0]
                else:
                    # If batch_size > 1, need to split features
                    # This is tricky without n_patch info
                    n_instances = features.shape[0]  # Approximation

                label = labels[i].item() if torch.is_tensor(labels) else labels[i]
                instance_counts.append((slide_id, n_instances, label))
        else:
            # Single item
            slide_id = slide_ids
            n_instances = features.shape[0]
            label = labels.item() if torch.is_tensor(labels) else labels
            instance_counts.append((slide_id, n_instances, label))

    return instance_counts


def compute_statistics(instance_counts):
    """
    Compute statistics from instance counts.

    Args:
        instance_counts: List of (slide_id, count, label) tuples

    Returns:
        Dictionary of statistics
    """
    counts = [count for _, count, _ in instance_counts]

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


def plot_histogram(instance_counts, stats, output_path, split_name='All'):
    """
    Plot histogram of instance counts.

    Args:
        instance_counts: List of (slide_id, count, label) tuples
        stats: Statistics dictionary
        output_path: Path to save the figure
        split_name: Name of the data split
    """
    matplotlib.use('Agg')

    # Set publication-ready style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10

    counts = [count for _, count, _ in instance_counts]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # Plot histogram
    n_bins = min(50, len(set(counts)))  # Adaptive bin count
    ax.hist(counts, bins=n_bins, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add mean line
    ax.axvline(stats['mean'], color='#F18F01', linestyle='--', linewidth=2,
               label=f"Mean: {stats['mean']:.1f}")

    # Add median line
    ax.axvline(stats['median'], color='#06A77D', linestyle='--', linewidth=2,
               label=f"Median: {stats['median']:.1f}")

    # Labels and title
    ax.set_xlabel('Number of Instances per Bag', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Instance Count Distribution ({split_name} Split)', fontsize=12, fontweight='bold')

    # Grid
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Histogram saved to: {output_path}")
    plt.close()


def save_statistics(stats, output_path):
    """
    Save statistics to CSV file.

    Args:
        stats: Statistics dictionary
        output_path: Path to save CSV
    """
    df = pd.DataFrame([stats])
    df.to_csv(output_path, index=False)
    print(f"Statistics saved to: {output_path}")


def save_instance_list(instance_counts, output_path):
    """
    Save detailed instance counts to CSV.

    Args:
        instance_counts: List of (slide_id, count, label) tuples
        output_path: Path to save CSV
    """
    df = pd.DataFrame(instance_counts, columns=['slide_id', 'n_instances', 'label'])
    df = df.sort_values('n_instances', ascending=False)
    df.to_csv(output_path, index=False)
    print(f"Instance list saved to: {output_path}")


def print_statistics(stats, split_name='All'):
    """
    Print statistics to console.

    Args:
        stats: Statistics dictionary
        split_name: Name of the data split
    """
    print(f"\n{'='*60}")
    print(f"  Instance Count Statistics ({split_name} Split)")
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
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze instance counts in MIL bags')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='csc_cl',
                        help='Dataset name (e.g., csc_cl, camelyon_cl)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for data')
    parser.add_argument('--feat_dir', type=str, default='embeddings',
                        help='Feature directory name')

    # Task arguments
    parser.add_argument('--n_tasks', type=int, default=2,
                        help='Number of tasks (for continual learning)')
    parser.add_argument('--task', type=int, default=0,
                        help='Task ID to analyze')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold number')
    parser.add_argument('--n_folds', type=int, default=1,
                        help='Total number of folds')

    # Analysis arguments
    parser.add_argument('--split', type=str, default='all', choices=['train', 'val', 'test', 'all'],
                        help='Which split to analyze')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                        help='Output directory for results')

    # Other arguments
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (usually 1 for MIL)')
    parser.add_argument('--n_classes', type=int, default=6,
                        help='Number of classes')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--csv_file', type=str, default='dataset.csv',
                        help='CSV file name (for some datasets)')
    parser.add_argument('--split_dir', type=str, default=None,
                        help='Split directory (default: data_root/splits)')

    args = parser.parse_args()

    # Set default split_dir if not provided
    if args.split_dir is None:
        args.split_dir = f'{args.data_root}/splits'

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create Args object for datamodule_gen
    dataset_args = Args(
        dataset=args.dataset,
        data_root=args.data_root,
        feat_dir=args.feat_dir,
        n_tasks=args.n_tasks,
        n_folds=args.n_folds,
        n_classes=args.n_classes,
        batch_size=args.batch_size,
        weighted_sample=False,
        # Additional required attributes
        cl_method='er',  # Default to experience replay (not joint)
        seed=args.seed,
        csv_file=args.csv_file,
        label_dict={'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5},  # Default label mapping
        split_dir=args.split_dir
    )

    print(f"\n{'='*60}")
    print(f"  Bag Instance Analysis")
    print(f"{'='*60}")
    print(f"  Dataset:   {args.dataset}")
    print(f"  Task:      {args.task}")
    print(f"  Fold:      {args.fold}")
    print(f"  Split:     {args.split}")
    print(f"{'='*60}\n")

    # Load dataset
    print("Loading dataset...")
    try:
        datamodule = datamodule_gen(dataset_args, fold=args.fold, task=args.task)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying without task parameter...")
        datamodule = datamodule_gen(dataset_args, fold=args.fold)

    # Collect instance counts
    all_instance_counts = []

    if args.split in ['train', 'all']:
        train_counts = collect_instance_counts(datamodule['train_loader'], 'train')
        all_instance_counts.extend(train_counts)

        if args.split == 'train':
            stats = compute_statistics(train_counts)
            print_statistics(stats, 'Train')

            # Save outputs
            plot_histogram(train_counts, stats,
                          f'{args.output_dir}/bag_instances_train_histogram.png', 'Train')
            save_statistics(stats, f'{args.output_dir}/bag_instances_train_stats.csv')
            save_instance_list(train_counts, f'{args.output_dir}/bag_instances_train_list.csv')

    if args.split in ['val', 'all']:
        val_counts = collect_instance_counts(datamodule['val_loader'], 'val')
        all_instance_counts.extend(val_counts)

        if args.split == 'val':
            stats = compute_statistics(val_counts)
            print_statistics(stats, 'Validation')

            # Save outputs
            plot_histogram(val_counts, stats,
                          f'{args.output_dir}/bag_instances_val_histogram.png', 'Validation')
            save_statistics(stats, f'{args.output_dir}/bag_instances_val_stats.csv')
            save_instance_list(val_counts, f'{args.output_dir}/bag_instances_val_list.csv')

    if args.split in ['test', 'all']:
        test_counts = collect_instance_counts(datamodule['test_loader'], 'test')
        all_instance_counts.extend(test_counts)

        if args.split == 'test':
            stats = compute_statistics(test_counts)
            print_statistics(stats, 'Test')

            # Save outputs
            plot_histogram(test_counts, stats,
                          f'{args.output_dir}/bag_instances_test_histogram.png', 'Test')
            save_statistics(stats, f'{args.output_dir}/bag_instances_test_stats.csv')
            save_instance_list(test_counts, f'{args.output_dir}/bag_instances_test_list.csv')

    # If analyzing all splits, compute combined statistics
    if args.split == 'all':
        stats = compute_statistics(all_instance_counts)
        print_statistics(stats, 'All')

        # Save combined outputs
        plot_histogram(all_instance_counts, stats,
                      f'{args.output_dir}/bag_instances_all_histogram.png', 'All')
        save_statistics(stats, f'{args.output_dir}/bag_instances_all_stats.csv')
        save_instance_list(all_instance_counts, f'{args.output_dir}/bag_instances_all_list.csv')

    print(f"\n✅ Analysis complete! Results saved to: {args.output_dir}\n")


if __name__ == '__main__':
    main()
