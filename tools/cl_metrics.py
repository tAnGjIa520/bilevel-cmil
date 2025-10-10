#!/usr/bin/env python
"""
持续学习指标计算模块
用于计算 AACC, BWT, IM 等持续学习性能指标
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


def calculate_aacc(accuracy_matrix: np.ndarray) -> float:
    """
    计算平均准确率 (Average Accuracy)

    AACC = 1/T * sum_{j=1}^{T} a_{T,j}

    Args:
        accuracy_matrix: shape (T, T) 的准确率矩阵
                        accuracy_matrix[t][j] 表示在训练完任务t后，在任务j上的准确率 a_{t,j}

    Returns:
        AACC 指标值
    """
    T = accuracy_matrix.shape[0]  # 总任务数
    # 取最后一行（训练完所有任务后）的所有任务准确率
    final_accuracies = accuracy_matrix[-1, :]
    aacc = np.mean(final_accuracies)
    return float(aacc)


def calculate_bwt(accuracy_matrix: np.ndarray) -> float:
    """
    计算后向迁移 (Backward Transfer)

    BWT = 1/(T-1) * sum_{j=1}^{T-1} (a_{T,j} - a_{j,j})

    Args:
        accuracy_matrix: shape (T, T) 的准确率矩阵
                        accuracy_matrix[t][j] 表示在训练完任务t后，在任务j上的准确率 a_{t,j}

    Returns:
        BWT 指标值
    """
    T = accuracy_matrix.shape[0]  # 总任务数

    if T <= 1:
        # 只有一个任务时，BWT 无意义
        return 0.0

    bwt_sum = 0.0
    for j in range(T - 1):  # j 从 0 到 T-2 (对应任务 1 到 T-1)
        a_T_j = accuracy_matrix[-1, j]  # 最终在任务 j 上的准确率
        a_j_j = accuracy_matrix[j, j]    # 刚训练完任务 j 时在任务 j 上的准确率
        bwt_sum += (a_T_j - a_j_j)

    bwt = bwt_sum / (T - 1)
    return float(bwt)


def calculate_im(accuracy_matrix: np.ndarray, joint_accuracies: Optional[np.ndarray] = None) -> float:
    """
    计算初始遗忘 (Initial Mistake/Intransigence Measure)

    IM = 1/T * sum_{j=1}^{T} (a_j^* - a_{j,j})

    Args:
        accuracy_matrix: shape (T, T) 的准确率矩阵
                        accuracy_matrix[t][j] 表示在训练完任务t后，在任务j上的准确率 a_{t,j}
        joint_accuracies: shape (T,) 的联合训练准确率数组
                         joint_accuracies[j] 表示任务 j 的联合训练准确率 a_j^*
                         如果为 None，则使用对角线的值作为近似

    Returns:
        IM 指标值
    """
    T = accuracy_matrix.shape[0]  # 总任务数

    # 如果没有提供联合训练准确率，使用对角线值（刚训练完时的准确率）作为上界估计
    if joint_accuracies is None:
        # 这种情况下 IM 会是 0，因为没有理想的联合训练基准
        # 实际应用中，应该提供 joint_accuracies
        joint_accuracies = np.diag(accuracy_matrix)

    im_sum = 0.0
    for j in range(T):
        a_j_star = joint_accuracies[j]  # 联合训练准确率
        a_j_j = accuracy_matrix[j, j]   # 刚训练完任务 j 时的准确率
        im_sum += (a_j_star - a_j_j)

    im = im_sum / T
    return float(im)


def parse_results_to_matrix(results: List[Dict], fold: int = 0) -> Tuple[np.ndarray, int]:
    """
    将结果列表解析为准确率矩阵

    Args:
        results: 结果字典列表，每个字典包含 'fold', 'task', 和各个任务的准确率
                例如: [{'fold': 0, 'task': 0, '0_acc': 1.0, '1_acc': 0.0, '2_acc': 0.0}, ...]
        fold: 要处理的 fold 编号

    Returns:
        accuracy_matrix: shape (T, T) 的准确率矩阵
        num_tasks: 任务数量 T
    """
    # 筛选出指定 fold 的结果
    fold_results = [r for r in results if r['fold'] == fold]

    if not fold_results:
        raise ValueError(f"No results found for fold {fold}")

    # 确定任务数量
    num_tasks = len(fold_results)

    # 提取所有可能的任务准确率键
    acc_keys = set()
    for r in fold_results:
        acc_keys.update([k for k in r.keys() if k.endswith('_acc')])

    # 确定实际的任务数（从准确率键中推断）
    task_indices = sorted([int(k.split('_')[0]) for k in acc_keys])
    T = len(task_indices)

    # 初始化准确率矩阵
    accuracy_matrix = np.zeros((num_tasks, T))

    # 填充准确率矩阵
    for result in fold_results:
        t = result['task']  # 当前训练到的任务
        for j in task_indices:
            acc_key = f'{j}_acc'
            if acc_key in result:
                accuracy_matrix[t, j] = result[acc_key]

    return accuracy_matrix, T


def compute_cl_metrics(results: List[Dict],
                       fold: int = 0,
                       joint_accuracies: Optional[np.ndarray] = None,
                       verbose: bool = True) -> Dict[str, float]:
    """
    计算持续学习指标

    Args:
        results: 结果字典列表
        fold: 要处理的 fold 编号
        joint_accuracies: 可选的联合训练准确率数组
        verbose: 是否打印详细信息

    Returns:
        包含各项指标的字典
    """
    # 解析结果为准确率矩阵
    accuracy_matrix, num_tasks = parse_results_to_matrix(results, fold)

    if verbose:
        print(f"\nFold {fold} - Accuracy Matrix:")
        print(f"Shape: {accuracy_matrix.shape} (Tasks x Tasks)")
        print("\nMatrix (rows=after training task t, cols=accuracy on task j):")
        for t in range(accuracy_matrix.shape[0]):
            print(f"Task {t}: {accuracy_matrix[t]}")
        print()

    # 计算指标
    aacc = calculate_aacc(accuracy_matrix)
    bwt = calculate_bwt(accuracy_matrix)
    im = calculate_im(accuracy_matrix, joint_accuracies)

    metrics = {
        'fold': fold,
        'num_tasks': num_tasks,
        'AACC': aacc,
        'BWT': bwt,
        'IM': im
    }

    if verbose:
        print(f"Fold {fold} Metrics:")
        print(f"  - Number of tasks (T): {num_tasks}")
        print(f"  - AACC (Average Accuracy): {aacc:.4f}")
        print(f"  - BWT (Backward Transfer): {bwt:.4f}")
        print(f"  - IM (Intransigence Measure): {im:.4f}")
        print()

    return metrics


def compute_all_folds_metrics(results: List[Dict],
                               joint_accuracies_dict: Optional[Dict[int, np.ndarray]] = None,
                               verbose: bool = True) -> Dict[str, any]:
    """
    计算所有 fold 的持续学习指标

    Args:
        results: 结果字典列表，格式为:
                 [{'fold': int, 'task': int, '0_acc': float, '1_acc': float, ...}, ...]
        joint_accuracies_dict: 字典，key 为 fold 编号，value 为该 fold 的联合训练准确率数组
                              例如: {0: np.array([0.95, 0.96, 0.94]), 1: np.array([0.96, 0.95, 0.93])}
        verbose: 是否打印详细信息

    Returns:
        包含所有 fold 指标的字典，格式为:
        {
            'per_fold': [
                {'fold': 0, 'num_tasks': 3, 'AACC': 0.85, 'BWT': -0.05, 'IM': 0.02},
                {'fold': 1, 'num_tasks': 3, 'AACC': 0.87, 'BWT': -0.03, 'IM': 0.01},
                ...
            ],
            'mean': {
                'AACC': 0.86,  # 所有fold的AACC均值
                'BWT': -0.04,  # 所有fold的BWT均值
                'IM': 0.015    # 所有fold的IM均值
            },
            'std': {
                'AACC': 0.01,  # 所有fold的AACC标准差
                'BWT': 0.01,   # 所有fold的BWT标准差
                'IM': 0.005    # 所有fold的IM标准差
            }
        }
    """
    # 找出所有的 fold 编号
    folds = sorted(set(r['fold'] for r in results))

    all_metrics = []

    for fold in folds:
        joint_acc = joint_accuracies_dict.get(fold) if joint_accuracies_dict else None
        metrics = compute_cl_metrics(results, fold, joint_acc, verbose)
        all_metrics.append(metrics)

    # 计算均值和标准差
    # 返回格式:
    # {
    #     'per_fold': [每个fold的详细指标],
    #     'mean': {'AACC': float, 'BWT': float, 'IM': float},  # 所有fold的均值
    #     'std': {'AACC': float, 'BWT': float, 'IM': float}    # 所有fold的标准差
    # }
    result_dict = {
        'per_fold': all_metrics,
        'mean': {},
        'std': {}
    }

    if len(all_metrics) > 0:
        # 计算所有fold的均值
        result_dict['mean'] = {
            'AACC': float(np.mean([m['AACC'] for m in all_metrics])),
            'BWT': float(np.mean([m['BWT'] for m in all_metrics])),
            'IM': float(np.mean([m['IM'] for m in all_metrics]))
        }

        # 计算所有fold的标准差
        result_dict['std'] = {
            'AACC': float(np.std([m['AACC'] for m in all_metrics])),
            'BWT': float(np.std([m['BWT'] for m in all_metrics])),
            'IM': float(np.std([m['IM'] for m in all_metrics]))
        }

    if verbose and len(all_metrics) > 1:
        print("=" * 60)
        print("Average Metrics across all folds:")
        print(f"  - Average AACC: {result_dict['mean']['AACC']:.4f} ± {result_dict['std']['AACC']:.4f}")
        print(f"  - Average BWT:  {result_dict['mean']['BWT']:.4f} ± {result_dict['std']['BWT']:.4f}")
        print(f"  - Average IM:   {result_dict['mean']['IM']:.4f} ± {result_dict['std']['IM']:.4f}")
        print("=" * 60)

    return result_dict


# 示例用法
if __name__ == "__main__":
    print("=" * 80)
    print("持续学习指标计算 - 示例演示")
    print("=" * 80)

    # ========== 示例 1: 单个 Fold 的简单示例 ==========
    print("\n【示例 1】单个 Fold 的简单示例")
    print("-" * 80)
    results_simple = [
        {'fold': 0, 'task': 0, '0_acc': 1.0, '1_acc': 0.0, '2_acc': 0.0},
        {'fold': 0, 'task': 1, '0_acc': 0.9, '1_acc': 0.95, '2_acc': 0.0},
        {'fold': 0, 'task': 2, '0_acc': 0.85, '1_acc': 0.88, '2_acc': 0.92}
    ]

    metrics_simple = compute_cl_metrics(results_simple, fold=0, verbose=True)
    print(f"返回的指标字典: {metrics_simple}\n")

    # ========== 示例 2: 多 Fold 示例（3个fold，4个任务）==========
    print("\n【示例 2】多 Fold 示例（3个fold，4个任务）")
    print("-" * 80)

    # 生成模拟数据：3个fold，每个fold有4个任务
    np.random.seed(42)
    results_multi_fold = []

    for fold in range(3):
        for task in range(4):
            result = {'fold': fold, 'task': task}
            for j in range(4):
                if j <= task:
                    # 已训练的任务有准确率
                    if j == task:
                        # 当前任务的准确率较高
                        acc = np.random.uniform(0.85, 0.95)
                    else:
                        # 之前任务的准确率（有遗忘）
                        forget_factor = (task - j) * 0.05  # 越早的任务遗忘越多
                        acc = np.random.uniform(0.75, 0.90) - forget_factor
                        acc = max(acc, 0.5)  # 确保不会太低
                else:
                    # 未训练的任务准确率为0
                    acc = 0.0
                result[f'{j}_acc'] = acc
            results_multi_fold.append(result)

    # 计算所有 fold 的指标
    all_metrics = compute_all_folds_metrics(results_multi_fold, verbose=True)

    print("\n返回的字典结构:")
    print(f"  - per_fold: {len(all_metrics['per_fold'])} 个 fold 的详细指标")
    print(f"  - mean: {all_metrics['mean']}")
    print(f"  - std: {all_metrics['std']}")

    # ========== 示例 3: 带联合训练准确率的示例 ==========
    print("\n【示例 3】带联合训练准确率（计算 IM 指标）")
    print("-" * 80)

    # 生成联合训练准确率（通常比顺序训练略高）
    joint_acc_dict = {}
    for fold in range(3):
        # 联合训练准确率设置为 0.92-0.98 之间
        joint_acc_dict[fold] = np.random.uniform(0.92, 0.98, size=4)

    all_metrics_with_joint = compute_all_folds_metrics(
        results_multi_fold,
        joint_accuracies_dict=joint_acc_dict,
        verbose=True
    )

    print("\n带联合训练准确率的指标:")
    print(f"  - mean: {all_metrics_with_joint['mean']}")
    print(f"  - std: {all_metrics_with_joint['std']}")

    # ========== 示例 4: 实际使用的用户数据格式 ==========
    print("\n【示例 4】用户提供的数据格式")
    print("-" * 80)

    user_results = [
        {'fold': 0, 'task': 0, '0_acc': 1.0, '1_acc': 0.0, '2_acc': 0.0},
        {'fold': 0, 'task': 1, '0_acc': 0.0, '1_acc': 0.1818181872367859, '2_acc': 0.0},
        {'fold': 0, 'task': 2, '0_acc': 0.0, '1_acc': 0.0, '2_acc': 0.3636363744735718}
    ]

    user_metrics = compute_all_folds_metrics(user_results, verbose=True)
    print(f"\n用户数据的指标结果:")
    print(f"  - mean: {user_metrics['mean']}")
    print(f"  - std: {user_metrics['std']}")

    print("\n" + "=" * 80)
    print("示例演示完成！")
    print("=" * 80)
