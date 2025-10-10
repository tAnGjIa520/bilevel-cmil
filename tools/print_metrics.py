#!/usr/bin/env python
"""
持续学习指标可视化工具
用于打印格式化的指标表格
"""


def print_cl_metrics_table(metrics_dict, method_name="Ours", model_name=None):
    """
    打印持续学习指标表格

    Args:
        metrics_dict: compute_all_folds_metrics 返回的字典，格式为:
                     {
                         'per_fold': [...],
                         'mean': {'AACC': float, 'BWT': float, 'IM': float},
                         'std': {'AACC': float, 'BWT': float, 'IM': float}
                     }
        method_name: 方法名称，默认为 "Ours"
        model_name: 模型名称（如 "CLAM", "TransMIL"），如果为 None 则不显示

    Example:
        >>> from cl_metrics import compute_all_folds_metrics
        >>> from tools.print_metrics import print_cl_metrics_table
        >>>
        >>> results = [
        ...     {'fold': 0, 'task': 0, '0_acc': 1.0, '1_acc': 0.0, '2_acc': 0.0},
        ...     {'fold': 0, 'task': 1, '0_acc': 0.9, '1_acc': 0.95, '2_acc': 0.0},
        ...     {'fold': 0, 'task': 2, '0_acc': 0.85, '1_acc': 0.88, '2_acc': 0.92}
        ... ]
        >>>
        >>> cl_metrics = compute_all_folds_metrics(results, verbose=False)
        >>> print_cl_metrics_table(cl_metrics, method_name="PREV", model_name="CLAM-SB")
    """
    mean = metrics_dict['mean']
    std = metrics_dict['std']

    # 构建表头和数据
    if model_name:
        col1_header = "Methods"
        col2_header = f"AACC↑ ({model_name})"
        col3_header = f"BWT↑ ({model_name})"
        col4_header = f"IM↓ ({model_name})"
    else:
        col1_header = "Methods"
        col2_header = "AACC↑"
        col3_header = "BWT↑"
        col4_header = "IM↓"

    # 格式化数值字符串
    aacc_str = f"{mean['AACC']:.4f} ± {std['AACC']:.4f}"
    bwt_str = f"{mean['BWT']:.4f} ± {std['BWT']:.4f}"
    im_str = f"{mean['IM']:.4f} ± {std['IM']:.4f}"

    # 计算列宽
    col1_width = max(len(col1_header), len(method_name)) + 2
    col2_width = max(len(col2_header), len(aacc_str)) + 2
    col3_width = max(len(col3_header), len(bwt_str)) + 2
    col4_width = max(len(col4_header), len(im_str)) + 2

    # 打印表格
    print("\n" + "=" * 100)
    print("持续学习指标汇总表")
    print("=" * 100)

    # 打印表头
    header_line = f"| {col1_header:<{col1_width}} | {col2_header:^{col2_width}} | {col3_header:^{col3_width}} | {col4_header:^{col4_width}} |"
    print(header_line)

    # 打印分隔线
    separator = f"|{'-' * (col1_width + 2)}|{'-' * (col2_width + 2)}|{'-' * (col3_width + 2)}|{'-' * (col4_width + 2)}|"
    print(separator)

    # 打印数据行
    data_line = f"| {method_name:<{col1_width}} | {aacc_str:^{col2_width}} | {bwt_str:^{col3_width}} | {im_str:^{col4_width}} |"
    print(data_line)

    print("=" * 100 + "\n")


def print_comparison_table(metrics_list, method_names=None, model_name=None):
    """
    打印多个方法的对比表格

    Args:
        metrics_list: 多个 metrics_dict 的列表
        method_names: 方法名称列表，如果为 None 则使用默认名称
        model_name: 模型名称（如 "CLAM", "TransMIL"），如果为 None 则不显示

    Example:
        >>> metrics_list = [metrics1, metrics2, metrics3]
        >>> method_names = ["ER", "LwF", "PREV"]
        >>> print_comparison_table(metrics_list, method_names, model_name="CLAM-SB")
    """
    if method_names is None:
        method_names = [f"Method {i+1}" for i in range(len(metrics_list))]

    if len(metrics_list) != len(method_names):
        raise ValueError("metrics_list 和 method_names 的长度必须相同")

    # 构建表头
    if model_name:
        col1_header = "Methods"
        col2_header = f"AACC↑ ({model_name})"
        col3_header = f"BWT↑ ({model_name})"
        col4_header = f"IM↓ ({model_name})"
    else:
        col1_header = "Methods"
        col2_header = "AACC↑"
        col3_header = "BWT↑"
        col4_header = "IM↓"

    # 格式化所有数据行
    data_rows = []
    for metrics, name in zip(metrics_list, method_names):
        mean = metrics['mean']
        std = metrics['std']
        aacc_str = f"{mean['AACC']:.4f} ± {std['AACC']:.4f}"
        bwt_str = f"{mean['BWT']:.4f} ± {std['BWT']:.4f}"
        im_str = f"{mean['IM']:.4f} ± {std['IM']:.4f}"
        data_rows.append((name, aacc_str, bwt_str, im_str))

    # 计算列宽
    col1_width = max(len(col1_header), max(len(row[0]) for row in data_rows)) + 2
    col2_width = max(len(col2_header), max(len(row[1]) for row in data_rows)) + 2
    col3_width = max(len(col3_header), max(len(row[2]) for row in data_rows)) + 2
    col4_width = max(len(col4_header), max(len(row[3]) for row in data_rows)) + 2

    # 打印表格
    print("\n" + "=" * 100)
    print("持续学习方法对比表")
    print("=" * 100)

    # 打印表头
    header_line = f"| {col1_header:<{col1_width}} | {col2_header:^{col2_width}} | {col3_header:^{col3_width}} | {col4_header:^{col4_width}} |"
    print(header_line)

    # 打印分隔线
    separator = f"|{'-' * (col1_width + 2)}|{'-' * (col2_width + 2)}|{'-' * (col3_width + 2)}|{'-' * (col4_width + 2)}|"
    print(separator)

    # 打印所有数据行
    for name, aacc_str, bwt_str, im_str in data_rows:
        data_line = f"| {name:<{col1_width}} | {aacc_str:^{col2_width}} | {bwt_str:^{col3_width}} | {im_str:^{col4_width}} |"
        print(data_line)

    print("=" * 100 + "\n")


if __name__ == "__main__":
    # 示例用法
    print("持续学习指标表格打印工具 - 示例")
    print("=" * 80)

    # 模拟单个方法的指标
    sample_metrics = {
        'per_fold': [
            {'fold': 0, 'num_tasks': 3, 'AACC': 0.8833, 'BWT': -0.0833, 'IM': 0.0},
            {'fold': 1, 'num_tasks': 3, 'AACC': 0.8667, 'BWT': -0.0667, 'IM': 0.0}
        ],
        'mean': {'AACC': 0.875, 'BWT': -0.075, 'IM': 0.0},
        'std': {'AACC': 0.0083, 'BWT': 0.0083, 'IM': 0.0}
    }

    print("\n1. 单个方法的指标表格:")
    print_cl_metrics_table(sample_metrics, method_name="PREV", model_name="CLAM-SB")

    # 模拟多个方法的指标对比
    metrics1 = {
        'mean': {'AACC': 0.7500, 'BWT': -0.2000, 'IM': 0.0500},
        'std': {'AACC': 0.0100, 'BWT': 0.0150, 'IM': 0.0050}
    }
    metrics2 = {
        'mean': {'AACC': 0.8000, 'BWT': -0.1500, 'IM': 0.0300},
        'std': {'AACC': 0.0120, 'BWT': 0.0100, 'IM': 0.0040}
    }
    metrics3 = {
        'mean': {'AACC': 0.8750, 'BWT': -0.0750, 'IM': 0.0100},
        'std': {'AACC': 0.0083, 'BWT': 0.0083, 'IM': 0.0020}
    }

    print("\n2. 多个方法的对比表格:")
    print_comparison_table(
        [metrics1, metrics2, metrics3],
        method_names=["ER", "LwF", "PREV"],
        model_name="CLAM-SB"
    )
