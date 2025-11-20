#!/usr/bin/env python3
"""
重新绘制梯度范数演化图表
使用新的配色方案
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ====== 配置 ======
DATA_PATH = "logs/csc_clam_cl/gradient_analysis.bak/gradient_norms_summary.csv"
OUTPUT_DIR = "logs/csc_clam_cl/gradient_analysis.bak"
OUTPUT_FILE_1 = "gradient_norms_evolution.png"      # 梯度范数演化图
OUTPUT_FILE_2 = "gradient_norms_ratio.png"          # 梯度比率图

# ====== 新配色方案 ======
# 使用更现代的配色
COLOR_CURRENT = '#FF6B6B'      # 珊瑚红 - 当前批次梯度
COLOR_ACCUMULATED = '#4ECDC4'  # 青绿色 - 累积梯度
COLOR_RATIO = '#95E1D3'        # 薄荷绿 - 比率
COLOR_TASK_BOUNDARY = '#FFD93D'  # 金黄色 - 任务边界
COLOR_BARS = '#A8E6CF'         # 浅绿色 - 柱状图

print("=" * 70)
print("  梯度范数演化图表 - 重新绘制")
print("=" * 70)

# ====== 1. 读取数据 ======
print(f"\n[1/4] 读取数据文件...")
print(f"  路径: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print(f"  ❌ 错误：文件不存在！")
    exit(1)

df = pd.read_csv(DATA_PATH)
print(f"  ✓ 成功读取 {len(df)} 条记录")
print(f"  ✓ 列: {list(df.columns)}")

# ====== 2. 数据预处理 ======
print(f"\n[2/4] 数据预处理...")

# 提取数据
all_update_steps = df['global_update_step'].values
all_current_norms = df['attn_grad_norm_current'].values
all_accumulated_norms = df['attn_grad_norm_accumulated'].values
all_buffer_counts = df['buffer_samples_used'].values
all_task_ids = df['task_id'].values

# 计算任务边界
unique_tasks = sorted(df['task_id'].unique())
task_boundaries = []
for task_id in unique_tasks[:-1]:  # 不包括最后一个任务
    task_data = df[df['task_id'] == task_id]
    if len(task_data) > 0:
        boundary = task_data['global_update_step'].max()
        task_boundaries.append(boundary)

print(f"  ✓ 任务数: {len(unique_tasks)}")
print(f"  ✓ 任务边界: {task_boundaries}")
print(f"  ✓ 更新步数范围: [{all_update_steps.min()}, {all_update_steps.max()}]")

# ====== 3. 创建图表 ======
print(f"\n[3/4] 绘制图表...")

# ====== 图表 1: 当前批次梯度范数 ======
print("  - 图表 1: 当前批次梯度范数")
fig1, ax1 = plt.subplots(figsize=(14, 5), dpi=300)
fig1.suptitle('Current Batch Gradient Norm Evolution',
              fontsize=16, fontweight='bold', y=0.98)

ax1.plot(all_update_steps, all_current_norms,
         marker='o', label='Current Batch Gradient Norm',
         color=COLOR_CURRENT, markersize=3, linewidth=2.5, alpha=0.8)

# 添加任务边界线
for boundary in task_boundaries:
    ax1.axvline(x=boundary, color=COLOR_TASK_BOUNDARY, linestyle='--',
                alpha=0.6, linewidth=2.5, label='Task Boundary' if boundary == task_boundaries[0] else '')

ax1.set_xlabel('Global Update Step', fontsize=12, fontweight='bold')
ax1.set_ylabel('Gradient Norm (L2)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 200)
ax1.legend(loc='best', fontsize=11, framealpha=0.95)
ax1.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
ax1.set_facecolor('#F8F9FA')
fig1.tight_layout()

# ====== 图表 2: 累积梯度范数 ======
print("  - 图表 2: 累积梯度范数（Batch + Buffer）")
fig2, ax2 = plt.subplots(figsize=(14, 5), dpi=300)
fig2.suptitle('Accumulated Gradient Norm (Current Batch + Buffer)',
              fontsize=16, fontweight='bold', y=0.98)

ax2.plot(all_update_steps, all_accumulated_norms,
         marker='s', label='Accumulated Gradient Norm (with Buffer)',
         color=COLOR_ACCUMULATED, markersize=3, linewidth=2.5, alpha=0.8)

# 添加任务边界线
for boundary in task_boundaries:
    ax2.axvline(x=boundary, color=COLOR_TASK_BOUNDARY, linestyle='--',
                alpha=0.6, linewidth=2.5, label='Task Boundary' if boundary == task_boundaries[0] else '')

ax2.set_xlabel('Global Update Step', fontsize=12, fontweight='bold')
ax2.set_ylabel('Gradient Norm (L2)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 200)
ax2.legend(loc='best', fontsize=11, framealpha=0.95)
ax2.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
ax2.set_facecolor('#F8F9FA')
fig2.tight_layout()

# ====== 4. 保存图表 ======
print(f"\n[4/4] 保存图表...")
output_path_1 = os.path.join(OUTPUT_DIR, OUTPUT_FILE_1)
output_path_2 = os.path.join(OUTPUT_DIR, OUTPUT_FILE_2)

fig1.savefig(output_path_1, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  ✓ 已保存 (图表1): {output_path_1}")

fig2.savefig(output_path_2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  ✓ 已保存 (图表2): {output_path_2}")

# ====== 5. 打印统计信息 ======
print(f"\n" + "=" * 70)
print("  统计信息")
print("=" * 70)
print(f"总更新步数: {len(all_update_steps)}")
print(f"任务数量: {len(unique_tasks)}")
print(f"当前批次梯度范数范围: [{all_current_norms.min():.6f}, {all_current_norms.max():.6f}]")
print(f"累积梯度范数范围: [{all_accumulated_norms.min():.6f}, {all_accumulated_norms.max():.6f}]")
print(f"缓冲区使用总次数: {(all_buffer_counts > 0).sum()}")
print("=" * 70)

print(f"\n✅ 完成！图表已保存到:")
print(f"   图表1: {output_path_1}")
print(f"   图表2: {output_path_2}\n")

plt.close(fig1)
plt.close(fig2)
