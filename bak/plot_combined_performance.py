import matplotlib.pyplot as plt
import numpy as np

# 第一组数据 (Q values)
Q_values = [1, 3, 5, 10]
Q_seed1 = [0.667, 0.641, 0.635, 0.673]
Q_seed2 = [0.621, 0.6198, 0.628, 0.635]
Q_seed3 = [0.652, 0.620, 0.623, 0.664]

# 第二组数据 (Inner loop)
inner_loop_values = [50, 100, 200, 300]
inner_seed1 = [0.644, 0.665, 0.526, 0.488]
inner_seed2 = [0.619, 0.630, 0.512, 0.512]
inner_seed3 = [0.630, 0.637, 0.535, 0.484]

# 计算第一组数据的均值、最大值、最小值
Q_means = [np.mean([Q_seed1[i], Q_seed2[i], Q_seed3[i]]) for i in range(len(Q_values))]
Q_maxs = [np.max([Q_seed1[i], Q_seed2[i], Q_seed3[i]]) for i in range(len(Q_values))]
Q_mins = [np.min([Q_seed1[i], Q_seed2[i], Q_seed3[i]]) for i in range(len(Q_values))]

# 计算第二组数据的均值、最大值、最小值
inner_means = [np.mean([inner_seed1[i], inner_seed2[i], inner_seed3[i]]) for i in range(len(inner_loop_values))]
inner_maxs = [np.max([inner_seed1[i], inner_seed2[i], inner_seed3[i]]) for i in range(len(inner_loop_values))]
inner_mins = [np.min([inner_seed1[i], inner_seed2[i], inner_seed3[i]]) for i in range(len(inner_loop_values))]

# 创建横向排列的子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

# ====== 子图 (a): Neumann-series Depth Q ======
# 绘制阴影区域（最大最小值之间）
ax1.fill_between(Q_values, Q_mins, Q_maxs, alpha=0.25, color='#B8C5D6', label='Min-Max Range')

# 绘制均值线
ax1.plot(Q_values, Q_means, marker='o', linewidth=2.5, markersize=8,
         color='#6B8BA8', label='Mean', zorder=3)

# 图表设置
ax1.set_xlabel('Neumann-series Depth Q', fontsize=9, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=9, fontweight='bold')
ax1.set_xticks(Q_values)
ax1.tick_params(labelsize=8)
ax1.grid(True, alpha=0.2, linestyle='--', color='#D0D0D0')
ax1.legend(loc='best', fontsize=8)
ax1.set_ylim(0.5, 0.8)
ax1.text(-0.15, -0.15, '(a)', transform=ax1.transAxes, fontsize=10, fontweight='bold')

# ====== 子图 (b): Inner Loop ======
# 绘制阴影区域（最大最小值之间）
ax2.fill_between(inner_loop_values, inner_mins, inner_maxs, alpha=0.25, color='#B8C5D6', label='Min-Max Range')

# 绘制均值线
ax2.plot(inner_loop_values, inner_means, marker='o', linewidth=2.5, markersize=8,
         color='#6B8BA8', label='Mean', zorder=3)

# 图表设置
ax2.set_xlabel('Inner Loop', fontsize=9, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=9, fontweight='bold')
ax2.set_xticks(inner_loop_values)
ax2.tick_params(labelsize=8)
ax2.grid(True, alpha=0.2, linestyle='--', color='#D0D0D0')
ax2.legend(loc='best', fontsize=8)
ax2.set_ylim(0.5, 0.8)
ax2.text(-0.15, -0.15, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('performance_combined.png', dpi=300, bbox_inches='tight')
print("组合图表已保存到 performance_combined.png")

# 打印统计数据
print("\n第一组数据统计 (Q):")
print("Q值\t均值\t\t最大值\t\t最小值\t\t范围")
for i, q in enumerate(Q_values):
    print(f"{q}\t{Q_means[i]:.4f}\t\t{Q_maxs[i]:.4f}\t\t{Q_mins[i]:.4f}\t\t{Q_maxs[i]-Q_mins[i]:.4f}")

print("\n第二组数据统计 (Inner Loop):")
print("Inner Loop\t均值\t\t最大值\t\t最小值\t\t范围")
for i, loop in enumerate(inner_loop_values):
    print(f"{loop}\t\t{inner_means[i]:.4f}\t\t{inner_maxs[i]:.4f}\t\t{inner_mins[i]:.4f}\t\t{inner_maxs[i]-inner_mins[i]:.4f}")
