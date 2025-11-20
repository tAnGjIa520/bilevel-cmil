import matplotlib.pyplot as plt
import numpy as np

# 第一组数据 (Neumann-series Depth Q)
Q_values = [1, 3, 5, 10]
Q_seed1 = [0.667, 0.641, 0.635, 0.673]
Q_seed2 = [0.621, 0.6198, 0.628, 0.635]
Q_seed3 = [0.652, 0.620, 0.623, 0.664]

# 第二组数据 (Buffer Slides)
buffer_slides_values = [50, 100, 200, 300]
buffer_seed1 = [0.644, 0.665, 0.526, 0.488]
buffer_seed2 = [0.619, 0.630, 0.512, 0.512]
buffer_seed3 = [0.630, 0.637, 0.535, 0.484]

# 第三组数据 (Inner Loop)
inner_loop_values = [1, 3, 5, 10]
inner_seed1 = [0.608, 0.6084, 0.6131, 0.6084]
inner_seed2 = [0.6439, 0.6439, 0.6439, 0.6132]
inner_seed3 = [0.6553, 0.5802, 0.6084, 0.6342]

# 第四组数据 (Lambda)
lambda_values = [1, 2, 3, 4]
lambda_seed1 = [0.676, 0.571, 0.615, 0.563]
lambda_seed2 = [0.641, 0.601, 0.592, 0.602]
lambda_seed3 = [0.630, 0.608, 0.599, 0.600]

# 计算第一组数据的均值、最大值、最小值
Q_means = [np.mean([Q_seed1[i], Q_seed2[i], Q_seed3[i]]) for i in range(len(Q_values))]
Q_maxs = [np.max([Q_seed1[i], Q_seed2[i], Q_seed3[i]]) for i in range(len(Q_values))]
Q_mins = [np.min([Q_seed1[i], Q_seed2[i], Q_seed3[i]]) for i in range(len(Q_values))]

# 计算第二组数据的均值、最大值、最小值
buffer_means = [np.mean([buffer_seed1[i], buffer_seed2[i], buffer_seed3[i]]) for i in range(len(buffer_slides_values))]
buffer_maxs = [np.max([buffer_seed1[i], buffer_seed2[i], buffer_seed3[i]]) for i in range(len(buffer_slides_values))]
buffer_mins = [np.min([buffer_seed1[i], buffer_seed2[i], buffer_seed3[i]]) for i in range(len(buffer_slides_values))]

# 计算第三组数据的均值、最大值、最小值
inner_means = [np.mean([inner_seed1[i], inner_seed2[i], inner_seed3[i]]) for i in range(len(inner_loop_values))]
inner_maxs = [np.max([inner_seed1[i], inner_seed2[i], inner_seed3[i]]) for i in range(len(inner_loop_values))]
inner_mins = [np.min([inner_seed1[i], inner_seed2[i], inner_seed3[i]]) for i in range(len(inner_loop_values))]

# 计算第四组数据的均值、最大值、最小值
lambda_means = [np.mean([lambda_seed1[i], lambda_seed2[i], lambda_seed3[i]]) for i in range(len(lambda_values))]
lambda_maxs = [np.max([lambda_seed1[i], lambda_seed2[i], lambda_seed3[i]]) for i in range(len(lambda_values))]
lambda_mins = [np.min([lambda_seed1[i], lambda_seed2[i], lambda_seed3[i]]) for i in range(len(lambda_values))]

# 创建2*2的子图布局（适配单栏宽度）
fig = plt.figure(figsize=(3.5, 3.5))

# 前三个子图使用GridSpec
gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)

# ====== 子图 (a): Neumann-series Depth Q ======
ax1 = fig.add_subplot(gs[0, 0])
ax1.fill_between(Q_values, Q_mins, Q_maxs, alpha=0.25, color='#B8C5D6', label='Min-Max Range')
ax1.plot(Q_values, Q_means, marker='o', linewidth=2.5, markersize=8,
         color='#6B8BA8', label='Mean', zorder=3)
ax1.set_xlabel('Neumann Q', fontsize=6.5)
ax1.set_ylabel('Accuracy', fontsize=6.5)
ax1.set_xticks(Q_values)
ax1.tick_params(labelsize=5.5)
ax1.grid(True, alpha=0.2, linestyle='--', color='#D0D0D0')
ax1.legend(loc='best', fontsize=5.5)
ax1.set_ylim(0.4, 0.8)
ax1.text(-0.2, -0.15, '(a)', transform=ax1.transAxes, fontsize=7)

# ====== 子图 (b): Buffer Slides ======
ax2 = fig.add_subplot(gs[0, 1])
ax2.fill_between(buffer_slides_values, buffer_mins, buffer_maxs, alpha=0.25, color='#B8C5D6', label='Min-Max Range')
ax2.plot(buffer_slides_values, buffer_means, marker='o', linewidth=2.5, markersize=8,
         color='#6B8BA8', label='Mean', zorder=3)
ax2.set_xlabel('Patches per Pseudo-bag', fontsize=6.5)
ax2.set_ylabel('Accuracy', fontsize=6.5)
ax2.set_xticks(buffer_slides_values)
ax2.tick_params(labelsize=5.5)
ax2.grid(True, alpha=0.2, linestyle='--', color='#D0D0D0')
ax2.legend(loc='best', fontsize=5.5)
ax2.set_ylim(0.4, 0.8)
ax2.text(-0.2, -0.15, '(b)', transform=ax2.transAxes, fontsize=7)

# ====== 子图 (c): Inner Loop ======
ax3 = fig.add_subplot(gs[1, 0])
ax3.fill_between(inner_loop_values, inner_mins, inner_maxs, alpha=0.25, color='#B8C5D6', label='Min-Max Range')
ax3.plot(inner_loop_values, inner_means, marker='o', linewidth=2.5, markersize=8,
         color='#6B8BA8', label='Mean', zorder=3)
ax3.set_xlabel('Inner Loop', fontsize=6.5)
ax3.set_ylabel('Accuracy', fontsize=6.5)
ax3.set_xticks(inner_loop_values)
ax3.tick_params(labelsize=5.5)
ax3.grid(True, alpha=0.2, linestyle='--', color='#D0D0D0')
ax3.legend(loc='best', fontsize=5.5)
ax3.set_ylim(0.4, 0.8)
ax3.text(-0.2, -0.15, '(c)', transform=ax3.transAxes, fontsize=7)

# ====== 子图 (d): Lambda ======
ax4 = fig.add_subplot(gs[1, 1])
ax4.fill_between(lambda_values, lambda_mins, lambda_maxs, alpha=0.25, color='#B8C5D6', label='Min-Max Range')
ax4.plot(lambda_values, lambda_means, marker='o', linewidth=2.5, markersize=8,
         color='#6B8BA8', label='Mean', zorder=3)
ax4.set_xlabel('$\lambda$', fontsize=6.5)
ax4.set_ylabel('Accuracy', fontsize=6.5)
ax4.set_xticks(lambda_values)
ax4.tick_params(labelsize=5.5)
ax4.grid(True, alpha=0.2, linestyle='--', color='#D0D0D0')
ax4.legend(loc='best', fontsize=5.5)
ax4.set_ylim(0.4, 0.8)
ax4.text(-0.2, -0.15, '(d)', transform=ax4.transAxes, fontsize=7)

plt.savefig('performance_four_combined.png', dpi=300, bbox_inches='tight')
print("四合一组合图表已保存到 performance_four_combined.png")

# 打印统计数据
print("\n第一组数据统计 (Neumann-series Depth Q):")
print("Q值\t均值\t\t最大值\t\t最小值")
for i, q in enumerate(Q_values):
    print(f"{q}\t{Q_means[i]:.4f}\t\t{Q_maxs[i]:.4f}\t\t{Q_mins[i]:.4f}")

print("\n第二组数据统计 (Buffer Slides):")
print("Buffer\t均值\t\t最大值\t\t最小值")
for i, bs in enumerate(buffer_slides_values):
    print(f"{bs}\t{buffer_means[i]:.4f}\t\t{buffer_maxs[i]:.4f}\t\t{buffer_mins[i]:.4f}")

print("\n第三组数据统计 (Inner Loop):")
print("Inner Loop\t均值\t\t最大值\t\t最小值")
for i, loop in enumerate(inner_loop_values):
    print(f"{loop}\t\t{inner_means[i]:.4f}\t\t{inner_maxs[i]:.4f}\t\t{inner_mins[i]:.4f}")

print("\n第四组数据统计 (Lambda):")
print("λ\t平均值\t\t最大值\t\t最小值")
for i, lam in enumerate(lambda_values):
    print(f"{lam}\t{lambda_means[i]:.4f}\t\t{lambda_maxs[i]:.4f}\t\t{lambda_mins[i]:.4f}")
