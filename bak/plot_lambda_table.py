import matplotlib.pyplot as plt
import numpy as np

# 数据
data = [
    ['$\lambda$', '1', '3', '5', '10'],
    ['seed1', '0.651', '0.595', '0.607', '0.482'],
    ['seed2', '0.647', '0.592', '0.600', '0.485'],
    ['seed3', '0.649', '0.591', '0.599', '0.484'],
]

# 创建图表
fig, ax = plt.subplots(figsize=(5, 2.5))
ax.axis('tight')
ax.axis('off')

# 创建表格
table = ax.table(cellText=data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# 设置表头样式（第一行）
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#6B8BA8')
    cell.set_text_props(weight='bold', color='white', fontsize=10)

# 设置首列样式（第一列）
for i in range(1, 4):
    cell = table[(i, 0)]
    cell.set_facecolor('#E8EEF5')
    cell.set_text_props(weight='bold', fontsize=10)

# 设置数据单元格样式
for i in range(1, 4):
    for j in range(1, 5):
        cell = table[(i, j)]
        cell.set_facecolor('white')
        cell.set_text_props(fontsize=10)
        # 添加边框
        cell.set_edgecolor('#D0D0D0')

# 添加边框到表头
for i in range(5):
    cell = table[(0, i)]
    cell.set_edgecolor('#6B8BA8')

plt.tight_layout()
plt.savefig('lambda_table.png', dpi=300, bbox_inches='tight')
print("表格已保存到 lambda_table.png")

# 计算并打印平均值
print("\n数据统计:")
print("λ\t平均值\t最大值\t最小值")
for col_idx in range(1, 5):
    values = [float(data[row_idx][col_idx]) for row_idx in range(1, 4)]
    avg = np.mean(values)
    max_val = np.max(values)
    min_val = np.min(values)
    lambda_val = data[0][col_idx]
    print(f"{lambda_val}\t{avg:.4f}\t{max_val:.4f}\t{min_val:.4f}")
