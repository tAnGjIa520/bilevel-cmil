#!/bin/bash

# 绘图脚本启动脚本
# 用法: bash run_plots.sh

set -e  # 任何命令失败都会退出

echo "========================================="
echo "开始生成所有图表..."
echo "========================================="

cd /mnt/shared-storage-user/tangjia/bilevel-cmil

# 检查Python版本
echo "Python 版本:"
python --version
echo ""

# 1. 生成双图组合（Neumann Q 和 Patches per Pseudo-bag）
echo "1. 生成 plot_combined_performance.py..."
python plot_combined_performance.py
echo "✓ 完成: performance_chart.png"
echo ""

# 2. 生成三图组合（Neumann Q + Patches per Pseudo-bag + Inner Loop）
echo "2. 生成 plot_three_combined_performance.py..."
python plot_three_combined_performance.py
echo "✓ 完成: performance_three_combined.png"
echo ""

# 3. 生成四图组合（Neumann Q + Patches per Pseudo-bag + Inner Loop + Lambda）
echo "3. 生成 plot_four_combined.py..."
python plot_four_combined.py
echo "✓ 完成: performance_four_combined.png"
echo ""

# 4. 生成Lambda表格
echo "4. 生成 plot_lambda_table.py..."
python plot_lambda_table.py
echo "✓ 完成: lambda_table.png"
echo ""

echo "========================================="
echo "✓ 所有图表已生成完成！"
echo "========================================="
echo ""
echo "生成的文件列表:"
echo "  1. performance_chart.png - 双图组合"
echo "  2. performance_three_combined.png - 三图组合"
echo "  3. performance_four_combined.png - 四图组合（推荐用于论文）"
echo "  4. lambda_table.png - Lambda 表格"
echo ""
echo "图表位置: /mnt/shared-storage-user/tangjia/bilevel-cmil/"
echo ""
