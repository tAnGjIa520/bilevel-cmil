#!/bin/bash
# 运行所有实验组合并汇总结果

# 默认参数
GPUS="0,1,2,3,4,5,6,7"
NEW_SIZE=10
BAG_SIZE_MIN=20
BAG_SIZE_MAX=50
NUM_BAGS=1000
EPOCHS1=10
EPOCHS2=10
LOG_DIR="logs/all_experiments_$(date +%Y%m%d_%H%M%S)"

echo "=================================================="
echo "运行所有实验组合"
echo "=================================================="
echo "GPU: $GPUS"
echo "新包大小: $NEW_SIZE"
echo "包大小范围: [$BAG_SIZE_MIN, $BAG_SIZE_MAX]"
echo "训练集包数: $NUM_BAGS"
echo "训练轮数: Stage1=$EPOCHS1, Stage2=$EPOCHS2"
echo "日志目录: $LOG_DIR"
echo "=================================================="
echo ""

python demo/run_all_experiments.py \
    --gpus "$GPUS" \
    --log_dir "$LOG_DIR" \
    --new_size $NEW_SIZE \
    --bag_size_min $BAG_SIZE_MIN \
    --bag_size_max $BAG_SIZE_MAX \
    --num_bags $NUM_BAGS \
    --epochs1 $EPOCHS1 \
    --epochs2 $EPOCHS2 \
    --kibo_outer_it 10 \
    --kibo_inner_it 2 \
    --kibo_lr_weight 0.1 \
    --output_csv "mismatch_summary.csv"

echo ""
echo "=================================================="
echo "所有实验完成！"
echo "结果保存在: $LOG_DIR"
echo "=================================================="
