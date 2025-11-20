#!/bin/bash
# KIBO超参数搜索调试脚本
# 使用8个GPU并行运行，每个GPU运行1个任务

# 设置参数
GPUS="0,1,2,3,4,5,6,7"  # 8个GPU
MAX_JOBS_PER_GPU=1       # 每个GPU同时运行1个任务
PRESET="configs/csc_clam_cl.yaml"  # 使用debug配置
LOG_DIR="kibo_hyperparam_logs"           # KIBO超参数日志目录
EXP_NAME_PREFIX="kibo_search"            # KIBO实验名称前缀

echo "============================================"
echo "KIBO 超参数搜索 - 调试脚本"
echo "============================================"
echo "可用GPU: ${GPUS}"
echo "配置文件: ${PRESET}"
echo "日志目录: ${LOG_DIR}"
echo "============================================"
echo ""

# ========================================
# KIBO核心超参数搜索
# ========================================

# 1. 搜索外层迭代次数 (bcsr_max_outer_it)
# 控制实例选择的优化迭代次数，影响选择质量
echo "搜索 mix_coarse_ratio"
python run_hyperparameter_search.py \
    --gpus ${GPUS} \
    --max-jobs-per-gpu ${MAX_JOBS_PER_GPU} \
    --preset ${PRESET} \
    --log-dir ${LOG_DIR}/mix_coarse_ratio \
    --exp-name-prefix ${EXP_NAME_PREFIX}_mix_coarse_ratio \
    --param-name mix_coarse_ratio \
    --param-values "2.0;3.0;1.1;1.5;2.5" \
    --notes "notes"

echo ""
echo "============================================"
echo "搜索完成！"
echo "结果保存在: ${LOG_DIR}"
echo "============================================"
