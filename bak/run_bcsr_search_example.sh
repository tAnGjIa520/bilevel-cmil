#!/bin/bash
# BCSR 超参数收敛性搜索 - 使用示例

# ========== 示例1: 快速测试（小规模搜索） ==========
# echo "示例1: 快速测试 - 少量超参数组合"
# python run_bcsr_convergence_search.py \
#     --gpus 0,1,2,3,4,5,6,7 \
#     --max-jobs-per-gpu 1 \
#     --lr-proxy 5,10 \
#     --beta 0.1 \
#     --max-inner-it 1 \
#     --weight-lr 0.05 \
#     --distall-lamda 0.1 \
#     --topk-method sigmoid,gumbel \
#     --topk-temperature 0.1 \
#     --normalize-method none,l2 \
#     --use-simplex-projection False \
#     --distill-target features
#     --dry-run

# ========== 示例2: 完整搜索（所有超参数） ==========
# 注意：这会生成 15552 个组合（约需130小时@4GPU），确保有足够的GPU和时间
# 建议：先运行示例1或--dry-run查看命令，再决定是否运行完整搜索
python run_bcsr_convergence_search.py \
    --gpus 0,1,2,3,4,5,6,7 \
    --max-jobs-per-gpu 10 \
    --lr-proxy 1,5,10 \
    --beta 0.01,0.1,1.0 \
    --max-inner-it 1,3 \
    --weight-lr 0.01,0.05,0.1 \
    --distall-lamda 0.1,0.5 \
    --topk-method sigmoid,gumbel,ste \
    --topk-temperature 0.1,0.5,1 \
    --normalize-method none,l2,softmax,zscore \
    --use-simplex-projection False,True \
    --distill-target logits,features,both

# ========== 示例3: 聚焦学习率搜索 ==========
# 固定其他参数，只搜索学习率相关参数
# python run_bcsr_convergence_search.py \
#     --gpus 0,1,2,3 \
#     --lr-proxy 1,3,5,7,10 \
#     --beta 0.1 \
#     --max-inner-it 1 \
#     --weight-lr 0.01,0.03,0.05,0.07,0.1 \
#     --distall-lamda 0.1 \
#     --topk-method sigmoid \
#     --topk-temperature 0.1 \
#     --normalize-method none \
#     --use-simplex-projection False \
#     --distill-target logits

# ========== 示例4: TopK方法对比 ==========
# 固定其他参数，对比不同的TopK方法
# python run_bcsr_convergence_search.py \
#     --gpus 0,1,2,3 \
#     --lr-proxy 5 \
#     --beta 0.1 \
#     --max-inner-it 1 \
#     --weight-lr 0.05 \
#     --distall-lamda 0.1 \
#     --topk-method sigmoid,gumbel,ste \
#     --topk-temperature 0.05,0.1,0.2,0.5 \
#     --normalize-method none,l2,softmax,zscore \
#     --use-simplex-projection False \
#     --distill-target logits

# ========== 示例5: 内层迭代次数优化 ==========
# 搜索最佳的内层迭代次数
# python run_bcsr_convergence_search.py \
#     --gpus 0,1,2,3 \
#     --lr-proxy 5 \
#     --beta 0.1 \
#     --max-inner-it 1,2,3,5 \
#     --weight-lr 0.05 \
#     --distall-lamda 0.1 \
#     --topk-method sigmoid \
#     --topk-temperature 0.1 \
#     --normalize-method none \
#     --use-simplex-projection False \
#     --distill-target logits

# ========== 示例6: 单纯形投影和蒸馏目标对比 ==========
# 对比单纯形投影和不同蒸馏目标的效果
# python run_bcsr_convergence_search.py \
#     --gpus 0,1,2,3 \
#     --lr-proxy 5 \
#     --beta 0.1 \
#     --max-inner-it 1 \
#     --weight-lr 0.05 \
#     --distall-lamda 0.1 \
#     --topk-method sigmoid \
#     --topk-temperature 0.1 \
#     --normalize-method none \
#     --use-simplex-projection False,True \
#     --distill-target logits,features,both

echo "脚本准备完成！取消注释相应的命令并运行"
