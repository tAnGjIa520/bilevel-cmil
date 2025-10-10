"""深入调试 STETopK"""
import torch
from tools.topk_selection import STETopK

# 重现测试3的场景
torch.manual_seed(42)
w = torch.randn(2000)
k = 100

print("=" * 80)
print("调试 STETopK")
print("=" * 80)

size = w.size(-1)
kth = size - k + 1  # 2000 - 100 + 1 = 1901

print(f"输入维度: {size}")
print(f"k = {k}")
print(f"kth = size - k + 1 = {kth}")

# 获取阈值
threshold = torch.kthvalue(w, kth, dim=-1, keepdim=True).values
print(f"阈值 (第{kth}小的值): {threshold.item():.6f}")

# 检查有多少个值 >= threshold
count_ge = (w >= threshold).sum().item()
count_gt = (w > threshold).sum().item()
count_eq = (w == threshold).sum().item()

print(f"\nw >= threshold: {count_ge} 个")
print(f"w > threshold: {count_gt} 个")
print(f"w == threshold: {count_eq} 个")

# 验证：top-k 应该有k个
topk_values = w.topk(k).values
print(f"\n使用 w.topk({k}):")
print(f"  最大值: {topk_values[0].item():.6f}")
print(f"  最小值 (第{k}大): {topk_values[-1].item():.6f}")
print(f"  阈值: {threshold.item():.6f}")
print(f"  第{k}大 == 阈值? {torch.isclose(topk_values[-1], threshold.squeeze())}")

# 现在测试STETopK
topk_model = STETopK(k=k, temperature=0.1)
output = topk_model(w)

print(f"\nSTETopK 输出:")
print(f"  选中的数量: {output.sum().item()}")
print(f"  期望: {k}")
print(f"  唯一值: {output.unique().tolist()}")

# 看看哪里出了问题
print(f"\n检查 STETopK 内部计算:")
print(f"  threshold shape: {threshold.shape}")
print(f"  w shape: {w.shape}")
print(f"  w >= threshold 的shape: {(w >= threshold).shape}")
print(f"  (w >= threshold).sum(): {(w >= threshold).sum().item()}")
