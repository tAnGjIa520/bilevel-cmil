"""测试 STETopK 的潜在问题"""
import torch
from tools.topk_selection import STETopK

# 测试用例1: 有重复值的情况
print("=" * 80)
print("测试1: 有重复值的情况")
print("=" * 80)

w = torch.tensor([1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0])
k = 3

topk = STETopK(k=k, temperature=0.1)
output = topk(w)

print(f"输入: {w.tolist()}")
print(f"k = {k}")
print(f"输出: {output.tolist()}")
print(f"选中的数量: {output.sum().item()}")
print(f"期望选中数量: {k}")
print(f"是否正确: {output.sum().item() == k}")

# 使用kthvalue看看阈值
size = w.size(-1)
kth = size - k + 1
threshold = torch.kthvalue(w, kth, dim=-1, keepdim=True).values
print(f"\n阈值 (第{kth}小的值): {threshold.item()}")
print(f"w >= threshold: {(w >= threshold).float().tolist()}")
print(f">= 会选中: {(w >= threshold).sum().item()} 个元素")

# 测试用例2: 没有重复值的情况
print("\n" + "=" * 80)
print("测试2: 没有重复值的情况")
print("=" * 80)

w2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
k2 = 3

topk2 = STETopK(k=k2, temperature=0.1)
output2 = topk2(w2)
print(f"输入: {w2.tolist()}")
print(f"k = {k2}")
print(f"输出: {output2.tolist()}")
print(f"选中的数量: {output2.sum().item()}")
print(f"期望选中数量: {k2}")
print(f"是否正确: {output2.sum().item() == k2}")

# 测试用例3: 大规模随机数据
print("\n" + "=" * 80)
print("测试3: 大规模随机数据（可能有重复）")
print("=" * 80)

torch.manual_seed(42)
w3 = torch.randn(2000)
k3 = 100

topk3 = STETopK(k=k3, temperature=0.1)
output3 = topk3(w3)
print(f"输入维度: {w3.shape}")
print(f"k = {k3}")
print(f"选中的数量: {output3.sum().item()}")
print(f"期望选中数量: {k3}")
print(f"是否正确: {output3.sum().item() == k3}")
print(f"输出的唯一值: {output3.unique().tolist()}")
