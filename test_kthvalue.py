"""测试 kthvalue 的行为"""
import torch

w = torch.tensor([5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0])
print(f"输入: {w.tolist()}")
print(f"排序后: {w.sort().values.tolist()}")

k = 3  # 我们想要 top-3

# 当前实现的计算方式
size = w.size(-1)
kth_current = size - k + 1  # 7 - 3 + 1 = 5
threshold_current = torch.kthvalue(w, kth_current).values

print(f"\n当前实现:")
print(f"  size = {size}, k = {k}")
print(f"  kth = size - k + 1 = {kth_current}")
print(f"  第{kth_current}小的值 (阈值): {threshold_current.item()}")
print(f"  w >= {threshold_current.item()}: {(w >= threshold_current).sum().item()} 个")
print(f"  这些值: {w[w >= threshold_current].sort(descending=True).values.tolist()}")

# 正确的应该是第(size-k+1)小 = 第k大
print(f"\n分析:")
print(f"  第5小 = 第3大? {w.sort().values[kth_current-1].item()} vs {w.sort(descending=True).values[k-1].item()}")
print(f"  第5小的值是: {w.sort().values[4].item()}")
print(f"  Top-3是: {w.topk(3).values.tolist()}")

# 另一种可能的正确实现
print(f"\n如果我们想选中top-k，阈值应该是第k大和第k+1大之间")
print(f"  第{k}大: {w.topk(k).values[-1].item()}")
print(f"  第{k+1}大: {w.topk(k+1).values[-1].item()}")

# 使用 > threshold 还是 >= threshold?
print(f"\n比较 > vs >=:")
threshold_test = w.topk(k).values[-1]  # 第k大的值
print(f"  阈值 = 第k大的值 = {threshold_test.item()}")
print(f"  w > threshold: {(w > threshold_test).sum().item()} 个 - {w[w > threshold_test].tolist()}")
print(f"  w >= threshold: {(w >= threshold_test).sum().item()} 个 - {w[w >= threshold_test].tolist()}")
