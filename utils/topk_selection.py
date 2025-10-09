"""
可微分的 Top-K Selection 算子

提供三种实现方法:
1. SigmoidTopK - 使用 Sigmoid + 动态阈值（推荐）
2. GumbelTopK - 使用 Gumbel 噪声增强
3. STETopK - 使用 Straight-Through Estimator

作者: Claude
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidTopK(nn.Module):
    """
    使用 Sigmoid + 动态阈值实现可微分 Top-K 选择

    原理:
    1. 找到第 k 大的值作为阈值
    2. 使用 Sigmoid 将 (w - threshold) 映射到 [0, 1]
    3. Top-K 项的 w > threshold, sigmoid 输出接近 1
    4. 其他项的 w < threshold, sigmoid 输出接近 0

    优点:
    - 完全可微分
    - 输出稳定
    - 温度可调
    - Top-K 项接近 1，其他项接近 0

    参数:
        k (int): 选择的 top 数量
        temperature (float): 温度参数，越小输出越"硬"（接近 0/1）
                            默认 0.1，推荐范围 [0.01, 1.0]
        dim (int): 在哪个维度上选择 top-k，默认 -1（最后一维）

    示例:
        >>> topk = SigmoidTopK(k=5, temperature=0.1)
        >>> w = torch.randn(100, requires_grad=True)
        >>> selection = topk(w)
        >>> print(selection.shape)  # torch.Size([100])
        >>> print(selection.max(), selection.min())  # 接近 1.0 和 0.0
    """

    def __init__(self, k: int, temperature: float = 0.1, dim: int = -1):
        super(SigmoidTopK, self).__init__()
        self.k = k
        self.temperature = temperature
        self.dim = dim

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            w: 输入权重，shape: (..., n)

        Returns:
            selection: Top-K 选择结果，shape 与 w 相同
                      Top-K 项接近 1，其他项接近 0
        """
        # 获取第 k 大的值作为阈值
        # kthvalue 返回第 k 小的值，所以用 (size - k + 1) 得到第 k 大
        size = w.size(self.dim)
        kth = size - self.k + 1

        # 计算阈值
        threshold = torch.kthvalue(w, kth, dim=self.dim, keepdim=True).values

        # 使用 Sigmoid 进行软选择
        # (w - threshold) / temperature:
        #   - Top-K 项: w > threshold, 值为正，sigmoid → 1
        #   - 其他项: w < threshold, 值为负，sigmoid → 0
        #   - temperature 越小，sigmoid 越陡峭，输出越接近 0/1
        scores = torch.sigmoid((w - threshold) / self.temperature)

        return scores

    def extra_repr(self) -> str:
        return f'k={self.k}, temperature={self.temperature}, dim={self.dim}'


class GumbelTopK(nn.Module):
    """
    使用 Gumbel 噪声增强的可微分 Top-K 选择

    原理:
    1. 在训练时加入 Gumbel 噪声，增加随机探索
    2. 在推理时不加噪声，输出确定性
    3. 使用 Sigmoid + 动态阈值进行软选择

    Gumbel 噪声: -log(-log(U)), U ~ Uniform(0, 1)

    优点:
    - 训练时有随机性，避免过早收敛
    - 推理时确定性
    - 适合强化学习等场景

    参数:
        k (int): 选择的 top 数量
        temperature (float): 温度参数
        gumbel_noise (float): Gumbel 噪声的缩放因子，默认 1.0
        dim (int): 在哪个维度上选择 top-k

    示例:
        >>> topk = GumbelTopK(k=5, temperature=0.1)
        >>> w = torch.randn(100, requires_grad=True)
        >>>
        >>> # 训练模式（有随机性）
        >>> topk.train()
        >>> selection1 = topk(w)
        >>> selection2 = topk(w)  # 每次结果可能不同
        >>>
        >>> # 推理模式（确定性）
        >>> topk.eval()
        >>> selection3 = topk(w)
        >>> selection4 = topk(w)  # 每次结果相同
    """

    def __init__(self, k: int, temperature: float = 0.1,
                 gumbel_noise: float = 1.0, dim: int = -1):
        super(GumbelTopK, self).__init__()
        self.k = k
        self.temperature = temperature
        self.gumbel_noise = gumbel_noise
        self.dim = dim

    def sample_gumbel(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """采样 Gumbel 噪声"""
        U = torch.rand(shape, device=device)
        # 避免 log(0)
        return -torch.log(-torch.log(U + 1e-10) + 1e-10)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            w: 输入权重，shape: (..., n)

        Returns:
            selection: Top-K 选择结果
        """
        size = w.size(self.dim)
        kth = size - self.k + 1

        # 在训练模式下加入 Gumbel 噪声
        if self.training:
            gumbel = self.sample_gumbel(w.shape, w.device)
            w_noisy = w + self.gumbel_noise * gumbel
            threshold = torch.kthvalue(w_noisy, kth, dim=self.dim, keepdim=True).values
            scores = torch.sigmoid((w_noisy - threshold) / self.temperature)
        else:
            # 推理模式：确定性输出
            threshold = torch.kthvalue(w, kth, dim=self.dim, keepdim=True).values
            scores = torch.sigmoid((w - threshold) / self.temperature)

        return scores

    def extra_repr(self) -> str:
        return (f'k={self.k}, temperature={self.temperature}, '
                f'gumbel_noise={self.gumbel_noise}, dim={self.dim}')


class STETopK(nn.Module):
    """
    使用 Straight-Through Estimator (STE) 的可微分 Top-K 选择

    原理:
    - 前向传播: 使用硬选择，输出精确的 0/1
    - 反向传播: 使用软近似计算梯度

    这是一个"欺骗"反向传播的技巧:
    1. 前向: output = hard_selection (0 or 1)
    2. 反向: grad = soft_approximation
    3. 实现: output = hard + (soft - soft.detach())
       - 前向计算: hard + (soft - soft) = hard
       - 反向梯度: grad(hard + soft - const) = grad(soft)

    优点:
    - 前向输出精确是 0/1
    - 反向仍然可以传递梯度
    - 适合需要精确二值化的场景

    缺点:
    - 前向和反向不一致，可能影响收敛

    参数:
        k (int): 选择的 top 数量
        temperature (float): 温度参数（仅用于反向传播的软近似）
        dim (int): 在哪个维度上选择 top-k

    示例:
        >>> topk = STETopK(k=5, temperature=0.1)
        >>> w = torch.randn(100, requires_grad=True)
        >>> selection = topk(w)
        >>> print(selection.unique())  # tensor([0., 1.]) 精确的 0/1
        >>>
        >>> loss = selection.sum()
        >>> loss.backward()
        >>> print(w.grad is not None)  # True，梯度成功回传
    """

    def __init__(self, k: int, temperature: float = 0.1, dim: int = -1):
        super(STETopK, self).__init__()
        self.k = k
        self.temperature = temperature
        self.dim = dim

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        前向传播（使用 STE 技巧）

        Args:
            w: 输入权重，shape: (..., n)

        Returns:
            selection: Top-K 选择结果
                      前向: 精确的 0/1
                      反向: 使用软近似的梯度
        """
        size = w.size(self.dim)
        kth = size - self.k + 1

        # 计算阈值
        threshold = torch.kthvalue(w, kth, dim=self.dim, keepdim=True).values

        # 硬选择：精确的 0/1
        # 注意: >= 保证至少有 k 个元素被选中
        hard = (w >= threshold).float()

        # 软近似：用于反向传播
        soft = torch.sigmoid((w - threshold) / self.temperature)

        # STE 技巧：前向用 hard，反向用 soft 的梯度
        # output = hard + (soft - soft.detach())
        # 前向: hard + (soft - soft) = hard
        # 反向: grad(output) = grad(hard) + grad(soft) - 0 = grad(soft)
        output = hard + (soft - soft.detach())

        return output

    def extra_repr(self) -> str:
        return f'k={self.k}, temperature={self.temperature}, dim={self.dim}'


# ============================================================================
# 辅助函数
# ============================================================================

def compare_topk_methods(w: torch.Tensor, k: int, temperature: float = 0.1):
    """
    比较三种 Top-K 方法的输出

    Args:
        w: 输入权重
        k: top-k 参数
        temperature: 温度参数

    Returns:
        dict: 包含三种方法输出的字典
    """
    sigmoid_topk = SigmoidTopK(k=k, temperature=temperature)
    gumbel_topk = GumbelTopK(k=k, temperature=temperature)
    ste_topk = STETopK(k=k, temperature=temperature)

    # 设置为推理模式（GumbelTopK 在推理模式下是确定性的）
    sigmoid_topk.eval()
    gumbel_topk.eval()
    ste_topk.eval()

    with torch.no_grad():
        sigmoid_out = sigmoid_topk(w)
        gumbel_out = gumbel_topk(w)
        ste_out = ste_topk(w)

    return {
        'sigmoid': sigmoid_out,
        'gumbel': gumbel_out,
        'ste': ste_out
    }


if __name__ == "__main__":
    # 简单测试
    print("=" * 80)
    print("可微分 Top-K Selection 模块")
    print("=" * 80)

    # 创建测试数据
    torch.manual_seed(42)
    w = torch.randn(20, requires_grad=True)
    k = 5

    print(f"\n输入权重 w: shape={w.shape}")
    print(f"Top-K: k={k}")
    print(f"输入值范围: [{w.min():.3f}, {w.max():.3f}]")

    # 测试三种方法
    print("\n" + "=" * 80)
    print("方法 1: SigmoidTopK")
    print("=" * 80)
    topk1 = SigmoidTopK(k=k, temperature=0.1)
    out1 = topk1(w)
    print(f"输出范围: [{out1.min():.4f}, {out1.max():.4f}]")
    print(f"Top-5 值: {out1.topk(5).values.tolist()}")
    print(f"Bottom-5 值: {out1.topk(5, largest=False).values.tolist()}")

    print("\n" + "=" * 80)
    print("方法 2: GumbelTopK")
    print("=" * 80)
    topk2 = GumbelTopK(k=k, temperature=0.1)
    topk2.eval()  # 推理模式
    out2 = topk2(w)
    print(f"输出范围: [{out2.min():.4f}, {out2.max():.4f}]")
    print(f"Top-5 值: {out2.topk(5).values.tolist()}")

    print("\n" + "=" * 80)
    print("方法 3: STETopK")
    print("=" * 80)
    topk3 = STETopK(k=k, temperature=0.1)
    out3 = topk3(w)
    print(f"输出范围: [{out3.min():.4f}, {out3.max():.4f}]")
    print(f"唯一值: {out3.unique().tolist()}")  # 应该只有 0 和 1
    print(f"被选中的数量: {out3.sum().item()}")

    print("\n" + "=" * 80)
    print("梯度测试")
    print("=" * 80)
    loss = out1.sum()
    loss.backward()
    print(f"梯度是否存在: {w.grad is not None}")
    print(f"梯度范围: [{w.grad.min():.4f}, {w.grad.max():.4f}]")

    print("\n测试完成！")
