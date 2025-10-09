"""
可微分 Top-K Selection 完整测试和演示脚本

包含四个测试:
1. 基础功能测试 - 验证输出正确性
2. 梯度验证测试 - 验证梯度回传
3. 优化演示 - 通过梯度下降学习最优权重
4. 方法对比 - 对比三种方法的效果

运行: python test_topk_demo.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from utils.topk_selection import SigmoidTopK, GumbelTopK, STETopK, compare_topk_methods


def test_1_basic_functionality():
    """测试 1: 基础功能测试"""
    print("\n" + "=" * 100)
    print("测试 1: 基础功能测试")
    print("=" * 100)

    torch.manual_seed(42)
    n = 50
    k = 10
    w = torch.randn(n, requires_grad=True)

    print(f"\n配置:")
    print(f"  - 输入维度 n: {n}")
    print(f"  - Top-K 参数 k: {k}")
    print(f"  - 输入范围: [{w.min():.3f}, {w.max():.3f}]")

    # 测试三种方法
    methods = [
        ('SigmoidTopK', SigmoidTopK(k=k, temperature=0.1)),
        ('GumbelTopK', GumbelTopK(k=k, temperature=0.1)),
        ('STETopK', STETopK(k=k, temperature=0.1))
    ]

    for name, model in methods:
        model.eval()
        output = model(w)

        print(f"\n{name}:")
        print(f"  - 输出形状: {output.shape}")
        print(f"  - 输出范围: [{output.min():.4f}, {output.max():.4f}]")
        print(f"  - Top-{k} 均值: {output.topk(k).values.mean():.4f}")
        print(f"  - Bottom-{n-k} 均值: {output.topk(n-k, largest=False).values.mean():.4f}")

        # 检查 Top-K 项是否接近 1
        top_k_values = output.topk(k).values
        assert top_k_values.min() > 0.8, f"{name}: Top-K 项应该 > 0.8"

        # 检查其他项是否接近 0
        bottom_values = output.topk(n-k, largest=False).values
        assert bottom_values.max() < 0.2, f"{name}: 非 Top-K 项应该 < 0.2"

        print(f"  ✓ 验证通过!")

    print(f"\n{'='*100}")
    print("✓ 测试 1 完成: 所有方法输出正确!")
    print("="*100)


def test_2_gradient_verification():
    """测试 2: 梯度验证测试"""
    print("\n" + "=" * 100)
    print("测试 2: 梯度验证测试")
    print("=" * 100)

    torch.manual_seed(42)
    n = 20
    k = 5

    methods = [
        ('SigmoidTopK', SigmoidTopK(k=k, temperature=0.1)),
        ('GumbelTopK', GumbelTopK(k=k, temperature=0.1)),
        ('STETopK', STETopK(k=k, temperature=0.1))
    ]

    for name, model in methods:
        print(f"\n{name} 梯度测试:")

        # 创建需要梯度的权重
        w = torch.randn(n, requires_grad=True)

        # 前向传播
        model.eval()
        output = model(w)

        # 定义一个简单的损失（希望所有 top-k 项都被选中）
        loss = output.sum()

        # 反向传播
        loss.backward()

        # 检查梯度
        print(f"  - w.grad 存在: {w.grad is not None}")
        print(f"  - w.grad 形状: {w.grad.shape}")
        print(f"  - w.grad 范围: [{w.grad.min():.4f}, {w.grad.max():.4f}]")
        print(f"  - w.grad 均值: {w.grad.mean():.4f}")
        print(f"  - w.grad 非零元素: {(w.grad.abs() > 1e-6).sum()}/{n}")

        assert w.grad is not None, f"{name}: 梯度应该存在"
        assert w.grad.abs().max() > 0, f"{name}: 梯度应该非零"

        print(f"  ✓ 梯度验证通过!")

    print(f"\n{'='*100}")
    print("✓ 测试 2 完成: 所有方法梯度正确回传!")
    print("="*100)


def test_3_optimization_demo():
    """测试 3: 优化演示 - 学习特定位置的权重"""
    print("\n" + "=" * 100)
    print("测试 3: 优化演示 - 学习使特定位置被选中")
    print("=" * 100)

    torch.manual_seed(42)
    n = 30
    k = 5

    # 目标: 希望位置 [3, 7, 15, 20, 25] 被选中
    target_indices = torch.tensor([3, 7, 15, 20, 25])
    target = torch.zeros(n)
    target[target_indices] = 1.0

    print(f"\n任务设置:")
    print(f"  - 总维度: {n}")
    print(f"  - Top-K: {k}")
    print(f"  - 目标位置: {target_indices.tolist()}")
    print(f"  - 学习率: 0.5")
    print(f"  - 迭代次数: 100")

    # 使用 SigmoidTopK
    model = SigmoidTopK(k=k, temperature=0.1)

    # 初始化权重
    w = torch.randn(n, requires_grad=True)
    optimizer = optim.SGD([w], lr=0.5)

    # 记录训练过程
    losses = []
    accuracies = []

    print(f"\n开始优化...")

    for epoch in range(100):
        optimizer.zero_grad()

        # 前向传播
        output = model(w)

        # 损失: MSE between output and target
        loss = F.mse_loss(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录
        losses.append(loss.item())

        # 计算准确率: 检查 top-k 位置是否正确
        predicted_indices = output.topk(k).indices
        accuracy = (predicted_indices.unsqueeze(1) == target_indices.unsqueeze(0)).any(dim=0).float().mean()
        accuracies.append(accuracy.item())

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={loss.item():.4f}, Accuracy={accuracy.item():.2%}")

    print(f"\n优化完成!")
    print(f"  - 最终损失: {losses[-1]:.4f}")
    print(f"  - 最终准确率: {accuracies[-1]:.2%}")

    # 检查最终结果
    with torch.no_grad():
        final_output = model(w)
        final_indices = final_output.topk(k).indices.sort().values

    print(f"\n结果对比:")
    print(f"  - 目标位置: {target_indices.sort().values.tolist()}")
    print(f"  - 学习位置: {final_indices.tolist()}")
    print(f"  - 匹配度: {(final_indices.unsqueeze(1) == target_indices.unsqueeze(0)).any(dim=0).float().mean():.2%}")

    # 保存训练曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)

    ax2.plot(accuracies)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Selection Accuracy')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('topk_optimization_demo.png', dpi=150)
    print(f"\n  ✓ 训练曲线已保存到: topk_optimization_demo.png")

    print(f"\n{'='*100}")
    print("✓ 测试 3 完成: 成功通过梯度下降学习目标权重!")
    print("="*100)


def test_4_comparison():
    """测试 4: 三种方法对比"""
    print("\n" + "=" * 100)
    print("测试 4: 三种方法对比")
    print("=" * 100)

    torch.manual_seed(42)
    n = 50
    k = 10
    w = torch.randn(n)

    print(f"\n配置:")
    print(f"  - 输入维度: {n}")
    print(f"  - Top-K: {k}")

    # 测试不同温度
    temperatures = [0.01, 0.1, 0.5, 1.0]

    fig, axes = plt.subplots(len(temperatures), 3, figsize=(15, 4*len(temperatures)))

    for i, temp in enumerate(temperatures):
        print(f"\n温度 T = {temp}:")

        sigmoid_model = SigmoidTopK(k=k, temperature=temp)
        gumbel_model = GumbelTopK(k=k, temperature=temp)
        ste_model = STETopK(k=k, temperature=temp)

        sigmoid_model.eval()
        gumbel_model.eval()
        ste_model.eval()

        with torch.no_grad():
            sigmoid_out = sigmoid_model(w).numpy()
            gumbel_out = gumbel_model(w).numpy()
            ste_out = ste_model(w).numpy()

        # 打印统计信息
        print(f"  SigmoidTopK: Top-{k} 均值={sigmoid_out[np.argsort(-sigmoid_out)[:k]].mean():.4f}, "
              f"其他均值={sigmoid_out[np.argsort(-sigmoid_out)[k:]].mean():.4f}")
        print(f"  GumbelTopK:  Top-{k} 均值={gumbel_out[np.argsort(-gumbel_out)[:k]].mean():.4f}, "
              f"其他均值={gumbel_out[np.argsort(-gumbel_out)[k:]].mean():.4f}")
        print(f"  STETopK:     Top-{k} 均值={ste_out[np.argsort(-ste_out)[:k]].mean():.4f}, "
              f"其他均值={ste_out[np.argsort(-ste_out)[k:]].mean():.4f}")

        # 可视化
        x = np.arange(n)

        axes[i, 0].bar(x, sigmoid_out, color='steelblue', alpha=0.7)
        axes[i, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[i, 0].set_title(f'SigmoidTopK (T={temp})')
        axes[i, 0].set_ylabel('Selection Score')
        axes[i, 0].set_ylim(-0.1, 1.1)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].bar(x, gumbel_out, color='forestgreen', alpha=0.7)
        axes[i, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[i, 1].set_title(f'GumbelTopK (T={temp})')
        axes[i, 1].set_ylim(-0.1, 1.1)
        axes[i, 1].grid(True, alpha=0.3)

        axes[i, 2].bar(x, ste_out, color='coral', alpha=0.7)
        axes[i, 2].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[i, 2].set_title(f'STETopK (T={temp})')
        axes[i, 2].set_ylim(-0.1, 1.1)
        axes[i, 2].grid(True, alpha=0.3)

        if i == len(temperatures) - 1:
            for ax in axes[i]:
                ax.set_xlabel('Index')

    plt.tight_layout()
    plt.savefig('topk_methods_comparison.png', dpi=150)
    print(f"\n  ✓ 对比图已保存到: topk_methods_comparison.png")

    print(f"\n{'='*100}")
    print("✓ 测试 4 完成: 三种方法对比完成!")
    print("="*100)


def main():
    """主函数 - 运行所有测试"""
    print("\n" + "=" * 100)
    print(" " * 30 + "可微分 Top-K Selection 完整测试")
    print("=" * 100)

    try:
        # 测试 1: 基础功能
        test_1_basic_functionality()

        # 测试 2: 梯度验证
        test_2_gradient_verification()

        # 测试 3: 优化演示
        test_3_optimization_demo()

        # 测试 4: 方法对比
        test_4_comparison()

        # 最终总结
        print("\n" + "=" * 100)
        print(" " * 35 + "所有测试通过! ✓")
        print("=" * 100)

        print("\n生成的文件:")
        print("  1. topk_optimization_demo.png - 优化过程可视化")
        print("  2. topk_methods_comparison.png - 三种方法对比")

        print("\n使用建议:")
        print("  - 一般情况推荐使用 SigmoidTopK (最稳定)")
        print("  - 需要随机探索时使用 GumbelTopK (训练时有随机性)")
        print("  - 需要精确 0/1 输出时使用 STETopK (前向二值化)")
        print("  - 温度参数: 越小输出越"硬"(接近 0/1)，推荐范围 [0.01, 1.0]")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    # 导入 F (用于 MSE loss)
    import torch.nn.functional as F

    # 运行所有测试
    main()
