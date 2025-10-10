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
from tools.topk_selection import SigmoidTopK, GumbelTopK, STETopK, compare_topk_methods


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
        print(f"topk={top_k_values}")
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
    """测试 3: 优化演示 - 三种方法对比学习特定位置的权重"""
    print("\n" + "=" * 100)
    print("测试 3: 优化演示 - 三种方法对比学习使特定位置被选中")
    print("=" * 100)

    torch.manual_seed(42)
    n = 2000
    k = 100

    # 目标: 希望前k个位置被选中
    target_indices = torch.arange(k)
    target = torch.zeros(n)
    target[target_indices] = 1.0

    print(f"\n任务设置:")
    print(f"  - 总维度: {n}")
    print(f"  - Top-K: {k}")
    print(f"  - 目标: 选中前{k}个位置")
    print(f"  - 初始学习率: 0.5")
    print(f"  - 学习率衰减: StepLR (每30步衰减0.5)")
    print(f"  - 迭代次数: 100")
    print(f"  - 输入归一化: 标准化 (减去均值，除以标准差)")

    # 三种方法
    methods = [
        ('SigmoidTopK', SigmoidTopK(k=k, temperature=0.5)),
        ('GumbelTopK', GumbelTopK(k=k, temperature=0.5)),
        ('STETopK', STETopK(k=k, temperature=0.5))
    ]

    # 存储所有方法的结果
    all_results = {}

    for method_name, model in methods:
        print(f"\n{'='*50}")
        print(f"训练 {method_name}")
        print(f"{'='*50}")

        # 初始化权重
        w = torch.randn(n, requires_grad=True)
        optimizer = optim.SGD([w], lr=0.5)

        # 添加学习率调度器 - 每30步学习率衰减为原来的0.5倍
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        # 记录训练过程
        losses = []
        accuracies = []
        learning_rates = []

        for epoch in range(100):
            optimizer.zero_grad()

            # 归一化输入 (标准化: 减去均值，除以标准差)
            w_mean = w.mean().detach()
            w_std = w.std().detach()
            w_normalized = (w - w_mean) / (w_std + 1e-8)

            # 前向传播
            output = model(w_normalized)

            # 损失: MSE between output and target
            loss = F.mse_loss(output, target)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 学习率递减
            scheduler.step()

            # 记录
            losses.append(loss.item())
            learning_rates.append(optimizer.param_groups[0]['lr'])

            # 计算准确率: 检查 top-k 位置是否正确
            predicted_indices = output.topk(k).indices
            accuracy = (predicted_indices.unsqueeze(1) == target_indices.unsqueeze(0)).any(dim=0).float().mean()
            accuracies.append(accuracy.item())

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}: Loss={loss.item():.4f}, Accuracy={accuracy.item():.2%}, LR={optimizer.param_groups[0]['lr']:.4f}")

        print(f"\n{method_name} 优化完成!")
        print(f"  - 最终损失: {losses[-1]:.4f}")
        print(f"  - 最终准确率: {accuracies[-1]:.2%}")

        # 检查最终结果
        with torch.no_grad():
            w_mean = w.mean()
            w_std = w.std()
            w_normalized = (w - w_mean) / (w_std + 1e-8)
            final_output = model(w_normalized)
            final_indices = final_output.topk(k).indices.sort().values

        print(f"\n结果对比:")
        print(f"  - 目标位置: [0, 1, 2, ..., {k-1}]")
        print(f"  - 学习位置前10: {final_indices[:10].tolist()}")
        print(f"  - 匹配度: {(final_indices.unsqueeze(1) == target_indices.unsqueeze(0)).any(dim=0).float().mean():.2%}")

        # 保存结果
        all_results[method_name] = {
            'losses': losses,
            'accuracies': accuracies,
            'learning_rates': learning_rates
        }

    # 保存对比训练曲线
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    colors = ['steelblue', 'forestgreen', 'coral']

    for i, (method_name, results) in enumerate(all_results.items()):
        # Loss
        axes[0, i].plot(results['losses'], color=colors[i], linewidth=2)
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].set_title(f'{method_name} - Training Loss')
        axes[0, i].grid(True, alpha=0.3)

        # Accuracy
        axes[1, i].plot(results['accuracies'], color=colors[i], linewidth=2)
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Accuracy')
        axes[1, i].set_title(f'{method_name} - Selection Accuracy')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_ylim(-0.05, 1.05)

        # Learning Rate
        axes[2, i].plot(results['learning_rates'], color=colors[i], linewidth=2)
        axes[2, i].set_xlabel('Epoch')
        axes[2, i].set_ylabel('Learning Rate')
        axes[2, i].set_title(f'{method_name} - Learning Rate Schedule')
        axes[2, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('topk_optimization_demo.png', dpi=150)
    print(f"\n  ✓ 训练曲线已保存到: topk_optimization_demo.png")

    print(f"\n{'='*100}")
    print("✓ 测试 3 完成: 三种方法对比训练完成!")
    print("="*100)


def test_4_weight_evolution_visualization(loss_type='bce'):
    """测试 4: 可视化训练过程中权重 w 的分布演化

    Args:
        loss_type: 损失函数类型，可选:
            - 'bce': Binary Cross Entropy
            - 'mse': Mean Squared Error
            - 'focal': Focal Loss
            - 'hinge': Hinge Loss
            - 'ranking': Ranking Loss (Pairwise Margin)
            - 'contrastive': Contrastive Loss
            - 'dice': Dice Loss
            - 'auc': AUC Loss
            - 'bce+ranking': BCE + Ranking混合
    """
    print("\n" + "=" * 100)
    print(f"测试 4: 可视化训练过程中权重 w 的分布演化 (Loss: {loss_type.upper()})")
    print("=" * 100)

    torch.manual_seed(42)
    n = 100
    k = 20

    # 目标: 希望前k个位置被选中
    target_indices = torch.arange(k)
    target = torch.zeros(n)
    target[target_indices] = 1.0

    print(f"\n任务设置:")
    print(f"  - 总维度: {n}")
    print(f"  - Top-K: {k}")
    print(f"  - 目标: 选中前{k}个位置")
    print(f"  - 迭代次数: 100")
    print(f"  - 损失函数: {loss_type.upper()}")

    # 使用 SigmoidTopK 作为示例 SigmoidTopK, GumbelTopK, STETopK
    model = SigmoidTopK(k=k, temperature=0.5)

    # 初始化权重
    w = torch.randn(n, requires_grad=True)
    optimizer = optim.SGD([w], lr=5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 记录训练过程中的权重快照
    weight_snapshots = []
    snapshot_epochs = [0, 10, 20, 30, 40, 50, 70, 90, 99]  # 每隔一段时间记录权重

    print(f"\n开始训练...")

    for epoch in range(100):
        # 保存权重快照（在训练前）
        if epoch in snapshot_epochs:
            with torch.no_grad():
                weight_snapshots.append((epoch, w.clone().detach().numpy()))

        optimizer.zero_grad()

        # 归一化输入
        w_mean = w.mean().detach()
        w_std = w.std().detach()
        w_normalized = (w - w_mean) / (w_std + 1e-8)

        # 前向传播
        output = model(w_normalized)

        # 根据 loss_type 计算不同的损失
        if loss_type == 'bce':
            # Binary Cross Entropy
            loss = F.binary_cross_entropy(output, target)

        elif loss_type == 'mse':
            # Mean Squared Error
            loss = F.mse_loss(output, target)

        elif loss_type == 'focal':
            # Focal Loss
            bce_loss = F.binary_cross_entropy(output, target, reduction='none')
            pt = torch.where(target == 1, output, 1 - output)
            alpha = 0.25
            gamma = 2.0
            focal_weight = alpha * (1 - pt) ** gamma
            loss = (focal_weight * bce_loss).mean()

        elif loss_type == 'hinge':
            # Hinge Loss (SVM-style)
            margin = 0.1
            target_hinge = 2 * target - 1  # 转换到 {-1, 1}
            output_hinge = 2 * output - 1
            loss = torch.clamp(margin - target_hinge * output_hinge, min=0).mean()

        elif loss_type == 'ranking':
            # Ranking Loss (Pairwise Margin)
            margin = 0.5
            target_weights = w_normalized[target_indices]
            nontarget_weights = w_normalized[k:]
            differences = target_weights[:, None] - nontarget_weights[None, :]
            loss = torch.clamp(margin - differences, min=0).mean()

        elif loss_type == 'contrastive':
            # Contrastive Loss
            margin = 1.0
            target_weights = w_normalized[:k]
            nontarget_weights = w_normalized[k:]

            target_variance = target_weights.var()
            target_mean = target_weights.mean()
            nontarget_mean = nontarget_weights.mean()
            separation = torch.clamp(margin - (target_mean - nontarget_mean), min=0)
            loss = target_variance + separation

        elif loss_type == 'dice':
            # Dice Loss
            smooth = 1e-6
            intersection = (output * target).sum()
            union = output.sum() + target.sum()
            dice = (2. * intersection + smooth) / (union + smooth)
            loss = 1 - dice

        elif loss_type == 'auc':
            # AUC Loss (approximate)
            pos_output = output[target == 1]
            neg_output = output[target == 0]
            differences = pos_output[:, None] - neg_output[None, :]
            auc = torch.sigmoid(differences).mean()
            loss = 1 - auc

        elif loss_type == 'bce+ranking':
            # BCE + Ranking 混合
            loss_bce = F.binary_cross_entropy(output, target)
            margin = 0.5
            target_weights = w_normalized[target_indices]
            nontarget_weights = w_normalized[k:]
            differences = target_weights[:, None] - nontarget_weights[None, :]
            loss_rank = torch.clamp(margin - differences, min=0).mean()
            loss = loss_bce + 0.5 * loss_rank

        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # 反向传播
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={loss.item():.4f}")

    print(f"\n训练完成!")

    # 创建可视化 - 3x3 网格展示9个epoch的权重排序图
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    for idx, (epoch, weights) in enumerate(weight_snapshots):
        ax = axes[idx]

        # 获取从大到小的排序索引
        sorted_indices = np.argsort(weights)[::-1]
        sorted_weights = weights[sorted_indices]

        # 确定每个权重的颜色：如果原始位置在[0, k)则为绿色（目标），否则为红色
        colors = ['green' if orig_idx < k else 'red' for orig_idx in sorted_indices]

        # 绘制每个柱子
        for i in range(n):
            ax.bar(i, sorted_weights[i], color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, edgecolor='black', label=f'Target positions (0-{k-1})'),
            Patch(facecolor='red', alpha=0.7, edgecolor='black', label=f'Non-target positions ({k}+)')
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

        # 添加 top-k 分隔线
        ax.axvline(x=k-0.5, color='blue', linestyle='--', linewidth=2, alpha=0.6)

        ax.set_xlabel('Sorted Index (Large to Small)', fontsize=11)
        ax.set_ylabel('Weight Value', fontsize=11)
        ax.set_title(f'Epoch {epoch}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Sorted Weight Values Evolution During Training (SigmoidTopK, {loss_type.upper()} Loss)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = f'weights_sorted_evolution_{loss_type}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ 权重排序演化可视化已保存到: {output_file}")

    print(f"\n{'='*100}")
    print("✓ 测试 4 完成: 权重分布演化可视化完成!")
    print("="*100)


def test_5_comparison():
    """测试 5: 三种方法对比"""
    print("\n" + "=" * 100)
    print("测试 5: 三种方法对比")
    print("=" * 100)

    torch.manual_seed(42)
    n = 2000
    k = 100
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

        # 可视化 - 按从大到小排序
        x = np.arange(n)

        # 对每个方法的输出进行排序（从大到小）
        sigmoid_sorted = np.sort(sigmoid_out)[::-1]
        gumbel_sorted = np.sort(gumbel_out)[::-1]
        ste_sorted = np.sort(ste_out)[::-1]

        axes[i, 0].bar(x, sigmoid_sorted, color='steelblue', alpha=0.7)
        axes[i, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[i, 0].set_title(f'SigmoidTopK (T={temp})')
        axes[i, 0].set_ylabel('Selection Score')
        axes[i, 0].set_ylim(-0.1, 1.1)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].bar(x, gumbel_sorted, color='forestgreen', alpha=0.7)
        axes[i, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[i, 1].set_title(f'GumbelTopK (T={temp})')
        axes[i, 1].set_ylim(-0.1, 1.1)
        axes[i, 1].grid(True, alpha=0.3)

        axes[i, 2].bar(x, ste_sorted, color='coral', alpha=0.7)
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
        # # 测试 1: 基础功能
        # test_1_basic_functionality()

        # # 测试 2: 梯度验证
        # test_2_gradient_verification()

        # # 测试 3: 优化演示
        # test_3_optimization_demo()

        # 测试 4: 权重演化可视化 - 不同损失函数对比
        # 可选: 'bce', 'mse', 'focal', 'hinge', 'ranking', 'contrastive', 'dice', 'auc', 'bce+ranking'

        for loss_to_test in ['bce', 'mse', 'focal', 'hinge', 'ranking', 'contrastive', 'dice', 'auc', 'bce+ranking']:
            test_4_weight_evolution_visualization(loss_type=loss_to_test)

        # 如果想测试多个损失函数，取消下面的注释：
        # for loss_type in ['bce', 'mse', 'focal', 'ranking', 'bce+ranking']:
        #     test_4_weight_evolution_visualization(loss_type=loss_type)

        # 测试 5: 方法对比
        # test_5_comparison()

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
        print("  - 温度参数: 越小输出越硬 (接近 0/1)，推荐范围 [0.01, 1.0]")

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
