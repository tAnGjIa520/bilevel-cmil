# SoftTopK 权重演化实验设置文档

## 📋 实验概述

本实验使用 **SoftTopK** 方法来研究在不同损失函数下，模型如何学会正确的 Top-K 选择。通过可视化权重在 100 个训练 epoch 中的演化过程，来理解平滑激活函数（Tanh）如何帮助模型进行柔和的渐进式选择。

**核心目标**: 观察权重如何从随机初始化逐步优化到能够正确选中目标位置。

---

## 1️⃣ 数据与任务设置

### 基本配置

```
总维度:           n = 100
选择数量:         k = 20
任务目标:         学会选中前20个位置（位置 0-19）
随机种子:         42 (确保实验可复现)
```

### 目标向量

```python
target = [1, 1, ..., 1 (×20), 0, 0, ..., 0 (×80)]
```

- **目标位置** (0-19): 标记为 1，应该被选中
- **非目标位置** (20-99): 标记为 0，不应该被选中
- 共有 100 维，其中 20 维被标记为目标

---

## 2️⃣ SoftTopK 方法详解

### 核心原理

SoftTopK 使用 **Tanh 激活函数** 替代 Sigmoid，提供更平滑的选择效果：

```
1. 找到第 k 大的值作为阈值
2. 计算: tanh((w - threshold) / temperature)
3. Tanh 输出 ∈ [-1, 1]，然后映射到 [0, 1]
4. 最终得分: (tanh_output + 1) / 2
```

### 优点对比

| 特性 | SigmoidTopK | GumbelTopK | SoftTopK |
|-----|-----------|-----------|---------|
| 可微性 | ✓ | ✓ | ✓ |
| 输出平滑度 | 中等 | 中等 | 高 |
| 训练随机性 | 无 | 有 (Gumbel) | 无 |
| 推理确定性 | ✓ | ✓ | ✓ |
| 梯度流通 | 好 | 好 | 优秀 |
| 收敛速度 | 快 | 中等 | 中等 |

### 参数配置

```python
SoftTopK(
    k=20,              # Top-K 参数，选择前20大的元素
    temperature=0.5,   # 温度参数，控制平滑度
                       # 越小 → 输出越"硬"（接近0/1）
                       # 越大 → 输出越"软"（接近0.5）
    dim=-1            # 操作维度（默认最后一维）
)
```

---

## 3️⃣ 损失函数配置

支持 9 种不同的损失函数进行对比：

### 1. **BCE（二值交叉熵）**
```python
loss = F.binary_cross_entropy(output, target)
```
- 标准的二分类损失
- Top-K 项应该接近 1，其他项接近 0

### 2. **MSE（均方误差）**
```python
loss = F.mse_loss(output, target)
```
- 直接最小化输出与目标的距离
- 对异常值更敏感

### 3. **Focal Loss（焦点损失）**
```python
pt = torch.where(target == 1, output, 1 - output)
loss = -((1 - pt) ** 2 * torch.log(pt + 1e-8)).mean()
```
- 关注困难样本
- 在类别不平衡时表现更好

### 4. **Hinge Loss（铰链损失）**
```python
loss = F.soft_margin_loss(output * (2 * target - 1), torch.ones_like(output))
```
- SVM 风格的损失
- 对边界的优化更敏感

### 5. **Ranking Loss（排序损失）**
```python
# Top-K 项与非 Top-K 项的排序损失
pos = (output * target).sum()
neg = (output * (1 - target)).sum()
loss = F.relu(neg - pos + 1)
```
- 强制 Top-K 项 > 非 Top-K 项
- 适合排序任务

### 6. **Contrastive Loss（对比损失）**
```python
# 目标项应该接近 1，非目标项应该接近 0
pos_loss = (1 - output) ** 2 * target
neg_loss = output ** 2 * (1 - target)
loss = (pos_loss + neg_loss).mean()
```
- 强制正样本聚集，负样本分散
- 适合度量学习

### 7. **Dice Loss（Dice 损失）**
```python
intersection = (output * target).sum()
loss = 1 - (2 * intersection) / (output.sum() + target.sum() + 1e-8)
```
- 常用于分割任务
- 对大背景敏感度低

### 8. **AUC Loss（面积损失）**
```python
# 基于 AUC 最大化的损失
sorted_idx = torch.argsort(output, descending=True)
auc_weight = torch.arange(1, n+1, dtype=torch.float32, device=device)
auc_loss = -(target[sorted_idx] * auc_weight).sum()
```
- 优化 AUC 度量
- 对排序敏感

### 9. **BCE + Ranking（混合损失）**
```python
bce = F.binary_cross_entropy(output, target)
pos = (output * target).sum()
neg = (output * (1 - target)).sum()
ranking = F.relu(neg - pos + 1)
loss = bce + 0.1 * ranking  # 加权混合
```
- 结合两种目标
- 既要输出接近目标，又要保持排序

---

## 4️⃣ 模型与优化配置

### 模型结构
```python
# 可学习的权重向量
w = torch.randn(100, requires_grad=True)

# SoftTopK 选择算子
topk_selector = SoftTopK(k=20, temperature=0.5)
```

### 优化器配置
```python
optimizer = optim.SGD([w], lr=5)
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,    # 每30个epoch衰减一次
    gamma=0.5        # 学习率乘以0.5
)
```

### 训练循环
```
总迭代数: 100 epoch
每epoch步骤:
  1. 零梯度
  2. 前向传播: output = topk_selector(w)
  3. 计算损失: loss = loss_fn(output, target)
  4. 反向传播: loss.backward()
  5. 优化: optimizer.step()
  6. 学习率调整: scheduler.step()
```

---

## 5️⃣ 权重快照与可视化

### 快照时刻

```python
snapshot_epochs = [0, 50, 99]
```

- **Epoch 0**: 初始化状态（随机权重）
- **Epoch 50**: 训练中期（学习过程中）
- **Epoch 99**: 最终收敛状态

### 可视化内容

对于每个快照，可视化权重从大到小排序后的分布：

```
X 轴: 排序后的索引 (0-99)
Y 轴: 权重值

颜色编码:
  🟢 绿色: 目标位置 (0-19) - 应该被选中
  🔴 红色: 非目标位置 (20-99) - 不应该被选中
  🔵 蓝线: Top-K 分界线（位置 20）
```

### 图表信息

```
整体尺寸: 7.5 × 2.4 英寸（1×3 布局，单栏论文格式）

标题: SoftTopK: Weight Evolution ({LOSS_TYPE})

子图标题: Epoch {epoch_number}

字体大小:
  - 主标题: 8pt
  - 子图标题: 7pt
  - 坐标轴标签: 6.5pt
  - 刻度标签: 5pt
  - 图例: 6pt

坐标轴:
  - X 轴: Index                    [排序后的索引]
  - Y 轴: Weight                   [权重值]

其他元素:
  - 网格: alpha=0.3，仅显示 Y 轴网格
  - 分界线: 蓝色虚线，标记 Top-K 边界
  - DPI: 150 (高清晰度)
  - 边框: bbox_inches='tight'
```

---

## 6️⃣ 实验执行

### 运行脚本

```bash
# 运行所有测试
python test_topk_demo.py

# 或指定某个测试（1-4）
python test_topk_demo.py --test 4  # 仅运行权重演化可视化
```

### 输出文件

生成的可视化文件保存到 `softtopk_analysis/` 目录：

```
softtopk_analysis/
├── softtopk_BCE_weight_evolution.png
├── softtopk_MSE_weight_evolution.png
├── softtopk_FOCAL_weight_evolution.png
├── softtopk_HINGE_weight_evolution.png
├── softtopk_RANKING_weight_evolution.png
├── softtopk_CONTRASTIVE_weight_evolution.png
├── softtopk_DICE_weight_evolution.png
├── softtopk_AUC_weight_evolution.png
└── softtopk_BCE+RANKING_weight_evolution.png
```

### 观察指标

| 指标 | 说明 |
|-----|------|
| 绿色柱子集中度 | 目标位置的权重是否聚集在左侧（大值） |
| 红色柱子集中度 | 非目标位置的权重是否聚集在右侧（小值） |
| 蓝线位置清晰度 | Top-K 边界是否明确分离绿色和红色 |
| 权重差异 | Top-K 项和非 Top-K 项的权重差距 |

---

## 7️⃣ 实验理论背景

### Tanh 激活函数的优势

```
Sigmoid: σ(x) = 1 / (1 + e^(-x))
         - 输出 ∈ [0, 1]
         - 中心在 0.5
         - S 型曲线较缓

Tanh:    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
         - 输出 ∈ [-1, 1]
         - 中心在 0
         - S 型曲线较陡峭
         - 梯度更强
```

### 为什么 SoftTopK 更好？

1. **更强的梯度信号**: Tanh 的梯度在中值处比 Sigmoid 更强
2. **更平滑的输出**: 在 [-1, 1] 范围内有更多中间值
3. **更好的收敛**: 强梯度帮助权重更快地分化
4. **减少梯度消失**: Tanh 的梯度比 Sigmoid 更不容易消失

---

## 8️⃣ 参数调整指南

### 如何调整参数

```python
# 修改采样时间点
snapshot_epochs = [0, 30, 60, 99]  # 4 个关键时刻

# 修改图表参数（在 test_4 函数中）
figsize=(7.5, 2.4)   # 1×3 布局，单栏论文格式 (宽, 高)

# 调整字体大小
fontsize_title = 8       # 主标题
fontsize_subtitle = 7    # 子图标题
fontsize_label = 6.5     # 坐标轴标签
fontsize_tick = 5        # 刻度标签

# 调整 SoftTopK 参数
temperature = 0.3       # 更小 → 更硬的选择
temperature = 1.0       # 更大 → 更软的选择

# 调整优化器参数
lr = 10                  # 更高的学习率
lr = 1                   # 更低的学习率
step_size = 50           # 更频繁的衰减
gamma = 0.3              # 更激进的衰减
```

---

## 9️⃣ 常见问题

### Q1: 为什么选择 SoftTopK 而不是 SigmoidTopK？

**A**: SoftTopK 提供更平滑的输出分布，有助于：
- 更好的梯度流通
- 更稳定的收敛
- 更自然的概率分布

### Q2: 为什么选择 SoftTopK 而不是 GumbelTopK？

**A**: SoftTopK 适合：
- 不需要探索性噪声的确定性任务
- 推理时不需要随机性
- 更简单的模型（无需噪声参数）

### Q3: 如何解释权重的演化过程？

**A**: 观察三个阶段：
1. **初始化** (Epoch 0): 权重随机分布
2. **学习** (Epoch 50): 目标位置权重增大
3. **收敛** (Epoch 99): 目标/非目标权重清晰分化

### Q4: 为什么某些损失函数收敛更快？

**A**: 不同的损失函数有不同的优化景观：
- Ranking Loss: 直接优化排序 → 最快收敛
- BCE: 通用损失 → 中等收敛
- Contrastive: 缓慢但稳定 → 最慢收敛

---

## 🔟 LaTeX 集成示例

### 论文中引用图表

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.9\columnwidth]{softtopk_analysis/softtopk_RANKING_weight_evolution.png}
    \caption{SoftTopK 在 Ranking Loss 下的权重演化过程。
             从左到右分别是初始化、训练中期和最终收敛状态。
             绿色柱子表示目标位置，红色表示非目标位置。}
    \label{fig:softtopk_evolution}
\end{figure}
```

### 方法对比表格

```latex
\begin{table}[h]
    \centering
    \begin{tabular}{c|cccc}
        \hline
        特性 & SigmoidTopK & GumbelTopK & STETopK & SoftTopK \\
        \hline
        可微性 & ✓ & ✓ & ✓ & ✓ \\
        平滑度 & 中等 & 中等 & 低 & 高 \\
        确定性 & 是 & 是 & 是 & 是 \\
        梯度强度 & 中等 & 中等 & 强 & 很强 \\
        \hline
    \end{tabular}
    \caption{不同 Top-K 选择方法的特性对比}
    \label{tab:topk_comparison}
\end{table}
```

### 尺寸配置

设计目标: 适配双栏论文中的单栏宽度

尺寸配置:
  - 图表宽度: 7.5 英寸 (≈ 3.5 英寸单栏宽度的 2.1 倍)
  - 图表高度: 2.4 英寸 (紧凑高度，节省页面空间)
  - 子图数量: 3 个 (1 行 × 3 列)
  - 每个子图约: ~2.5 × 2.4 英寸

在论文中的显示:
  - 如果设置为单栏宽度: 图表会自动缩放到 ~3.5 英寸宽
  - 高度会相应缩放到 ~1.35 英寸

---

## 📊 完整工作流

```
1. 初始化权重 w ← 高斯随机
   ↓
2. 前向传播: output = SoftTopK(w)
   ├─ 计算第20大的阈值
   ├─ Tanh 映射: tanh((w - threshold) / 0.5)
   └─ 缩放到 [0, 1]
   ↓
3. 计算损失: loss = loss_fn(output, target)
   (9种损失函数选择)
   ↓
4. 反向传播: loss.backward()
   ├─ 计算 ∂loss/∂output
   └─ 计算 ∂loss/∂w
   ↓
5. 优化: w ← w - lr * ∂loss/∂w
   ↓
6. 学习率调整: lr *= gamma (每30个epoch)
   ↓
7. 快照: 记录 epoch [0, 50, 99] 的权重
   ↓
8. 可视化: 排序权重并绘制 1×3 对比图
   ↓
9. 输出: 保存到 softtopk_analysis/{loss_type}_weight_evolution.png
```

---

## 🎯 总结

SoftTopK 通过使用 **Tanh 激活函数** 提供更平滑、更稳定的 Top-K 选择。与 Sigmoid 相比，Tanh 提供：

- ✓ 更强的梯度信号
- ✓ 更好的收敛性能
- ✓ 更自然的概率分布
- ✓ 更适合递进式选择

通过可视化 100 个 epoch 中三个关键时刻的权重分布，我们可以清晰地观察模型如何学习正确地选择 Top-K 项，以及不同损失函数对这一过程的影响。

