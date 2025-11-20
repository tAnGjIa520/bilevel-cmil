# GumbelTopK 权重演化实验设置文档

## 📋 实验概述

本实验使用 **GumbelTopK** 方法来研究在不同损失函数下，模型如何学会正确的 Top-K 选择。通过可视化权重在 100 个训练 epoch 中的演化过程，来理解 Gumbel 噪声如何帮助模型进行探索和收敛。

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

## 2️⃣ 模型设置 (GumbelTopK)

### 模型初始化

```python
model = GumbelTopK(
    k=20,              # 选择前 20 个最大值
    temperature=0.5,   # 温度参数（控制输出硬度）
    gumbel_noise=1.0   # Gumbel 噪声强度
)
model.train()          # 启用训练模式（使用 Gumbel 噪声）
```

### GumbelTopK 工作原理

```
训练模式 (self.training = True):
  ├─ 采样 Gumbel 噪声: gumbel ~ -log(-log(U))，U ~ Uniform(0,1)
  ├─ 加入噪声: w_noisy = w + gumbel_noise × gumbel
  ├─ 计算阈值: threshold = kth_value(w_noisy, k)
  └─ 软选择: output = sigmoid((w_noisy - threshold) / temperature)

特点:
  ✓ 每次迭代的噪声都不同，增加随机探索
  ✓ 避免过早收敛到局部最优
  ✓ 适合需要探索的优化问题
```

### 权重初始化

```python
w = torch.randn(n, requires_grad=True)
# 从标准正态分布 N(0, 1) 初始化
```

---

## 3️⃣ 优化设置

### 优化器配置

```python
优化器:        SGD (随机梯度下降)
初始学习率:    lr = 5.0
学习率调度:    StepLR
  - step_size = 30 (每 30 个 epoch 调整一次)
  - gamma = 0.5 (每次乘以 0.5)

学习率时间表:
  ┌─ epoch [0-29]:   lr = 5.0
  ├─ epoch [30-59]:  lr = 2.5
  ├─ epoch [60-89]:  lr = 1.25
  └─ epoch [90-99]:  lr = 0.625
```

### 训练总长度

```
总 epoch 数: 100
总迭代次数: 100
```

---

## 4️⃣ 前向传播处理

每个训练迭代中的处理流程：

```python
# 第1步: 梯度清零
optimizer.zero_grad()

# 第2步: 归一化权重 (Z-score 标准化)
w_mean = w.mean().detach()
w_std = w.std().detach()
w_normalized = (w - w_mean) / (w_std + 1e-8)

# 第3步: 前向传播
output = model(w_normalized)
# 输出 shape: [100]
# 输出范围: [0, 1]（软选择，接近 0/1）

# 第4步: 计算损失 (9 种损失函数之一)
loss = compute_loss(output, target, loss_type)

# 第5步: 反向传播
loss.backward()

# 第6步: 参数更新
optimizer.step()
scheduler.step()
```

---

## 5️⃣ 9 种损失函数详解

### 1. Binary Cross Entropy (BCE)

```python
loss = F.binary_cross_entropy(output, target)

特点:
  - 标准的二分类交叉熵损失
  - 直接衡量预测分布与目标分布的差异
  - 广泛应用于分类问题
```

### 2. Mean Squared Error (MSE)

```python
loss = F.mse_loss(output, target)

特点:
  - 均方误差，回归风格的损失
  - 对大错误的惩罚更强
  - 更关注精确的数值匹配
```

### 3. Focal Loss

```python
bce_loss = F.binary_cross_entropy(output, target, reduction='none')
pt = torch.where(target == 1, output, 1 - output)
alpha = 0.25
gamma = 2.0
focal_weight = alpha * (1 - pt) ** gamma
loss = (focal_weight * bce_loss).mean()

特点:
  - 难样本挖掘（难分类的样本获得更大权重）
  - 处理类不平衡问题
  - 通过调整 gamma 改变困难程度
```

### 4. Hinge Loss (SVM-style)

```python
margin = 0.1
target_hinge = 2 * target - 1          # 转换到 {-1, 1}
output_hinge = 2 * output - 1
loss = torch.clamp(margin - target_hinge * output_hinge, min=0).mean()

特点:
  - SVM 风格的最大间隔学习
  - 鼓励目标与非目标有清晰的分离
  - margin 参数控制分离程度
```

### 5. Ranking Loss (Pairwise Margin)

```python
margin = 0.5
target_weights = w_normalized[target_indices]
nontarget_weights = w_normalized[k:]
differences = target_weights[:, None] - nontarget_weights[None, :]
loss = torch.clamp(margin - differences, min=0).mean()

特点:
  - 成对排序损失
  - 确保目标位置的权重总是大于非目标位置
  - 直接优化选择的正确性
```

### 6. Contrastive Loss

```python
margin = 1.0
target_weights = w_normalized[:k]
nontarget_weights = w_normalized[k:]

target_variance = target_weights.var()
target_mean = target_weights.mean()
nontarget_mean = nontarget_weights.mean()
separation = torch.clamp(margin - (target_mean - nontarget_mean), min=0)
loss = target_variance + separation

特点:
  - 对比学习方法
  - 同时优化类内聚性和类间分离
  - target_variance: 目标组内的紧凑程度
  - separation: 两个组之间的分离程度
```

### 7. Dice Loss

```python
smooth = 1e-6
intersection = (output * target).sum()
union = output.sum() + target.sum()
dice = (2. * intersection + smooth) / (union + smooth)
loss = 1 - dice

特点:
  - Dice 系数（骰子相似度）
  - 衡量预测与目标的重叠程度
  - 常用于医学图像分割
```

### 8. AUC Loss (近似)

```python
pos_output = output[target == 1]
neg_output = output[target == 0]
differences = pos_output[:, None] - neg_output[None, :]
auc = torch.sigmoid(differences).mean()
loss = 1 - auc

特点:
  - 近似优化 AUC (Area Under Curve)
  - 衡量模型的排序能力
  - 与排序相关，但更平滑
```

### 9. BCE + Ranking 混合

```python
loss_bce = F.binary_cross_entropy(output, target)
margin = 0.5
target_weights = w_normalized[target_indices]
nontarget_weights = w_normalized[k:]
differences = target_weights[:, None] - nontarget_weights[None, :]
loss_rank = torch.clamp(margin - differences, min=0).mean()
loss = loss_bce + 0.5 * loss_rank

特点:
  - 结合 BCE 和 Ranking 损失
  - 同时优化概率匹配和排序正确性
  - 0.5 权重平衡两个目标
```

---

## 6️⃣ 权重快照采样

### 采样时间点

```python
snapshot_epochs = [0, 50, 99]
```

| Epoch | 训练进度 | 描述 |
|-------|---------|------|
| **0** | 0% | 初始化：权重完全随机 |
| **50** | 50% | 中期：权重分化明显，目标与非目标分离 |
| **99** | 100% | 最终：完全收敛，权重稳定 |

### 快照内容

```python
# 每个快照记录: (epoch_number, weight_array)
weight_snapshots.append((epoch, w.clone().detach().numpy()))
```

---

## 7️⃣ 可视化设置

### 图表结构

```
1 × 3 网格布局（横向排列）
├─ 行: 1 行
├─ 列: 3 列
└─ 共 3 个子图，每个子图显示一个 epoch 的权重分布

子图排列:
  [1] Epoch 0    [2] Epoch 50    [3] Epoch 99
  初始化          中期             最终

  从左到右展示权重从随机到有序的演化过程
```

### 单个子图的构成

```python
# 对权重进行排序（从大到小）
sorted_indices = np.argsort(weights)[::-1]
sorted_weights = weights[sorted_indices]

# 根据原始位置给柱子着色
colors = []
for orig_idx in sorted_indices:
    if orig_idx < 20:           # 目标位置
        colors.append('green')
    else:                        # 非目标位置
        colors.append('red')

# 绘制柱状图
for i in range(100):
    ax.bar(i, sorted_weights[i], color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)

# 添加分界线（Top-K 边界）
ax.axvline(x=k-0.5, color='blue', linestyle='--', linewidth=2, alpha=0.6)

# 添加图例
legend_elements = [
    Patch(facecolor='green', label=f'Target positions (0-{k-1})'),
    Patch(facecolor='red', label=f'Non-target positions ({k}+)')
]
ax.legend(handles=legend_elements, fontsize=9, loc='upper right')
```

### 颜色编码

| 颜色 | 含义 |
|------|------|
| 🟢 **绿色** | 目标位置 (0-19) - 应该被选中 |
| 🔴 **红色** | 非目标位置 (20-99) - 不应该被选中 |
| 🔵 **蓝线** | Top-K 分界线（位置 20） |

### 图表信息

```
整体尺寸: 7.5 × 2.4 英寸（1×3 布局，单栏论文格式）

标题: GumbelTopK: Weight Evolution ({LOSS_TYPE})

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

## 8️⃣ 完整实验流程

```
┌─────────────────────────────────────────┐
│  初始化                                 │
├─────────────────────────────────────────┤
│  1. 设置随机种子 (seed=42)              │
│  2. 初始化目标向量                      │
│  3. 初始化 GumbelTopK 模型              │
│  4. 初始化权重 w ~ N(0,1)              │
│  5. 创建优化器和学习率调度器            │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  训练循环 (100 epochs)                   │
├─────────────────────────────────────────┤
│  for epoch in range(100):               │
│    ├─ 【快照】保存权重                  │
│    ├─ 【归一化】z-score 标准化          │
│    ├─ 【前向】GumbelTopK(w_norm)       │
│    ├─ 【损失】计算 9 种损失之一         │
│    ├─ 【反向】loss.backward()           │
│    ├─ 【优化】optimizer.step()          │
│    └─ 【调度】scheduler.step()          │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  可视化                                 │
├─────────────────────────────────────────┤
│  1. 创建 3×3 子图网格                   │
│  2. 对每个快照的权重排序                │
│  3. 按目标/非目标着色                   │
│  4. 绘制柱状图和分界线                  │
│  5. 添加图例和标签                      │
│  6. 保存到 gumbel_topk_analysis/        │
└─────────────────────────────────────────┘
```

---

## 9️⃣ 核心研究问题

### 问题 1: 学习能力
**GumbelTopK 能否学会正确的 Top-K 选择？**
- 观察: 绿色柱子是否逐渐聚集在左侧（前 20 个位置）
- 指标: 目标位置的权重排名

### 问题 2: 损失函数影响
**不同损失函数如何影响权重的演化过程？**
- 观察: 9 张图表的权重分布是否有显著差异
- 比较: 哪种损失函数导致最快的收敛？最平滑的演化？

### 问题 3: 噪声作用
**Gumbel 噪声如何帮助探索和收敛？**
- 特性: 训练时噪声增加随机性，避免局部最优
- 动态: 权重从混乱到有序的过程

### 问题 4: 训练动态
**权重如何从随机初始化演化到最终状态？**
- 观察三个关键阶段:
  - **Epoch 0** (初始): 权重完全随机，没有明显模式
  - **Epoch 50** (中期): 权重分化明显，目标与非目标位置开始清晰分离，绿色柱子逐步向左聚集
  - **Epoch 99** (最终): 完全收敛，目标位置权重稳定在最前面（左侧），形成清晰的绿红分界

---

## 🔟 输出文件

### 生成的图表

```
gumbel_topk_analysis/
├── gumbel_topk_BCE_weight_evolution.png
├── gumbel_topk_MSE_weight_evolution.png
├── gumbel_topk_FOCAL_weight_evolution.png
├── gumbel_topk_HINGE_weight_evolution.png
├── gumbel_topk_RANKING_weight_evolution.png
├── gumbel_topk_CONTRASTIVE_weight_evolution.png
├── gumbel_topk_DICE_weight_evolution.png
├── gumbel_topk_AUC_weight_evolution.png
└── gumbel_topk_BCE+RANKING_weight_evolution.png
```

### 文件命名规则

```
gumbel_topk_{LOSS_TYPE}_weight_evolution.png

示例:
  - gumbel_topk_BCE_weight_evolution.png
  - gumbel_topk_FOCAL_weight_evolution.png
  - ...
```

---

## 1️⃣1️⃣ 运行方式

### 命令

```bash
cd /mnt/shared-storage-user/tangjia/bilevel-cmil
python test_topk_demo.py
```

### 运行时间
- 估计: 5-10 分钟（取决于硬件）

### 日志输出
```
====================================================
可微分 Top-K Selection 完整测试
====================================================

测试 4: 可视化训练过程中权重 w 的分布演化 (Loss: BCE)
        (使用 GumbelTopK 方法)
====================================================

任务设置:
  - 总维度: 100
  - Top-K: 20
  - 目标: 选中前20个位置
  - 迭代次数: 100
  - 损失函数: BCE

开始训练...
  Epoch  20: Loss=0.6234
  Epoch  40: Loss=0.4821
  ...

✓ 权重排序演化可视化已保存到: gumbel_topk_analysis/gumbel_topk_BCE_weight_evolution.png
```

---

## 1️⃣2️⃣ 参数调整指南

### 如果想要修改实验设置

```python
# 修改数据大小
n = 100          # 总维度
k = 20           # Top-K 数量

# 修改 GumbelTopK 参数
temperature=0.5      # 越小输出越"硬"
gumbel_noise=1.0     # 越大噪声越强

# 修改优化器参数
lr=5.0               # 初始学习率
step_size=30         # 学习率衰减周期
gamma=0.5            # 学习率衰减因子

# 修改采样时间点
snapshot_epochs = [0, 50, 99]  # 3 个关键时刻：初始、中期、最终

# 修改图表参数（在 test_4 函数中）
figsize=(7.5, 2.4)   # 1×3 布局，单栏论文格式 (宽, 高)
# 调整字体大小
fontsize_title = 8       # 主标题
fontsize_subtitle = 7    # 子图标题
fontsize_label = 6.5     # 坐标轴标签
fontsize_tick = 5        # 刻度标签
fontsize_legend = 6      # 图例
```

---

## 1️⃣3️⃣ 相关文件

- **主脚本**: `test_topk_demo.py`
- **模型代码**: `tools/topk_selection.py`
- **输出目录**: `gumbel_topk_analysis/`
- **本文档**: `GUMBEL_TOPK_EXPERIMENT.md`

---

---

## 📐 单栏论文格式设计说明

### 图表尺寸优化

```
设计目标: 适配双栏论文中的单栏宽度

尺寸配置:
  - 图表宽度: 7.5 英寸 (≈ 3.5 英寸单栏宽度的 2.1 倍)
  - 图表高度: 2.4 英寸 (紧凑高度，节省页面空间)
  - 子图数量: 3 个 (1 行 × 3 列)
  - 每个子图约: ~2.5 × 2.4 英寸

在论文中的显示:
  - 如果设置为单栏宽度: 图表会自动缩放到 ~3.5 英寸宽
  - 高度会相应缩放到 ~1.35 英寸
  - 保持所有文字清晰可读
  - 三个训练阶段横向排列，简洁展示权重演化
```

### 字体优化

```
设计原则: 论文级别的清晰度与可读性

字体大小对应关系:
  ┌─ 主标题 (8pt)
  │  └─ 论文图表通常要求 10-12pt
  │     在缩放到 3.5 英寸时约 4-5pt (可读)
  │
  ├─ 子图标题 (7pt)  → 缩放后 3.5-4pt (清晰)
  ├─ 坐标轴标签 (6.5pt) → 缩放后 3.3pt (清晰)
  ├─ 刻度标签 (5pt)  → 缩放后 2.5pt (勉强可读)
  └─ 图例 (6pt)   → 缩放后 3pt (清晰)

优化建议:
  - 所有文字均未使用粗体，保持简洁
  - 标签文字简化 ("Sorted Index" → "Index")
  - 主标题简化 ("Weight Distribution Evolution" → "Weight Evolution")
  - 这样在印刷版本中会更加清晰
```

### 论文中的使用

1. **插入到 LaTeX 文档**:
```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.9\columnwidth]{gumbel_topk_BCE_weight_evolution.png}
  \caption{GumbelTopK 权重演化过程（BCE 损失）}
  \label{fig:gumbel_topk_bce}
\end{figure}
```

2. **调整宽度**:
```latex
% 如果需要更小的宽度
\includegraphics[width=0.8\columnwidth]{...}

% 如果需要双栏宽度
\includegraphics[width=0.45\textwidth]{...}  % 放在 minipage 中
```

3. **多个图表排列**:
```latex
\begin{figure}
  \begin{minipage}{0.9\columnwidth}
    \includegraphics[width=\columnwidth]{gumbel_topk_BCE_weight_evolution.png}
  \end{minipage}\\
  \begin{minipage}{0.9\columnwidth}
    \includegraphics[width=\columnwidth]{gumbel_topk_MSE_weight_evolution.png}
  \end{minipage}
  \caption{不同损失函数的权重演化对比}
\end{figure}
```

---

**最后更新**: 2025-01-12
