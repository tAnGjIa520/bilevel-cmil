# 实例选择方法对比实验 - 使用指南

## 📁 文件说明

- `attention_mil_harder_tasks111.py`: 主程序，支持三种实例选择方法
- `README_instance_selection_methods.md`: 方法原理和理论背景

## 🚀 快速开始

### 1. 使用Attention方法（默认，速度快）

```bash
python demo/attention_mil_harder_tasks111.py \
    --task triplet \
    --selection_method attention \
    --top_k 3 --bottom_k 2 \
    --epochs1 20 --epochs2 20 \
    --num_bags 500
```

### 2. 使用KIBO双层优化方法（理论最优）

```bash
python demo/attention_mil_harder_tasks111.py \
    --task triplet \
    --selection_method kibo \
    --top_k 5 \
    --kibo_outer_it 10 \
    --kibo_inner_it 2 \
    --kibo_lr_weight 0.1 \
    --epochs1 20 --epochs2 20 \
    --num_bags 500
```

### 3. 使用Random随机方法（baseline）

```bash
python demo/attention_mil_harder_tasks111.py \
    --task triplet \
    --selection_method random \
    --top_k 5 \
    --epochs1 20 --epochs2 20 \
    --num_bags 500
```

### 4. 对比所有三种方法（推荐！）

```bash
python demo/attention_mil_harder_tasks111.py \
    --task triplet \
    --selection_method compare \
    --top_k 5 --bottom_k 2 \
    --epochs1 30 --epochs2 30 \
    --num_bags 1000
```

## 📊 命令行参数详解

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | `triplet` | 任务类型: `triplet`, `sum`, `increasing`, `large_range`, `all` |
| `--selection_method` | `attention` | 选择方法: `attention`, `kibo`, `random`, `compare` |
| `--epochs1` | 50 | 第一阶段（原始数据）训练轮数 |
| `--epochs2` | 50 | 第二阶段（过滤数据）训练轮数 |
| `--num_bags` | 2000 | 训练集包数量 |

### Attention方法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--top_k` | 3 | 选择最高attention的实例数 |
| `--bottom_k` | 2 | 选择最低attention的实例数 |

### KIBO方法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--top_k` | 3 | 选择的实例总数 |
| `--kibo_outer_it` | 10 | 外层优化迭代次数 |
| `--kibo_inner_it` | 2 | 内层训练迭代次数 |
| `--kibo_lr_weight` | 0.1 | 权重学习率 |

### Random方法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--top_k` | 3 | 随机选择的实例数 |

## 🎯 任务类型说明

### 1. `triplet` - 三元组检测（推荐入门）
- **正包**: 存在连续三元组 (如 3,4,5)
- **负包**: 不存在连续三元组
- **难度**: 中等
- **推荐包大小**: 8-25

### 2. `sum` - 和为10检测
- **正包**: 存在两数字和为10 (如 3+7)
- **负包**: 不存在这样的数字对
- **难度**: 中等
- **推荐包大小**: 5-20

### 3. `increasing` - 递增序列检测
- **正包**: 存在3+个数字的递增序列
- **负包**: 不存在这样的序列
- **难度**: 较难
- **推荐包大小**: 5-20

### 4. `large_range` - 大范围数字检测
- **正包**: 存在连续数字对 (0-99)
- **负包**: 不存在连续数字对
- **难度**: 难（输入维度100）
- **推荐包大小**: 5-20

## 📈 实验示例与预期结果

### 示例1: 快速测试（小数据集）

```bash
python demo/attention_mil_harder_tasks111.py \
    --task triplet \
    --selection_method compare \
    --epochs1 10 --epochs2 10 \
    --num_bags 200 \
    --top_k 4
```

**预期输出**:
- Attention: AUC ~0.95, 标签不一致率 ~30%
- KIBO: AUC ~0.96, 标签不一致率 ~25%
- Random: AUC ~0.90, 标签不一致率 ~45%

**运行时间**: ~5分钟

### 示例2: 标准实验（中等数据集）

```bash
python demo/attention_mil_harder_tasks111.py \
    --task triplet \
    --selection_method compare \
    --epochs1 30 --epochs2 30 \
    --num_bags 1000 \
    --top_k 5
```

**预期输出**:
- Attention: AUC ~0.97, 标签不一致率 ~25%
- KIBO: AUC ~0.98, 标签不一致率 ~20%
- Random: AUC ~0.92, 标签不一致率 ~40%

**运行时间**: ~20分钟

### 示例3: 完整实验（大数据集）

```bash
python demo/attention_mil_harder_tasks111.py \
    --task triplet \
    --selection_method compare \
    --epochs1 50 --epochs2 50 \
    --num_bags 2000 \
    --top_k 5 --bottom_k 2
```

**预期输出**:
- Attention: AUC ~0.98, 标签不一致率 ~20%
- KIBO: AUC ~0.99, 标签不一致率 ~15%
- Random: AUC ~0.93, 标签不一致率 ~38%

**运行时间**: ~40分钟

## 🔍 输出解读

### 标签一致性检查

```
标签一致性检查:
  总包数: 1000
  标签不一致: 250 (25.00%)
    - 正包→负包 (丢失连续对): 200 (20.00%)
    - 负包→正包 (意外产生连续对): 50 (5.00%)
  标签一致: 750 (75.00%)
```

**关键指标**:
- **总不一致率**: 越低越好，说明选择的实例保留了原包的语义
- **正包→负包**: 丢失了关键实例
- **负包→正包**: 意外选中了形成模式的实例组合

### 性能对比

```
结果对比:
第一阶段 (原始数据):
  Accuracy: 0.9500, AUC: 0.9650, F1: 0.9480
第二阶段 (Attention过滤):
  Accuracy: 0.9600, AUC: 0.9750, F1: 0.9580
性能变化:
  Accuracy: +0.0100
  AUC: +0.0100
  F1: +0.0100
```

**理想情况**: 第二阶段性能≥第一阶段
**实际情况**:
- 如果选择方法好，性能应该持平或提升
- 如果标签不一致率太高，性能可能下降

## 💡 调参建议

### Attention方法
- `top_k`: 3-5（选太少丢失信息，选太多失去筛选意义）
- `bottom_k`: 1-3（提供对比信息，不宜太多）
- 适合快速实验和原型验证

### KIBO方法
- `top_k`: 4-6（总选择数量）
- `kibo_outer_it`: 10-20（越多越好，但计算慢）
- `kibo_inner_it`: 2-5（太多容易过拟合）
- `kibo_lr_weight`: 0.05-0.2（太大不稳定，太小收敛慢）
- 适合追求最优性能

### Random方法
- `top_k`: 4-6
- 仅作为baseline对比

## ⚠️ 注意事项

1. **内存使用**: KIBO方法会复制模型，需要更多GPU内存
2. **计算时间**: KIBO >> Attention >> Random
3. **参数敏感性**: KIBO对学习率和迭代次数较敏感
4. **数据集大小**: 小数据集上KIBO可能过拟合
5. **随机种子**: 固定为42，保证可复现

## 🐛 常见问题

### Q1: KIBO方法运行很慢？
A: 正常现象。可以减少 `--kibo_outer_it` 或使用更少的包 `--num_bags`

### Q2: 标签不一致率很高怎么办？
A:
- 增加 `--top_k` 保留更多实例
- 使用KIBO方法（理论上不一致率更低）
- 检查任务定义是否合理

### Q3: 第二阶段性能下降？
A: 可能原因：
- 标签不一致率过高
- 选择的实例数太少
- 第二阶段训练不充分（增加 `--epochs2`）

### Q4: GPU内存不足？
A:
- 减少 `--num_bags`
- 减少 batch_size（需修改代码）
- 使用 Random 或 Attention 方法

## 📝 实验记录建议

建议记录以下信息：
- 任务类型和选择方法
- 所有命令行参数
- 标签不一致率
- 两阶段性能对比
- 运行时间

示例记录格式：
```
实验: Triplet + KIBO
参数: top_k=5, outer_it=15, num_bags=1000
结果:
  - 标签不一致: 18%
  - Stage1 AUC: 0.965
  - Stage2 AUC: 0.982 (+0.017)
  - 时间: 25min
```

## 🎓 扩展实验

1. **参数搜索**: 尝试不同的 `top_k`, `outer_it` 组合
2. **任务对比**: 在所有4个任务上运行对比
3. **数据规模**: 测试不同 `num_bags` 的影响
4. **消融实验**: 只用top-k不用bottom-k（设置 `--bottom_k 0`）

祝实验顺利！ 🚀
