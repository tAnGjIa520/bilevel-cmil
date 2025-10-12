# 实例选择方法对比实验

本文档说明在 `attention_mil_harder_tasks copy.py` 中实现的三种实例选择方法。

## 🎯 三种实例选择方法

### 1. **Attention-based Selection** (原方法)
**原理**: 使用训练好的模型的attention权重来选择实例
- 选择top-k个最高attention的实例
- 选择bottom-k个最低attention的实例
- 简单快速，依赖模型学到的注意力机制

**优点**:
- 计算效率高
- 直观，基于模型已学习的特征重要性

**缺点**:
- 依赖attention质量
- 可能保留不相关的低attention实例

### 2. **KIBO Bilevel Optimization** (双层优化)
**原理**: 通过双层优化学习最优的实例权重
- **内层优化**: 在加权实例上训练代理模型，最小化加权损失
- **外层优化**: 优化实例权重，使验证损失最小
- 通过隐式梯度计算权重梯度

**核心思想**:
```
min_{w} L_val(θ*(w))
s.t. θ*(w) = argmin_{θ} L_train(θ, w)
```

**优点**:
- 理论更优，直接优化验证性能
- 不依赖attention质量
- 考虑实例间的相互作用

**缺点**:
- 计算开销大（需要多次迭代）
- 参数敏感（学习率、迭代次数）

### 3. **Random Selection** (随机抽取baseline)
**原理**: 从每个包中随机选择固定数量的实例

**优点**:
- 无偏，不依赖模型
- 计算最快
- 作为baseline对比

**缺点**:
- 可能丢失关键实例
- 性能通常较差

## 📊 代码结构

### 核心类和函数

```python
# 双层优化选择器
class BilevelInstanceSelection:
    def __init__(self, model, lr_model, lr_weight, max_outer_it, max_inner_it, device)
    def train_inner(self, data, labels, weights)  # 内层训练
    def train_outer(self, data, labels, weights, topk)  # 外层优化
    def select_instances(self, data, label, topk)  # 选择实例

# 统一接口
def extract_instances(model, dataset, device, check_label_fn, selection_method, ...)

# 三种实现
def extract_attention_based_instances(...)
def extract_kibo_based_instances(...)
def extract_random_instances(...)
```

## 🚀 使用示例

```python
# 方法1: Attention-based
bags, labels, stats = extract_instances(
    model, dataset, device, check_fn,
    selection_method='attention',
    top_k=3, bottom_k=2
)

# 方法2: KIBO
bags, labels, stats = extract_instances(
    model, dataset, device, check_fn,
    selection_method='kibo',
    top_k=5,
    bilevel_params={'max_outer_it': 10, 'max_inner_it': 2, 'lr_weight': 0.1}
)

# 方法3: Random
bags, labels, stats = extract_instances(
    model, dataset, device, check_fn,
    selection_method='random',
    top_k=5
)
```

## 📈 预期实验结果

### 标签不一致率
- **Random**: 最高（随机选择容易丢失关键实例）
- **Attention**: 中等（取决于attention质量）
- **KIBO**: 可能最低（优化目标直接相关）

### 第二阶段性能
- **Random**: baseline性能
- **Attention**: 如果attention学得好，性能较好
- **KIBO**: 理论上最优，但受参数影响

### 计算时间
- **Random**: 最快（~1秒/千包）
- **Attention**: 快（~5秒/千包）
- **KIBO**: 慢（~60秒/千包，取决于迭代次数）

## 🔬 关键参数说明

### Attention方法
- `top_k`: 选择最高attention的实例数（默认3）
- `bottom_k`: 选择最低attention的实例数（默认2）

### KIBO方法
- `top_k`: 选择的实例总数（默认5）
- `max_outer_it`: 外层迭代次数（默认10-20）
- `max_inner_it`: 内层迭代次数（默认2-3）
- `lr_weight`: 权重学习率（默认0.1）
- `lr_model`: 模型学习率（默认0.01）

### Random方法
- `num_select`: 随机选择的实例数（默认5）

## 🎓 理论背景

### KIBO双层优化
KIBO源自coreset selection和bilevel optimization研究：

1. **Coreset选择**: 从大数据集中选择代表性子集
2. **双层优化**:
   - 内层：模型参数优化
   - 外层：样本权重优化
3. **隐式梯度**: 使用链式法则计算权重对验证损失的梯度

### Attention机制
Attention-MIL假设：
- 模型学习为重要实例分配高权重
- Top attention实例是正样本的关键
- Bottom attention实例提供对比信息

## 💡 使用建议

1. **快速原型**: 使用Random作为baseline
2. **平衡性能**: 使用Attention（速度快，效果好）
3. **最优性能**: 使用KIBO（计算慢，但理论最优）
4. **小数据集**: KIBO可能过拟合，使用Attention
5. **大数据集**: KIBO优势明显，但计算成本高

## 📝 待完成工作

- [ ] 在run_experiment函数中集成三种方法
- [ ] 添加命令行参数 `--selection_method`
- [ ] 实现三种方法的对比实验
- [ ] 绘制性能对比图表
- [ ] 分析标签不一致率与最终性能的关系
