# Patch Selection Analysis Tool

这个工具用于分析在第一个任务结束后，每个bag中patch挑选的详细过程。

## 文件说明

- **main_cl_analyze.py**: 修改后的主程序，包含patch selection分析功能
- **原始文件**: main_cl.py (保持不变)

## 主要功能

### 1. PatchSelectionAnalyzer 类
负责记录和分析每个bag的patch挑选过程：

**记录的信息**：
- Bag索引和Slide ID
- 样本标签
- 原始patch数量 vs 挑选后patch数量
- 挑选比例
- 挑选方法（random, max, maxrand, kibo, mix等）
- 注意力权重统计（mean, std, max, min）
- 被选中的patch索引
- 挑选耗时
- KIBO方法的特有信息（如果使用）

### 2. 修改后的 distill_slide() 函数

新增 `return_analysis` 参数：
- `False` (默认): 向后兼容，仅返回选中的patches
- `True`: 返回详细的分析字典，包含：
  - `selected_patches`: 选中的patch特征
  - `indices`: 被选中的patch索引
  - `analysis_info`: 分析信息
  - `kibo_info`: KIBO方法的额外信息（如果适用）
  - `attn_weights`: 注意力权重

### 3. 集成到 Buffer Selection 阶段

在buffer selection过程中（第一个任务结束后），自动：
1. 初始化 `PatchSelectionAnalyzer`
2. 为每个添加到buffer的bag记录patch挑选信息
3. 实时打印进度
4. 任务结束时保存分析结果

## 使用方法

### 运行分析版本

```bash
# 使用与原版相同的命令，只需将 main_cl.py 替换为 main_cl_analyze.py
python main_cl_analyze.py --preset your_config.yaml
```

### 输出文件

程序会在每个任务的日志目录下生成以下文件：

```
{log_dir}/{exp_name}/fold_{fold}_task_{task}/
├── patch_selection_summary.csv      # 统计摘要（CSV格式）
├── patch_selection_details.json     # 详细信息（JSON格式）
└── buffer_labels.csv                 # 原有的buffer类别分布
```

### 输出示例

#### 1. patch_selection_summary.csv
```csv
bag_idx,slide_id,label,original_patches,selected_patches,selection_ratio,method,attn_mean,attn_std,attn_max,attn_min,selection_time
0,slide_001,1,5000,500,0.1,kibo,0.0012,0.0008,0.0145,0.0001,2.345
1,slide_002,0,4800,480,0.1,kibo,0.0011,0.0007,0.0132,0.0002,2.187
...
```

#### 2. patch_selection_details.json
```json
[
  {
    "bag_idx": 0,
    "slide_id": "slide_001",
    "label": 1,
    "original_patches": 5000,
    "selected_patches": 500,
    "selection_ratio": 0.1,
    "method": "kibo",
    "attn_mean": 0.0012,
    "attn_std": 0.0008,
    "attn_max": 0.0145,
    "attn_min": 0.0001,
    "selected_indices": [234, 567, 891, ...],
    "selection_time": 2.345,
    "kibo_info": {
      "max_outer_it": 50,
      "max_inner_it": 1,
      "lr_proxy_model": 0.0001,
      "weight_lr": 0.1,
      "beta": 0.5
    }
  },
  ...
]
```

#### 3. 控制台输出示例

```
============================================================
  Patch Selection Analyzer Initialized
  Fold: 0 | Task: 0
  Save dir: logs/exp_name/fold_0_task_0
============================================================

  [KIBO] Starting selection: 5000 -> 500 patches
  ├─ Inner[1/1]: 0.2345 ✓
  └─ Outer[50/50]: 0.1234 ✓
inner loss:0.234, outer loss:0.123
  [KIBO] Selection complete
  [Analyzer] Recorded 10 bags | Last: 5000->500 patches (kibo)
  ...
  [Analyzer] Recorded 100 bags | Last: 4800->480 patches (kibo)

[Analyzer] Summary saved to: logs/exp_name/fold_0_task_0/patch_selection_summary.csv

============================================================
  Patch Selection Statistics
============================================================
  Total bags analyzed: 156
  Selection method: kibo
  Avg original patches: 4850.2
  Avg selected patches: 485.0
  Avg selection ratio: 10.00%
  Avg attention weight: 0.0011
  Avg selection time: 2.156s

  Statistics by class:
    Class 0: 78 bags, 482.5 patches (avg), 9.95% ratio (avg)
    Class 1: 78 bags, 487.5 patches (avg), 10.05% ratio (avg)
============================================================

[Analyzer] Details saved to: logs/exp_name/fold_0_task_0/patch_selection_details.json
```

## 分析要点

### 从Task 0结束后开始分析

修改后的代码专注于分析第一个任务（Task 0）训练完成后的patch挑选过程。关键阶段：

1. **训练完成**: 模型在Task 0上训练完毕
2. **加载最佳模型**: 加载验证集上表现最好的模型权重
3. **Buffer Selection开始**:
   - 初始化 `PatchSelectionAnalyzer`
   - 遍历训练集的每个bag
   - 使用模型提取注意力权重
   - 调用 `distill_slide()` 挑选代表性patches
   - 记录挑选过程的详细信息
4. **保存分析结果**: 生成CSV和JSON文件

### 支持的挑选方法

所有方法都被分析记录：
- **random**: 随机挑选
- **max**: 选择注意力权重最高的patches
- **maxmin**: 选择最高和最低注意力权重的patches
- **maxrand**: 混合最高注意力和随机选择
- **kibo**: 使用双层优化的coreset selection（会记录额外的优化信息）
- **mix**: 先用maxrand粗选，再用kibo精选

### KIBO方法的特殊输出

如果使用KIBO或mix方法，会额外记录：
- Inner loop和Outer loop的迭代次数
- 学习率参数
- Beta参数（正则化系数）
- 优化过程的损失值

## 注意事项

1. **向后兼容**: 修改不影响原有功能，`distill_slide()`的默认行为保持不变
2. **性能影响**: 分析模式会略微增加运行时间（主要来自数据记录，不影响核心算法）
3. **存储空间**: JSON文件可能较大（每个bag存储最多100个索引）
4. **仅在有buffer时运行**: 只有在 `args.buffer_size > 0` 且 `args.cl_method != 'joint'` 时才会进行分析

## 后续分析建议

使用生成的数据可以进行：

1. **挑选效率分析**: 不同方法的平均挑选时间对比
2. **注意力分布研究**: 被选中vs未选中patches的注意力权重分布
3. **类别差异分析**: 不同类别样本的patch挑选模式
4. **KIBO优化过程**: 双层优化的收敛情况
5. **可视化**:
   - 注意力权重热图
   - 挑选比例分布
   - 类别间的patch数量对比

## 示例分析脚本

```python
import pandas as pd
import json
import matplotlib.pyplot as plt

# 读取数据
summary = pd.read_csv('logs/exp_name/fold_0_task_0/patch_selection_summary.csv')
with open('logs/exp_name/fold_0_task_0/patch_selection_details.json') as f:
    details = json.load(f)

# 分析1: 挑选时间统计
print(f"Average selection time: {summary['selection_time'].mean():.3f}s")
print(f"Total selection time: {summary['selection_time'].sum():.2f}s")

# 分析2: 按类别统计
for label in summary['label'].unique():
    label_data = summary[summary['label'] == label]
    print(f"Class {label}: {len(label_data)} bags, "
          f"avg {label_data['selected_patches'].mean():.1f} patches")

# 分析3: 注意力权重分布
plt.figure(figsize=(10, 6))
for label in summary['label'].unique():
    label_data = summary[summary['label'] == label]
    plt.hist(label_data['attn_mean'], alpha=0.5, label=f'Class {label}', bins=20)
plt.xlabel('Mean Attention Weight')
plt.ylabel('Frequency')
plt.legend()
plt.title('Attention Weight Distribution by Class')
plt.savefig('attention_distribution.png')
```

## 问题排查

如果遇到问题：

1. **没有生成分析文件**: 检查是否满足条件：
   - `args.buffer_size > 0`
   - `args.buffer_slide_size > 0`
   - `args.cl_method != 'joint'`

2. **内存不足**: 对于大规模数据集，可以修改 `PatchSelectionAnalyzer.record_selection()` 中的 `[:100]` 限制，减少保存的索引数量

3. **运行时间过长**: KIBO方法的分析会增加少量时间，可以先用简单方法（如random或maxrand）进行测试

## 版本历史

- v1.0 (2025-01-12): 初始版本，支持完整的patch selection分析功能
