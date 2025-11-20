# BCSR 超参数配置说明

本文档说明了 BCSR (Bilevel Coreset Selection and Refinement) 算法中所有可配置的超参数。

## 超参数列表

### 1. 核心优化参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `bcsr_lr_proxy_model` | float | 5.0 | 代理模型的学习率 |
| `bcsr_beta` | float | 0.1 | 正则化系数，平衡损失和正则项 |
| `bcsr_max_outer_it` | int | 10 | 外层优化的最大迭代次数 |
| `bcsr_max_inner_it` | int | 1 | 内层优化的最大迭代次数 |
| `bcsr_weight_lr` | float | 0.05 | 样本权重的学习率 |

### 2. 知识蒸馏参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `distall_lamda` | float | 0.1 | 知识蒸馏损失的系数 |
| `distill_target` | str | 'logits' | 知识蒸馏的目标对象 |

**distill_target 可选值：**
- `'logits'`: 只蒸馏输出层的 logits（使用 KL 散度）
- `'features'`: 只蒸馏中间层的 features（使用 MSE 损失）
- `'both'`: 同时蒸馏 logits 和 features（总损失 = KL散度 + MSE）

### 3. TopK 选择器参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `topk_method` | str | 'sigmoid' | TopK 选择方法 |
| `topk_temperature` | float | 0.1 | TopK 温度参数，控制软选择的平滑度 |

**topk_method 可选值：**
- `'sigmoid'`: Sigmoid 软选择方法
- `'gumbel'`: Gumbel-Softmax 方法
- `'ste'`: Straight-Through Estimator 方法

### 4. 权重归一化参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `normalize_method` | str | 'none' | 权重归一化方法 |

**normalize_method 可选值：**
- `'none'`: 不进行归一化
- `'l2'`: L2 归一化（除以 L2 范数）
- `'softmax'`: Softmax 归一化（所有权重和为1）
- `'zscore'`: Z-score 标准化（均值0，标准差1）

### 5. 权重约束参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use_simplex_projection` | bool | False | 是否使用单纯形投影 |

**单纯形投影说明：**
- 当设置为 `True` 时，会将权重投影到单纯形上（所有权重和为1，且非负）
- 投影会在权重初始化后和每次梯度更新后执行
- 可以保证权重满足概率分布的约束

### 6. 其他参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `draw_curve` | bool | False | 是否绘制训练曲线 |
| `buffer_slide_size` | int | 50 | Buffer 中每个 slide 的 patch 数量 |

## 使用方法

### 1. 通过 YAML 配置文件

编辑 `configs/csc_clam_cl_debug_buffer_ana.yaml`:

```yaml
# BCSR Coreset hyperparameters
bcsr_lr_proxy_model: 5
bcsr_beta: 0.1
bcsr_max_outer_it: 10
bcsr_max_inner_it: 1
bcsr_weight_lr: 0.05
distall_lamda: 0.1
topk_method: 'sigmoid'
topk_temperature: 0.1
normalize_method: 'none'
use_simplex_projection: False
distill_target: 'logits'
```

### 2. 通过命令行参数

```bash
python main_cl_buffer_analysis.py \
    --preset configs/csc_clam_cl_debug_buffer_ana.yaml \
    --bcsr_lr_proxy_model 5 \
    --bcsr_beta 0.1 \
    --use_simplex_projection True \
    --distill_target features
```

### 3. 超参数搜索

使用 `run_bcsr_convergence_search.py` 进行网格搜索：

```bash
# 快速测试（8个组合）
python run_bcsr_convergence_search.py \
    --gpus 0,1 \
    --lr-proxy 5,10 \
    --beta 0.1 \
    --topk-method sigmoid,gumbel \
    --normalize-method none,l2 \
    --use-simplex-projection False \
    --distill-target logits

# 对比单纯形投影和蒸馏目标（6个组合）
python run_bcsr_convergence_search.py \
    --gpus 0,1,2,3 \
    --lr-proxy 5 \
    --beta 0.1 \
    --topk-method sigmoid \
    --normalize-method none \
    --use-simplex-projection False,True \
    --distill-target logits,features,both
```

## 超参数调优建议

### 学习率相关参数

1. **bcsr_lr_proxy_model (1-10)**
   - 较小值 (1-3): 更稳定但收敛慢
   - 较大值 (7-10): 收敛快但可能不稳定
   - 推荐: 5

2. **bcsr_weight_lr (0.01-0.1)**
   - 权重更新步长，通常小于代理模型学习率
   - 推荐: 0.05

### 迭代次数参数

1. **bcsr_max_outer_it (5-20)**
   - 控制双层优化的外层迭代次数
   - 更多迭代会提高收敛质量但增加计算时间
   - 推荐: 10

2. **bcsr_max_inner_it (1-5)**
   - 控制代理模型训练的内层迭代次数
   - 通常设为1就足够
   - 推荐: 1

### TopK 选择器参数

1. **topk_method**
   - `sigmoid`: 平滑可微，梯度稳定（推荐）
   - `gumbel`: 引入随机性，可能帮助探索
   - `ste`: 直通估计器，梯度近似

2. **topk_temperature (0.05-0.5)**
   - 较小值: 更接近硬选择（离散）
   - 较大值: 更平滑的软选择
   - 推荐: 0.1

### 权重归一化参数

1. **normalize_method**
   - `none`: 不归一化，保留原始权重分布（推荐开始尝试）
   - `l2`: 控制权重的尺度
   - `softmax`: 权重和为1，类似概率分布
   - `zscore`: 标准化，适合处理异常值

### 知识蒸馏参数

1. **distill_target**
   - `logits`: 保留输出层的知识（推荐开始尝试）
   - `features`: 保留中间层的表示
   - `both`: 最全面但可能过约束

2. **distall_lamda (0.1-0.5)**
   - 控制蒸馏损失的权重
   - 推荐: 0.1

### 单纯形投影

1. **use_simplex_projection**
   - `False`: 允许权重自由更新（推荐开始尝试）
   - `True`: 强制权重满足单纯形约束（和为1，非负）

## 收敛性指标

搜索脚本会自动提取以下收敛指标：

- **score**: 综合收敛得分 (0-1)
- **status**: 收敛状态 (Excellent/Good/Fair/Poor/Very Poor)
- **decrease**: 损失下降率
- **monotonicity**: 单调性得分
- **stability**: 稳定性得分
- **oscillation**: 振荡得分

## 故障排查

### 收敛不佳

1. 降低学习率 (`bcsr_lr_proxy_model`, `bcsr_weight_lr`)
2. 增加迭代次数 (`bcsr_max_outer_it`)
3. 尝试 `use_simplex_projection=True`
4. 调整 `normalize_method`

### 训练不稳定

1. 降低温度参数 (`topk_temperature`)
2. 使用 `topk_method='sigmoid'`
3. 使用 `normalize_method='l2'` 或 `'softmax'`

### 蒸馏损失过大

1. 降低 `distall_lamda`
2. 改变 `distill_target` (尝试只用 `'logits'` 或 `'features'`)

## 参考文档

- `run_bcsr_search_example.sh`: 包含6个示例场景
- `run_bcsr_convergence_search.py`: 网格搜索脚本
- `PATCH_SELECTION_ANALYSIS_README.md`: Patch 选择分析文档
