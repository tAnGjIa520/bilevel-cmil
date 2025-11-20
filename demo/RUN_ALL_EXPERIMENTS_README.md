# 并行运行所有实验组合

## 概述

`run_all_experiments.py` 和 `run_all_experiments.sh` 用于并行运行所有实验组合（3种方法 × 4个任务 = 12个实验），并自动汇总标签不一致率结果。

## 实验组合

**方法 (Methods):**
- `attention`: 基于注意力的实例选择
- `kibo`: 基于双层优化的实例选择
- `random`: 随机实例选择

**任务 (Tasks):**
- `large_range`: 扩展数字范围检测 (0-99连续对)
- `triplet`: 三元组检测 (连续三数字)
- `sum`: 和为10检测 (两数字和)
- `increasing`: 递增序列检测 (3+个数字)

**总计**: 12 个实验组合

## 快速开始

### 方法1: 使用Shell脚本（推荐）

```bash
# 使用默认参数
./demo/run_all_experiments.sh

# 或者自定义参数
GPUS="0,1,2,3" NEW_SIZE=15 NUM_BAGS=2000 ./demo/run_all_experiments.sh
```

### 方法2: 直接使用Python脚本

```bash
python demo/run_all_experiments.py \
    --gpus 0,1,2,3,4,5,6,7 \
    --new_size 10 \
    --bag_size_min 20 \
    --bag_size_max 50 \
    --num_bags 1000 \
    --epochs1 10 \
    --epochs2 10 \
    --log_dir logs/my_experiments
```

## 参数说明

### GPU和输出配置
- `--gpus`: 可用的GPU ID列表，逗号分隔（默认: 0,1,2,3,4,5,6,7）
- `--log_dir`: 日志文件保存目录（默认: logs/all_experiments_[时间戳]）
- `--output_csv`: 结果汇总CSV文件名（默认: mismatch_summary.csv）

### 实验参数
- `--new_size`: 新包大小，选择的实例数量（默认: 10）
- `--bag_size_min`: 原始包大小最小值（默认: 20）
- `--bag_size_max`: 原始包大小最大值（默认: 50）
- `--num_bags`: 训练集包数量（默认: 1000）
- `--epochs1`: 第一阶段训练轮数（默认: 10）
- `--epochs2`: 第二阶段训练轮数（默认: 10）
- `--kibo_outer_it`: KIBO外层迭代次数（默认: 10）
- `--kibo_inner_it`: KIBO内层迭代次数（默认: 2）
- `--kibo_lr_weight`: KIBO权重学习率（默认: 0.1）

## 工作原理

1. **任务分配**: 脚本会自动将12个实验分配到可用的GPU上
2. **并行执行**: 使用GPU池，每个GPU同时运行一个实验
3. **自动排队**: 当某个GPU完成任务后，自动从队列中取下一个实验
4. **日志记录**: 每个实验的完整输出保存到独立的日志文件
5. **结果解析**: 自动从日志中提取标签不一致率数据
6. **汇总生成**: 生成CSV表格和透视表

## 输出文件

运行完成后，会在 `log_dir` 目录下生成以下文件：

```
logs/all_experiments_20250113_143022/
├── large_range_attention.log      # 各实验的详细日志
├── large_range_kibo.log
├── large_range_random.log
├── triplet_attention.log
├── triplet_kibo.log
├── triplet_random.log
├── sum_attention.log
├── sum_kibo.log
├── sum_random.log
├── increasing_attention.log
├── increasing_kibo.log
├── increasing_random.log
├── mismatch_summary.csv           # 完整汇总表
└── mismatch_pivot.csv             # 透视表（方法×任务）
```

## 结果示例

### 完整汇总表 (mismatch_summary.csv)

```
Task         Method      Mismatch Rate (%)  Pos→Neg  Neg→Pos  Log File
large_range  attention   12.50              62       63       logs/.../large_range_attention.log
large_range  kibo        8.30               40       43       logs/.../large_range_kibo.log
large_range  random      15.20              75       77       logs/.../large_range_random.log
triplet      attention   10.10              50       51       logs/.../triplet_attention.log
...
```

### 透视表 (mismatch_pivot.csv)

```
Method       attention  kibo   random
Task
large_range  12.50      8.30   15.20
triplet      10.10      7.50   13.80
sum          9.20       6.90   12.40
increasing   11.30      8.10   14.60
```

### 统计摘要

脚本会自动输出：
- 平均不一致率
- 最高/最低不一致率及对应的实验
- 各方法的平均不一致率
- 各任务的平均不一致率

## 监控进度

运行时会实时显示进度：

```
[GPU 0] 启动实验: large_range_attention
[GPU 1] 启动实验: large_range_kibo
[GPU 2] 启动实验: large_range_random
...
[GPU 0] ✓ 完成: large_range_attention
[GPU 3] 启动实验: sum_attention
...
进度: 8/12 | 运行中: sum_kibo, increasing_random
```

## 示例场景

### 场景1: 快速测试（小规模数据）

```bash
python demo/run_all_experiments.py \
    --gpus 0,1,2,3 \
    --num_bags 500 \
    --epochs1 5 \
    --epochs2 5 \
    --new_size 8
```

### 场景2: 完整实验（大规模数据）

```bash
python demo/run_all_experiments.py \
    --gpus 0,1,2,3,4,5,6,7 \
    --num_bags 2000 \
    --epochs1 50 \
    --epochs2 50 \
    --new_size 15 \
    --bag_size_min 30 \
    --bag_size_max 60
```

### 场景3: 对比不同包大小

```bash
# 小包
python demo/run_all_experiments.py --new_size 5 --log_dir logs/small_bags

# 中包
python demo/run_all_experiments.py --new_size 10 --log_dir logs/medium_bags

# 大包
python demo/run_all_experiments.py --new_size 20 --log_dir logs/large_bags
```

## 注意事项

1. **GPU数量**: 建议至少使用4个GPU，可以同时运行4个实验，加快速度
2. **内存占用**: 每个实验大约需要2-4GB GPU内存，根据实际情况调整
3. **运行时间**:
   - 默认参数（1000个包，10轮训练）：约15-30分钟/实验
   - 12个实验在8个GPU上并行：约30-60分钟总计
4. **日志文件**: 每个日志文件可能较大（几MB到几十MB），注意磁盘空间
5. **错误处理**: 如果某个实验失败，会显示失败信息，但不影响其他实验继续运行

## 故障排除

### 问题1: GPU不足
```bash
# 减少并行数量，使用更少的GPU
python demo/run_all_experiments.py --gpus 0,1,2,3
```

### 问题2: 内存不足
```bash
# 减少包数量或包大小
python demo/run_all_experiments.py --num_bags 500 --bag_size_max 30
```

### 问题3: 某个实验失败
```bash
# 查看对应的日志文件找到错误原因
cat logs/all_experiments_*/[task]_[method].log
```

## 扩展

如果需要添加更多任务或方法，修改 `run_all_experiments.py` 中的：

```python
# 第61-62行
tasks = ['large_range', 'triplet', 'sum', 'increasing', 'your_new_task']
methods = ['attention', 'kibo', 'random', 'your_new_method']
```

## 联系

如有问题或建议，请查看主项目README或提交issue。
