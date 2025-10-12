# 🚀 快速参考卡片

## 一键运行命令

### 🔬 对比三种方法（推荐！）
```bash
python demo/attention_mil_harder_tasks111.py \
    --task triplet \
    --selection_method compare \
    --epochs1 30 --epochs2 30 \
    --num_bags 1000 --top_k 5
```

### ⚡ Attention方法（快速）
```bash
python demo/attention_mil_harder_tasks111.py \
    --selection_method attention \
    --top_k 3 --bottom_k 2
```

### 🎯 KIBO方法（最优）
```bash
python demo/attention_mil_harder_tasks111.py \
    --selection_method kibo \
    --top_k 5 --kibo_outer_it 15
```

### 🎲 Random方法（baseline）
```bash
python demo/attention_mil_harder_tasks111.py \
    --selection_method random \
    --top_k 5
```

## 三种方法对比表

| 特性 | Attention | KIBO | Random |
|------|-----------|------|--------|
| **原理** | 基于注意力权重 | 双层优化 | 随机抽取 |
| **速度** | 🚀🚀🚀 快 | 🚀 慢 | 🚀🚀🚀🚀 最快 |
| **性能** | ⭐⭐⭐⭐ 好 | ⭐⭐⭐⭐⭐ 最优 | ⭐⭐⭐ 一般 |
| **标签不一致率** | ~25% | ~20% | ~40% |
| **适用场景** | 快速原型 | 追求最优 | Baseline |
| **参数复杂度** | 低 | 高 | 极低 |

## 关键参数速查

### 通用参数
```
--task          : triplet / sum / increasing / large_range
--epochs1       : 第一阶段轮数 (默认50)
--epochs2       : 第二阶段轮数 (默认50)
--num_bags      : 训练包数量 (默认2000)
```

### Attention参数
```
--top_k         : 最高attention数 (推荐3-5)
--bottom_k      : 最低attention数 (推荐2-3)
```

### KIBO参数
```
--top_k              : 选择总数 (推荐4-6)
--kibo_outer_it      : 外层迭代 (推荐10-20)
--kibo_inner_it      : 内层迭代 (推荐2-3)
--kibo_lr_weight     : 权重学习率 (推荐0.1)
```

## 快速测试（5分钟）
```bash
python demo/attention_mil_harder_tasks111.py \
    --task triplet --selection_method compare \
    --epochs1 10 --epochs2 10 --num_bags 200 --top_k 4
```

## 标准实验（30分钟）
```bash
python demo/attention_mil_harder_tasks111.py \
    --task triplet --selection_method compare \
    --epochs1 30 --epochs2 30 --num_bags 1000 --top_k 5
```

## 完整实验（2小时）
```bash
python demo/attention_mil_harder_tasks111.py \
    --task triplet --selection_method compare \
    --epochs1 50 --epochs2 50 --num_bags 2000 --top_k 5
```

## 文件导航

- `attention_mil_harder_tasks111.py` - 主程序
- `README_instance_selection_methods.md` - 方法原理
- `USAGE_GUIDE.md` - 详细使用指南
- `SUMMARY.md` - 完成总结
- `QUICK_REFERENCE.md` - 本文件

## 输出解读速查

### 标签一致性
```
标签不一致: 250/1000 (25.00%)
  - 正包→负包: 200 (丢失关键实例)
  - 负包→正包: 50  (意外形成模式)
```
👉 **越低越好**，说明选择保留了原包语义

### 性能对比
```
性能变化:
  Accuracy: +0.0100
  AUC: +0.0100
  F1: +0.0100
```
👉 **正值表示提升**，负值表示过滤损失信息

## 常见问题速查

**Q: 哪个方法最好？**
A: KIBO理论最优，Attention性价比高，Random作baseline

**Q: KIBO太慢怎么办？**
A: 减少--kibo_outer_it或--num_bags

**Q: 标签不一致率高怎么办？**
A: 增加--top_k或使用KIBO方法

**Q: 内存不足？**
A: 减少--num_bags或使用Attention/Random

---

📚 **详细文档**: 见 `USAGE_GUIDE.md`
🎓 **理论背景**: 见 `README_instance_selection_methods.md`
