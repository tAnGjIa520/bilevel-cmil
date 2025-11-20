# 持续学习的双层核心集选择方法 (Bilevel-CMIL)

本仓库实现了一个针对多示例学习（Multiple Instance Learning, MIL）的持续学习框架，采用双层核心集选择与重加权（BCSR）方法。

## 概述

本项目旨在解决全切片图像（Whole Slide Image, WSI）分类中的持续学习挑战，使模型能够从不同的医学影像数据集中顺序学习，同时避免灾难性遗忘。我们采用双层优化方法进行智能核心集选择和实例重加权。

### 主要特性

- **双层核心集选择与重加权（BCSR）**：用于有效持续学习的智能样本选择和重加权策略
- **多种 MIL 架构**：支持 CLAM（聚类约束注意力多示例学习）和 TransMIL（基于 Transformer 的 MIL）
- **基于回放的持续学习**：具有多种采样策略的内存缓冲区管理
- **知识蒸馏**：多种蒸馏策略（注意力、logits、特征）以保留先前知识
- **全面的评估指标**：包括准确率、AUC、后向/前向迁移、遗忘度等指标

## 环境配置

我们使用类似于 [CLAM](https://github.com/mahmoodlab/CLAM) 的 Python 环境，并添加了 [MICIL](https://github.com/cvblab/MICIL) 和 [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) 的额外依赖包。

### 安装步骤

```bash
# 创建 conda 环境
conda create -n cmil python=3.8
conda activate cmil

# 安装 PyTorch（根据需要调整 CUDA 版本）
pip install torch torchvision torchaudio

# 安装其他依赖
pip install lightning
pip install wandb
pip install torchmetrics
pip install pandas pyyaml
pip install opencv-python
pip install scikit-learn
pip install tensorboard
```

## 数据准备

### 皮肤癌数据集（CSC）

从 [MICIL](https://github.com/cvblab/MICIL) 下载数据。修改以下配置文件中的数据路径：
- `configs/csc_clam_cl.yaml`
- `configs/csc_transmil_cl.yaml`

### Camelyon16 和 TCGA 数据集

按照 [CLAM](https://github.com/mahmoodlab/CLAM) 的说明准备数据。修改以下配置文件中的数据路径：
- `configs/c16_lung_rcc_clam_cl.yaml`
- `configs/c16_lung_rcc_transmil_cl.yaml`

### 数据划分

数据划分文件存储在 `splits/` 目录中。仓库包含了 Camelyon-TCGA 实验的预生成划分。您可以按照 CLAM 的划分生成方法创建自定义划分文件，并适配持续学习场景。

## 训练

### 基本用法

主训练脚本是 `main_cl.py`。可以通过 YAML 文件和命令行参数指定配置。

### 皮肤癌数据集

```bash
# TransMIL + BCSR 缓冲区
python main_cl.py --preset configs/csc_transmil_cl.yaml \
    --cl_method prev \
    --buffer_size 42 \
    --exp_name csc_transmil_cl_buf42_attn_logit

# CLAM + BCSR 缓冲区
python main_cl.py --preset configs/csc_clam_cl.yaml \
    --cl_method prev \
    --buffer_size 42 \
    --exp_name csc_clam_cl_buf42_attn_logit
```

### Camelyon16-TCGA 数据集

```bash
# TransMIL + 补丁平衡缓冲区
python main_cl.py --preset configs/c16_lung_rcc_transmil_cl.yaml \
    --cl_method prev \
    --buffer_size 128 \
    --buffer_slide_size 0.05 \
    --distill_method maxrand \
    --exp_name c16_lung_rcc_transmil_cl_pbbuf128_005_attn_logits_maxrand

# CLAM + 补丁平衡缓冲区
python main_cl.py --preset configs/c16_lung_rcc_clam_cl.yaml \
    --cl_method prev \
    --buffer_size 128 \
    --buffer_slide_size 0.05 \
    --distill_method maxrand \
    --exp_name c16_lung_rcc_clam_cl_pbbuf128_005_attn_logits_maxrand
```

### 主要参数说明

- `--preset`: YAML 配置文件路径
- `--cl_method`: 持续学习方法（`prev`、`none` 等）
- `--buffer_size`: 回放缓冲区大小
- `--buffer_slide_size`: 每个切片存储的补丁比例
- `--distill_method`: 知识蒸馏策略（`maxrand`、`attention`、`logits`）
- `--exp_name`: 实验名称（用于日志记录）

## 结果输出

训练结果和日志保存在 `logs/<exp_name>/` 目录中：
- 模型检查点
- 训练指标（CSV 格式）
- WandB 日志（如果启用）
- 持续学习指标（准确率、AUC、BWT、FWT、遗忘度）

## 项目结构

```
.
├── configs/              # 配置文件
├── datasets/             # 数据集加载器和数据模块
├── models/               # MIL 模型（CLAM、TransMIL）
├── tools/                # 指标和可视化工具
├── images/               # 生成的图表和图形
├── main_cl.py            # 主训练脚本
├── mil_bcsr_coreset.py   # BCSR 核心集实现
├── mil_bcsr_training.py  # BCSR 训练模块
├── utils.py              # 工具函数
└── docs/                 # 附加文档
```

## 文档

- [BCSR 超参数指南](BCSR_HYPERPARAMETERS.md)
- [损失函数指南](loss_functions_guide.md)
- [Top-k 方法对比](TOPK_METHODS_COMPARISON.md)
- [梯度分析报告](gradient_analysis_report.md)

## 超参数搜索

我们提供了超参数优化脚本：

```bash
# BCSR 收敛性搜索
python run_bcsr_convergence_search.py

# 种子搜索（用于可重复性）
python run_seed_search.py

# 消融实验
python run_ablation_study.py
```

## 分析工具

提供了多种分析脚本：

```bash
# 分析包级实例分布
python analyze_bag_instances.py

# 绘制性能指标
python plot_combined_performance.py
python plot_gradient_norms.py

# 比较不同配置
python plot_four_combined.py
```

## 引用

如果您觉得这项工作有用，请引用：

```bibtex
@article{your_paper,
  title={Bilevel Coreset Selection for Continual Multiple Instance Learning},
  author={Your Name},
  journal={Your Journal/Conference},
  year={2024}
}
```

## 致谢

本项目基于以下工作：
- [CLAM](https://github.com/mahmoodlab/CLAM) - 聚类约束注意力多示例学习
- [MICIL](https://github.com/cvblab/MICIL) - 多示例持续学习框架
- [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) - 轻量级 PyTorch 训练框架

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

## 联系方式

如有问题或建议，请在 GitHub 上提交 issue 或联系作者。
