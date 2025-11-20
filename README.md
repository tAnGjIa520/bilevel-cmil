# Bilevel Coreset Selection for Continual Multiple Instance Learning

This repository contains the implementation of a continual learning framework for Multiple Instance Learning (MIL) with bilevel coreset selection and reweighting (BCSR).

## Overview

This project addresses the challenge of continual learning in the context of whole slide image (WSI) classification, where the model needs to learn sequentially from different medical imaging datasets without forgetting previously learned knowledge. We employ a bilevel optimization approach for intelligent coreset selection and instance reweighting.

### Key Features

- **Bilevel Coreset Selection and Reweighting (BCSR)**: Intelligent sample selection and reweighting strategy for effective continual learning
- **Multiple MIL Architectures**: Support for CLAM (Clustering-constrained attention multiple instance learning) and TransMIL (Transformer-based MIL)
- **Replay-based Continual Learning**: Memory buffer management with various sampling strategies
- **Knowledge Distillation**: Multiple distillation strategies (attention, logits, features) to preserve previous knowledge
- **Comprehensive Evaluation**: Metrics for accuracy, AUC, backward/forward transfer, and forgetting

## Environment Setup

We use a Python environment similar to [CLAM](https://github.com/mahmoodlab/CLAM) with additional packages from [MICIL](https://github.com/cvblab/MICIL) and [Lightning Fabric](https://lightning.ai/docs/fabric/stable/).

### Installation

```bash
# Create conda environment
conda create -n cmil python=3.8
conda activate cmil

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio

# Install other dependencies
pip install lightning
pip install wandb
pip install torchmetrics
pip install pandas pyyaml
pip install opencv-python
pip install scikit-learn
pip install tensorboard
```

## Data Preparation

### Skin Cancer Dataset (CSC)

Download the data from [MICIL](https://github.com/cvblab/MICIL). Modify the data root paths in:
- `configs/csc_clam_cl.yaml`
- `configs/csc_transmil_cl.yaml`

### Camelyon16 and TCGA Datasets

Follow the instructions from [CLAM](https://github.com/mahmoodlab/CLAM) for data preparation. Modify the data root paths in:
- `configs/c16_lung_rcc_clam_cl.yaml`
- `configs/c16_lung_rcc_transmil_cl.yaml`

### Data Splits

Data split files are stored in the `splits/` directory. The repository includes pre-generated splits for Camelyon-TCGA experiments. You can generate custom split files following the CLAM split generation approach, adapted for continual learning scenarios.

## Training

### Basic Usage

The main training script is `main_cl.py`. You can specify configurations through YAML files and command-line arguments.

### Skin Cancer Dataset

```bash
# TransMIL with BCSR buffer
python main_cl.py --preset configs/csc_transmil_cl.yaml \
    --cl_method prev \
    --buffer_size 42 \
    --exp_name csc_transmil_cl_buf42_attn_logit

# CLAM with BCSR buffer
python main_cl.py --preset configs/csc_clam_cl.yaml \
    --cl_method prev \
    --buffer_size 42 \
    --exp_name csc_clam_cl_buf42_attn_logit
```

### Camelyon16-TCGA Dataset

```bash
# TransMIL with patch-balanced buffer
python main_cl.py --preset configs/c16_lung_rcc_transmil_cl.yaml \
    --cl_method prev \
    --buffer_size 128 \
    --buffer_slide_size 0.05 \
    --distill_method maxrand \
    --exp_name c16_lung_rcc_transmil_cl_pbbuf128_005_attn_logits_maxrand

# CLAM with patch-balanced buffer
python main_cl.py --preset configs/c16_lung_rcc_clam_cl.yaml \
    --cl_method prev \
    --buffer_size 128 \
    --buffer_slide_size 0.05 \
    --distill_method maxrand \
    --exp_name c16_lung_rcc_clam_cl_pbbuf128_005_attn_logits_maxrand
```

### Key Arguments

- `--preset`: Path to YAML configuration file
- `--cl_method`: Continual learning method (`prev`, `none`, etc.)
- `--buffer_size`: Size of the replay buffer
- `--buffer_slide_size`: Proportion of patches to store per slide
- `--distill_method`: Knowledge distillation strategy (`maxrand`, `attention`, `logits`)
- `--exp_name`: Experiment name for logging

## Results

Training results and logs are saved in `logs/<exp_name>/`:
- Model checkpoints
- Training metrics (CSV format)
- WandB logs (if enabled)
- Continual learning metrics (accuracy, AUC, BWT, FWT, forgetting)

## Project Structure

```
.
├── configs/              # Configuration files
├── datasets/             # Dataset loaders and data modules
├── models/               # MIL models (CLAM, TransMIL)
├── tools/                # Utility tools for metrics and visualization
├── images/               # Generated plots and figures
├── main_cl.py            # Main training script
├── mil_bcsr_coreset.py   # BCSR coreset implementation
├── mil_bcsr_training.py  # BCSR training module
├── utils.py              # Utility functions
└── docs/                 # Additional documentation
```

## Documentation

- [BCSR Hyperparameters Guide](BCSR_HYPERPARAMETERS.md)
- [Loss Functions Guide](loss_functions_guide.md)
- [Top-k Methods Comparison](TOPK_METHODS_COMPARISON.md)
- [Gradient Analysis Report](gradient_analysis_report.md)

## Hyperparameter Search

We provide scripts for hyperparameter optimization:

```bash
# BCSR convergence search
python run_bcsr_convergence_search.py

# Seed search for reproducibility
python run_seed_search.py

# Ablation studies
python run_ablation_study.py
```

## Analysis Tools

Various analysis scripts are provided:

```bash
# Analyze bag-level instance distributions
python analyze_bag_instances.py

# Plot performance metrics
python plot_combined_performance.py
python plot_gradient_norms.py

# Compare different configurations
python plot_four_combined.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{your_paper,
  title={Bilevel Coreset Selection for Continual Multiple Instance Learning},
  author={Your Name},
  journal={Your Journal/Conference},
  year={2024}
}
```

## Acknowledgements

This project builds upon:
- [CLAM](https://github.com/mahmoodlab/CLAM) - Clustering-constrained attention multiple instance learning
- [MICIL](https://github.com/cvblab/MICIL) - Multiple instance continual learning framework
- [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) - Lightweight PyTorch training framework

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
