import torch
from datasets.dataset import datamodule_gen
from models.clam import CLAM_SB, CLAM_MB
from models.transmil import TransMIL
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from utils import CustomEarlyStopping as EarlyStopping
from utils import CustomProgressBar
import yaml
import wandb
import os
import pandas as pd
from torchmetrics import AUROC, Accuracy, MeanMetric
from lightning.fabric import Fabric, seed_everything, is_wrapped
from lightning.fabric.utilities import move_data_to_device
from lightning.fabric.loggers import CSVLogger
import copy
from collections import defaultdict
import random
from mil_bcsr_coreset import BCSR_Coreset
from tools.cl_metrics import compute_all_folds_metrics
from tools.print_metrics import print_cl_metrics_table
import time
from tools.tb import AutoStepWriter


# use pure pytorch instead of pytorch-lightning
class SimpleBuffer():
    def __init__(self, buffer_size, device='cpu', **kwargs):
        self.buffer = []  # List of samples
        self.buffer_size = buffer_size
        self.device = device
        self.sample_selection_strategy = 'reservoir'
        self.n_seen_samples = 0
        self.epoch_indices = None
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0
        self.n_patches_total = 0
        self.labels = {}  # Count of samples per class

    def to(self, device):
        old_device = self.device
        self.device = device
        if old_device != device:
            for i in range(len(self.buffer)):
                self.buffer[i] = move_data_to_device(self.buffer[i], device)
        return self

    def adjust_buffer_size_by_new_classes(self, n_new_classes: int):
        # In this simple version, we do not adjust the buffer size
        pass

    def add_judge(self, sample_label):
        """Determine whether to add a new sample to the buffer using reservoir sampling."""
        n_seen = self.n_seen_samples
        self.n_seen_samples += 1
        if len(self.buffer) < self.buffer_size:
            # Buffer not full yet
            return len(self.buffer)
        else:
            # Buffer full, apply reservoir sampling
            rand_idx = np.random.randint(0, n_seen)
            if rand_idx < self.buffer_size:
                return rand_idx
            else:
                return -1  # Do not add the sample

    def add(self, sample, idx):
        """Add a sample to the buffer at the given index."""
        assert idx != -1, "The sample is not added to the buffer"
        sample = move_data_to_device(sample, self.device)
        label = sample['label'].item()
        if idx == len(self.buffer):
            # Adding a new sample to the buffer
            self.buffer.append(sample)
            self.n_patches_total += sample['features'].size(0)
            self.labels[label] = self.labels.get(label, 0) + 1
        else:
            # Replacing an existing sample in the buffer
            self.n_patches_total = self.n_patches_total - self.buffer[idx]['features'].size(0) + sample['features'].size(0)
            old_label = self.buffer[idx]['label'].item()
            self.labels[old_label] -= 1
            self.buffer[idx] = sample
            self.labels[label] = self.labels.get(label, 0) + 1

    def start_epoch(self):
        """Reset indices for a new epoch."""
        self.epoch_indices = np.random.permutation(len(self.buffer))
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0

    def get_next_batch(self, batch_size=1):
        """Get the next batch of samples for the current epoch."""
        if self.epoch_indices is None:
            self.start_epoch()
        remaining = len(self.buffer) - self.current_epoch_position
        if remaining == 0:
            return None  # Epoch is finished
        actual_batch_size = min(batch_size, remaining)
        batch_indices = self.epoch_indices[self.current_epoch_position:
                                           self.current_epoch_position + actual_batch_size]
        batch = [self.buffer[i] for i in batch_indices]
        self.current_epoch_position += actual_batch_size
        self.samples_output_in_epoch += actual_batch_size
        return batch

    def get_samples_output_count(self):
        """Return the number of samples output in the current epoch."""
        return self.samples_output_in_epoch

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        """Support iteration over all samples in buffer"""
        return iter(self.buffer)


class ClsssIncrementalBuffer():
    def __init__(self, buffer_size, device='cpu', **kwargs):
        self.buffers = {}  # Dict mapping class labels to lists of samples
        self.buffer_size = buffer_size
        self.device = device
        self.sample_selection_strategy = 'reservoir'
        self.n_seen_samples_per_class = defaultdict(int)
        self.num_classes = 0
        self.buffer_size_per_class = None
        self.epoch_indices = None
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0
        self.n_patches_total = 0
        self.labels = {}  # Count of samples per class

    def to(self, device):
        old_device = self.device
        self.device = device
        if old_device != device:
            for l in self.buffers:
                for i in range(len(self.buffers[l])):
                    self.buffers[l][i] = move_data_to_device(self.buffers[l][i], device)
        return self

    # Given the number of new classes will be adding, adjust the buffer size per class
    def adjust_buffer_size_by_new_classes(self, n_new_classes: int):
        self.buffer_size_per_class = self.buffer_size // (self.num_classes + n_new_classes)
        for l in self.buffers:
            if len(self.buffers[l]) > self.buffer_size_per_class:
                self.buffers[l] = random.sample(self.buffers[l], self.buffer_size_per_class)
                self.labels[l] = self.buffer_size_per_class
        # calculate the number of patches in the buffer
        self.n_patches_total = sum([sample['features'].size(0) for samples in self.buffers.values() for sample in samples])

    # Judge whether to add the new sample to the buffer by reservoir sampling
    # If add, return the index
    def add_judge(self, sample_label):
        if sample_label not in self.buffers:
            # New class encountered
            self.buffers[sample_label] = []
            self.num_classes += 1

        self.n_seen_samples_per_class[sample_label] += 1
        n_seen = self.n_seen_samples_per_class[sample_label]
        if len(self.buffers[sample_label]) < self.buffer_size_per_class:
            return len(self.buffers[sample_label])
        else:
            # Buffer full, apply reservoir sampling
            rand_idx = np.random.randint(0, n_seen)
            if rand_idx < self.buffer_size_per_class:
                return rand_idx
            else:
                return -1
            
    def add(self, sample, idx):
        assert idx != -1, "The sample is not added to the buffer"
        sample = move_data_to_device(sample, self.device)
        label = sample['label'].item()
        if idx == len(self.buffers[label]):
            self.buffers[label].append(sample)
            self.n_patches_total += sample['features'].size(0)
            self.labels[label] = self.labels.get(label, 0) + 1
        else:
            self.n_patches_total = self.n_patches_total - self.buffers[label][idx]['features'].size(0) + sample['features'].size(0)
            self.buffers[label][idx] = sample
            # Labels count remains the same

    def start_epoch(self):
        """Reset indices for a new epoch"""
        # Flatten buffers into a single list
        self.all_samples = []
        for samples in self.buffers.values():
            self.all_samples.extend(samples)
        self.epoch_indices = np.random.permutation(len(self.all_samples))
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0

    def get_next_batch(self, batch_size=1):
        """Get next batch of samples for the current epoch"""
        if self.epoch_indices is None:
            self.start_epoch()
        remaining = len(self.all_samples) - self.current_epoch_position
        if remaining == 0:
            return None  # Epoch is finished
        actual_batch_size = min(batch_size, remaining)
        batch_indices = self.epoch_indices[self.current_epoch_position:
                                           self.current_epoch_position + actual_batch_size]
        batch = [self.all_samples[i] for i in batch_indices]
        self.current_epoch_position += actual_batch_size
        self.samples_output_in_epoch += actual_batch_size
        return batch

    def get_samples_output_count(self):
        """Return the number of samples output in the current epoch"""
        return self.samples_output_in_epoch

    def __len__(self):
        return sum(len(samples) for samples in self.buffers.values())

    def __iter__(self):
        """Support iteration over all samples in buffer"""
        for samples in self.buffers.values():
            for sample in samples:
                yield sample

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

    
def parse_list_arg(value_str):
    """Parse list parameters from command line, e.g., '[1e-4,1e-4,1e-4]' or '[1, 2, 3]'"""
    import ast
    try:
        # Try to parse string safely using ast.literal_eval
        parsed_value = ast.literal_eval(value_str)
        return parsed_value
    except (ValueError, SyntaxError):
        # If parsing fails, return original string
        return value_str

def add_argument(parser, name, value):
    """Helper function to add an argument to the parser if it doesn't already exist."""
    if not any(arg.dest == name for arg in parser._actions):
        if isinstance(value, bool):
            parser.add_argument(f'--{name}', action='store_false' if value else 'store_true', default=value)
        elif isinstance(value, list):
            # For list types, use custom parsing function
            parser.add_argument(f'--{name}', type=parse_list_arg, default=value)
        elif isinstance(value, str):
            # Handle scientific notation strings from YAML (e.g., '1e-8' → 1e-8)
            try:
                float_value = float(value)
                parser.add_argument(f'--{name}', type=float, default=float_value)
            except ValueError:
                # If conversion fails, keep as string
                parser.add_argument(f'--{name}', type=str, default=value)
        else:
            parser.add_argument(f'--{name}', type=type(value), default=value)

def load_config_from_yaml(file_path):
    """Loads configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file) or {}

def init_args():
    parser = argparse.ArgumentParser(description='MIL-CL')
    parser.add_argument('--preset', type=str, default=None, help='preset config')

    args, remaining_argv = parser.parse_known_args()

    # Load configuration from YAML if preset is provided
    config = load_config_from_yaml(args.preset) if args.preset else {}

    # Update parser with options from YAML configuration
    for k, v in config.items():
        add_argument(parser, k, v)

    args = parser.parse_args()

    # update args
    if args.debug == 'False':
        args.debug = False
    elif args.debug == 'True':
        args.debug = 'lite'
    if args.debug == 'lite':
        args.n_batches = 10
        args.epochs = 2
        args.n_folds = 1
        args.wandb_mode = 'disabled'
    elif args.debug == 'full':
        args.n_batches = 20
        args.epochs = 3
        args.wandb_mode = 'disabled'


    os.environ['WANDB_MODE'] = args.wandb_mode
    wandb.require("core")

    if args.folds_start == -1:
        args.folds_start = 0
    if args.folds_end == -1:
        if hasattr(args, 'n_folds'):
            args.folds_end = args.n_folds
        else:
            args.folds_end = 1

    # Auto-calculate buffer_size (based on full bag capacity)
    if hasattr(args, 'full_bag_buffer_size') and hasattr(args, 'avg_instances_per_bag') and hasattr(args, 'buffer_slide_size'):
        import math

        # Calculate total capacity (in instances)
        total_buffer_instances = args.full_bag_buffer_size * args.avg_instances_per_bag

        # Calculate how many pseudo-bags can be stored
        args.buffer_size = math.ceil(total_buffer_instances / args.buffer_slide_size)

        # Calculate compression ratio and efficiency gain
        compression_ratio = args.buffer_slide_size / args.avg_instances_per_bag * 100
        efficiency_gain = args.buffer_size / args.full_bag_buffer_size

        print(f"\n{'='*70}")
        print(f"Buffer Configuration:")
        print(f"  Full-bag buffer capacity: {args.full_bag_buffer_size} bags")
        print(f"  Avg instances per original bag: {args.avg_instances_per_bag}")
        print(f"  Total buffer capacity: {total_buffer_instances} instances")
        print(f"  Instances per pseudo-bag: {args.buffer_slide_size}")
        print(f"  → Auto-calculated buffer_size: {args.buffer_size} pseudo-bags")
        print(f"  Compression ratio: {compression_ratio:.2f}%")
        print(f"  Efficiency gain: {efficiency_gain:.2f}x")
        print(f"{'='*70}\n")

    print(args)
    return args

def load_model(args):
    # if args.net == 'abmil':
        # model = Attn_Net(args)
    # elif args.net == 'abmil_gated':
        # model = Attn_Net_Gated(args)
    # elif args.net == 'dtfd':
        # model = DTFD_CL(args)
    if args.net == 'clam_sb':
        model = CLAM_SB(D_feat=args.D_feat,
                        L=args.L,
                        K=args.K,
                        dropout=args.dropout,
                        gate=args.gate,
                        k_sample=args.k_sample,
                        n_classes=args.n_classes,
                        instance_loss_name=args.instance_loss_name,
                        subtyping=args.subtyping,)
    elif args.net == 'clam_mb':
        model = CLAM_MB(D_feat=args.D_feat,
                        L=args.L,
                        K=args.K,
                        dropout=args.dropout,
                        gate=args.gate,
                        k_sample=args.k_sample,
                        n_classes=args.n_classes,
                        instance_loss_name=args.instance_loss_name,
                        subtyping=args.subtyping)
    # elif args.net == 'acmil':
        # model = ACMIL_CL(args)
    elif args.net == 'transmil':
        model = TransMIL(D_feat=args.D_feat,
                         D_inner=args.L,
                         n_classes=args.n_classes,)
    else:
        raise NotImplementedError

    return model

def compute_convergence_score(loss_list):
    """
    Compute convergence score for loss sequence

    Args:
        loss_list: Loss sequence (list)

    Returns:
        dict: Dictionary containing convergence score and status
    """
    import numpy as np

    if len(loss_list) < 2:
        return {
            'score': 0.0,
            'status': 'Insufficient data',
            'details': {}
        }

    losses = np.array(loss_list)

    # 1. Overall decrease rate
    total_decrease = losses[0] - losses[-1]
    relative_decrease = total_decrease / (abs(losses[0]) + 1e-10)
    decrease_score = np.clip(relative_decrease, 0, 1)

    # 2. Monotonicity score (whether loss keeps decreasing)
    gradients = np.diff(losses)
    decreasing_steps = np.sum(gradients < 0)
    monotonicity_score = decreasing_steps / len(gradients) if len(gradients) > 0 else 0

    # 3. Recent stability (volatility in last few steps)
    window_size = min(5, len(losses))
    recent_losses = losses[-window_size:]
    recent_std = np.std(recent_losses)
    recent_mean = np.mean(recent_losses)
    recent_cv = recent_std / (abs(recent_mean) + 1e-10)
    stability_score = 1.0 / (1.0 + recent_cv)

    # 4. Oscillation index (degree of back-and-forth oscillation)
    if len(gradients) > 1:
        sign_changes = np.sum(np.diff(np.sign(gradients)) != 0)
        oscillation_index = sign_changes / (len(gradients) - 1)
        oscillation_score = 1 - oscillation_index
    else:
        oscillation_score = 1.0

    # Composite score (weighted average)
    convergence_score = (
        0.3 * decrease_score +      # Overall decrease 30%
        0.3 * monotonicity_score +   # Monotonicity 30%
        0.2 * stability_score +      # Stability 20%
        0.2 * oscillation_score      # No oscillation 20%
    )

    # Determine convergence status
    if convergence_score >= 0.8:
        status = 'Excellent'
    elif convergence_score >= 0.6:
        status = 'Good'
    elif convergence_score >= 0.4:
        status = 'Fair'
    elif convergence_score >= 0.2:
        status = 'Poor'
    else:
        status = 'Very Poor'

    return {
        'score': float(convergence_score),
        'status': status,
        'details': {
            'decrease': float(decrease_score),
            'monotonicity': float(monotonicity_score),
            'stability': float(stability_score),
            'oscillation': float(oscillation_score),
            'initial_loss': float(losses[0]),
            'final_loss': float(losses[-1]),
            'total_decrease': float(total_decrease),
            'loss_history': losses.tolist()
        }
    }

def kd_loss_fn(train_logits, prev_logits, ta = 2, softmax = True):
    """
    KD_LOSS: Compute distillation loss between output of the current model and the output of the previous (saved) model.
        + Inputs:
            - train_logits: Logits of model in training phase only for active units
            - prev_logits: Logits of model previous experience only for active units
        + Outputs:
            - dist_loss: Knowledge distrillation loss
    """
    assert prev_logits.size() == train_logits.size()
    if softmax:
        q = torch.softmax(prev_logits / ta, dim = -1)
        log_p = torch.log_softmax(train_logits / ta, dim = -1)
    else:
        q = prev_logits
        log_p = torch.log(train_logits)
    dist_loss = torch.nn.functional.kl_div(log_p, q, reduction = "batchmean")
    return dist_loss

def adaptive_distill(slide,  size=1e5, method='random',model=None):
    pass
import copy
# TODO: Critical - select subset from a large set of patches
def distill_slide(slide, attn=None, size=1e5, method='random',model=None,label=None,task_id=None,seen_classes=None,tb_writer=None):
    assert len(slide.shape) == 2, f"slide shape: {slide.shape}"
    
    print("slide={}".format(size))
    # exit(0)
    
    size = int(min(size, slide.size(0)))
    if method == 'random':
        idx = torch.randperm(slide.size(0))[:size]
    elif method == 'max':
        idx = torch.topk(attn, size)[1][-1]
    elif method == 'maxmin':
        size = size // 2
        top_p_ids = torch.topk(attn, size)[1][-1]
        top_n_ids = torch.topk(-attn, size, dim=1)[1][-1]
        idx = torch.cat((top_p_ids, top_n_ids))
    elif method == 'maxminrand':
        size = size // 4
        top_p_ids = torch.topk(attn, size)[1][-1]
        top_n_ids = torch.topk(-attn, size, dim=1)[1][-1]
        rand_ids = torch.randperm(slide.size(0))[:size*2].to(top_n_ids.device)
        idx = torch.cat((top_p_ids, top_n_ids, rand_ids))
    elif method == 'maxrand':
        size = size // 2
        top_p_ids = torch.topk(attn, size)[1][-1]
        rand_ids = torch.randperm(slide.size(0))[:size].to(top_p_ids.device)
        idx = torch.cat((top_p_ids, rand_ids))
    elif method=="kibo":
        proxy_model=copy.deepcopy(model)
        for param in proxy_model.parameters():
            param.requires_grad = True

        # Create temperature scheduler (if enabled)
        temp_scheduler = None
        if hasattr(args, 'use_temp_scheduler') and args.use_temp_scheduler:
            from tools.temperature_scheduler import TemperatureScheduler
            temp_scheduler = TemperatureScheduler(
                initial_temp=getattr(args, 'temp_initial', 2.0),
                final_temp=getattr(args, 'temp_final', 0.5),
                max_iterations=args.bcsr_max_outer_it,
                strategy=getattr(args, 'temp_strategy', 'cosine')
            )

        # Create learning rate scheduler (if enabled)
        lr_scheduler = None
        if hasattr(args, 'use_lr_scheduler') and args.use_lr_scheduler:
            from tools.lr_scheduler import LearningRateScheduler
            lr_init = getattr(args, 'lr_initial', None) or args.weight_adam_lr
            lr_fin = getattr(args, 'lr_final', None) or (args.weight_adam_lr * 0.1)
            lr_scheduler = LearningRateScheduler(
                initial_lr=lr_init,
                final_lr=lr_fin,
                max_iterations=args.bcsr_max_outer_it,
                strategy=getattr(args, 'lr_strategy', 'cosine')
            )

        BCSR_Coreset_selector = BCSR_Coreset(
                            proxy_model,
                            lr_proxy_model=args.bcsr_lr_proxy_model,
                            beta=args.bcsr_beta,
                            max_outer_it=args.bcsr_max_outer_it,
                            max_inner_it=args.bcsr_max_inner_it,
                            weight_lr=args.bcsr_weight_lr,
                            distall_lamda=args.distall_lamda,
                            tb_writer=tb_writer,
                            draw_curve=args.draw_curve,
                            topk=size,
                            topk_method=args.topk_method,
                            topk_temperature=args.topk_temperature,
                            normalize_method=args.normalize_method,
                            use_simplex_projection=args.use_simplex_projection,
                            distill_target=args.distill_target,
                            temp_scheduler=temp_scheduler,
                            weight_optimizer_type=getattr(args, 'weight_optimizer_type', 'sgd'),
                            weight_adam_lr=getattr(args, 'weight_adam_lr', 0.001),
                            weight_adam_betas=getattr(args, 'weight_adam_betas', (0.9, 0.999)),
                            weight_adam_eps=getattr(args, 'weight_adam_eps', 1e-8),
                            use_lr_scheduler=getattr(args, 'use_lr_scheduler', False),
                            lr_initial=getattr(args, 'lr_initial', None),
                            lr_final=getattr(args, 'lr_final', None),
                            lr_strategy=getattr(args, 'lr_strategy', 'cosine'),
                            coreset_weight_init=getattr(args, 'coreset_weight_init', 'uniform_random'),
                            model_type=args.net,
                            init_hyperparams=getattr(args, 'init_hyperparams', {}),
                            neumann_series_depth=getattr(args, 'bcsr_neumann_series_depth', 3))
        # Optimization: Directly pass GPU tensor to avoid CPU-GPU data transfer

        idx, loss_list, inner_loss_list = BCSR_Coreset_selector.coreset_select(proxy_model, slide, label, task_id=task_id,
                                                 topk=size, seen_classes=seen_classes)
        # Note: inner_loss_list is received but not used in main_cl.py

        # idx is already on the correct device, ensure consistency with slide device
        idx = idx.to(slide.device)

        # Evaluate convergence quality of BCSR coreset selection
        # convergence = compute_convergence_score(loss_list)
        # print(f"  ├─ Convergence Score: {convergence['score']:.4f} ({convergence['status']})")
        # print(f"  ├─ Initial Loss: {convergence['details']['initial_loss']:.4f}")
        # print(f"  ├─ Final Loss: {convergence['details']['final_loss']:.4f}")
        # print(f"  └─ Total Decrease: {convergence['details']['total_decrease']:.4f}")

        # # Log to TensorBoard
        # if tb_writer is not None:
        #     tb_writer.add_scalar('bcsr/convergence_score', convergence['score'])
        #     tb_writer.add_scalar('bcsr/convergence_decrease', convergence['details']['decrease'])
        #     tb_writer.add_scalar('bcsr/convergence_monotonicity', convergence['details']['monotonicity'])
        #     tb_writer.add_scalar('bcsr/convergence_stability', convergence['details']['stability'])

        # # Save convergence metrics and terminate program
        # print("\n" + "="*70)
        # print("CONVERGENCE METRICS COLLECTED - TERMINATING PROGRAM")
        # print("="*70)
        # print(f"Score: {convergence['score']:.4f}")
        # print(f"Status: {convergence['status']}")
        # print(f"Decrease Rate: {convergence['details']['decrease']:.4f}")
        # print(f"Monotonicity: {convergence['details']['monotonicity']:.4f}")
        # print(f"Stability: {convergence['details']['stability']:.4f}")
        # print(f"Oscillation Score: {convergence['details']['oscillation']:.4f}")
        # print("="*70 + "\n")

        # # Save to file
        # import json
        # convergence_file = f'{args.log_dir}/{args.exp_name}/convergence_metrics.json'
        # os.makedirs(os.path.dirname(convergence_file), exist_ok=True)
        # with open(convergence_file, 'w') as f:
        #     json.dump(convergence, f, indent=2)
        # print(f"Convergence metrics saved to: {convergence_file}")

        # Terminate program
    
    

    elif method=="mix":
        # Hybrid strategy: coarse filtering with maxrand, then fine filtering with kibo
        # Stage 1: Use maxrand to coarsely select more candidate samples (e.g., mix_coarse_ratio*size)

        mix_coarse_ratio = getattr(args, 'mix_coarse_ratio', 2)  # Default: coarse selection is 2x fine selection
        coarse_size = int(min(size * mix_coarse_ratio, slide.size(0)))  # Coarse sample count
        coarse_size = coarse_size // 2  # maxrand internally divides by 2
        top_p_ids = torch.topk(attn, coarse_size)[1][-1]
        rand_ids = torch.randperm(slide.size(0))[:coarse_size].to(top_p_ids.device)
        coarse_idx = torch.cat((top_p_ids, rand_ids))

        # Get candidate samples after coarse selection
        candidate_slide = slide[coarse_idx]

        # Stage 2: Use kibo for fine filtering within candidate samples
        proxy_model = copy.deepcopy(model)
        for param in proxy_model.parameters():
            param.requires_grad = True

        # Create temperature scheduler (if enabled)
        temp_scheduler = None
        if hasattr(args, 'use_temp_scheduler') and args.use_temp_scheduler:
            from tools.temperature_scheduler import TemperatureScheduler
            temp_scheduler = TemperatureScheduler(
                initial_temp=getattr(args, 'temp_initial', 2.0),
                final_temp=getattr(args, 'temp_final', 0.5),
                max_iterations=args.bcsr_max_outer_it,
                strategy=getattr(args, 'temp_strategy', 'cosine')
            )

        BCSR_Coreset_selector = BCSR_Coreset(
            proxy_model,
            lr_proxy_model=args.bcsr_lr_proxy_model,
            beta=args.bcsr_beta,
            max_outer_it=args.bcsr_max_outer_it,
            max_inner_it=args.bcsr_max_inner_it,
            weight_lr=args.bcsr_weight_lr,
            distall_lamda=args.distall_lamda,
            draw_curve=args.draw_curve,
            tb_writer=tb_writer,
            topk_method=args.topk_method,
            topk_temperature=args.topk_temperature,
            normalize_method=args.normalize_method,
            use_simplex_projection=args.use_simplex_projection,
            distill_target=args.distill_target,
            temp_scheduler=temp_scheduler,
            weight_optimizer_type=getattr(args, 'weight_optimizer_type', 'sgd'),
            weight_adam_lr=getattr(args, 'weight_adam_lr', 0.001),
            weight_adam_betas=getattr(args, 'weight_adam_betas', (0.9, 0.999)),
            weight_adam_eps=getattr(args, 'weight_adam_eps', 1e-8),
            coreset_weight_init=getattr(args, 'coreset_weight_init', 'uniform_random'),
            model_type=args.net,
            init_hyperparams=getattr(args, 'init_hyperparams', {}),
            neumann_series_depth=getattr(args, 'bcsr_neumann_series_depth', 3))

        # Perform fine filtering within candidate samples
        fine_idx, outer_loss = BCSR_Coreset_selector.coreset_select(
            proxy_model, candidate_slide, label, task_id=task_id,
            topk=size, out_loss=None, seen_classes=seen_classes)

        # Map fine filtering indices back to original sample indices
        # Ensure fine_idx and coarse_idx are on the same device
        fine_idx = fine_idx.to(coarse_idx.device)
        idx = coarse_idx[fine_idx].to(slide.device)


    else:
        raise NotImplementedError

    return slide[idx]

def one_fold(args, fold=0):
    # ====== 1. Initialization Block ======
    # Setup distributed training framework and random seed
    fabric = Fabric(devices=1, accelerator="auto")
    seed_everything(args.seed)

    # Create TensorBoard writer
    tb_writer = AutoStepWriter(logdir=f'{args.log_dir}/{args.exp_name}/tb_logs/fold_{fold}')

    # Load and initialize model
    model = load_model(args)
    model = fabric.to_device(model)

    # Prepare test dataloaders for all tasks for final evaluation
    test_dataloaders = [datamodule_gen(args, fold=fold, task=task)['test_loader'] for task in range(args.n_tasks)]
    # test_dataloaders = fabric.setup_dataloaders(*test_dataloaders)

    # Setup memory buffer for continual learning (stores historical samples for replay)
    if hasattr(args, 'cl_method') and args.cl_method != 'joint':
        if hasattr(args, 'buffer_size') and args.buffer_size > 0:
            if args.buffer_balance:
                # Class-balanced buffer: equal storage space for each class
                buffer = ClsssIncrementalBuffer(buffer_size=args.buffer_size)
            else:
                # Simple buffer: no class balancing, random sample storage
                buffer = SimpleBuffer(buffer_size=args.buffer_size)

    # Initialize result storage and seen classes tracking
    results = []
    seen_classes = np.empty(0, dtype=int)

    # ====== 2. Main Task Sequence Training Loop ======
    for task in range(args.n_tasks):

        # ====== 2.1 Current Task Initialization Block ======
        # Clean up old model from previous task and release GPU memory
        if 'old_model' in locals():
            del old_model
        torch.cuda.empty_cache()

        # Clean up previous log files and setup logger for current task
        if os.path.exists(f'{args.log_dir}/{args.exp_name}/fold_{fold}_task_{task}/metrics.csv'):
            os.remove(f'{args.log_dir}/{args.exp_name}/fold_{fold}_task_{task}/metrics.csv')
        logger = CSVLogger(root_dir=f'{args.log_dir}', name=f'{args.exp_name}', version=f'fold_{fold}_task_{task}')

        # ====== 2.2 Data Preparation and Class Management Block ======
        # Get data module for current task (includes train, val, test data)
        datamodule = datamodule_gen(args, fold=fold, task=task)
        # Extract classes contained in current task
        cur_classes = np.asarray(np.unique(datamodule['train_dataset'].targets), dtype=int)
        # Save previously seen classes (for knowledge distillation)
        old_seen_classes = seen_classes
        # Update seen classes set
        seen_classes = np.append(seen_classes, cur_classes)
        # Calculate unseen classes (for masking unseen class outputs)
        unseen_classes = list(set(np.arange(args.n_classes)) - set(seen_classes))

        # ====== 2.3 Optimizer Configuration Block ======
        # Select learning rate based on config (supports different lr per task)
        if isinstance(args.lr, list):
            lr = float(args.lr[task])
        else:
            lr = float(args.lr)

        # Create optimizer (supports Adam and SGD)
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        elif args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)

        # Setup data loaders
        # model, optimizer = fabric.setup(model, optimizer)
        # train_loader, val_loader = fabric.setup_dataloaders(datamodule['train_loader'], datamodule['val_loader'])
        train_loader, val_loader = datamodule['train_loader'], datamodule['val_loader']

        # ====== 2.4 Continual Learning Method Configuration Block ======
        # Configure continual learning settings (knowledge distillation, weight normalization, etc.)
        if hasattr(args, 'cl_method') and args.cl_method != 'joint':
            # Save old model for knowledge distillation methods (LwF: Learning without Forgetting, MICIL, prev)
            if args.cl_method in ['LwF', 'MICIL', 'prev'] and task > 0:
                old_model = copy.deepcopy(model)
                # Freeze old model parameters, only use for inference
                for param in old_model.parameters():
                    param.requires_grad = False
                old_model.eval()
                old_model = fabric.to_device(old_model)

            # Weight Normalization - prevent representation drift
            if args.wn and task > 0:
                with torch.no_grad():
                    if args.net == 'transmil':
                        model._fc2.weight.data = F.normalize(model._fc2.weight.data)
                    elif args.net in ['clam_sb', 'clam_mb']:
                        model.classifiers.weight.data = F.normalize(model.classifiers.weight.data)
                    else:
                        raise NotImplementedError

        # ====== 3. Model Training Main Loop Block ======
        # Determine number of training epochs for current task
        if isinstance(args.epochs, list):
            epochs = args.epochs[task]
        else:
            epochs = args.epochs

        # Configure early stopping mechanism
        if args.early_stop:
            early_stop = EarlyStopping(patience=args.patience, stop_epoch=epochs//2)

        # ====== 3.1 Training Epoch Loop ======
        for i in range(epochs):
            # ====== 3.1.1 Training Metrics Initialization ======
            # Initialize training and validation loss recorders
            train_loss_metric = MeanMetric()
            val_loss_metric = MeanMetric()

            # Initialize accuracy calculator based on classification task type
            if args.n_classes == 2:
                val_acc_metric = fabric.to_device(Accuracy(task='binary'))
                # val_auc_metric = fabric.to_device(AUROC(task='binary'))
            else:
                val_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=args.n_classes, average='micro'))
                # val_auc_metric = fabric.to_device(AUROC(task='multiclass', num_classes=average='weighted'))

            # ====== 3.1.2 Training Batch Loop ======
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                # Limit batch count in debug mode
                if args.n_batches > 0 and batch_idx > args.n_batches:
                    break

                # Prepare logging and zero gradients
                logger_batch = {'epoch': i, 'batch': batch_idx}
                # print(batch['features'].shape)
                optimizer.zero_grad()
                batch = fabric.to_device(batch)

                # ====== 3.1.2.1 TransMIL Model Forward Pass and Loss Computation ======
                if args.net == 'transmil':
                    # Forward pass
                    out = model(batch['features'])
                    # Mask unseen class outputs (prevent high confidence predictions on unseen classes)
                    out['logits'][:, unseen_classes] = -100
                    # Calculate classification loss
                    loss = F.cross_entropy(out['logits'], batch['label'])

                    # ====== Continual Learning Loss Computation (Current Task Data) ======
                    if hasattr(args, 'cl_method') and args.cl_on_current:
                        if args.cl_method == 'LwF' and task > 0:
                            # Learning without Forgetting: knowledge distillation loss
                            old_out = old_model(batch['features'])
                            kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            loss = loss + kd_loss
                            logger_batch.update({'kd_loss': kd_loss.item()})
                        elif args.cl_method == 'MICIL' and task > 0:
                            # MICIL: knowledge distillation + feature distillation
                            old_out = old_model(batch['features'])
                            kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            em_loss = F.mse_loss(out['features'], old_out['features'])
                            loss = loss + 10 * kd_loss + em_loss
                            logger_batch.update({'kd_loss': 10 * kd_loss.item(), 'em_loss': em_loss.item()})
                        elif args.cl_method == 'prev' and task > 0:
                            # Attention preservation method: maintain attention patterns
                            old_out = old_model(batch['features'], return_attn=True)
                            # logits_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            attn_loss = kd_loss_fn(out['attn1'], old_out['attn1']) + kd_loss_fn(out['attn2'], old_out['attn2']) + \
                                        F.mse_loss(out['h1'], old_out['h1']) + F.mse_loss(out['h2'], old_out['h2'])
                            # em_loss = F.mse_loss(out['features'], old_out['features'])
                            loss = loss + attn_loss
                            logger_batch.update({'attn_loss': attn_loss.item()})

                    # Update training loss and log
                    train_loss_metric.update(loss.item())
                    logger_batch.update({'loss': loss.item()})
                    logger.log_metrics(logger_batch)

                # ====== 3.1.2.2 CLAM Model Forward Pass and Loss Computation ======
                elif args.net in ['clam_sb', 'clam_mb']:
                    # print(batch['features'].device)
                    # print(batch['label'])

                    # CLAM forward pass: includes bag-level and instance-level predictions
                    out = model(batch['features'], batch['label'], instance_eval=True, return_features=True, seen_classes=seen_classes)
                    bag_loss = F.cross_entropy(out['logits'], batch['label'])  # Bag-level loss
                    inst_loss = out['instance_loss']  # Instance-level loss
                    loss = 0.7*bag_loss + 0.3*inst_loss  # Weighted combination

                    # ====== Continual Learning Loss Computation (Current Task Data) ======
                    if hasattr(args, 'cl_method') and args.cl_on_current:
                        if args.cl_method == 'LwF' and task > 0:
                            # Learning without Forgetting: knowledge distillation loss
                            old_out = old_model(batch['features'], batch['label'], instance_eval=True, seen_classes=old_seen_classes)
                            kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            loss = loss + kd_loss
                            logger_batch.update({'kd_loss': kd_loss.item()})
                        elif args.cl_method == 'MICIL' and task > 0:
                            # MICIL: knowledge distillation + feature distillation
                            old_out = old_model(batch['features'], batch['label'], instance_eval=True, return_features=True, seen_classes=old_seen_classes)
                            kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            em_loss = F.mse_loss(out['features'], old_out['features'])
                            loss = loss + 10 * kd_loss + em_loss
                            logger_batch.update({'kd_loss': 10 * kd_loss.item(), 'em_loss': em_loss.item()})
                        elif args.cl_method == 'prev' and task > 0:
                            # Attention preservation method: maintain attention weights
                            old_out = old_model(batch['features'], batch['label'], instance_eval=True, return_features=True, seen_classes=old_seen_classes)
                            # logits_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            attn_loss = kd_loss_fn(out['A'], old_out['A'])  # Attention weight distillation
                            # em_loss = F.mse_loss(out['features'], old_out['z'])
                            loss = loss + attn_loss
                            logger_batch.update({'attn_loss': attn_loss.item()})

                    # Update training loss and log
                    train_loss_metric.update(loss.item())
                    # Safely log losses (handle potential None values)
                    log_dict = {'bag_loss': bag_loss.item(), 'loss': loss.item()}
                    if inst_loss is not None:
                        log_dict['inst_loss'] = inst_loss.item()
                    logger_batch.update(log_dict)
                    logger.log_metrics(logger_batch)
                else:
                    raise NotImplementedError

                # ====== 3.1.2.3 Backpropagation and Parameter Update ======
                fabric.backward(loss)
                optimizer.step()

                # ====== 3.1.2.4 Weight Normalization (Prevent Representation Drift) ======
                if args.wn and task > 0 and hasattr(args, 'cl_method'):
                    with torch.no_grad():
                        if args.net == 'transmil':
                            model._fc2.weight.data = F.normalize(model._fc2.weight.data)
                        elif args.net in ['clam_sb', 'clam_mb']:
                            model.classifiers.weight.data = F.normalize(model.classifiers.weight.data)
                        else:
                            raise NotImplementedError

                # ====== 3.1.2.5 Memory Buffer Replay Training ======
                # From second task onwards, use historical samples from buffer for replay training
                if task > 0 and hasattr(args, 'cl_method') and hasattr(args, 'buffer_size') and args.buffer_size > 0:
                    # Reset buffer sampling order at the start of each epoch
                    if batch_idx == 0: buffer.start_epoch()

                    # Ensure balanced training ratio between buffer and current task samples
                    # When current batch progress >= buffer sample output progress, sample from buffer
                    while batch_idx / len(train_loader) >= buffer.get_samples_output_count() / len(buffer):
                        old_batch = buffer.get_next_batch()
                        if old_batch is None: raise StopIteration  # All buffer samples have been traversed
                        old_batch = old_batch[0]
                        logger_batch = {'epoch': i, 'batch': batch_idx}
                        old_batch = fabric.to_device(old_batch)
                        optimizer.zero_grad()

                        # ====== 3.1.2.5.1 TransMIL Buffer Sample Training ======
                        if args.net == 'transmil':
                            out = model(old_batch['features'], return_attn=True)
                            out['logits'][:, unseen_classes] = -100
                            loss = F.cross_entropy(out['logits'], old_batch['label'])

                            if hasattr(args, 'cl_method'):
                                if args.cl_method == 'LwF' and task > 0:
                                    old_out = old_model(old_batch['features'])
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    loss = loss + kd_loss
                                    logger_batch.update({'logits_loss': kd_loss.item()})
                                elif args.cl_method == 'derpp' and task > 0:
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_batch['logits'][:, old_seen_classes])
                                    loss = loss + kd_loss
                                    logger_batch.update({'logits_loss': kd_loss.item()})
                                elif args.cl_method == 'MICIL' and task > 0:
                                    old_out = old_model(old_batch['features'])
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    em_loss = F.mse_loss(out['features'], old_out['features'])
                                    loss = loss + 10 * kd_loss + em_loss
                                    logger_batch.update({'kd_loss': 10 * kd_loss.item(), 'em_loss': em_loss.item()})
                                elif args.cl_method == 'prev' and task > 0:
                                    old_out = old_model(old_batch['features'], return_attn=True)
                                    p_loss = 0.
                                    logits_loss = 10 * kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    attn_loss = 10 * (kd_loss_fn(out['attn1'], old_out['attn1'], ta=1) + kd_loss_fn(out['attn2'], old_out['attn2'], ta=1))
                                    h_loss = 10 * (F.mse_loss(out['h1'], old_out['h1']) + F.mse_loss(out['h2'], old_out['h2']))
                                    # p_loss +=  kd_loss_fn(out['vv1'], old_out['vv1']) + kd_loss_fn(out['vv2'], old_out['vv2'])
                                    # p_loss += F.mse_loss(out['features'], old_out['features'])
                                    loss = loss + logits_loss + attn_loss + h_loss
                                    logger_batch.update({'logits_loss': logits_loss.item(), 'attn_loss': attn_loss.item(), 'h_loss': h_loss.item()})
                                elif args.cl_method == 'ER':
                                    pass
                                else:
                                    raise NotImplementedError

                            train_loss_metric.update(loss.item())
                        elif args.net in ['clam_sb', 'clam_mb']:
                            out = model(old_batch['features'], old_batch['label'], instance_eval=True, return_features=True, seen_classes=seen_classes)
                            bag_loss = F.cross_entropy(out['logits'], old_batch['label'])
                            inst_loss = out['instance_loss']
                            loss = 0.7*bag_loss + 0.3*inst_loss
                            logger_batch.update({'bag_loss': bag_loss.item(), 'inst_loss': inst_loss.item()})

                            if hasattr(args, 'cl_method'):
                                if args.cl_method == 'LwF' and task > 0:
                                    old_out = old_model(old_batch['features'], old_batch['label'], instance_eval=True, return_features=True, seen_classes=old_seen_classes)
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    loss = loss + kd_loss
                                    logger_batch.update({'logits_loss': kd_loss.item()})
                                elif args.cl_method == 'derpp' and task > 0:
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_batch['logits'][:, old_seen_classes])
                                    loss = loss + kd_loss
                                    logger_batch.update({'logits_loss': kd_loss.item()})
                                elif args.cl_method == 'MICIL' and task > 0:
                                    old_out = old_model(old_batch['features'], old_batch['label'], instance_eval=True, return_features=True, seen_classes=old_seen_classes)
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    em_loss = F.mse_loss(out['features'], old_out['features'])
                                    loss = loss + 10 * kd_loss + em_loss
                                    logger_batch.update({'logits_loss': 10 * kd_loss.item(), 'features_loss': em_loss.item()})
                                elif args.cl_method == 'prev' and task > 0:
                                    old_out = old_model(old_batch['features'], old_batch['label'], instance_eval=True, return_features=True, seen_classes=old_seen_classes)
                                    logits_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    attn_loss = kd_loss_fn(out['A'], old_out['A'])
                                    loss = loss + attn_loss + 10 * logits_loss
                                    logger_batch.update({'attn_loss': attn_loss.item(), 'logits_loss': 10 * logits_loss.item()})
                                elif args.cl_method == 'ER':
                                    pass
                                else:
                                    raise NotImplementedError

                            train_loss_metric.update(loss.item())
                        else:
                            raise NotImplementedError
                        
                        logger_batch.update({'loss': loss.item()})
                        logger.log_metrics(logger_batch)

                        fabric.backward(loss)
                        optimizer.step()

                        if args.wn and task > 0 and hasattr(args, 'cl_method'):
                            with torch.no_grad():
                                if args.net == 'transmil':
                                    model._fc2.weight.data = F.normalize(model._fc2.weight.data)
                                elif args.net in ['clam_sb', 'clam_mb']:
                                    model.classifiers.weight.data = F.normalize(model.classifiers.weight.data)
                                else:
                                    raise NotImplementedError
                                    

            # ====== 3.2 Validation Phase ======
            model.eval()
            for batch_idx, batch in enumerate(val_loader):
                # Limit validation batch count in debug mode
                if args.n_batches > 0 and batch_idx > args.n_batches:
                    break

                batch = fabric.to_device(batch)

                # ====== 3.2.1 CLAM Model Validation ======
                if args.net in ['clam_sb', 'clam_mb']:
                    with torch.no_grad():
                        out = model(batch['features'], batch['label'], instance_eval=True, seen_classes=seen_classes)
                    bag_loss = F.cross_entropy(out['logits'], batch['label'])
                    inst_loss = out['instance_loss']
                    loss = 0.7*bag_loss + 0.3*inst_loss
                    y_prob = F.softmax(out['logits'], dim=-1)  # Prediction probabilities
                    y_hat = out['logits'].argmax(dim=-1)  # Predicted classes

                    val_loss_metric.update(loss.item())
                    val_acc_metric.update(y_hat, batch['label'])
                    # val_auc_metric.update(y_prob, batch['label'])

                # ====== 3.2.2 TransMIL Model Validation ======
                elif args.net == 'transmil':
                    with torch.no_grad():
                        out = model(batch['features'])
                    out['logits'][:, unseen_classes] = -100  # Mask unseen classes
                    loss = F.cross_entropy(out['logits'], batch['label'])
                    y_prob = F.softmax(out['logits'], dim=-1)  # Prediction probabilities
                    y_hat = out['logits'].argmax(dim=-1)  # Predicted classes

                    val_loss_metric.update(loss.item())
                    val_acc_metric.update(y_hat, batch['label'])
                    # val_auc_metric.update(y_prob, batch['label'])
                else:
                    raise NotImplementedError
                

            # ====== 3.3 Logging and Early Stopping Check After Epoch ======
            if args.net in ['clam_sb', 'clam_mb', 'transmil']:
                # Compute and log training and validation metrics
                train_loss = train_loss_metric.compute().item()
                val_loss = val_loss_metric.compute().item()
                val_acc = val_acc_metric.compute().item()
                # val_auc = val_auc_metric.compute().item()
                print(f'Epoch {i}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, val_acc={val_acc:.5f}\n')

                # Reset metric recorders for next epoch
                train_loss_metric.reset()
                val_loss_metric.reset()
                val_acc_metric.reset()
                # val_auc_metric.reset()
                logger.log_metrics({'epoch': i, 'train_loss_epoch': train_loss, 'val_loss_epoch': val_loss, 'val_acc_epoch': val_acc})
            else:
                train_loss = train_loss_metric.compute().item()
                val_loss = val_loss_metric.compute().item()
                print(f'Epoch {i}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}\n')

                train_loss_metric.reset()
                val_loss_metric.reset()
                logger.log_metrics({'epoch': i, 'train_loss_epoch': train_loss, 'val_loss_epoch': val_loss})

            # ====== 3.4 Early Stopping Check ======
            if args.early_stop:
                early_stop(i, val_loss)
                # If current validation loss is minimum, save model
                if early_stop.val_loss_min == val_loss:
                    torch.save(model.state_dict(), f'{args.log_dir}/{args.exp_name}/fold_{fold}_task_{task}.pt')
                # Check if early stopping condition is met
                if early_stop.early_stop:
                    print("Early stopping")
                    break

        # ====== 4. Model Saving and Loading Best Weights ======
        # If early stopping not used, save model after training ends
        if not args.early_stop:
            torch.save(model.state_dict(), f'{args.log_dir}/{args.exp_name}/fold_{fold}_task_{task}.pt')

        # Load best model weights and switch to evaluation mode
        model.load_state_dict(torch.load(f'{args.log_dir}/{args.exp_name}/fold_{fold}_task_{task}.pt'))
        model.eval()
        torch.cuda.empty_cache()

        # TODO: Select examples

        # ====== 5. Memory Buffer Data Addition Block ======
        # Add representative samples from current task to buffer for future task replay training
        # Note: Last task doesn't need buffer save (no subsequent tasks)
        if hasattr(args, 'cl_method') and hasattr(args, 'buffer_size') and args.buffer_size > 0 and task < args.n_tasks - 1:
            # Start monitoring buffer selection
            buffer_start_time = time.time()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            seed_everything(args.seed)
            # Adjust buffer size based on number of new classes
            buffer.adjust_buffer_size_by_new_classes(len(cur_classes))
            n_seen_samples_per_task = min(args.buffer_size, len(train_loader))

            # ====== 5.1 Traverse Training Data and Selectively Add to Buffer ======
            for batch_idx, batch in enumerate(train_loader):
                # seed_everything(args.seed + batch_idx)
                # Use reservoir sampling to judge whether to add current sample to buffer
                buffer_idx = buffer.add_judge(batch['label'].item())
                if buffer_idx == -1: continue  # Don't add current sample
                # ====== 5.2 Sample Feature Compression (Optional) ======
                # If buffer sample size limit is configured, perform feature compression
                if args.buffer_slide_size > 0:
                    # with torch.no_grad():
                    # Calculate compressed feature size
                    buffer_slide_size = args.buffer_slide_size * batch['features'].size(0) if isinstance(args.buffer_slide_size, float) else args.buffer_slide_size
                    batch = fabric.to_device(batch)

                    # ====== 5.2.1 CLAM Model Feature Distillation ======
                    if args.net in ['clam_sb', 'clam_mb']:
                        with torch.no_grad():
                            out = model(batch['features'], batch['label'], seen_classes=seen_classes)
                            attn = out['A'].mean(dim=0, keepdim=True).view(1, -1)  # Get attention weights
                        # Select representative features based on attention weights
                        slide = distill_slide(batch['features'], attn, size=buffer_slide_size, method=args.distill_method,model=model,label=batch['label'],task_id=task,seen_classes=seen_classes,tb_writer=tb_writer)
                        batch['features'] = slide

                    # ====== 5.2.2 TransMIL Model Feature Distillation ======
                    elif args.net == 'transmil':
                        out = model(batch['features'], return_attn=True)
                        # Normalize attention weights
                        attn1, attn2 = out['attn1']/(out['attn1'].max(dim=1, keepdim=True)[0]), out['attn2']/(out['attn2'].max(dim=1, keepdim=True)[0])
                        attn = torch.ones(attn1.shape).to(attn1.device)
                        attn = (attn2 + 1) * (attn1 + 1) / 4  # Combine two layers of attention
                        attn = attn.mean(dim=0, keepdim=True).view(1, -1)
                        # Select representative features based on attention weights
                        slide = distill_slide(batch['features'], attn, size=buffer_slide_size, method=args.distill_method,model=model,label=batch['label'],seen_classes=seen_classes,tb_writer=tb_writer)
                        batch['features'] = slide
                    else:
                        raise NotImplementedError

                # ====== 5.3 DER++ Method Logits Storage ======
                # If using DER++ method, need to store current model output logits
                if args.cl_method in ['derpp']:
                    with torch.no_grad():
                        batch = fabric.to_device(batch)
                        if args.net in ['clam_sb', 'clam_mb']:
                            out = model(batch['features'], batch['label'], seen_classes=seen_classes)
                        elif args.net == 'transmil':
                            out = model(batch['features'])
                    batch['logits'] = out['logits']  # Save logits for future distillation

                # ====== 5.4 Add Sample to Buffer ======
                buffer.add(batch, buffer_idx)
                # if batch_idx >= n_seen_samples_per_task:
                #     break

            # ====== 5.5 Buffer Status Logging ======
            print(f'Buffer size: {len(buffer)}')
            print(f'Number of patches in buffer: {buffer.n_patches_total}')
            print(f'Labels in buffer: {buffer.labels}')
            # Save buffer class distribution to CSV file
            buffer_labels_dir = f'{args.log_dir}/{args.exp_name}/fold_{fold}_task_{task}'
            os.makedirs(buffer_labels_dir, exist_ok=True)
            with open(f'{buffer_labels_dir}/buffer_labels.csv', 'a') as f:
                for key in buffer.labels.keys():
                    f.write("%s,%s\n"%(key,buffer.labels[key]))

            # End monitoring and log
            buffer_end_time = time.time()
            buffer_duration = buffer_end_time - buffer_start_time
            buffer_peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

            print(f'Buffer selection duration: {buffer_duration:.2f} seconds')
            print(f'Buffer selection peak memory: {buffer_peak_memory:.2f} MB')

            logger.log_metrics({'buffer_size': len(buffer), 'n_patches_in_buffer': buffer.n_patches_total,
                              'buffer_selection_duration_seconds': buffer_duration,
                              'buffer_selection_peak_memory_mb': buffer_peak_memory})

        # ====== 6. Memory Cleanup ======
        del train_loader, val_loader, optimizer, datamodule

        # ====== 7. Model Testing and Evaluation Block ======
        # Evaluate current model performance on test sets from all tasks
        model.eval()
        # Initialize test accuracy calculator based on classification task type
        if args.n_classes == 2:
            test_acc_metric = fabric.to_device(Accuracy(task='binary'))
            # test_auc_metric = fabric.to_device(AUROC(task='binary'))
        else:
            test_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=args.n_classes, average='micro'))
            # test_auc_metric = fabric.to_device(AUROC(task='multiclass', num_classes=args.n_classes, average='weighted'))

        # ====== 7.1 Evaluate on All Task Test Sets ======
        result = {'fold': fold, 'task': task}
        for idx, test_loader in enumerate(test_dataloaders):
            # Traverse test set of each task (test forward and backward transfer)
            for batch_idx, batch in enumerate(test_loader):
                # Limit test batch count in debug mode
                if args.n_batches > 0 and batch_idx > args.n_batches:
                    break

                batch = fabric.to_device(batch)

                # ====== 7.1.1 CLAM Model Test Inference ======
                if args.net in ['clam_sb', 'clam_mb']:
                    with torch.no_grad():
                        out = model(batch['features'], seen_classes=seen_classes)
                    # bag_loss = F.cross_entropy(out['logits'], batch['label'])
                    y_hat = out['logits'].argmax(dim=-1)  # Predicted classes
                    y_prob = F.softmax(out['logits'], dim=-1)  # Prediction probabilities

                    test_acc_metric.update(y_hat, batch['label'])
                    # test_auc_metric.update(y_prob, batch['label'])

                # ====== 7.1.2 TransMIL Model Test Inference ======
                elif args.net == 'transmil':
                    with torch.no_grad():
                        out = model(batch['features'])
                    out['logits'][:, unseen_classes] = -100  # Mask unseen classes
                    y_hat = out['logits'].argmax(dim=-1)  # Predicted classes
                    y_prob = F.softmax(out['logits'], dim=-1)  # Prediction probabilities

                    test_acc_metric.update(y_hat, batch['label'])
                    # test_auc_metric.update(y_prob, batch['label'])
                else:
                    raise NotImplementedError

            # ====== 7.1.3 Log Current Task Test Results ======
            result.update({f'{idx}_acc': test_acc_metric.compute().item()})
            test_acc_metric.reset()
            # test_auc_metric.reset()

        # ====== 7.2 Save and Output Results ======
        print(result)
        results.append(result)
        # Save results to CSV file
        df = pd.DataFrame(results)
        df.to_csv(f'{args.log_dir}/{args.exp_name}/fold_{fold}_results.csv', index=False)

        # Log task completion status
        logger.finalize(f"Success on fold {fold} task {task}!")

    # ====== 8. Return Test Results from All Tasks ======
    return results # [{'fold': 0, 'task': 0, '0_auc': 0.9, '0_acc': 0.8}, {...}, ...]


def one_fold_jt(args, fold=0):
    fabric = Fabric(devices=1, accelerator="auto")
    seed_everything(args.seed)

    model = load_model(args)
    model = fabric.to_device(model)

    # Prepare test dataloaders from all tasks
    datamodule = datamodule_gen(args, fold=fold, task=args.n_tasks)

    # optimizer
    if isinstance(args.lr, list):
        lr = float(args.lr[0])
    else:
        lr = float(args.lr)
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)

    # model, optimizer = fabric.setup(model, optimizer)
    # train_loader, val_loader = fabric.setup_dataloaders(datamodule['train_loader'], datamodule['val_loader'])
    train_loader, val_loader = datamodule['train_loader'], datamodule['val_loader']

    logger = CSVLogger(root_dir=f'{args.log_dir}', name=f'{args.exp_name}', version=f'fold_{fold}_JT')

    # Fitting
    if isinstance(args.epochs, list):
        epochs = args.epochs[0]
    else:
        epochs = args.epochs
    if args.early_stop:
        early_stop = EarlyStopping(patience=args.patience, stop_epoch=epochs//2)

    for i in range(epochs):
        # init logging
        train_loss_metric = MeanMetric()
        val_loss_metric = MeanMetric()
        if args.n_classes == 2:
            val_acc_metric = fabric.to_device(Accuracy(task='binary'))
            # val_auc_metric = fabric.to_device(AUROC(task='binary'))
        else:
            val_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=args.n_classes, average='micro'))
            # val_auc_metric = fabric.to_device(AUROC(task='multiclass', num_classes=args.n_classes, average='weighted'))

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            if args.n_batches > 0 and batch_idx > args.n_batches:
                break
            
            logger_batch = {'epoch': i, 'batch': batch_idx}

            optimizer.zero_grad()
            batch = fabric.to_device(batch)

            if args.net == 'transmil':
                out = model(batch['features'])
                loss = F.cross_entropy(out['logits'], batch['label'])

                train_loss_metric.update(loss.item())
                logger_batch.update({'loss': loss.item()})
                logger.log_metrics(logger_batch)

            elif args.net in ['clam_sb', 'clam_mb']:
                out = model(batch['features'], batch['label'], instance_eval=True, return_features=True)
                bag_loss = F.cross_entropy(out['logits'], batch['label'])
                inst_loss = out['instance_loss']
                loss = 0.7*bag_loss + 0.3*inst_loss

                train_loss_metric.update(loss.item())
                logger_batch.update({'bag_loss': bag_loss.item(), 'inst_loss': inst_loss.item(), 'loss': loss.item()})
                logger.log_metrics(logger_batch)
            else:
                raise NotImplementedError

            fabric.backward(loss)
            optimizer.step()

        # validation
        model.eval()
        for batch_idx, batch in enumerate(val_loader):
            if args.n_batches > 0 and batch_idx > args.n_batches:
                break

            batch = fabric.to_device(batch)

            if args.net in ['clam_sb', 'clam_mb']:
                with torch.no_grad():
                    out = model(batch['features'], batch['label'], instance_eval=True)
                bag_loss = F.cross_entropy(out['logits'], batch['label'])
                inst_loss = out['instance_loss']
                loss = 0.7*bag_loss + 0.3*inst_loss
                y_prob = F.softmax(out['logits'], dim=-1)
                y_hat = out['logits'].argmax(dim=-1)

                val_loss_metric.update(loss.item())
                val_acc_metric.update(y_hat, batch['label'])
                # val_auc_metric.update(y_prob, batch['label'])
            elif args.net == 'transmil':
                with torch.no_grad():
                    out = model(batch['features'])
                loss = F.cross_entropy(out['logits'], batch['label'])
                y_prob = F.softmax(out['logits'], dim=-1)
                y_hat = out['logits'].argmax(dim=-1)

                val_loss_metric.update(loss.item())
                val_acc_metric.update(y_hat, batch['label'])
                # val_auc_metric.update(y_prob, batch['label'])
            else:
                raise NotImplementedError
            

        # logging each epoch
        if args.net in ['clam_sb', 'clam_mb', 'transmil']:
            train_loss = train_loss_metric.compute().item()
            val_loss = val_loss_metric.compute().item()
            val_acc = val_acc_metric.compute().item()
            # val_auc = val_auc_metric.compute().item()
            print(f'Epoch {i}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, val_acc={val_acc:.5f}\n')

            train_loss_metric.reset()
            val_loss_metric.reset()
            val_acc_metric.reset()
            # val_auc_metric.reset()
            logger.log_metrics({'epoch': i, 'train_loss_epoch': train_loss, 'val_loss_epoch': val_loss, 'val_acc_epoch': val_acc})
        else:
            train_loss = train_loss_metric.compute().item()
            val_loss = val_loss_metric.compute().item()
            print(f'Epoch {i}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}\n')

            train_loss_metric.reset()
            val_loss_metric.reset()
            logger.log_metrics({'epoch': i, 'train_loss_epoch': train_loss, 'val_loss_epoch': val_loss})

        if args.early_stop:
            early_stop(i, val_loss)
            if early_stop.val_loss_min == val_loss:
                torch.save(model.state_dict(), f'{args.log_dir}/{args.exp_name}/fold_{fold}_JT.pt')
            if early_stop.early_stop:
                print("Early stopping")
                break

    del train_loader, val_loader, optimizer
    # testing, log acc and auc
    model.load_state_dict(torch.load(f'{args.log_dir}/{args.exp_name}/fold_{fold}_JT.pt'))
    model.eval()

    if args.n_classes == 2:
        test_acc_metric = fabric.to_device(Accuracy(task='binary'))
        test_acc_metric_task = fabric.to_device(Accuracy(task='binary'))
    else:
        test_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=args.n_classes, average='micro'))
        test_acc_metric_task = fabric.to_device(Accuracy(task='multiclass', num_classes=args.n_classes, average='micro'))
    
    test_dataloaders = fabric.setup_dataloaders(*datamodule['test_loader'])
    result = {'fold': fold}
    for idx, test_loader in enumerate(test_dataloaders):
        for batch_idx, batch in enumerate(test_loader):
            if args.n_batches > 0 and batch_idx > args.n_batches:
                break

            if args.net in ['clam_sb', 'clam_mb']:
                with torch.no_grad():
                    out = model(batch['features'])
                # bag_loss = F.cross_entropy(out['logits'], batch['label'])
                y_hat = out['logits'].argmax(dim=-1)
                # y_prob = F.softmax(out['logits'], dim=-1)

                test_acc_metric.update(y_hat, batch['label'])
                test_acc_metric_task.update(y_hat, batch['label'])
                # test_auc_metric.update(y_prob, batch['label'])
            elif args.net == 'transmil':
                with torch.no_grad():
                    out = model(batch['features'])
                y_hat = out['logits'].argmax(dim=-1)
                # y_prob = F.softmax(out['logits'], dim=-1)

                test_acc_metric.update(y_hat, batch['label'])
                test_acc_metric_task.update(y_hat, batch['label'])
                # test_auc_metric.update(y_prob, batch['label'])
            else:
                raise NotImplementedError
        result.update({f'{idx}_acc': test_acc_metric_task.compute().item()})

        test_acc_metric_task.reset()
        # test_auc_metric.reset()

    result.update({'acc': test_acc_metric.compute().item()})

    print(result)
    df = pd.DataFrame(result, index=[0])
    df.to_csv(f'{args.log_dir}/{args.exp_name}/fold_{fold}_results_JT.csv', index=False)
    logger.finalize(f"Success on fold {fold}!")
    return result

def test_on_one_fold(args, fold=0):
    fabric = Fabric(devices=1, accelerator="auto")
    model = load_model(args)
    test_dataloaders = [datamodule_gen(args, fold=fold, task=task)['test_loader'] for task in range(args.n_tasks)]
    test_dataloaders = fabric.setup_dataloaders(*test_dataloaders)

    results = []
    seen_classes = np.empty(0, dtype=int)
    for model_idx in range(args.n_tasks):
        cur_classes = np.asarray(np.unique(test_dataloaders[model_idx].dataset.targets), dtype=int)
        seen_classes = np.append(seen_classes, cur_classes)

        if args.load is not None:
            model.load_state_dict(torch.load(args.load))
        else:
            model.load_state_dict(torch.load(f'{args.log_dir}/{args.exp_name}/fold_{fold}_task_{model_idx}.pt'))
        model = fabric.to_device(model)
        model.eval()

        result = {'fold': fold, 'model': model_idx}

        for idx, test_loader in enumerate(test_dataloaders[:model_idx+1]):
            cur_classes = np.asarray(np.unique(test_loader.dataset.targets), dtype=int)
            if len(seen_classes) <= 2:
                ci_acc_metric = fabric.to_device(Accuracy(task='binary'))
                ci_auc_metric = fabric.to_device(AUROC(task='binary'))
            else:
                ci_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=len(seen_classes), average='micro'))
                ci_auc_metric = fabric.to_device(AUROC(task='multiclass', num_classes=len(seen_classes), average='weighted'))
            if len(cur_classes) <= 2:
                ti_acc_metric = fabric.to_device(Accuracy(task='binary'))
                ti_auc_metric = fabric.to_device(AUROC(task='binary'))
            else:
                ti_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=len(cur_classes), average='micro'))
                ti_auc_metric = fabric.to_device(AUROC(task='multiclass', num_classes=len(cur_classes), average='weighted'))

            for batch_idx, batch in enumerate(test_loader):
                if args.n_batches > 0 and batch_idx > args.n_batches:
                    break

                if args.net in ['clam_sb', 'clam_mb', 'transmil']:
                    with torch.no_grad():
                        out = model(batch['features'])

                    ci_logit = out['logits'][:, seen_classes]
                    ci_label = F.one_hot(batch['label'], num_classes=args.n_classes)[:, seen_classes]
                    ci_label = ci_label.argmax(dim=-1) 
                    y_hat = ci_logit.argmax(dim=-1)
                    y_prob = F.softmax(ci_logit, dim=-1)
                    if len(seen_classes) <= 2: y_prob = y_prob[:, -1]
                    ci_acc_metric.update(y_hat, ci_label)
                    ci_auc_metric.update(y_prob, ci_label)

                    ti_logit = out['logits'][:, cur_classes]
                    ti_label = F.one_hot(batch['label'], num_classes=args.n_classes)[:, cur_classes]
                    ti_label = ti_label.argmax(dim=-1)
                    y_hat = ti_logit.argmax(dim=-1)
                    y_prob = F.softmax(ti_logit, dim=-1)
                    if len(cur_classes) <= 2: y_prob = y_prob[:, -1]
                    ti_acc_metric.update(y_hat, ti_label)
                    ti_auc_metric.update(y_prob, ti_label)


            result.update({f'{idx}_ci_auc': ci_auc_metric.compute().item(), 
                           f'{idx}_ci_acc': ci_acc_metric.compute().item(),
                           f'{idx}_ti_auc': ti_auc_metric.compute().item(), 
                           f'{idx}_ti_acc': ti_acc_metric.compute().item()})
                
        print(result)
        results.append(result)

    return results

def test(args):
    seed_everything(args.seed)

    if args.load is None:
        results = []
        # Test models from all tasks
        for fold in range(args.folds_start, args.folds_end):
            result = test_on_one_fold(args, fold=fold)
            results.extend(result)
    else:
        fold = max(args.folds_start, 0)
        # model_fold = args.load.split('_')[-2]
        # assert fold == int(model_fold), f'Fold mismatch: {fold} vs {model_fold}'
        results = test_on_one_fold(args, fold=fold)

    # convert to csv
    df = pd.DataFrame(results)
    log_path = f'{args.log_dir}/{args.exp_name}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    df.to_csv(f'{log_path}/test_results.csv', index=False)

def main(args):
    log_path = f'{args.log_dir}/{args.exp_name}'
    if not os.path.exists(log_path): # Log directory
        os.makedirs(log_path)
    # Save args to yaml
    with open(f'{log_path}/args.yaml', 'w') as f: # Write parameters
        yaml.dump(vars(args), f)

    results = []
    for fold in range(args.folds_start, args.folds_end):
        if 'jt' in args.dataset or (hasattr(args, 'cl_method') and args.cl_method == 'joint'):
            result = one_fold_jt(args, fold=fold)
        else:
            result = one_fold(args, fold=fold)

        results.extend(result)

    # Convert to csv
    df = pd.DataFrame(results)
    df.to_csv(f'{log_path}/results.csv', index=False)

    # ====== Compute and Print Continual Learning Metrics ======
    if 'jt' not in args.dataset and (not hasattr(args, 'cl_method') or args.cl_method != 'joint'):
        # Only compute metrics in continual learning scenarios (not joint training)
        print("\nComputing continual learning metrics...")

        # Compute metrics for all folds
        cl_metrics = compute_all_folds_metrics(results, joint_accuracies_dict=None, verbose=False)

        # Determine model name
        model_name_map = {
            'clam_sb': 'CLAM-SB',
            'clam_mb': 'CLAM-MB',
            'transmil': 'TransMIL'
        }
        model_name = model_name_map.get(args.net, args.net.upper())

        # Print table
        method_name = getattr(args, 'cl_method', 'Ours').upper()
        print_cl_metrics_table(cl_metrics, method_name=method_name, model_name=model_name)

        # Save metrics to file
        metrics_df = pd.DataFrame({
            'Method': [method_name],
            'Model': [model_name],
            'AACC_mean': [cl_metrics['mean']['AACC']],
            'AACC_std': [cl_metrics['std']['AACC']],
            'BWT_mean': [cl_metrics['mean']['BWT']],
            'BWT_std': [cl_metrics['std']['BWT']],
            'IM_mean': [cl_metrics['mean']['IM']],
            'IM_std': [cl_metrics['std']['IM']]
        })
        metrics_df.to_csv(f'{log_path}/cl_metrics.csv', index=False)
        print(f"Continual learning metrics saved to: {log_path}/cl_metrics.csv\n")

        # Return AACC mean
        return cl_metrics['mean']['AACC']

    return None
    
    
if __name__ == '__main__':
    args = init_args()

    if args.testing:
        test(args)
    else:
        main(args)
