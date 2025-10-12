"""
Sample Selection Strategies for Continual Learning.

This module provides various methods for selecting representative samples
from MIL bags for buffer storage, including attention-based and bilevel optimization methods.
"""

import torch
import copy
from mil_bcsr_coreset import BCSR_Coreset


def adaptive_distill(slide, size=1e5, method='random', model=None):
    """
    Placeholder for adaptive distillation method.

    Args:
        slide: Feature tensor of shape (n_patches, feature_dim)
        size: Target number of patches to select
        method: Selection method
        model: Model for adaptive selection

    Returns:
        Selected feature tensor
    """
    pass


def distill_slide(slide, attn=None, size=1e5, method='random', model=None, label=None, task_id=None, seen_classes=None, args=None):
    """
    Select a subset of patches from a slide based on various strategies.

    Args:
        slide (torch.Tensor): Feature tensor of shape (n_patches, feature_dim)
        attn (torch.Tensor, optional): Attention weights for patches
        size (int/float): Target number of patches to select (or fraction if float < 1)
        method (str): Selection strategy:
            - 'random': Random sampling
            - 'max': Top-k by attention
            - 'maxmin': Half max, half min attention
            - 'maxminrand': Mix of max, min, and random
            - 'maxrand': Half max attention, half random
            - 'kibo': Bilevel optimization using BCSR coreset
        model: Model for bilevel optimization (used with 'kibo' method)
        label: Sample label (used with 'kibo' method)
        task_id: Current task ID (used with 'kibo' method)
        seen_classes: Previously seen classes (used with 'kibo' method)
        args: Arguments object containing hyperparameters (used with 'kibo' method)

    Returns:
        torch.Tensor: Selected subset of patches
    """
    assert len(slide.shape) == 2, f"slide shape: {slide.shape}"
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

    elif method == 'adaptive':
        pass

    elif method == 'kibo':
        # Bilevel optimization coreset selection
        proxy_model = copy.deepcopy(model)
        for param in proxy_model.parameters():
            param.requires_grad = True

        # Initialize BCSR coreset selector with bilevel optimization
        if args is not None:
            BCSR_Coreset_selector = BCSR_Coreset(
                proxy_model,
                lr_proxy_model=args.bcsr_lr_proxy_model,
                beta=args.bcsr_beta,
                out_dim=args.bcsr_out_dim,
                max_outer_it=args.bcsr_max_outer_it,
                max_inner_it=args.bcsr_max_inner_it,
                weight_lr=args.bcsr_weight_lr,
                candidate_batch_size=args.bcsr_candidate_batch_size,
                logging_period=args.bcsr_logging_period,
                distall_lamda=args.distall_lamda,
                draw_curve=args.draw_curve)
            topk = args.buffer_size
        else:
            # Default parameters
            BCSR_Coreset_selector = BCSR_Coreset(
                proxy_model,
                lr_proxy_model=10,
                beta=0.1,
                out_dim=100,
                max_outer_it=5,
                max_inner_it=1,
                weight_lr=10,
                candidate_batch_size=size,
                logging_period=1000)
            topk = size

        idx, outer_loss = BCSR_Coreset_selector.coreset_select(
            proxy_model,
            slide.cpu().numpy(),
            label.cpu().numpy(),
            task_id=task_id,
            topk=topk,
            out_loss=None,
            seen_classes=seen_classes)

        idx = idx.cpu()

    else:
        raise NotImplementedError(f"Method '{method}' not implemented")

    return slide[idx]
