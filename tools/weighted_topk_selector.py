import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.topk_selection import SigmoidTopK, GumbelTopK, STETopK


class WeightedTopKSelector(nn.Module):
    """
    A module that applies weighted topk selection to input data.

    This module performs three operations in sequence:
    1. Normalize sample weights
    2. Apply differentiable topk selection
    3. Weight the input data with selected weights

    Args:
        k: Number of top samples to select
        temperature: Temperature parameter for topk selection (default: 0.1)
        normalize_method: Method for normalizing weights ('l2', 'softmax', 'zscore', 'none')
        topk_method: TopK selection method ('sigmoid', 'gumbel', 'ste')
            - 'sigmoid': Smooth selection, outputs values close to 0/1 (推荐)
            - 'gumbel': Adds Gumbel noise during training for exploration
            - 'ste': Hard 0/1 selection with soft gradients
    """

    def __init__(self, k, temperature=0.1, normalize_method='l2', topk_method='sigmoid'):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.normalize_method = normalize_method
        self.topk_method = topk_method

        # 根据选择的方法创建 TopK 选择器
        if topk_method == 'sigmoid':
            self.topk_selector = SigmoidTopK(k=k, temperature=temperature)
        elif topk_method == 'gumbel':
            self.topk_selector = GumbelTopK(k=k, temperature=temperature)
        elif topk_method == 'ste':
            self.topk_selector = STETopK(k=k, temperature=temperature)
        else:
            raise ValueError(f"Unknown topk_method: {topk_method}. Choose from ['sigmoid', 'gumbel', 'ste']")

    def normalize_weights(self, weights):
        """
        Normalize sample weights using specified method.

        Args:
            weights: Weight tensor of shape (n_samples,)

        Returns:
            Normalized weight tensor of shape (n_samples,)
        """
        if self.normalize_method == 'l2':
            # L2 normalization
            return F.normalize(weights.unsqueeze(0), p=2, dim=1).squeeze(0)
        elif self.normalize_method == 'softmax':
            # Softmax normalization
            return F.softmax(weights, dim=0)
        elif self.normalize_method == 'zscore':
            # Z-score normalization (standardization): (x - mean) / std
            mean = weights.mean()
            std = weights.std(unbiased=False)
            # Add small epsilon to avoid division by zero
            return (weights - mean) / (std + 1e-8)
        elif self.normalize_method == 'none':
            # No normalization
            return weights
        else:
            raise ValueError(f"Unknown normalize method: {self.normalize_method}")

    def forward(self, data, sample_weights):
        """
        Apply weighted topk selection to data.

        Args:
            data: Input data tensor of shape (n_samples, feature_dim)
            sample_weights: Weight tensor of shape (n_samples,)

        Returns:
            Weighted data tensor of shape (n_samples, feature_dim)
        """
        # 1. Normalize weights
        normalized_weights = self.normalize_weights(sample_weights)

        # 2. Apply topk selection
        topk_weights = self.topk_selector(normalized_weights)

        # 3. Weight the data
        weighted_data = data * topk_weights.unsqueeze(1)

        return weighted_data

    def update_k(self, new_k):
        """Update the k value and reinitialize topk selector."""
        self.k = new_k

        # 根据当前的方法重新创建选择器
        if self.topk_method == 'sigmoid':
            self.topk_selector = SigmoidTopK(k=new_k, temperature=self.temperature)
        elif self.topk_method == 'gumbel':
            self.topk_selector = GumbelTopK(k=new_k, temperature=self.temperature)
        elif self.topk_method == 'ste':
            self.topk_selector = STETopK(k=new_k, temperature=self.temperature)
