"""
核心集权重初始化模块

提供多种初始化策略，支持配置化选择和超参数调节
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any


class CoresetWeightInitializer:
    """
    核心集权重初始化器

    支持多种初始化策略：
    1. uniform_random: 均匀随机分布 U(0,1)
    2. uniform_equal: 均匀权重 (所有样本权重相同)
    3. gaussian: 高斯分布
    4. attention_based: 基于模型注意力机制
    5. entropy_based: 基于预测熵
    6. hybrid: 混合策略

    Args:
        init_method: 初始化方法名称
        model_type: 模型类型 ('clam_sb', 'clam_mb', 'transmil')
        hyperparams: 方法特定的超参数字典
    """

    def __init__(
        self,
        init_method: str = 'uniform_random',
        model_type: str = 'clam_sb',
        hyperparams: Optional[Dict[str, Any]] = None
    ):
        self.init_method = init_method.lower()
        self.model_type = model_type.lower()
        self.hyperparams = hyperparams or {}

        # 验证初始化方法
        valid_methods = [
            'uniform_random',
            'uniform_equal',
            'gaussian',
            'attention_based',
            'entropy_based',
            'hybrid'
        ]
        if self.init_method not in valid_methods:
            raise ValueError(f"Unknown init method: {self.init_method}. "
                           f"Choose from {valid_methods}")

        # 设置默认超参数
        self._set_default_hyperparams()

    def _set_default_hyperparams(self):
        """设置各方法的默认超参数"""
        defaults = {
            'uniform_random': {
                'low': 0.0,
                'high': 1.0,
            },
            'uniform_equal': {
                'base_value': None,  # 如果None，则为1/n
            },
            'gaussian': {
                'mean': 0.5,
                'std': 0.2,
                'activation': 'sigmoid',  # sigmoid 或 relu
            },
            'attention_based': {
                'aggregation': 'mean',  # mean, max, weighted_mean
                'normalize': True,
                'temperature': 1.0,
            },
            'entropy_based': {
                'normalization': 'minmax',  # minmax 或 softmax
                'inverse': True,  # 低熵高权重
                'eps': 1e-8,
            },
            'hybrid': {
                'attention_weight': 0.7,  # 注意力权重占比
                'entropy_weight': 0.3,    # 熵权重占比
                'temperature': 1.0,
            }
        }

        # 合并默认值和用户指定的超参数
        method_defaults = defaults[self.init_method]
        method_defaults.update(self.hyperparams)
        self.hyperparams = method_defaults

    def initialize(
        self,
        n_samples: int,
        device: torch.device,
        model: Optional[torch.nn.Module] = None,
        X: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        初始化核心集权重

        Args:
            n_samples: 样本数量
            device: 设备
            model: 模型（对于基于模型的初始化方法）
            X: 输入数据（对于基于模型的初始化方法）
            **kwargs: 其他参数

        Returns:
            权重张量 [n_samples]，shape为(n_samples,)，requires_grad=True
        """

        if self.init_method == 'uniform_random':
            return self._uniform_random(n_samples, device)

        elif self.init_method == 'uniform_equal':
            return self._uniform_equal(n_samples, device)

        elif self.init_method == 'gaussian':
            return self._gaussian(n_samples, device)

        elif self.init_method == 'attention_based':
            if model is None or X is None:
                raise ValueError("attention_based requires model and X")
            return self._attention_based(n_samples, device, model, X)

        elif self.init_method == 'entropy_based':
            if model is None or X is None:
                raise ValueError("entropy_based requires model and X")
            return self._entropy_based(n_samples, device, model, X)

        elif self.init_method == 'hybrid':
            if model is None or X is None:
                raise ValueError("hybrid requires model and X")
            return self._hybrid(n_samples, device, model, X)

        else:
            raise NotImplementedError(f"Method {self.init_method} not implemented")

    def _uniform_random(
        self,
        n_samples: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        均匀随机分布初始化

        权重 ~ U(low, high)，默认 U(0, 1)
        """
        low = self.hyperparams['low']
        high = self.hyperparams['high']

        weights = torch.rand(n_samples, device=device) * (high - low) + low
        return weights.requires_grad_(True)

    def _uniform_equal(
        self,
        n_samples: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        均匀等权重初始化

        所有样本权重相同，默认为 1/n
        """
        base_value = self.hyperparams['base_value']

        if base_value is None:
            weights = torch.ones(n_samples, device=device) / n_samples
        else:
            weights = torch.ones(n_samples, device=device) * base_value

        return weights.requires_grad_(True)

    def _gaussian(
        self,
        n_samples: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        高斯分布初始化

        权重 ~ N(mean, std)，通过激活函数映射到 [0,1]
        """
        mean = self.hyperparams['mean']
        std = self.hyperparams['std']
        activation = self.hyperparams['activation'].lower()

        # 从正态分布采样
        raw_weights = torch.randn(n_samples, device=device) * std + mean

        # 通过激活函数映射
        if activation == 'sigmoid':
            weights = torch.sigmoid(raw_weights)
        elif activation == 'relu':
            weights = F.relu(raw_weights)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        return weights.requires_grad_(True)

    def _attention_based(
        self,
        n_samples: int,
        device: torch.device,
        model: torch.nn.Module,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        基于注意力机制初始化

        使用模型的注意力权重作为初始值
        """
        aggregation = self.hyperparams['aggregation']
        normalize = self.hyperparams['normalize']
        temperature = self.hyperparams['temperature']

        with torch.no_grad():
            # 获取注意力权重
            if self.model_type in ['clam_sb', 'clam_mb']:
                out = model(X)
                attn = out['A']  # [batch, n_patches] 或 [n_patches]

                # 聚合注意力
                if aggregation == 'mean':
                    weights = attn.mean(dim=0)
                elif aggregation == 'max':
                    weights = attn.max(dim=0)[0]
                elif aggregation == 'weighted_mean':
                    # 加权平均：用最大注意力作为权重
                    max_attn = attn.max(dim=0, keepdim=True)[0]
                    normalized_attn = attn / (max_attn + 1e-8)
                    weights = (attn * normalized_attn).sum(dim=0) / (normalized_attn.sum(dim=0) + 1e-8)
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")

            elif self.model_type == 'transmil':
                out = model(X, return_attn=True)
                attn1 = out['attn1']  # [batch, n_patches]
                attn2 = out['attn2']  # [batch, n_patches]

                if aggregation == 'mean':
                    weights = (attn1.mean(dim=0) + attn2.mean(dim=0)) / 2
                elif aggregation == 'max':
                    weights = (attn1.max(dim=0)[0] + attn2.max(dim=0)[0]) / 2
                elif aggregation == 'weighted_mean':
                    max_attn1 = attn1.max(dim=0, keepdim=True)[0]
                    max_attn2 = attn2.max(dim=0, keepdim=True)[0]
                    norm_attn1 = attn1 / (max_attn1 + 1e-8)
                    norm_attn2 = attn2 / (max_attn2 + 1e-8)
                    w1 = (attn1 * norm_attn1).sum(dim=0) / (norm_attn1.sum(dim=0) + 1e-8)
                    w2 = (attn2 * norm_attn2).sum(dim=0) / (norm_attn2.sum(dim=0) + 1e-8)
                    weights = (w1 + w2) / 2
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # 应用温度系数
            weights = weights ** (1.0 / temperature)

            # 归一化
            if normalize:
                weights = F.softmax(weights, dim=-1)
            else:
                weights = weights / (weights.sum() + 1e-8)

        return weights.clone().detach().requires_grad_(True)

    def _entropy_based(
        self,
        n_samples: int,
        device: torch.device,
        model: torch.nn.Module,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        基于预测熵初始化

        低熵样本（高置信度）权重高，高熵样本（低置信度）权重低
        """
        normalization = self.hyperparams['normalization']
        inverse = self.hyperparams['inverse']
        eps = self.hyperparams['eps']

        with torch.no_grad():
            # 获取模型预测
            if self.model_type in ['clam_sb', 'clam_mb']:
                out = model(X)
                logits = out['logits']  # [batch, n_classes]
            elif self.model_type == 'transmil':
                out = model(X)
                logits = out['logits']  # [batch, n_classes]
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # 计算熵
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)  # [batch]

            # 取平均（如果有多个batch）
            if entropy.dim() > 0:
                entropy = entropy.mean(dim=0)

            # 反转：低熵 → 高权重
            if inverse:
                max_entropy = torch.log(torch.tensor(logits.shape[-1], device=device, dtype=torch.float))
                weights = 1.0 - (entropy / (max_entropy + eps))
            else:
                weights = entropy

            # 归一化
            if normalization == 'minmax':
                min_w, max_w = weights.min(), weights.max()
                weights = (weights - min_w) / (max_w - min_w + eps)
            elif normalization == 'softmax':
                weights = F.softmax(weights, dim=-1)
            else:
                raise ValueError(f"Unknown normalization: {normalization}")

        return weights.clone().detach().requires_grad_(True)

    def _hybrid(
        self,
        n_samples: int,
        device: torch.device,
        model: torch.nn.Module,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        混合初始化

        结合注意力和熵的加权组合
        """
        attn_weight = self.hyperparams['attention_weight']
        entropy_weight = self.hyperparams['entropy_weight']

        # 确保权重和为1
        total = attn_weight + entropy_weight
        attn_weight = attn_weight / total
        entropy_weight = entropy_weight / total

        # 获取基于注意力的权重
        old_hyper = self.hyperparams.copy()
        self.init_method = 'attention_based'
        self._set_default_hyperparams()
        attn_weights = self._attention_based(n_samples, device, model, X).detach()

        # 获取基于熵的权重
        self.init_method = 'entropy_based'
        self._set_default_hyperparams()
        entropy_weights = self._entropy_based(n_samples, device, model, X).detach()

        # 恢复原始配置
        self.init_method = 'hybrid'
        self.hyperparams = old_hyper

        # 混合
        weights = attn_weight * attn_weights + entropy_weight * entropy_weights
        weights = F.softmax(weights, dim=-1)  # 确保和为1

        return weights.requires_grad_(True)


def create_initializer(
    init_method: str = 'uniform_random',
    model_type: str = 'clam_sb',
    **hyperparams
) -> CoresetWeightInitializer:
    """
    工厂函数，创建初始化器

    Args:
        init_method: 初始化方法
        model_type: 模型类型
        **hyperparams: 超参数

    Returns:
        CoresetWeightInitializer 实例
    """
    return CoresetWeightInitializer(
        init_method=init_method,
        model_type=model_type,
        hyperparams=hyperparams
    )
