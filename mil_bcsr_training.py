import torch
import torch.nn.functional as F
import numpy as np
import math
from tools.weighted_topk_selector import WeightedTopKSelector
import matplotlib.pyplot as plt

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


class Training():

    def __init__(self, proxy_model, beta, device, lr_proxy_model, lr_weights, distall_lamda=0.1, draw_curve=False, topk=10, topk_method='sigmoid', topk_temperature=0.1, normalize_method='none', distill_target='logits'):
        """
        初始化训练模块

        参数:
            proxy_model: 代理模型
            beta: 正则化系数
            device: 设备
            lr_proxy_model: 代理模型学习率
            lr_weights: 权重学习率
            distall_lamda: 知识蒸馏损失系数
            draw_curve: 是否绘制曲线
            topk: top-k 参数
            topk_method: TopK 选择方法 ('sigmoid', 'gumbel', 'ste')
            topk_temperature: TopK 温度参数
            normalize_method: 权重归一化方法 ('l2', 'softmax', 'zscore', 'none')
            distill_target: 知识蒸馏对象 ('logits', 'features', 'both')
        """
        self.proxy_model = proxy_model
        self.origin_model = None
        self.lr_p = lr_proxy_model  # 代理模型的学习率
        self.lr_w = lr_weights  # w的学习率
        self.optimizer_theta_p_model = torch.optim.SGD(self.proxy_model.parameters(), lr=self.lr_p)  # 代理模型优化器
        self.weight_optimizer = None
        self.device = device

        self.beta = beta  # 正则化项的系数
        self.buffer = []
        self.identity = []
        self.distall_lamda = distall_lamda
        self.draw_curve = draw_curve
        self.topk = topk

        # 可微分 topk 选择器参数
        self.topk_temperature = topk_temperature
        self.topk_method = topk_method
        self.normalize_method = normalize_method
        self.distill_target = distill_target

        # 创建 TopK 选择器
        self.weighted_topk_selector = WeightedTopKSelector(
            k=topk,
            temperature=self.topk_temperature,
            normalize_method=normalize_method,
            topk_method=topk_method
        )
      
        

    # @ 初始化proxymodel，内部无调用
    def init_proxy_model(self):
        for m in self.proxy_model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)

    # 内层训练，外部调用，data_S, target_S分别是训练的batch,
    # sample_weights是样本的权重，也就是coreset的权重
    # 里面有一个task id

    def train_inner(self, data_S, target_S, task_id, sample_weights, inner_epchos,seen_classes=None,topk=10):
        # 核心训练过程，
        # 通过代理模型对样本权重进行更新
        # 加权 w 是一个常数

        # 数据准备 - 移到循环外，避免重复的 CPU-GPU 数据传输
        # 优化：检查是否已在目标设备上，避免不必要的传输
        if data_S.device != self.device:
            data = data_S.to(self.device).type(torch.float)
        else:
            data = data_S.type(torch.float)

        if target_S.device != self.device:
            target = target_S.to(self.device).type(torch.long)
        else:
            target = target_S.type(torch.long)

        if sample_weights.device != self.device:
            sample_weights = sample_weights.to(self.device).type(torch.float).detach()
        else:
            sample_weights = sample_weights.type(torch.float).detach()

        # 创建优化器 - 只创建一次，避免重置优化器状态
        optimizer = torch.optim.SGD(self.proxy_model.parameters(), lr=self.lr_p)

        loss_value = 0.0
        for i in range(inner_epchos):
            self.proxy_model.train()
            optimizer.zero_grad()

            # 前向传播
            # 使用 unsqueeze 替代转置操作，更高效
            weighted_data = self.weighted_topk_selector(data, sample_weights)
            output = self.proxy_model(
                weighted_data,
                target,
                instance_eval=True,
                return_features=True,
                seen_classes=seen_classes
            )["logits"]

            # 损失计算 - 使用 mean 代替 none
            loss = F.cross_entropy(output, target, reduction='mean')

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 保存损失值用于返回
            loss_value = loss.item()

            # 打印inner loop的loss
            if i < inner_epchos - 1:
                print(f"  ├─ Inner[{i+1}/{inner_epchos}]: {loss_value:.4f}")
            else:
                print(f"  └─ Inner[{i+1}/{inner_epchos}]: {loss_value:.4f} ✓")

        # 返回最后一次迭代的损失（作为标量tensor，保持与原代码接口兼容）
        return torch.tensor(loss_value, device=self.device)


    def compute_outer_loss(self, input_train, target_train, input_selected, sample_weights, seen_classes=None):
        """
        计算外层优化的损失函数

        参数:
            input_train: 训练数据
            target_train: 训练标签
            input_selected: 选中的样本数据
            sample_weights: 样本权重
            seen_classes: 已见类别

        返回:
            loss_outer_avg: 平均外层损失
            loss_outer: 每个样本的损失（用于梯度计算）
        """
        self.proxy_model.train()

        # 1. 计算代理模型在训练数据上的输出
        proxy_output = self.proxy_model(input_train, target_train, return_features=True, seen_classes=seen_classes)

        # 2. 计算分类损失（每个样本）
        loss_outer = F.cross_entropy(proxy_output["logits"][:, seen_classes], target_train, reduction='none')

        # 3. 计算知识蒸馏损失（如果有原始模型）
        distall_loss = 0.0
        if self.origin_model is not None:
            # 获取加权输入
            weighted_input = self.weighted_topk_selector(input_selected, sample_weights)

            # 原始模型输出（不需要梯度）
            with torch.no_grad():
                origin_output = self.origin_model(weighted_input, target_train, return_features=True, seen_classes=seen_classes)

            # 代理模型输出（需要梯度）
            new_output = self.proxy_model(weighted_input, target_train, return_features=True, seen_classes=seen_classes)

            # 根据 distill_target 参数计算不同的蒸馏损失
            if self.distill_target == 'logits':
                # 只蒸馏 logits
                distall_loss = kd_loss_fn(
                    new_output["logits"][:, seen_classes],
                    origin_output["logits"][:, seen_classes],
                    ta=2,
                    softmax=True
                )
            elif self.distill_target == 'features':
                # 只蒸馏 features
                distall_loss = F.mse_loss(new_output["features"], origin_output["features"])
            elif self.distill_target == 'both':
                # 同时蒸馏 logits 和 features
                logits_loss = kd_loss_fn(
                    new_output["logits"][:, seen_classes],
                    origin_output["logits"][:, seen_classes],
                    ta=2,
                    softmax=True
                )
                features_loss = F.mse_loss(new_output["features"], origin_output["features"])
                distall_loss = logits_loss + features_loss
            else:
                raise ValueError(f"Invalid distill_target: {self.distill_target}. Must be 'logits', 'features', or 'both'.")

        # 4. 组合总损失
        loss_outer_avg = torch.mean(loss_outer) + self.distall_lamda * distall_loss
        # print(loss_outer_avg, loss_outer)
        # print("Classification Loss: {:.4f}, Distillation Loss: {:.4f}".format(torch.mean(loss_outer).item(), distall_loss.item()))
        return loss_outer_avg, loss_outer

    # train_outer， 更新w
    def train_outer(self, data, target, data_weights, seen_classes=None):
        """
        外层训练入口，更新样本权重

        参数:
            data: 输入数据
            target: 标签
            data_weights: 初始样本权重
            seen_classes: 已见类别
        """
        # 优化：检查设备并避免重复传输
        if data.device != self.device:
            data = data.to(self.device)

        if target.device != self.device:
            target = target.to(self.device).type(torch.long)
        else:
            target = target.type(torch.long)

        if data_weights.device != self.device:
            sample_weights = data_weights.to(self.device)
        else:
            sample_weights = data_weights

        # 调用更新权重函数
        return self.update_sample_weights(
            input_data=data,
            target_data=target,
            sample_weights=sample_weights,
            seen_classes=seen_classes
        )


    # 更新权重w，内部调用
    def update_sample_weights(self, input_data, target_data, sample_weights, seen_classes=None):
        """
        使用双层优化更新样本权重

        参数:
            input_data: 输入数据
            target_data: 标签数据
            sample_weights: 样本权重
            seen_classes: 已见类别

        返回:
            sample_weights: 更新后的样本权重
            gradient: 权重梯度
            loss_outer_avg: 外层损失
        """

        # 计算外层损失
        loss_outer_avg, loss_outer = self.compute_outer_loss(
            input_train=input_data,
            target_train=target_data,
            input_selected=input_data,
            sample_weights=sample_weights,
            seen_classes=seen_classes
        )

        print(f"  ├─ Outer: {loss_outer_avg.item():.4f}")

        d_g_d_w=torch.autograd.grad(loss_outer_avg, sample_weights,allow_unused=True,retain_graph=True)[0]

        d_g_d_theta = torch.autograd.grad(loss_outer_avg, self.proxy_model.parameters(),allow_unused=True)
        v_0  = d_g_d_theta


        # 使用 WeightedTopKSelector 模块进行加权操作
        weighted_input = self.weighted_topk_selector(input_data, sample_weights)

        loss_inner = torch.mean( F.cross_entropy(
            self.proxy_model(weighted_input, target_data,  return_features=True, seen_classes=seen_classes)["logits"][:,seen_classes], target_data, reduction='none'))
        grads_theta = torch.autograd.grad(loss_inner, self.proxy_model.parameters(), create_graph=True,allow_unused=True)
        G_theta = []
        for p, g in zip(self.proxy_model.parameters(), grads_theta):
            if g == None:
                # G_theta.append(None)
                pass
            else:
                G_theta.append(p-self.lr_p*g)
        
        v_Q = v_0
        for _ in range(3):

            para_list=self.proxy_model.parameters()
            v_new = torch.autograd.grad(G_theta, para_list, grad_outputs=v_0, retain_graph=True,allow_unused=True)
            v_0 = [i.detach() for i in v_new if i is not None]
            for i in range(len(v_0)):
                v_Q[i].add_(v_0[i].detach())
        grads_theta=[i for i in grads_theta if i is not None]
        jacobian = -torch.autograd.grad(grads_theta, sample_weights, grad_outputs=v_Q,allow_unused=True)[0]

        gradient = d_g_d_w+jacobian
        # gradient=gradient/ (torch.norm(gradient)+epsilon)
        with torch.no_grad():
            sample_weights -= self.lr_w * gradient

        return  sample_weights, gradient, loss_outer_avg

