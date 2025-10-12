import torch
import torch.nn.functional as F
import numpy as np
import math
from tools.topk_selection import SigmoidTopK
import matplotlib.pyplot as plt

class Training():

    def __init__(self, proxy_model, beta, device, lr_proxy_model, lr_weights,distall_lamda=0.1,draw_curve=False):
        self.proxy_model = proxy_model
        self.origin_model=None
        self.lr_p =  lr_proxy_model # 代理模型的学习率
        self.lr_w =  lr_weights # w的学习率
        self.optimizer_theta_p_model = torch.optim.SGD(self.proxy_model.parameters(), lr=self.lr_p) # 代理模型优化器
        self.weight_optimizer = None
        self.device = device
        self.eta = 0.5 #
        self.beta = beta # 正则化项的系数
        self.buffer = []
        self.identity = []
        self.distall_lamda=distall_lamda
        self.draw_curve = draw_curve
        
        # 可微分 topk 选择器缓存（支持不同的k值）
        self.topk_selectors = {}
        self.topk_temperature = 0.1  # 温度参数

    # @ 初始化proxymodel，内部无调用
    def init_proxy_model(self):
        for m in self.proxy_model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)

    # 内层训练，外部调用，data_S, target_S分别是训练的batch,
    # sample_weights是样本的权重，也就是coreset的权重
    # 里面有一个task id

    def train_inner(self, data_S, target_S, task_id, sample_weights, inner_epchos,seen_classes=None):
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
            weighted_data = data * sample_weights.unsqueeze(1)
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


    # train_outer， 更新w，refx这里永远是None
    def train_outer(self, data, target, task_id, data_weights, topk, ref_x=None, ref_y=None,seen_classes=None):
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

        # 优化：直接使用，无需再次 .to(device)
        X_S = data[:]
        y_S = target[:]

        return self.update_sample_weights(data, target, task_id, X_S, y_S, sample_weights, topk, beta=self.beta, ref_x=ref_x, ref_y=ref_y,seen_classes=seen_classes)


    # 更新权重w，内部调用
    def update_sample_weights(self, input_train, target_train, task_id, input_selected, target_selected,  sample_weights, topk, beta, seen_classes=None,epsilon=1e-3, ref_x=None, ref_y=None):
        z = torch.normal(0, 1, size=[topk]).cuda()

        proxy_output = self.proxy_model(input_train, target_train, instance_eval=True, return_features=True, seen_classes=seen_classes)

        loss_outer = F.cross_entropy(proxy_output["logits"],target_train, reduction='none')
        
        topk_weights, ind = sample_weights.topk(topk)
        
        if self.origin_model is not None:

            with torch.no_grad():
                origin_output=self.origin_model(input_train, target_train, instance_eval=True, return_features=True, seen_classes=seen_classes)

            # 使用余弦相似度损失：1 - cosine_similarity
            # cosine_similarity 返回 [-1, 1]，我们用 1 - sim 得到 [0, 2] 的损失
            cosine_sim = F.cosine_similarity(
                origin_output["features"].detach(),
                proxy_output["features"],
                dim=-1
            )
            distall_loss = (1 - cosine_sim).mean()
            

        
        loss_outer_avg = torch.mean(loss_outer) - beta*(topk_weights + epsilon*z).sum()+self.distall_lamda*distall_loss


        d_theta = torch.autograd.grad(loss_outer_avg, self.proxy_model.parameters(),allow_unused=True)
        v_0  = d_theta

        # 使用可微分 topk 算子选择 top-k 权重（带缓存优化）
        if topk not in self.topk_selectors:
            self.topk_selectors[topk] = SigmoidTopK(k=topk, temperature=self.topk_temperature)

        # projectesd_weights = self.topk_selectors[topk](sample_weights)
        # 归一化 sample_weights 防止梯度爆炸

        normalized_weights = F.normalize(sample_weights.unsqueeze(0), p=2, dim=1).squeeze(0)
        topk_weights = self.topk_selectors[topk](normalized_weights)

        # 使用 unsqueeze 替代转置操作，更高效
        weighted_input = input_selected * topk_weights.unsqueeze(1)
        
        loss_inner = torch.mean( F.cross_entropy(
            self.proxy_model(weighted_input, target_selected, instance_eval=True, return_features=True, seen_classes=seen_classes)["logits"], target_selected, reduction='none'))
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
        
        
        
        with torch.no_grad():
            sample_weights -= self.lr_w * jacobian

        return  sample_weights, jacobian, loss_outer

