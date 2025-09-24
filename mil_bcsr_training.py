import torch
import torch.nn.functional as F
import numpy as np
import math

def _concat(xs):
    """
    将多个张量展平并拼接成一个一维张量

    该函数用于将模型参数或梯度张量列表转换为单一的扁平张量，
    常用于梯度计算和参数更新的中间步骤。

    参数:
        xs: 张量列表，每个张量可以是任意形状

    返回:
        torch.Tensor: 拼接后的一维张量

    示例:
        >>> x1 = torch.tensor([[1, 2], [3, 4]])  # 形状: (2, 2)
        >>> x2 = torch.tensor([5, 6])           # 形状: (2,)
        >>> _concat([x1, x2])                   # 输出: tensor([1, 2, 3, 4, 5, 6])
    """
    return torch.cat([x.view(-1) for x in xs])

class Training():
    """
    BCSR 双层优化训练器

    该类实现了 BCSR (Bilevel Coreset Selection with Representativity) 的核心训练逻辑。
    通过双层优化框架：内层优化模型参数，外层优化样本权重，实现智能的核心集选择。

    双层优化原理:
        - 内层问题: 给定样本权重，优化模型参数 θ
        - 外层问题: 给定模型参数，优化样本权重 w
        - 目标: 选择的样本子集能够最大化模型在验证集上的性能
    """

    def __init__(self, proxy_model, beta, device, lr_proxy_model, lr_weights):
        """
        初始化 BCSR 训练器

        参数:
            proxy_model: 代理模型，用于评估样本选择质量的神经网络模型
            beta: 正则化系数，控制样本选择的多样性 (0 < beta < 1)
            device: 计算设备 ('cuda' 或 'cpu')
            lr_proxy_model: 代理模型的学习率，控制内层优化的步长
            lr_weights: 样本权重的学习率，控制外层优化的步长

        属性初始化:
            - proxy_model: 存储传入的代理模型
            - lr_p: 内层优化学习率 (模型参数)
            - lr_w: 外层优化学习率 (样本权重)
            - optimizer_theta_p_model: 代理模型的 SGD 优化器
            - eta: 内部超参数，用于梯度计算
            - beta: 正则化参数，平衡损失和样本多样性
        """
        self.proxy_model = proxy_model                      # 代理模型
        self.lr_p = lr_proxy_model                         # 代理模型的学习率
        self.lr_w = lr_weights                             # 样本权重的学习率
        self.optimizer_theta_p_model = torch.optim.SGD(   # 代理模型优化器
            self.proxy_model.parameters(), lr=self.lr_p
        )
        self.weight_optimizer = None                       # 权重优化器 (预留)
        self.device = device                               # 计算设备
        self.eta = 0.5                                     # 内部超参数
        self.beta = beta                                   # 正则化系数
        self.buffer = []                                   # 缓冲区 (预留)
        self.identity = []                                 # 身份标识 (预留)

    def init_proxy_model(self):
        """
        初始化代理模型的权重参数

        使用 Xavier 均匀分布初始化卷积层和线性层的权重，有助于：
        1. 避免梯度消失或梯度爆炸问题
        2. 提供更好的训练起点
        3. 提高双层优化的收敛性

        输入:
            无 (使用类内部的 self.proxy_model)

        输出:
            无 (直接修改模型权重)

        初始化策略:
            - Conv2d 层: Xavier 均匀分布初始化
            - Linear 层: Xavier 均匀分布初始化
            - 其他层: 保持默认初始化

        注意:
            该方法目前在代码中未被调用，属于预留功能
        """
        for m in self.proxy_model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)

    def train_inner(self, data_S, target_S, task_id, sample_weights, inner_epochs, seen_classes=None):
        """
        内层优化：固定样本权重，优化模型参数

        这是双层优化中的内层问题，目标是在给定样本权重的情况下，
        通过加权损失训练代理模型参数，使模型能够适应当前的样本选择策略。

        参数:
            data_S: 输入特征数据，形状为 [n_samples, n_features]
            target_S: 目标标签，形状为 [n_samples]
            task_id: 任务标识符，用于多任务学习中的标签偏移
            sample_weights: 样本权重向量，形状为 [n_samples]，表示每个样本的重要性
            inner_epochs: 内层优化的迭代次数
            seen_classes: 已见过的类别列表，用于持续学习场景

        返回:
            loss: 最后一次迭代的损失值 (torch.Tensor)

        内层优化流程:
            1. 对每个 epoch 重新初始化优化器（避免动量积累）
            2. 将数据和权重移动到指定设备
            3. 计算加权特征：weight * data (逐样本加权)
            4. 通过代理模型前向传播得到预测结果
            5. 计算交叉熵损失（带任务偏移）
            6. 反向传播并更新模型参数

        数学原理:
            内层问题: min_θ L_inner(θ, w) = Σ w_i * CrossEntropy(f_θ(x_i), y_i + task_id*2)
            其中 w_i 是样本权重，f_θ 是代理模型
        """
        loss = math.inf  # 初始化损失值

        for _ in range(inner_epochs):
            # 每个 epoch 重新初始化优化器，避免动量对权重更新的干扰
            self.optimizer_theta_p_model = torch.optim.SGD(
                self.proxy_model.parameters(), lr=self.lr_p
            )

            # 设置模型为训练模式
            self.proxy_model.train()
            self.optimizer_theta_p_model.zero_grad()

            # 数据类型转换和设备迁移
            data = data_S.to(self.device).type(torch.float)
            target = target_S.to(self.device).type(torch.long)
            sample_weights = sample_weights.to(self.device).type(torch.float).detach()

            # 计算加权输入特征：每个样本乘以对应的权重
            # (sample_weights * data.T).T 实现了逐样本的权重缩放
            weighted_data = (sample_weights * data.T).T #

            # 通过代理模型进行前向传播
            output = self.proxy_model(
                weighted_data, target,
                instance_eval=True,
                return_features=True,
                seen_classes=seen_classes
            )["logits"]

            # 计算交叉熵损失
            # target + task_id*2: 为不同任务添加标签偏移，避免类别冲突
            loss = F.cross_entropy(output, target + task_id*2, reduction='none')

            # 反向传播和参数更新
            loss.backward()
            self.optimizer_theta_p_model.step()
            self.proxy_model.zero_grad()

        return loss


    def train_outer(self, data, target, task_id, data_weights, topk, ref_x=None, ref_y=None, seen_classes=None):
        """
        外层优化：固定模型参数，优化样本权重

        这是双层优化中的外层问题，目标是在固定代理模型参数的情况下，
        更新样本权重以最大化模型在验证集上的性能。

        参数:
            data: 输入特征数据，形状为 [n_samples, n_features]
            target: 目标标签，形状为 [n_samples]
            task_id: 任务标识符，用于多任务学习
            data_weights: 当前样本权重，形状为 [n_samples]
            topk: 选择的top-k样本数量
            ref_x: 参考输入数据 (当前未使用，预留接口)
            ref_y: 参考标签数据 (当前未使用，预留接口)
            seen_classes: 已见过的类别列表，用于持续学习

        返回:
            tuple: (更新后的样本权重, 权重梯度, 外层损失)
                - sample_weights: 更新后的样本权重向量
                - jacobian: 样本权重的梯度信息
                - loss_outer: 外层目标函数的损失值

        外层优化流程:
            1. 数据预处理和设备迁移
            2. 调用 update_sample_weights 进行核心权重更新
            3. 返回更新结果

        数学原理:
            外层问题: min_w L_outer(θ*(w), w)
            其中 θ*(w) 是内层优化得到的最优参数
        """
        # 数据预处理：类型转换和设备迁移
        data = data.to(self.device)
        target = target.to(self.device).type(torch.long)
        sample_weights = data_weights.to(self.device)

        # 准备选择集数据（这里选择全部数据）
        X_S = data[:].to(self.device)
        y_S = target[:].to(self.device).type(torch.long)

        # 调用核心权重更新函数
        return self.update_sample_weights(
            data, target, task_id, X_S, y_S, sample_weights, topk,
            beta=self.beta, ref_x=ref_x, ref_y=ref_y, seen_classes=seen_classes
        )


    def update_sample_weights(self, input_train, target_train, task_id, input_selected, target_selected,
                             sample_weights, topk, beta, seen_classes=None, epsilon=1e-3, ref_x=None, ref_y=None):
        """
        更新样本权重的核心函数 - BCSR 双层优化的关键实现

        该函数实现了双层优化中最复杂的权重更新逻辑，通过超梯度 (hypergradient)
        方法计算样本权重相对于外层目标函数的梯度，并更新权重以提升模型性能。

        参数:
            input_train: 训练输入数据，形状为 [n_samples, n_features]
            target_train: 训练标签，形状为 [n_samples]
            task_id: 任务标识符，用于多任务学习
            input_selected: 选择的输入数据，形状为 [n_selected, n_features]
            target_selected: 选择的标签数据，形状为 [n_selected]
            sample_weights: 当前样本权重，形状为 [n_samples]
            topk: 选择的前k个样本数量
            beta: 正则化系数，控制样本多样性
            seen_classes: 已见类别列表，用于持续学习
            epsilon: 噪声项系数，增加随机性
            ref_x: 参考输入 (预留，当前未使用)
            ref_y: 参考标签 (预留，当前未使用)

        返回:
            tuple: (更新后的样本权重, Jacobian梯度, 外层损失)
                - sample_weights: 更新后的权重向量
                - jacobian: 权重相对于外层损失的梯度
                - loss_outer: 外层目标函数的损失值

        双层优化数学原理:
            外层目标: L_outer(θ*(w), w) = L_val(θ*(w)) - β * Σ(top-k weights)
            内层目标: L_inner(θ, w) = Σ w_i * L(f_θ(x_i), y_i)

            权重更新: w ← w - lr_w * ∇_w L_outer
            其中 ∇_w L_outer 通过超梯度方法计算
        """

        # ====== 1. 外层损失计算 ======
        # 添加随机噪声项，增加选择的随机性
        z = torch.normal(0, 1, size=[topk]).cuda()

        # 计算验证集上的损失 (外层目标的主要部分)
        loss_outer = F.cross_entropy(
            self.proxy_model(input_train, target_train, instance_eval=True,
                           return_features=True, seen_classes=seen_classes)["logits"],
            target_train, reduction='none'
        )

        # 获取权重最大的 top-k 样本
        topk_weights, _ = sample_weights.topk(topk)

        # 计算外层平均损失：验证损失 - 正则化项 (鼓励选择多样化样本)
        loss_outer_avg = torch.mean(loss_outer) - beta * (topk_weights + epsilon * z).sum()

        # ====== 2. 超梯度计算准备 ======
        # 计算外层损失相对于模型参数的梯度
        d_theta = torch.autograd.grad(loss_outer_avg, self.proxy_model.parameters(), allow_unused=True)
        v_0 = d_theta  # 初始化超梯度向量

        # 计算内层损失 (用于后续的二阶导数计算)
        loss_inner = torch.mean(F.cross_entropy(
            self.proxy_model(
                (F.softmax(sample_weights, dim=-1) * input_selected.T).T,
                target_selected, instance_eval=True,
                return_features=True, seen_classes=seen_classes
            )["logits"],
            target_selected, reduction='none'
        ))

        # 计算内层损失相对于模型参数的梯度
        grads_theta = torch.autograd.grad(
            loss_inner, self.proxy_model.parameters(),
            create_graph=True, allow_unused=True
        )

        # ====== 3. 构建梯度更新操作 G(θ) = θ - lr * ∇_θ L_inner ======
        G_theta = []
        for p, g in zip(self.proxy_model.parameters(), grads_theta):
            if g is not None:
                G_theta.append(p - self.lr_p * g)  # SGD 更新步骤

        v_Q = v_0  # 初始化累积梯度

        # ====== 4. 多步超梯度近似计算 ======
        # 通过多次迭代近似计算 (I - lr * ∇²L)^(-1) * ∇L
        # 这是求解双层优化问题的关键步骤
        for _ in range(3):  # 进行3次迭代近似
            para_list = self.proxy_model.parameters()

            # 计算 ∇_θ G_θ * v_0 (链式法则的应用)
            v_new = torch.autograd.grad(
                G_theta, para_list, grad_outputs=v_0,
                retain_graph=True, allow_unused=True
            )

            # 更新并累积梯度信息
            v_0 = [i.detach() for i in v_new if i is not None]
            for i in range(len(v_0)):
                v_Q[i].add_(v_0[i].detach())

        # ====== 5. 计算样本权重的最终梯度 ======
        # 过滤掉 None 梯度
        grads_theta = [i for i in grads_theta if i is not None]

        # 计算 Jacobian: ∂L_inner/∂w 通过超梯度方法
        jacobian = -torch.autograd.grad(
            grads_theta, sample_weights,
            grad_outputs=v_Q, allow_unused=True
        )[0]

        # ====== 6. 权重更新 ======
        # 使用梯度下降更新样本权重
        with torch.no_grad():
            sample_weights -= self.lr_w * jacobian

        return sample_weights, jacobian, loss_outer

