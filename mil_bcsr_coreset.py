import numpy as np
import torch
from mil_bcsr_training import Training


class BCSR_Coreset:
    """
    基于双层优化的核心集选择 (BCSR: Bilevel Coreset Selection with Representativity)

    该类实现了一种智能的样本选择策略，用于从大量候选样本中选择最具代表性的子集。
    通过双层优化框架：外层优化选择样本权重，内层优化训练模型，确保选择的核心集
    既能保持数据的代表性，又能最大化模型在验证集上的性能。

    参数说明:
        proxy_model: 用于核心集选择的代理模型
        lr_proxy_model: 代理模型的学习率
        beta: 平衡主损失和正则化项的权重参数
        out_dim: 输出特征维度
        max_outer_it: 双层优化的外层最大迭代次数
        max_inner_it: 双层优化的内层最大迭代次数
        weight_lr: 更新样本权重的步长
        candidate_batch_size: 候选核心集的样本数量
        logging_period: 日志记录周期
        device: 计算设备 ('cuda' 或 'cpu')
    """
    def __init__(self, proxy_model, lr_proxy_model, beta, out_dim=10, max_outer_it=50, max_inner_it=1, weight_lr=1e-1,
                candidate_batch_size=600, logging_period=1000, device='cuda'):
        """
        初始化 BCSR 核心集选择器

        参数:
            proxy_model: 代理模型，用于评估样本选择的质量
            lr_proxy_model: 代理模型的学习率
            beta: 正则化参数，平衡主损失和正则化项
            out_dim: 输出特征维度
            max_outer_it: 外层优化最大迭代次数
            max_inner_it: 内层优化最大迭代次数
            weight_lr: 样本权重更新的学习率
            candidate_batch_size: 候选样本的批次大小
            logging_period: 日志记录的周期
            device: 计算设备
        """
        # 存储关键参数
        self.out_dim = out_dim                              # 输出特征维度
        self.max_outer_it = max_outer_it                    # 外层优化迭代次数
        self.max_inner_it = max_inner_it                    # 内层优化迭代次数
        self.weight_lr = weight_lr                          # 权重学习率
        self.candidate_batch_size = candidate_batch_size    # 候选批次大小
        self.logging_period = logging_period                # 日志周期

        # 初始化其他属性
        self.nystrom_batch = None                           # Nystrom 方法相关
        self.nystrom_normalization = None                   # Nystrom 归一化
        self.param_size = []                                # 模型参数尺寸列表
        self.seed = 0                                       # 随机种子
        self.lr_proxy_model = lr_proxy_model               # 代理模型学习率

        # 初始化训练操作器
        self.training_model_op = Training(proxy_model, beta, device, lr_proxy_model, lr_weights=self.weight_lr)

        # 记录模型参数尺寸
        for p in self.training_model_op.proxy_model.parameters():
            self.param_size.append(p.size())


    def outer_loss(self, X, y, task_id, topk, ref_x=None, ref_y=None):
        """
        计算外层损失函数

        该函数计算在给定样本权重下的外层优化目标，用于评估当前样本选择策略的质量。
        外层损失通常反映模型在验证集或参考数据上的性能。

        参数:
            X: 输入特征矩阵，形状为 [n_samples, n_features]
            y: 标签向量，形状为 [n_samples]
            task_id: 任务标识符，用于多任务学习场景
            topk: 要选择的top-k样本数量
            ref_x: 参考输入特征 (可选)
            ref_y: 参考标签 (可选)

        返回:
            outer_loss: 计算得到的外层损失值
        """
        # 数据类型转换：确保输入为 PyTorch 张量
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        # 获取样本数量并初始化均匀权重
        n = X.shape[0]
        coreset_weights = 1.0 / n * torch.ones([n], dtype=torch.float, requires_grad=True)

        # 调用训练操作器计算外层损失
        _, _, outer_loss = self.training_model_op.train_outer(X, y, task_id, coreset_weights, topk, ref_x, ref_y)
        return outer_loss

    def projection_onto_simplex(self, v, b=1):
        """
        将向量投影到单纯形上

        该函数实现了将任意向量投影到概率单纯形的算法。投影后的向量满足：
        1. 所有元素非负
        2. 所有元素的和等于 b (通常为1)
        这确保了样本权重构成一个有效的概率分布。

        参数:
            v: 输入向量 (PyTorch tensor)，需要被投影的权重向量
            b: 单纯形的约束常数，默认为1 (表示概率分布)

        返回:
            w: 投影后的权重向量 (PyTorch tensor)，满足单纯形约束

        算法原理:
            使用 Euclidean 投影算法将向量投影到标准单纯形
            {x: x_i >= 0, sum(x_i) = b}
        """
        # 转换为 numpy 数组进行计算
        v = v.cpu().detach().numpy()
        n_features = v.shape[0]

        # 按降序排列
        u = np.sort(v)[::-1]

        # 计算累积和减去约束值
        cssv = np.cumsum(u) - b

        # 创建索引数组
        ind = np.arange(n_features) + 1

        # 找到满足条件的元素
        cond = u - cssv / ind > 0

        # 找到 rho 值（最后一个满足条件的索引）
        rho = ind[cond][-1]

        # 计算阈值 theta
        theta = cssv[cond][-1] / float(rho)

        # 执行投影：max(v_i - theta, 0)
        w = np.maximum(v - theta, 0)

        # 转换回 PyTorch tensor 并移到 GPU
        w = torch.from_numpy(w).cuda()
        w.requires_grad = True
        return w

    def coreset_select(self, model, X, y, task_id, topk, out_loss=None, ref_x=None, ref_y=None, seen_classes=None):
        """
        执行核心集选择的主函数

        该函数实现了基于双层优化的核心集选择算法。通过交替优化样本权重（外层）
        和模型参数（内层），找到最具代表性的样本子集。选择的核心集能够最大化
        模型在验证数据上的性能，同时保持数据分布的代表性。

        参数:
            model: 原始模型，用于初始化代理模型的参数
            X: 输入特征矩阵，形状为 [n_samples, n_features]
            y: 标签向量，形状为 [n_samples]
            task_id: 任务标识符，用于多任务学习场景
            topk: 要选择的样本数量
            out_loss: 输出损失列表 (可选)，用于记录优化过程中的损失值
            ref_x: 参考输入特征 (可选)，用于验证集
            ref_y: 参考标签 (可选)，用于验证集
            seen_classes: 已见过的类别列表 (可选)，用于持续学习

        返回:
            tuple: (selected_indices, out_loss)
                - selected_indices: 选中样本的索引 (PyTorch tensor)
                - out_loss: 更新后的输出损失列表

        算法流程:
            1. 初始化样本权重为均匀分布
            2. 迭代执行双层优化：
               - 内层：固定样本权重，优化模型参数
               - 外层：固定模型参数，优化样本权重
            3. 根据最终的样本权重概率采样得到核心集
        """
        # 设置随机种子确保结果可复现
        np.random.seed(self.seed)

        # 数据类型转换：确保输入为 PyTorch 张量
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        n = X.shape[0]  # 样本总数

        # 用原始模型参数初始化代理模型
        self.training_model_op.proxy_model.load_state_dict(model.state_dict())

        # 初始化样本权重为均匀分布
        coreset_weights = 1.0/n * torch.ones([n], dtype=torch.float, requires_grad=True)

        # 将样本权重投影到概率单纯形上
        coreset_weights = self.projection_onto_simplex(coreset_weights)

        # 设置学习率
        self.training_model_op.lr_p = self.lr_proxy_model  # 代理模型学习率
        self.training_model_op.lr_w = self.weight_lr       # 权重学习率

        # 双层优化主循环
        # outer loop
        for i in range(self.max_outer_it):
            
            # inner loop
            # 内层优化：固定样本权重，训练模型参数
            inner_loss = self.training_model_op.train_inner(
                X, y, task_id, coreset_weights, self.max_inner_it, seen_classes=seen_classes
            )

            # 外层优化：固定模型参数，更新样本权重
            coreset_weights, _, outer_loss = self.training_model_op.train_outer(
                X, y, task_id, coreset_weights, topk, ref_x, ref_y, seen_classes=seen_classes
            )

            # 重新投影权重到单纯形上
            coreset_weights = self.projection_onto_simplex(coreset_weights)

            # 计算总损失
            total_loss = torch.mean(outer_loss).item()

        # 打印最终的损失值
        print('inner loss:{:.3f}, outer loss:{:.3f}'.format(inner_loss.item(), total_loss))

        # 记录损失值（仅在特定条件下）
        if out_loss is not None and n == 50:
            out_loss.append(total_loss)

        # 根据样本权重概率分布采样得到核心集索引
        selected_indices = torch.multinomial(coreset_weights, topk, replacement=False)

        return selected_indices, out_loss
