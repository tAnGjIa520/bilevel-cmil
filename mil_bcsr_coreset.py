import numpy as np
import torch
from mil_bcsr_training import Training
import copy
from tools.coreset_weight_init import CoresetWeightInitializer
def get_total_gradient_norm(tensors, norm_type: float = 2.0) -> float:
      """
      计算多个tensor的总梯度范数
      
      Args:
          tensors: tensor的列表或迭代器
          norm_type: 范数类型
      
      Returns:
          所有tensor梯度的总范数
      """
      gradients = [t.grad for t in tensors if t.grad is not None]

      if not gradients:
          return 0.0

      # 将所有梯度展平并连接
      grad_vec = torch.cat([g.flatten() for g in gradients])

      return torch.norm(grad_vec, p=norm_type).item()
  

class BCSR_Coreset:
    """"
    Coreset selection basede on bilevel optimzation

    Args:
        proxy_model: model for coreset selection
        lr_proxy_model: learning rare for proxy_model
        beta: balance the loss and regularizer

        max_outer_it: outer loops for bilevel optimizaiton
        max_inner_it: inner loops for bilevel optimizaiton
        weight_lr: step size for updating samlple weights
       
    """
    def __init__(self, proxy_model, lr_proxy_model,  beta, max_outer_it=50, max_inner_it=1, weight_lr=1e-1, device='cuda',distall_lamda=0.1,draw_curve=False,tb_writer=None,topk=10,topk_method='sigmoid',topk_temperature=0.1,normalize_method='none',use_simplex_projection=False,distill_target='logits',temp_scheduler=None,weight_optimizer_type='sgd',weight_adam_lr=0.001,weight_adam_betas=(0.9, 0.999),weight_adam_eps=1e-8,use_lr_scheduler=False,lr_initial=None,lr_final=None,lr_strategy='cosine',coreset_weight_init='uniform_random',model_type='clam_sb',init_hyperparams=None,neumann_series_depth=3):

        self.max_outer_it = max_outer_it
        self.max_inner_it = max_inner_it
        self.weight_lr = weight_lr
        self.use_simplex_projection = use_simplex_projection

        self.param_size=  []
        self.seed = 0
        self.lr_proxy_model = lr_proxy_model
        self.device = device

        # 创建权重初始化器
        init_hyperparams = init_hyperparams or {}
        self.weight_initializer = CoresetWeightInitializer(
            init_method=coreset_weight_init,
            model_type=model_type,
            hyperparams=init_hyperparams
        )

        # 创建学习率调度器（如果启用）
        lr_scheduler = None
        if use_lr_scheduler:
            from tools.lr_scheduler import LearningRateScheduler
            # 如果 lr_initial 或 lr_final 为 None，使用默认值
            lr_init = lr_initial if lr_initial is not None else weight_adam_lr
            lr_fin = lr_final if lr_final is not None else (weight_adam_lr * 0.1)
            lr_scheduler = LearningRateScheduler(
                initial_lr=lr_init,
                final_lr=lr_fin,
                max_iterations=max_outer_it,
                strategy=lr_strategy
            )

        self.training_model_op = Training(proxy_model, beta, device, lr_proxy_model, lr_weights=self.weight_lr,distall_lamda=distall_lamda,draw_curve=draw_curve,topk=topk,topk_method=topk_method,topk_temperature=topk_temperature,normalize_method=normalize_method,distill_target=distill_target,temp_scheduler=temp_scheduler,weight_optimizer_type=weight_optimizer_type,weight_adam_lr=weight_adam_lr,weight_adam_betas=weight_adam_betas,weight_adam_eps=weight_adam_eps,lr_scheduler=lr_scheduler,neumann_series_depth=neumann_series_depth)
        self.training_model_op.origin_model=copy.deepcopy(proxy_model)
        self.topk=topk

        self.tb_writer=tb_writer

        for p in self.training_model_op.proxy_model.parameters():
            self.param_size.append(p.size())


    def outer_loss(self, X, y, task_id, topk, ref_x=None, ref_y=None):
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        n = X.shape[0]
        coreset_weights = 1.0 / n * torch.ones([n], dtype=torch.float, requires_grad=True)

        _, _, outer_loss = self.training_model_op.train_outer(X, y, coreset_weights)
        return outer_loss

    def projection_onto_simplex(self, v, b=1):
        """
        投影到单纯形的优化实现（GPU加速版本）
        解决问题：x 属于单纯形 {x: sum(x)=1, x>=0}
        基于投影梯度下降法
        """
        device = v.device
        n_features = v.shape[0]

        # 对权重进行排序（GPU上的PyTorch操作）
        u, _ = torch.sort(v, descending=True)

        # 计算累积和（GPU上的PyTorch操作）
        cssv = torch.cumsum(u, dim=0) - b

        # 计算 rho（最大的 j 使得 u_j - cssv[j]/j > 0）
        ind = torch.arange(1, n_features + 1, dtype=torch.float, device=device)
        cond = u - cssv / ind > 0

        # 找到满足条件的最后一个索引
        rho_indices = torch.nonzero(cond, as_tuple=False)
        if len(rho_indices) > 0:
            rho = rho_indices[-1].item() + 1
            theta = cssv[rho_indices[-1].item()].item() / float(rho)
        else:
            # 如果没有满足条件的，使用第一个（这种情况很少发生）
            rho = 1
            theta = u[0].item() - b

        # 投影：w = max(v - theta, 0)
        w = torch.clamp(v - theta, min=0.0)
        w.requires_grad = True

        return w

    def coreset_select(self, model, X, y, task_id,  topk,  ref_x=None, ref_y=None,seen_classes=None):
        np.random.seed(self.seed)

        # 重置权重优化器状态，确保每个 distill_slide 调用都独立
        self.training_model_op.weight_optimizer = None

        # 重置学习率调度器（如果存在）
        if self.training_model_op.lr_scheduler is not None:
            self.training_model_op.lr_scheduler.reset()

        # 优化：统一处理输入数据，确保是 tensor 并在正确的设备上
        device = next(model.parameters()).device  # 获取模型所在设备

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(device)
        elif isinstance(X, torch.Tensor):
            X = X.to(device)  # 确保在正确设备上
        else:
            raise TypeError(f"X must be numpy array or torch tensor, got {type(X)}")

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float().to(device)
        elif isinstance(y, torch.Tensor):
            y = y.to(device)  # 确保在正确设备上
        else:
            raise TypeError(f"y must be numpy array or torch tensor, got {type(y)}")

        print("Starting coreset selection using BCSR...")
        print(y)
        
        # exit()
        
        n = X.shape[0]
        topk=min(topk, n)
        self.training_model_op.proxy_model.load_state_dict(model.state_dict())
        self.training_model_op.origin_model.load_state_dict(model.state_dict())

        # 初始化样本权重（支持多种初始化方法）
        coreset_weights = self.weight_initializer.initialize(
            n_samples=n,
            device=device,
            model=model,
            X=X
        )
        # project sample weights onto simplex
        if self.use_simplex_projection:
            coreset_weights = self.projection_onto_simplex(coreset_weights)

        self.training_model_op.lr_p = self.lr_proxy_model # 学习率
        self.training_model_op.lr_w = self.weight_lr # 学习率 get_total_gradient_norm(coreset_weights) coreset_weights==coreset_weights_bak

        # 初始化权重历史记录（用于跟踪每个实例的权重演变）
        if self.tb_writer is not None:
            self.tb_writer.add_histogram('bcsr/coreset_weights_iter_0', coreset_weights)

        # solve the bilevel problem
        out_loss=0
        loss_list=[]
        inner_loss_list=[]
        for i in range(self.max_outer_it):
            # 打印当前温度（如果有 scheduler）
            if self.training_model_op.temp_scheduler is not None:
                temp = self.training_model_op.temp_scheduler.get_temperature()
                print(f'Outer[{i+1}/{self.max_outer_it}] temp={temp:.3f}')

            inner_loss = self.training_model_op.train_inner(X, y, task_id, coreset_weights, self.max_inner_it,seen_classes=seen_classes,topk=topk)
            coreset_weights, gradient, outer_loss = self.training_model_op.train_outer(X, y, coreset_weights, seen_classes=seen_classes)

            # project sample weights onto simplex after gradient update
            

            total_loss = torch.mean(outer_loss).item()
            inner_loss_value = inner_loss.item()

            # 降低 TensorBoard 日志频率，每 5 次迭代记录一次，避免频繁的 GPU->CPU 数据传输
            if self.tb_writer is not None and i % 5 == 0:
                # 记录损失和梯度
                self.tb_writer.add_scalar('bcsr/inner_loss', inner_loss_value)
                self.tb_writer.add_scalar('bcsr/outer_loss', total_loss)
                self.tb_writer.add_scalar('bcsr/weight_gradient_norm', gradient.norm().item())

                # 记录每次迭代的权重分布（直方图）
                self.tb_writer.add_histogram(f'bcsr/coreset_weights_iter_{i+1}', coreset_weights)

                # 分步计算 topk 权重用于可视化
                weighted_topk_selector = self.training_model_op.weighted_topk_selector
                normalized_weights = weighted_topk_selector.normalize_weights(coreset_weights)
                topk_weights = weighted_topk_selector.topk_selector(normalized_weights)

                # 记录 topk 权重统计信息
                topk_weights_array = topk_weights.detach().cpu().numpy() if isinstance(topk_weights, torch.Tensor) else topk_weights
                self.tb_writer.add_scalar('bcsr/topk_weights_mean', topk_weights_array.mean())
                self.tb_writer.add_scalar('bcsr/topk_weights_std', topk_weights_array.std())
                self.tb_writer.add_scalar('bcsr/topk_weights_max', topk_weights_array.max())
                self.tb_writer.add_scalar('bcsr/topk_weights_min', topk_weights_array.min())

                # 绘制每个topk位置的权重变化折线图（参照demo中的实例权重跟踪）
                for pos_idx in range(min(len(topk_weights_array), 10)):  # 只记录前10个以避免过多曲线
                    self.tb_writer.add_scalar(f'bcsr/topk_weight_pos_{pos_idx}', topk_weights_array[pos_idx])

                # 记录所有原始权重的折线图（可选，用于详细分析）
                coreset_weights_array = coreset_weights.detach().cpu().numpy()
                # 找出权重最大的前5个实例和最小的前5个实例
                top5_indices = np.argsort(coreset_weights_array)[-5:]
                bottom5_indices = np.argsort(coreset_weights_array)[:5]
                for idx in top5_indices:
                    self.tb_writer.add_scalar(f'bcsr/weight_top_instance_{idx}', coreset_weights_array[idx])
                for idx in bottom5_indices:
                    self.tb_writer.add_scalar(f'bcsr/weight_bottom_instance_{idx}', coreset_weights_array[idx])

            print('inner loss:{:.3f}, outer loss:{:.3f}'.format(inner_loss_value, total_loss))
            loss_list.append(total_loss)
            inner_loss_list.append(inner_loss_value)

            # 更新 temperature scheduler
            if self.training_model_op.temp_scheduler is not None:
                self.training_model_op.temp_scheduler.step()


        return torch.topk(coreset_weights, topk).indices, loss_list, inner_loss_list
