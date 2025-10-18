import numpy as np
import torch
from mil_bcsr_training import Training
import copy
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
    def __init__(self, proxy_model, lr_proxy_model,  beta, max_outer_it=50, max_inner_it=1, weight_lr=1e-1, device='cuda',distall_lamda=0.1,draw_curve=False,tb_writer=None,topk=10,topk_method='sigmoid',topk_temperature=0.1,normalize_method='none',use_simplex_projection=False,distill_target='logits'):

        self.max_outer_it = max_outer_it
        self.max_inner_it = max_inner_it
        self.weight_lr = weight_lr
        self.use_simplex_projection = use_simplex_projection

        self.param_size=  []
        self.seed = 0
        self.lr_proxy_model = lr_proxy_model
        self.training_model_op = Training(proxy_model, beta, device, lr_proxy_model, lr_weights=self.weight_lr,distall_lamda=distall_lamda,draw_curve=draw_curve,topk=topk,topk_method=topk_method,topk_temperature=topk_temperature,normalize_method=normalize_method,distill_target=distill_target)
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
        # v 权重
        device = v.device  # 优化：记住原始设备
        v = v.cpu().detach().numpy()
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - b
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        w = torch.from_numpy(w).to(device)  # 优化：恢复到原始设备而非硬编码 cuda
        w.requires_grad = True
        return w

    def coreset_select(self, model, X, y, task_id,  topk,  ref_x=None, ref_y=None,seen_classes=None):
        np.random.seed(self.seed)
        
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

        n = X.shape[0]
        topk=min(topk, n)
        self.training_model_op.proxy_model.load_state_dict(model.state_dict())
        self.training_model_op.origin_model.load_state_dict(model.state_dict())
        
        # initialize sample weights from uniform distribution
        # 优化：直接在目标设备上创建，避免后续传输
        coreset_weights = torch.rand([n], dtype=torch.float, requires_grad=True, device=device)
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
        for i in range(self.max_outer_it):
            inner_loss = self.training_model_op.train_inner(X, y, task_id, coreset_weights, self.max_inner_it,seen_classes=seen_classes,topk=topk)
            coreset_weights_bak=coreset_weights.clone()
            coreset_weights, gradient, outer_loss = self.training_model_op.train_outer(X, y, coreset_weights, seen_classes=seen_classes)

            # project sample weights onto simplex after gradient update
            if self.use_simplex_projection:
                coreset_weights = self.projection_onto_simplex(coreset_weights)

            total_loss = torch.mean(outer_loss).item()
            if self.tb_writer is not None:
                # 记录损失和梯度
                self.tb_writer.add_scalar('bcsr/inner_loss', inner_loss.item())
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

            print('inner loss:{:.3f}, outer loss:{:.3f}'.format(inner_loss.item(), total_loss))
            loss_list.append(total_loss)
 


        return torch.topk(coreset_weights, topk).indices, loss_list
