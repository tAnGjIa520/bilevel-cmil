import numpy as np
import torch
from mil_bcsr_training import Training
import copy

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
    def __init__(self, proxy_model, lr_proxy_model,  beta, max_outer_it=50, max_inner_it=1, weight_lr=1e-1, device='cuda',distall_lamda=0.1,draw_curve=False):

        self.max_outer_it = max_outer_it
        self.max_inner_it = max_inner_it
        self.weight_lr = weight_lr


        self.nystrom_batch = None
        self.nystrom_normalization = None
        self.param_size=  []
        self.seed = 0
        self.lr_proxy_model = lr_proxy_model
        self.training_model_op = Training(proxy_model, beta, device, lr_proxy_model, lr_weights=self.weight_lr,distall_lamda=distall_lamda,draw_curve=draw_curve)
        self.training_model_op.origin_model=copy.deepcopy(proxy_model)
        
        
        for p in self.training_model_op.proxy_model.parameters():
            self.param_size.append(p.size())


    def outer_loss(self, X, y, task_id, topk, ref_x=None, ref_y=None):
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        n = X.shape[0]
        coreset_weights = 1.0 / n * torch.ones([n], dtype=torch.float, requires_grad=True)

        _, _, outer_loss = self.training_model_op.train_outer(X, y, task_id, coreset_weights, topk, ref_x,ref_y)
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

    def coreset_select(self, model, X, y, task_id,  topk, out_loss=None, ref_x=None, ref_y=None,seen_classes=None):
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
        # coreset_weights = self.projection_onto_simplex(coreset_weights) # origin

        self.training_model_op.lr_p = self.lr_proxy_model # 学习率
        self.training_model_op.lr_w = self.weight_lr # 学习率

        # solve the bilevel problem
        for i in range(self.max_outer_it):
            inner_loss = self.training_model_op.train_inner(X, y, task_id, coreset_weights, self.max_inner_it,seen_classes=seen_classes)
            coreset_weights, _, outer_loss = self.training_model_op.train_outer(X, y, task_id, coreset_weights, topk, ref_x, ref_y,seen_classes=seen_classes)
            # coreset_weights = self.projection_onto_simplex(coreset_weights)
            total_loss = torch.mean(outer_loss).item()
        print('inner loss:{:.3f}, outer loss:{:.3f}'.format(inner_loss.item(), total_loss))
        if out_loss != None and n==50:
            out_loss.append(total_loss)


        return torch.topk(coreset_weights, topk).indices, out_loss
