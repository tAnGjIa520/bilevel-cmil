import torch
import torch.nn.functional as F
import numpy as np
import math
def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

class Training():

    def __init__(self, proxy_model, beta, device, lr_proxy_model, lr_weights):
        self.proxy_model = proxy_model
        self.lr_p =  lr_proxy_model # 代理模型的学习率
        self.lr_w =  lr_weights # w的学习率
        self.optimizer_theta_p_model = torch.optim.SGD(self.proxy_model.parameters(), lr=self.lr_p) # 代理模型优化器
        self.weight_optimizer = None
        self.device = device
        self.eta = 0.5 #
        self.beta = beta # 正则化项的系数
        self.buffer = []
        self.identity = []

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
        loss = math.inf
        for _ in range(inner_epchos):
            self.optimizer_theta_p_model = torch.optim.SGD(self.proxy_model.parameters(), lr=self.lr_p)
            self.proxy_model.train()
            self.optimizer_theta_p_model.zero_grad()
            data = data_S.to(self.device).type(torch.float)
            target = target_S.to(self.device).type(torch.long)
            sample_weights = sample_weights.to(self.device).type(torch.float).detach()
            # ==================================
            output = self.proxy_model((sample_weights * data.T).T, target,instance_eval=True, return_features=True, seen_classes=seen_classes)["logits"]
            # ==================================
            # model(batch['features'], batch['label'], instance_eval=True, return_features=True, seen_classes=seen_classes)


            loss = F.cross_entropy(output, target+task_id*2, reduction='none')
            loss.backward()
            self.optimizer_theta_p_model.step()
            self.proxy_model.zero_grad()
        return loss


    # train_outer， 更新w，refx这里永远是None
    def train_outer(self, data, target, task_id, data_weights, topk, ref_x=None, ref_y=None,seen_classes=None):
        data = data.to(self.device)
        target = target.to(self.device).type(torch.long)
        sample_weights = data_weights.to(self.device)
        X_S = data[:].to(self.device)
        y_S = target[:].to(self.device).type(torch.long)
        return self.update_sample_weights(data, target, task_id, X_S, y_S, sample_weights, topk, beta=self.beta, ref_x=ref_x, ref_y=ref_y,seen_classes=seen_classes)


    # 更新权重w，内部调用
    def update_sample_weights(self, input_train, target_train, task_id, input_selected, target_selected,  sample_weights, topk, beta, seen_classes=None,epsilon=1e-3, ref_x=None, ref_y=None):
        z = torch.normal(0, 1, size=[topk]).cuda()
        loss_outer = F.cross_entropy(self.proxy_model(input_train,target_train,instance_eval=True, return_features=True, seen_classes=seen_classes)["logits"],target_train, reduction='none')
        topk_weights, ind = sample_weights.topk(topk)
        loss_outer_avg = torch.mean(loss_outer) - beta*(topk_weights + epsilon*z).sum()
        # if ref_x != None:
        #     loss_buff = []
        #     for i in range(task_id-1):
        #         loss_buff += F.cross_entropy(self.proxy_model(ref_x[i].to(self.device), i+1), ref_y[i].to(self.device), reduction='none')
        #     loss_buff_avg = torch.mean(torch.Tensor(loss_buff))
        #     alpha = 0.1
        #     loss_outer_avg +=  alpha * loss_buff_avg

        # grads = torch.autograd.grad(loss_outer_avg, list(self.proxy_model.parameters()), allow_unused=True)
        #
        # for name, param, grad in zip(self.proxy_model.state_dict().keys(), self.proxy_model.parameters(), grads):
        #     if grad is None:
        #         print(f"未使用的参数: {name} (形状: {param.shape})")


        d_theta = torch.autograd.grad(loss_outer_avg, self.proxy_model.parameters(),allow_unused=True)
        v_0  = d_theta
        loss_inner = torch.mean( F.cross_entropy(
            self.proxy_model((F.softmax(sample_weights, dim=-1)*input_selected.T).T,target_selected, instance_eval=True, return_features=True, seen_classes=seen_classes)["logits"], target_selected, reduction='none'))
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

