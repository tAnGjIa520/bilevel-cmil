import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce
from torch.optim.optimizer import Optimizer
from collections import defaultdict
from torchmetrics import AUROC, Accuracy

class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(fast_p.data - slow, alpha=group['lookahead_alpha'])
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        #assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)
# helper functions

def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main attention class

class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        # attn1 [b, h, padding+n, m], attn2_inv [b, h, m, m], attn3 [b, h, m, padding+n], v [b, h, padding+n, dim_head]
        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out = out + self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            # attn = attn1 @ attn2_inv @ attn3
            # attn = attn[..., padding:padding + n, padding:padding + n]
            # attn = attn[..., -n:, -n:]
            # v = v[..., -n:, :]
            # vv = v @ v.transpose(-2, -1)
            # vv = vv[..., -n:, -n:]
            # reduce memory version
            attn = attn1 @ attn2_inv # [b, h, n, m]
            attn = attn[..., padding, :].unsqueeze(-2) @ attn3[..., -n:] # [b, h, 1, n]
            # vv = v[..., -n:, :] # [b, h, n, dim_head]
            return out, attn

        return out

# transformer

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Nystromformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        attn_values_residual = True,
        attn_values_residual_conv_kernel = 33,
        attn_dropout = 0.,
        ff_dropout = 0.   
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, NystromAttention(dim = dim, dim_head = dim_head, heads = heads, num_landmarks = num_landmarks, pinv_iterations = pinv_iterations, residual = attn_values_residual, residual_conv_kernel = attn_values_residual_conv_kernel, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x
        return x
    
class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            out, attn = self.attn(self.norm(x), return_attn=return_attn)
            return x + out, attn
        x = x + self.attn(self.norm(x), return_attn=return_attn)
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, D_feat=1024, D_inner=512, n_classes=2):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=D_inner)
        if D_feat != D_inner:
            self._fc1 = nn.Sequential(nn.Linear(D_feat, D_inner), nn.ReLU())
        else:
            self._fc1 = nn.Identity()
        self.cls_token = nn.Parameter(torch.randn(1, 1, D_inner))
        self.n_classes =n_classes
        self.layer1 = TransLayer(dim=D_inner)
        self.layer2 = TransLayer(dim=D_inner)
        self.norm = nn.LayerNorm(D_inner)
        self._fc2 = nn.Linear(D_inner, n_classes)
        # self.lr = lr or 0.0002
        # self.weight_decay = weight_decay or 0.00001

    def forward(self, x, return_attn=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        h = self._fc1(x)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        if return_attn:
            h, attn1 = self.layer1(h, return_attn=return_attn)
            h1 = h[:, 0]
        else:
            h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        if return_attn:
            h, attn2 = self.layer2(h, return_attn=return_attn)
            h2 = h[:, 0]
        else:
            h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim=1)
        
       
        
        if return_attn:
            # attn: [B, n_heads, padding+1+H+add_length, padding+1+H+add_length]
            end_idx = - add_length if add_length > 0 else None
            # attn1 = attn1[0, :, -1-H-add_length, -H-add_length:end_idx] # [n_heads, n]
            # attn2 = attn2[0, :, -1-H-add_length, -H-add_length:end_idx] # [n_heads, n]
            # vv1 = vv1[0, :, :end_idx, :end_idx] # [n_heads, N, N]
            # vv2 = vv2[0, :, :end_idx, :end_idx] # [n_heads, N, N]
            # reduce memory version
            attn1 = attn1.squeeze()[:, -H-add_length:end_idx] # [n_heads, H]
            attn2 = attn2.squeeze()[:, -H-add_length:end_idx]
            # vv1 = vv1[0, :, :end_idx, :] # [n_heads, N, dim_head]
            # vv2 = vv2[0, :, :end_idx, :]
            # p?   exit()
            return {'logits': logits, 'features': h, 'attn1': attn1, 'attn2': attn2, 'h1': h1, 'h2': h2}
        else:
            return {'logits': logits, 'features': h}

    # def configure_optimizers(self):

    #     def restore_optimizers(optimizers):
    #         if hasattr(self, 'optimizer_state_dict'):
    #             for optimizer, state_dict in zip(optimizers, self.optimizer_state_dict):
    #                 optimizer.load_state_dict(state_dict)
    #                 for param_group in optimizer.param_groups:
    #                     param_group['lr'] = param_group['lr'] * self.lr_rate
    #             del self.optimizer_state_dict
    #         return optimizers

    #     optimizer = Lookahead(torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay))

    #     optimizers = restore_optimizers([optimizer])
    #     return optimizers


# class TransMIL(pl.LightningModule):

#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.save_hyperparameters()
#         self.loss_fn = nn.CrossEntropyLoss()

#         if args.feature_extractor == 'mnist_cnn':
#             self.feature_extractor = MNIST_CNN(dim_in=args.dim_in, dim_out=args.L)
#         elif args.feature_extractor == 'mnist_mlp':
#             self.feature_extractor = MNIST_MLP(dim_in=args.dim_in, dim_out=args.L)
#         elif args.feature_extractor == 'digit_cnn':
#             self.feature_extractor = Digit_CNN(dim_in=args.dim_in, dim_out=args.L)
#         elif args.feature_extractor == 'lite_resnet':
#             self.feature_extractor = LiteResNet(dim_in=args.dim_in, dim_out=args.L)
#         else:
#             self.feature_extractor = None

#         if args.net == 'transmil':
#             from types import SimpleNamespace
#             conf = SimpleNamespace(D_feat=args.D_feat, D_inner=args.L, n_class=args.n_classes, lr=args.lr, weight_decay=args.weight_decay)
#             self.net = TransMIL_Warpper(conf)
#         else:
#             raise ValueError('Unknown model:', args.model)

#         if args.n_classes == 2:
#             self.val_auc = AUROC(task='binary')
#             self.val_acc = Accuracy(task='binary')
#             self.test_auc = AUROC(task='binary')
#             self.test_acc = Accuracy(task='binary')
#         else:
#             self.val_auc = AUROC(task='multiclass', num_classes=args.n_classes)
#             self.val_acc = Accuracy(task='multiclass', num_classes=args.n_classes)
#             self.test_auc = AUROC(task='multiclass', num_classes=args.n_classes)
#             self.test_acc = Accuracy(task='multiclass', num_classes=args.n_classes)

#     # def update_weight_from_checkpoint(self, checkpoint_path, optimizer=True, lr_rate=1.0):
#     #     checkpoint = torch.load(checkpoint_path, map_location=self.device)
#     #     if 'state_dict' in checkpoint:
#     #         self.load_state_dict(checkpoint['state_dict'])
#     #         if optimizer and 'optimizer_states' in checkpoint:
#     #             self.optimizer_state_dict = checkpoint['optimizer_states']
#     #             self.lr_rate = lr_rate
#     #     else:
#     #         self.load_state_dict(checkpoint)

#     def forward(self, batch, **kwargs):
#         features = batch['features']
#         if self.feature_extractor is not None:
#             features = self.feature_extractor(features)

#         return self.net.forward(features, **batch, **kwargs)

#     def training_step(self, batch, batch_idx):
#         out = self.forward(batch)

#         loss = self.loss_fn(out['logits'], batch['label'])

#         self.log('train/loss', loss, on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         return loss

#     def configure_optimizers(self):

#         # def restore_optimizers(optimizers):
#         #     if hasattr(self, 'optimizer_state_dict'):
#         #         for optimizer, state_dict in zip(optimizers, self.optimizer_state_dict):
#         #             optimizer.load_state_dict(state_dict)
#         #             for param_group in optimizer.param_groups:
#         #                 param_group['lr'] = param_group['lr'] * self.lr_rate
#         #         del self.optimizer_state_dict
#         #     return optimizers

#         lr = getattr(self.args, 'lr', 1e-3)
#         opt_name = getattr(self.args, 'opt', 'adam')
#         weight_decay = getattr(self.args, 'weight_decay', 1e-5)
#         momentum = getattr(self.args, 'momentum', 0.9)

#         if opt_name == 'adam':
#             optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
#         elif opt_name == 'adamw':
#             optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
#         elif opt_name == 'sgd':
#             optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
#         elif opt_name == 'lookahead_radam':
#             optimizer = Lookahead(torch.optim.RAdam(self.parameters(), lr=lr, weight_decay=weight_decay))
#         else:
#             raise ValueError('Unknown optimizer:', opt_name)

#         # optimizer = restore_optimizers(optimizer)
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35], gamma=0.1)
#         return [optimizer], [scheduler]

#     def validation_step(self, batch, batch_idx, dataloader_idx=0):
#         out = self.forward(batch)

#         loss = self.loss_fn(out['logits'], batch['label'])
#         y_hat = out['logits'].argmax(dim=-1)
#         y_prob = F.softmax(out['logits'], dim=-1)

#         self.log('val/loss', loss, on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         self.val_auc.update(y_prob[:, 1], batch['label'])
#         self.val_acc.update(y_hat, batch['label'])
#         self.log('val/auc', self.val_auc, on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         # self.val_loss.append(loss.item())
#         return loss

#     def test_step(self, batch, batch_idx):
#         out = self.forward(batch)

#         loss = self.loss_fn(out['logits'], batch['label'])
#         y_hat = out['logits'].argmax(dim=-1)
#         y_prob = F.softmax(out['logits'], dim=-1)

#         self.log('test/loss', loss, on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         self.test_auc.update(y_prob[:, 1], batch['label'])
#         self.test_acc.update(y_hat, batch['label'])
#         self.log('test/auc', self.test_auc, on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         return loss


# class TransMIL_CL(ContinualModel, TransMIL):

#     def __init__(self, args):
#         super().__init__(args)
#         if args.n_classes == 2:
#             self.test_auc = nn.ModuleList([AUROC(task='binary') for _ in range(args.n_tasks)])
#             self.test_acc = nn.ModuleList([Accuracy(task='binary') for _ in range(args.n_tasks)])
#         else:
#             self.test_auc = nn.ModuleList([AUROC(task='multiclass', num_classes=args.n_classes) for _ in range(args.n_tasks)])
#             self.test_acc = nn.ModuleList([Accuracy(task='multiclass', num_classes=args.n_classes) for _ in range(args.n_tasks)])

#     def training_step(self, batch, batch_idx):
#         out = self.forward(batch)

#         loss = 0.

#         if self.cl_method in ['der', 'derpp'] and 'logits' in batch.keys(): # derpp method
#             l2_loss = F.mse_loss(out['logits'], batch['logits'])
#             loss += l2_loss
#             if self.cl_method == 'derpp':
#                 loss += self.loss_fn(out['logits'], batch['label'])
#         else:
#             loss += self.loss_fn(out['logits'], batch['label'])

#         if 'kl' in self.cl_method and 'A' in batch.keys() and batch['Y_hat'] == batch['label']: # kl method
#             # kl_loss = F.kl_div(torch.log_softmax(out['A'], dim=-1), batch['A'], reduction='batchmean')
#             T = 1.
#             kl_loss = F.kl_div(F.log_softmax(out['A'] / T, dim=-1), F.softmax(batch['A'] / T), reduction='batchmean') * T * T
#             loss += kl_loss
#             self.log('train/kl_loss', kl_loss, batch_size=self.args.batch_size)


#         self.log('train/loss', loss, batch_size=self.args.batch_size)
#         return loss

#     def test_step(self, batch, batch_idx, dataloader_idx=0):
#         out = self.forward(batch)

#         loss = self.loss_fn(out['logits'], batch['label'])
#         y_hat = out['logits'].argmax(dim=-1)
#         y_prob = F.softmax(out['logits'], dim=-1)

#         self.log('test/loss', loss, on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         self.test_auc[dataloader_idx].update(y_prob[:, 1], batch['label'])
#         self.test_acc[dataloader_idx].update(y_hat, batch['label'])
#         self.log('test/auc', self.test_auc[dataloader_idx], on_step=False, on_epoch=True,
#                  batch_size=self.args.batch_size)
#         self.log('test/acc', self.test_acc[dataloader_idx], on_step=False, on_epoch=True,
#                  batch_size=self.args.batch_size)
#         return loss