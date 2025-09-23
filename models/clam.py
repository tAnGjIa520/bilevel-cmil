import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from torchmetrics import AUROC, Accuracy
# from models.model import ContinualModel
# from models.model import MNIST_CNN, MNIST_MLP, Digit_CNN, LiteResNet
# from utils_cl import EwcOn, NativeReplay, distill_slide
import os
import pandas as pd
import wandb


"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, D_feat=1024, L=512, D=256, K=1, dropout=False, gate=True, k_sample=8, n_classes=2, instance_loss_name='cross_entropy', subtyping=False):
        super(CLAM_SB, self).__init__()
        self.D_feat = D_feat
        self.L = L
        self.D = D
        self.K = K
        self.n_classes = n_classes
        self.subtyping = subtyping

        if n_classes > 2 and not subtyping:
            print("Warning: n_classes > 2 but not subtyping, will ignore the out-of-the-class instances")

        fc = [nn.Linear(D_feat, L), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L, D, dropout, K)
        else:
            attention_net = Attn_Net(L, D, dropout, K)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(L, n_classes)
        instance_classifiers = [nn.Linear(L, 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        if instance_loss_name == "cross_entropy":
            self.instance_loss_fn = nn.CrossEntropyLoss()
        elif instance_loss_name == "svm":
            from topk.svm import SmoothTop1SVM
            self.instance_loss_fn = SmoothTop1SVM(n_classes=2)

    # def relocate(self):
    #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.attention_net = self.attention_net.to(device)
    #     self.classifiers = self.classifiers.to(device)
    #     self.instance_classifiers = self.instance_classifiers.to(device)
    #     if device == "cuda":
    #         self.instance_loss_fn = self.instance_loss_fn.cuda()
    #     else:
    #         self.instance_loss_fn = self.instance_loss_fn.cpu()
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        
        k_sample = min(self.k_sample, A.shape[1]//2)
        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k_sample, device)
        n_targets = self.create_negative_targets(k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        # print(logits.device)
        # print(all_targets.device)

        # print(self.device)
        instance_loss = self.instance_loss_fn(logits.cpu(), all_targets.cpu())
        return instance_loss, all_preds, all_targets, logits
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        k_sample = min(self.k_sample, A.shape[1])
        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits.cpu(), p_targets.cpu())
        return instance_loss, p_preds, p_targets, logits

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, **kwargs):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N, [B, N]

        seen_classes = kwargs.get('seen_classes', range(self.n_classes))
        if instance_eval:
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            all_logits = []
            # inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in seen_classes:
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets, instance_logits = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_logits.append(instance_logits)
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets, instance_logits = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                        all_logits.append(instance_logits)
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(seen_classes)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M) # [B, n_classes]
        # mask unseen classes
        if len(seen_classes) < self.n_classes:
            unseen_mask = torch.ones(self.n_classes, dtype=torch.bool)
            unseen_mask[seen_classes] = False  # Set seen classes to False 
            logits[:, unseen_mask] = -100 
        results_dict = {'logits': logits, 'A': A_raw}
        if instance_eval:
            results_dict.update({'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 'inst_preds': np.array(all_preds), 'instance_logits': torch.cat(all_logits, dim=0)})
        if return_features:
            results_dict.update({'features': M})
        return results_dict

class CLAM_MB(CLAM_SB):
    def __init__(self, D_feat=1024, L=512, D=256, K=1, dropout=False, gate=True, k_sample=8, n_classes=2, instance_loss_name='cross_entropy', subtyping=False):
        nn.Module.__init__(self)
        self.D_feat = D_feat
        self.L = L
        self.D = D
        self.K = K
        self.n_classes = n_classes
        self.subtyping = subtyping

        fc = [nn.Linear(D_feat, L), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L, D, K, dropout)
        else:
            attention_net = Attn_Net(L, D, K, dropout)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(L, 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(L, 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        if instance_loss_name == "cross_entropy":
            self.instance_loss_fn = nn.CrossEntropyLoss()
        elif instance_loss_name == "svm":
            from topk.svm import SmoothTop1SVM
            self.instance_loss_fn = SmoothTop1SVM(n_classes)
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, **kwargs):
        device = h.device
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = [] 
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'A_raw': A_raw}
        if instance_eval:
            results_dict.update({'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                                 'inst_preds': np.array(all_preds)})
        if return_features:
            results_dict.update({'features': M})
        return results_dict


# class CLAM(pl.LightningModule):

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

#         if args.net == 'clam_sb':
#             self.net = CLAM_SB(D_feat=args.D_feat, L=args.L, D=args.D, K=args.K, dropout=args.dropout,
#                                  gate=args.gate, k_sample=args.k_sample, n_classes=args.n_classes,
#                                  instance_loss_name=args.instance_loss_name, subtyping=args.subtyping)
#         elif args.net == 'clam_mb':
#             self.net = CLAM_MB(D_feat=args.D_feat, L=args.L, D=args.D, K=args.K, dropout=args.dropout,
#                                  gate=args.gate, k_sample=args.k_sample, n_classes=args.n_classes,
#                                  instance_loss_name=args.instance_loss_name, subtyping=args.subtyping)
#         else:
#             raise ValueError('Unknown model:', args.model)

#         if args.n_classes == 2:
#             self.val_auc = AUROC(task='binary')
#             self.val_acc = Accuracy(task='binary')
#             self.test_auc = AUROC(task='binary')
#             self.test_acc = Accuracy(task='binary')
#         else:
#             self.val_auc = AUROC(task='multiclass', num_classes=args.n_classes)
#             self.val_acc = Accuracy(task='multiclass', num_classes=args.n_classes, average='micro')
#             self.test_auc = AUROC(task='multiclass', num_classes=args.n_classes)
#             self.test_acc = Accuracy(task='multiclass', num_classes=args.n_classes, average='micro')

#     # def on_fit_start(self):
#     #     # move loss_fn to device
#     #     if self.device.type == 'cuda':
#     #         self.net.instance_loss_fn = self.net.instance_loss_fn.cuda()
#     #     else:
#     #         self.net.instance_loss_fn = self.net.instance_loss_fn.cpu()

#     def forward(self, batch, **kwargs):
#         features = batch['features']
#         if self.feature_extractor is not None:
#             features = self.feature_extractor(features)

#         return self.net.forward(features, **batch, **kwargs)

#     def training_step(self, batch, batch_idx):
#         out = self.forward(batch, instance_eval=True)

#         bag_loss = self.loss_fn(out['logits'], batch['label'])
#         instance_loss = out['instance_loss']
#         loss = self.args.bag_weight * bag_loss + (1 - self.args.bag_weight) * instance_loss

#         self.log('train/loss', bag_loss, on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         self.log('train/instance_loss', instance_loss, on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         # self.train_loss.append(loss.item())
#         return loss


#     def validation_step(self, batch, batch_idx):
#         out = self.forward(batch)

#         loss = self.loss_fn(out['logits'], batch['label'])
#         y_hat = out['logits'].argmax(dim=-1)
#         y_prob = F.softmax(out['logits'], dim=-1)

#         self.log('val/loss', loss, on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         if self.args.n_classes == 2:
#             self.val_auc.update(y_prob[:, 1], batch['label'])
#         else:
#             self.val_auc.update(y_prob, batch['label'])
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
#         if self.args.n_classes == 2:
#             self.test_auc.update(y_prob[:, 1], batch['label'])
#         else:
#             self.test_auc.update(y_prob, batch['label'])
#         self.test_acc.update(y_hat, batch['label'])
#         self.log('test/auc', self.test_auc, on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         return loss

#     def predict_step(self, batch, batch_idx):
#         out = self.forward(batch)

#         y_hat = out['logits'].argmax(dim=-1)
#         y_prob = F.softmax(out['logits'], dim=-1)

#         out = {'pred': y_hat, 'prob': y_prob}
#         if 'label' in batch.keys():
#             out['label'] = batch['label']
#         if 'slide_id' in batch.keys():
#             out['slide_id'] = batch['slide_id']
#         if 'n_patch' in batch.keys():
#             out['bag_size'] = batch['n_patch']
#         return out

#     def configure_optimizers(self):

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
#         else:
#             raise ValueError('Unknown optimizer:', opt_name)

#         # optimizer = restore_optimizers(optimizer)
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35], gamma=0.1)
#         return [optimizer], [scheduler]

#     # def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
#     #     optimizer.zero_grad()
#     #
#     # def on_validation_model_zero_grad(self) -> None:
#     #     self.zero_grad()


# class CLAM_EwcOn(EwcOn):

#     def end_task(self, train_loader, optimizer):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.net = self.net.to(self.device)
#         fish = torch.zeros_like(self.get_params(self.net), device=self.device)

#         instance_loss_fn = self.net.net.instance_loss_fn
#         self.net.net.instance_loss_fn = nn.CrossEntropyLoss(reduction='none')
#         for batch_idx, batch in enumerate(train_loader):
#             # only work for batch_size = 1 MIL!
#             # batch_size = inputs.size(0)
#             batch = self.net.transfer_batch_to_device(batch, self.device, 0)
#             label = batch['label']
#             optimizer.zero_grad()

#             # work for clam
#             out = self.net(batch, instance_eval=True)

#             # bag
#             loss = - F.cross_entropy(self.logsoft(out['logits']), label, reduction='none')
#             loss = loss * self.args.bag_weight
#             exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
#             loss = torch.mean(loss)
#             loss.backward(retain_graph=True)
#             fish += exp_cond_prob * self.get_grads(self.net) ** 2
            
#             # instance
#             loss = - out['instance_loss'] * (1 - self.args.bag_weight)
#             exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
#             loss = torch.mean(loss)
#             loss.backward()
#             fish += exp_cond_prob * self.get_grads(self.net) ** 2

#         self.net.net.instance_loss_fn = instance_loss_fn

#         fish /= (len(train_loader) * 1)

#         if self.fish is None:
#             self.fish = fish
#         else:
#             self.fish = self.fish * self.gamma + fish

#         self.checkpoint = self.get_params(self.net).data.clone()


# import matplotlib.pyplot as plt
# class CLAM_CL(ContinualModel, CLAM):

#     def __init__(self, args):
#         # init
#         super().__init__(args)
#         if args.n_classes == 2:
#             self.test_auc = nn.ModuleList([AUROC(task='binary') for _ in range(args.n_tasks)])
#             self.test_acc = nn.ModuleList([Accuracy(task='binary') for _ in range(args.n_tasks)])
#         else:
#             self.test_auc = nn.ModuleList(
#                 [AUROC(task='multiclass', num_classes=args.n_classes, average='weighted') for _ in range(args.n_tasks)])
#             self.test_acc = nn.ModuleList(
#                 [Accuracy(task='multiclass', num_classes=args.n_classes, average='micro') for _ in range(args.n_tasks)])

#         if args.cl_method == 'ewc_on':
#             self.cl_plugin = CLAM_EwcOn(self, args)
#             self.end_task = self.cl_plugin.end_task
#         # elif args.cl_method == 'clser':
#         #     self.on_fit_start = self.clser_on_fit_start

#     def on_train_batch_start(self, batch, batch_idx):
#         if self.args.log_model:
#             self.tmp_dict = None

#     def on_fit_start(self) -> None:
#         if self.args.log_model:
#             self.all_A_grad = []
#             self.all_logits_grad = []
#             self.all_classifier_grad = []

#     def on_fit_end(self) -> None:
#         if self.args.log_model:
#             # save the gradients to wandb artifact
#             wandb_logger = self.logger.experiment
#             torch.save({'all_A_grad': self.all_A_grad, 'all_logits_grad': self.all_logits_grad, 'all_classifier_grad': self.all_classifier_grad}, os.path.join(wandb_logger.dir, 'grads.pt'))
#             # artifact = wandb.Artifact("gradient_data", type="data")
#             # artifact.add_file(os.path.join(wandb_logger.dir, 'grads.pt'))
#             # wandb_logger.log_artifact(artifact)

#             # Prepare data for plotting
#             steps = len(self.all_A_grad)

#             # Creating the hist plots
#             fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 16))

#             # Collecting data for the histograms
#             all_gradients_A = torch.cat(self.all_A_grad).cpu().numpy()
#             all_steps_A = np.repeat(np.arange(steps), [len(grad) for grad in self.all_A_grad])
#             all_gradients_logits = torch.cat(self.all_logits_grad).cpu().numpy()
#             all_steps_logits = np.repeat(np.arange(steps), [len(grad) for grad in self.all_logits_grad])
#             all_gradients_classifier = torch.cat(self.all_classifier_grad).cpu().numpy()
#             all_steps_classifier = np.repeat(np.arange(steps), [len(grad) for grad in self.all_classifier_grad])

#             # Define new bins
#             edge = 1
#             gradient_bins = np.linspace(-edge, edge, 21)  # 20 bins between -1 and 1
#             gradient_bins = np.concatenate([[-np.inf], gradient_bins, [np.inf]])  # Add bins for < -1 and > 1

#             # Compute histograms
#             hA, xedgesA, yedgesA = np.histogram2d(all_steps_A, all_gradients_A, bins=[steps, gradient_bins])
#             hLogits, xedgesL, yedgesL = np.histogram2d(all_steps_logits, all_gradients_logits, bins=[steps, gradient_bins])
#             hClassifier, xedgesC, yedgesC = np.histogram2d(all_steps_classifier, all_gradients_classifier, bins=[steps, gradient_bins])

#             # Normalize histograms by column (step)
#             hA_norm = hA / hA.sum(axis=1, keepdims=True) * 100
#             hLogits_norm = hLogits / hLogits.sum(axis=1, keepdims=True) * 100
#             hClassifier_norm = hClassifier / hClassifier.sum(axis=1, keepdims=True) * 100

#             # Determine the common maximum percentage for consistent color scaling
#             max_density = max(hA_norm.max(), hLogits_norm.max(), hClassifier_norm.max())

#             # Custom y-tick locations and labels
#             y_ticks = np.arange(23)
#             y_tick_labels = [f'-∞ to -{edge}'] + [f'{x:.2f}' for x in np.linspace(-edge, edge, 21)] + [f'{edge} to ∞']

#             # Plot the normalized histograms
#             imA = axes[0].imshow(hA_norm.T, aspect='auto', origin='lower', extent=[0, steps, 0, 22], cmap='viridis', vmin=0, vmax=max_density)
#             axes[0].set_title('Percentage Distribution of A gradients over Steps')
#             axes[0].set_xlabel('Step')
#             axes[0].set_ylabel('Gradient Value')
#             axes[0].set_yticks(y_ticks)
#             axes[0].set_yticklabels(y_tick_labels, fontsize=6)

#             imLogits = axes[1].imshow(hLogits_norm.T, aspect='auto', origin='lower', extent=[0, steps, 0, 22], cmap='viridis', vmin=0, vmax=max_density)
#             axes[1].set_title('Percentage Distribution of Logits gradients over Steps')
#             axes[1].set_xlabel('Step')
#             axes[1].set_ylabel('Gradient Value')
#             axes[1].set_yticks(y_ticks)
#             axes[1].set_yticklabels(y_tick_labels, fontsize=6)

#             imClassifier = axes[2].imshow(hClassifier_norm.T, aspect='auto', origin='lower', extent=[0, steps, 0, 22], cmap='viridis', vmin=0, vmax=max_density)
#             axes[2].set_title('Percentage Distribution of Classifier gradients over Steps')
#             axes[2].set_xlabel('Step')
#             axes[2].set_ylabel('Gradient Value')
#             axes[2].set_yticks(y_ticks)
#             axes[2].set_yticklabels(y_tick_labels, fontsize=6)

#             # Add a common colorbar
#             cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
#             fig.colorbar(imLogits, cax=cbar_ax, label='Density (%)')

#             plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the main plot to make room for the colorbar

#             self.logger.log_image('grad_distributions', [fig])


#             # Creating the box plots
#             fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 16))

#             # Define colors
#             colors = ['blue', 'red', 'green', 'orange', 'purple']

#             five_number_summary_logits = np.array([np.percentile(data.cpu().numpy(), [0, 25, 50, 75, 100]) for data in self.all_logits_grad]).T
#             five_number_summary_A = np.array([np.percentile(data.cpu().numpy(), [0, 25, 50, 75, 100]) for data in self.all_A_grad]).T
#             five_number_summary_classifier = np.array([np.percentile(data.cpu().numpy(), [0, 25, 50, 75, 100]) for data in self.all_classifier_grad]).T

#             # Plot data using scatter
#             for i, color in enumerate(colors):
#                 axes[0].scatter(np.arange(steps), five_number_summary_A[i], c=[color], s=1, alpha=0.5, label=['Minimum', '25% Quartile', 'Median', '75% Quartile', 'Maximum'][i])
#                 axes[1].scatter(np.arange(steps), five_number_summary_logits[i], c=[color], s=1, alpha=0.5, label=['Minimum', '25% Quartile', 'Median', '75% Quartile', 'Maximum'][i])
#                 axes[2].scatter(np.arange(steps), five_number_summary_classifier[i], c=[color], s=1, alpha=0.5, label=['Minimum', '25% Quartile', 'Median', '75% Quartile', 'Maximum'][i])


#             axes[0].set_title('Percentage Distribution of A gradients over Steps')
#             axes[0].set_xlabel('Step')
#             axes[0].set_ylabel('Data Distribution Change Over Steps')

#             axes[1].set_title('Percentage Distribution of Logits gradients over Steps')
#             axes[1].set_xlabel('Step')

#             axes[2].set_title('Percentage Distribution of Classifier gradients over Steps')
#             axes[2].set_xlabel('Step')

#             # Add legend
#             legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
#                             markerfacecolor=color, markersize=8) 
#                             for color, label in zip(colors[::-1], ['Minimum', '25% Quartile', 'Median', '75% Quartile', 'Maximum'][::-1])]
#             axes[0].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

#             self.logger.log_image('Grad_distributions 2', [fig])


#     def on_before_optimizer_step(self, optimizer):
#         # Compute the 2-norm for each layer
#         # If using mixed precision, the gradients are already unscaled here
#         if self.args.log_model:
#             norms = grad_norm(self, norm_type=2)
#             self.log_dict(norms)
#             if self.tmp_dict is not None:
#                 A_grad_norms = self.tmp_dict['A'].grad.data.norm(2)
#                 A_grad_max = self.tmp_dict['A'].grad.data.abs().max()
#                 logist_grad_norms = self.tmp_dict['logits'].grad.data.norm(2)
#                 logist_grad_max = self.tmp_dict['logits'].grad.data.abs().max()
#                 z_norms = self.tmp_dict['z'].data.norm(2)
#                 classifier_grad_norms = self.net.classifiers.weight.grad.data.norm(2)
#                 classifier_grad_max = self.net.classifiers.weight.grad.data.abs().max()
#                 out_dict = {'grad_A_norm': A_grad_norms, 'grad_logits_norm': logist_grad_norms,
#                             'grad_A_max': A_grad_max, 'grad_logits_max': logist_grad_max, 'z_norm': z_norms,
#                             'grad_classifier_norm': classifier_grad_norms, 'grad_classifier_max': classifier_grad_max}
#                 self.log_dict(out_dict)
#                 self.all_A_grad.append(self.tmp_dict['A'].grad.data.cpu().flatten())
#                 self.all_logits_grad.append(self.tmp_dict['logits'].grad.data.cpu().flatten())
#                 self.all_classifier_grad.append(self.net.classifiers.weight.grad.data.cpu().flatten())


#     def training_step(self, batch, batch_idx):
#         out = self.forward(batch, instance_eval=True, return_features=True)

#         if self.args.log_model:
#             A = out['attention']
#             A.retain_grad()
#             logits = out['logits']
#             logits.retain_grad()
#             self.tmp_dict = {'A': A, 'logits': logits, 'z': out['features']}

#         loss = (1 - self.args.bag_weight) * out['instance_loss']

#         if self.cl_method in ['der', 'derpp'] and 'logits' in batch.keys():
#             l2_loss = F.mse_loss(out['logits'], batch['logits'])
#             loss += l2_loss
#             if self.cl_method == 'derpp':
#                 loss += self.args.bag_weight * self.loss_fn(out['logits'], batch['label'])
#         else:
#             bag_loss = self.loss_fn(out['logits'], batch['label'])
#             loss += self.args.bag_weight * bag_loss
#             self.log('train/bag_loss', self.loss_fn(out['logits'], batch['label']), batch_size=self.args.batch_size)

#         if 'kl' in self.cl_method and 'A' in batch.keys() and batch['Y_hat'] == batch['label']:
#             # kl_loss = F.kl_div(torch.log_softmax(out['A'], dim=-1), batch['A'], reduction='batchmean')
#             T = 1.
#             kl_loss = F.kl_div(F.log_softmax(out['A'] / T, dim=-1), F.softmax(batch['A'] / T, dim=-1), reduction='batchmean') * T * T
#             loss += kl_loss
#             self.log('train/kl_loss', kl_loss, batch_size=self.args.batch_size)

#         self.log('train/loss', loss, batch_size=self.args.batch_size)
#         self.log('train/instance_loss', out['instance_loss'], batch_size=self.args.batch_size)
#         return loss

#     def ewc_training_step(self, batch, batch_idx):
#         loss = super().training_step(batch, batch_idx)
#         loss = self.cl_plugin.add_penalty_in_loss(loss)
#         return loss

#     # def clser_on_fit_start(self):
#     #     # move loss_fn to device
#     #     if self.device.type == 'cuda':
#     #         self.net.instance_loss_fn = self.net.instance_loss_fn.cuda()
#     #         self.stable_model.instance_loss_fn = self.stable_model.instance_loss_fn.cuda()
#     #         self.plastic_model.instance_loss_fn = self.plastic_model.instance_loss_fn.cuda()
#     #     else:
#     #         self.net.instance_loss_fn = self.net.instance_loss_fn.cpu()
#     #         self.stable_model.instance_loss_fn = self.stable_model.instance_loss_fn.cpu()
#     #         self.plastic_model.instance_loss_fn = self.plastic_model.instance_loss_fn.cpu()

#     def clser_training_step(self, batch, batch_idx):
#         out = self.forward(batch, instance_eval=True)

#         bag_loss = self.loss_fn(out['logits'], batch['label'])
#         instance_loss = out['instance_loss']
#         loss = self.args.bag_weight * bag_loss + (1 - self.args.bag_weight) * instance_loss

#         if 'Y_hat' in batch.keys(): # batch in memery
#             features = batch['features']
#             if self.feature_extractor is not None:
#                 features = self.feature_extractor(features)
#             stable_model_out = self.stable_model.forward(features, label=batch['label'], instance_eval=True)
#             plastic_model_out = self.plastic_model.forward(features, label=batch['label'], instance_eval=True)

#             # bag
#             stable_model_prob = F.softmax(stable_model_out['logits'], dim=-1)
#             plastic_model_prob = F.softmax(plastic_model_out['logits'], dim=-1)

#             label_mask = F.one_hot(batch['label'], num_classes=self.args.n_classes) > 0
#             sel_idx = stable_model_prob[label_mask] > plastic_model_prob[label_mask]
#             sel_idx = sel_idx.unsqueeze(1)

#             ema_logits = torch.where(
#                 sel_idx,
#                 stable_model_out['logits'],
#                 plastic_model_out['logits'],
#             ).detach()

#             l_reg = torch.mean(self.consistency_loss(out['logits'], ema_logits))
#             l_reg = self.reg_weight * l_reg * self.args.bag_weight
#             loss += l_reg

#             # instance may change
#             # stable_model_prob = F.softmax(stable_model_out['instance_logits'], dim=-1)
#             # plastic_model_prob = F.softmax(plastic_model_out['instance_logits'], dim=-1)

#             # label_mask = F.one_hot(batch['inst_labels'], num_classes=self.args.n_classes) > 0
#             # sel_idx = stable_model_prob[label_mask] > plastic_model_prob[label_mask]
#             # sel_idx = sel_idx.unsqueeze(1)

#             # ema_logits = torch.where(
#             #     sel_idx,
#             #     stable_model_out['instance_logits'],
#             #     plastic_model_out['instance_logits'],
#             # ).detach()

#             # l_reg = torch.mean(self.consistency_loss(out['instance_logits'], ema_logits))
#             # l_reg = self.reg_weight * l_reg * (1 - self.args.bag_weight)
#             # loss += l_reg

#         self.log('train/loss', loss, batch_size=self.args.batch_size)
#         return loss

#     def clser_on_train_batch_end(self, *args, **kwargs):

#         def update_model(model, target_model, alpha):
#             alpha = min(1 - 1 / (self.trainer.global_step + 1), alpha)
#             for param, target_param in zip(model.parameters(), target_model.parameters()):
#                 param.data = alpha * param.data + (1 - alpha) * target_param.data

#         # update plastic and stable model
#         if torch.randn(1) < self.plastic_model_update_freq:
#             update_model(self.plastic_model, self.net, self.plastic_model_alpha)
#         if torch.randn(1) < self.stable_model_update_freq:
#             update_model(self.stable_model, self.net, self.stable_model_alpha)

#     def test_step(self, batch, batch_idx, dataloader_idx=0):
#         out = self.forward(batch)

#         loss = self.loss_fn(out['logits'], batch['label'])
#         y_hat = out['logits'].argmax(dim=-1)
#         y_prob = F.softmax(out['logits'], dim=-1)

#         self.log('test/loss', loss, on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         if self.args.n_classes == 2:
#             self.test_auc[dataloader_idx].update(y_prob[:, 1], batch['label'])
#         else:
#             self.test_auc[dataloader_idx].update(y_prob, batch['label'])
#         self.test_acc[dataloader_idx].update(y_hat, batch['label'])
#         self.log('test/auc', self.test_auc[dataloader_idx], on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         self.log('test/acc', self.test_acc[dataloader_idx], on_step=False, on_epoch=True, batch_size=self.args.batch_size)
#         return loss
    
#     def predict_step(self, batch, batch_idx, dataloader_idx=0):
#         out = self.forward(batch)

#         y_hat = out['logits'].argmax(dim=-1)
#         y_prob = F.softmax(out['logits'], dim=-1)

#         out = {'pred': y_hat, 'prob': y_prob}
#         if 'label' in batch.keys():
#             out['label'] = batch['label']
#         if 'slide_id' in batch.keys():
#             out['slide_id'] = batch['slide_id']
#         if 'n_patch' in batch.keys():
#             out['bag_size'] = batch['n_patch']
#         return out