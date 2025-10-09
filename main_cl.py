import torch
from datasets.dataset import datamodule_gen
from models.clam import CLAM_SB, CLAM_MB
from models.transmil import TransMIL
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from utils import CustomEarlyStopping as EarlyStopping
from utils import CustomProgressBar
import yaml
import wandb
import os
import pandas as pd
from torchmetrics import AUROC, Accuracy, MeanMetric
from lightning.fabric import Fabric, seed_everything, is_wrapped
from lightning.fabric.utilities import move_data_to_device
from lightning.fabric.loggers import CSVLogger
import copy
from collections import defaultdict
import random
from mil_bcsr_coreset import BCSR_Coreset


# use pure pytorch instead of pytorch-lightning
class SimpleBuffer():
    def __init__(self, buffer_size, device='cpu', **kwargs):
        self.buffer = []  # List of samples
        self.buffer_size = buffer_size
        self.device = device
        self.sample_selection_strategy = 'reservoir'
        self.n_seen_samples = 0
        self.epoch_indices = None
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0
        self.n_patches_total = 0
        self.labels = {}  # Count of samples per class

    def to(self, device):
        old_device = self.device
        self.device = device
        if old_device != device:
            for i in range(len(self.buffer)):
                self.buffer[i] = move_data_to_device(self.buffer[i], device)
        return self

    def adjust_buffer_size_by_new_classes(self, n_new_classes: int):
        # In this simple version, we do not adjust the buffer size
        pass

    def add_judge(self, sample_label):
        """Determine whether to add a new sample to the buffer using reservoir sampling."""
        n_seen = self.n_seen_samples
        self.n_seen_samples += 1
        if len(self.buffer) < self.buffer_size:
            # Buffer not full yet
            return len(self.buffer)
        else:
            # Buffer full, apply reservoir sampling
            rand_idx = np.random.randint(0, n_seen)
            if rand_idx < self.buffer_size:
                return rand_idx
            else:
                return -1  # Do not add the sample

    def add(self, sample, idx):
        """Add a sample to the buffer at the given index."""
        assert idx != -1, "The sample is not added to the buffer"
        sample = move_data_to_device(sample, self.device)
        label = sample['label'].item()
        if idx == len(self.buffer):
            # Adding a new sample to the buffer
            self.buffer.append(sample)
            self.n_patches_total += sample['features'].size(0)
            self.labels[label] = self.labels.get(label, 0) + 1
        else:
            # Replacing an existing sample in the buffer
            self.n_patches_total = self.n_patches_total - self.buffer[idx]['features'].size(0) + sample['features'].size(0)
            old_label = self.buffer[idx]['label'].item()
            self.labels[old_label] -= 1
            self.buffer[idx] = sample
            self.labels[label] = self.labels.get(label, 0) + 1

    def start_epoch(self):
        """Reset indices for a new epoch."""
        self.epoch_indices = np.random.permutation(len(self.buffer))
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0

    def get_next_batch(self, batch_size=1):
        """Get the next batch of samples for the current epoch."""
        if self.epoch_indices is None:
            self.start_epoch()
        remaining = len(self.buffer) - self.current_epoch_position
        if remaining == 0:
            return None  # Epoch is finished
        actual_batch_size = min(batch_size, remaining)
        batch_indices = self.epoch_indices[self.current_epoch_position:
                                           self.current_epoch_position + actual_batch_size]
        batch = [self.buffer[i] for i in batch_indices]
        self.current_epoch_position += actual_batch_size
        self.samples_output_in_epoch += actual_batch_size
        return batch

    def get_samples_output_count(self):
        """Return the number of samples output in the current epoch."""
        return self.samples_output_in_epoch

    def __len__(self):
        return len(self.buffer)
    

class ClsssIncrementalBuffer():
    def __init__(self, buffer_size, device='cpu', **kwargs):
        self.buffers = {}  # Dict mapping class labels to lists of samples
        self.buffer_size = buffer_size
        self.device = device
        self.sample_selection_strategy = 'reservoir'
        self.n_seen_samples_per_class = defaultdict(int)
        self.num_classes = 0
        self.buffer_size_per_class = None
        self.epoch_indices = None
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0
        self.n_patches_total = 0
        self.labels = {}  # Count of samples per class

    def to(self, device):
        old_device = self.device
        self.device = device
        if old_device != device:
            for l in self.buffers:
                for i in range(len(self.buffers[l])):
                    self.buffers[l][i] = move_data_to_device(self.buffers[l][i], device)
        return self

    # Given the number of new classes will be adding, adjust the buffer size per class
    def adjust_buffer_size_by_new_classes(self, n_new_classes: int):
        self.buffer_size_per_class = self.buffer_size // (self.num_classes + n_new_classes)
        for l in self.buffers:
            if len(self.buffers[l]) > self.buffer_size_per_class:
                self.buffers[l] = random.sample(self.buffers[l], self.buffer_size_per_class)
                self.labels[l] = self.buffer_size_per_class
        # calculate the number of patches in the buffer
        self.n_patches_total = sum([sample['features'].size(0) for samples in self.buffers.values() for sample in samples])

    # Judge whether to add the new sample to the buffer by reservoir sampling
    # If add, return the index
    def add_judge(self, sample_label):
        if sample_label not in self.buffers:
            # New class encountered
            self.buffers[sample_label] = []
            self.num_classes += 1

        self.n_seen_samples_per_class[sample_label] += 1
        n_seen = self.n_seen_samples_per_class[sample_label]
        if len(self.buffers[sample_label]) < self.buffer_size_per_class:
            return len(self.buffers[sample_label])
        else:
            # Buffer full, apply reservoir sampling
            rand_idx = np.random.randint(0, n_seen)
            if rand_idx < self.buffer_size_per_class:
                return rand_idx
            else:
                return -1
            
    def add(self, sample, idx):
        assert idx != -1, "The sample is not added to the buffer"
        sample = move_data_to_device(sample, self.device)
        label = sample['label'].item()
        if idx == len(self.buffers[label]):
            self.buffers[label].append(sample)
            self.n_patches_total += sample['features'].size(0)
            self.labels[label] = self.labels.get(label, 0) + 1
        else:
            self.n_patches_total = self.n_patches_total - self.buffers[label][idx]['features'].size(0) + sample['features'].size(0)
            self.buffers[label][idx] = sample
            # Labels count remains the same

    def start_epoch(self):
        """Reset indices for a new epoch"""
        # Flatten buffers into a single list
        self.all_samples = []
        for samples in self.buffers.values():
            self.all_samples.extend(samples)
        self.epoch_indices = np.random.permutation(len(self.all_samples))
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0

    def get_next_batch(self, batch_size=1):
        """Get next batch of samples for the current epoch"""
        if self.epoch_indices is None:
            self.start_epoch()
        remaining = len(self.all_samples) - self.current_epoch_position
        if remaining == 0:
            return None  # Epoch is finished
        actual_batch_size = min(batch_size, remaining)
        batch_indices = self.epoch_indices[self.current_epoch_position:
                                           self.current_epoch_position + actual_batch_size]
        batch = [self.all_samples[i] for i in batch_indices]
        self.current_epoch_position += actual_batch_size
        self.samples_output_in_epoch += actual_batch_size
        return batch

    def get_samples_output_count(self):
        """Return the number of samples output in the current epoch"""
        return self.samples_output_in_epoch

    def __len__(self):
        return sum(len(samples) for samples in self.buffers.values())

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

    
def add_argument(parser, name, value):
    """Helper function to add an argument to the parser if it doesn't already exist."""
    if not any(arg.dest == name for arg in parser._actions):
        if isinstance(value, bool):
            parser.add_argument(f'--{name}', action='store_false' if value else 'store_true', default=value)
        else:
            parser.add_argument(f'--{name}', type=type(value), default=value)

def load_config_from_yaml(file_path):
    """Loads configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file) or {}

def init_args():
    parser = argparse.ArgumentParser(description='MIL-CL')
    parser.add_argument('--preset', type=str, default=None, help='preset config')

    args, remaining_argv = parser.parse_known_args()

    # Load configuration from YAML if preset is provided
    config = load_config_from_yaml(args.preset) if args.preset else {}

    # Update parser with options from YAML configuration
    for k, v in config.items():
        add_argument(parser, k, v)

    args = parser.parse_args()

    # update args
    if args.debug == 'False':
        args.debug = False
    elif args.debug == 'True':
        args.debug = 'lite'
    if args.debug == 'lite':
        args.n_batches = 10
        args.epochs = 2
        args.n_folds = 1
        args.wandb_mode = 'disabled'

        if hasattr(args, 'n_tasks'):
            args.n_tasks = 2
    elif args.debug == 'full':
        args.n_batches = 20
        args.epochs = 3
        args.wandb_mode = 'disabled'


    os.environ['WANDB_MODE'] = args.wandb_mode
    wandb.require("core")

    if args.folds_start == -1:
        args.folds_start = 0
    if args.folds_end == -1:
        if hasattr(args, 'n_folds'):
            args.folds_end = args.n_folds
        else:
            args.folds_end = 1

    print(args)
    return args

def load_model(args):
    # if args.net == 'abmil':
        # model = Attn_Net(args)
    # elif args.net == 'abmil_gated':
        # model = Attn_Net_Gated(args)
    # elif args.net == 'dtfd':
        # model = DTFD_CL(args)
    if args.net == 'clam_sb':
        model = CLAM_SB(D_feat=args.D_feat,
                        L=args.L,
                        K=args.K,
                        dropout=args.dropout,
                        gate=args.gate,
                        k_sample=args.k_sample,
                        n_classes=args.n_classes,
                        instance_loss_name=args.instance_loss_name,
                        subtyping=args.subtyping,)
    elif args.net == 'clam_mb':
        model = CLAM_MB(D_feat=args.D_feat,
                        L=args.L,
                        K=args.K,
                        dropout=args.dropout,
                        gate=args.gate,
                        k_sample=args.k_sample,
                        n_classes=args.n_classes,
                        instance_loss_name=args.instance_loss_name,
                        subtyping=args.subtyping)
    # elif args.net == 'acmil':
        # model = ACMIL_CL(args)
    elif args.net == 'transmil':
        model = TransMIL(D_feat=args.D_feat,
                         D_inner=args.L,
                         n_classes=args.n_classes,)
    else:
        raise NotImplementedError

    return model

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

def adaptive_distill(slide,  size=1e5, method='random',model=None):
    pass
import copy
# todo:关键，从一个大的patch 里面挑选出子集
def distill_slide(slide, attn=None, size=1e5, method='random',model=None,label=None,task_id=None,seen_classes=None):
    assert len(slide.shape) == 2, f"slide shape: {slide.shape}"
    size = int(min(size, slide.size(0)))
    if method == 'random':
        idx = torch.randperm(slide.size(0))[:size]
    elif method == 'max':
        idx = torch.topk(attn, size)[1][-1]
    elif method == 'maxmin':
        size = size // 2
        top_p_ids = torch.topk(attn, size)[1][-1]
        top_n_ids = torch.topk(-attn, size, dim=1)[1][-1]
        idx = torch.cat((top_p_ids, top_n_ids))
    elif method == 'maxminrand':
        size = size // 4
        top_p_ids = torch.topk(attn, size)[1][-1]
        top_n_ids = torch.topk(-attn, size, dim=1)[1][-1]
        rand_ids = torch.randperm(slide.size(0))[:size*2].to(top_n_ids.device)
        idx = torch.cat((top_p_ids, top_n_ids, rand_ids))
    elif method == 'maxrand':
        size = size // 2
        top_p_ids = torch.topk(attn, size)[1][-1]
        rand_ids = torch.randperm(slide.size(0))[:size].to(top_p_ids.device)
        idx = torch.cat((top_p_ids, rand_ids))
    elif method=="adaptive":
        pass
    elif method=="bilevel":
        # 模型 输入数据x 输出数据y，任务(用于key alue 映射) ，选择几个，outer loss是什么
        # def coreset_select(self, model, X, y, task_id,  topk, out_loss=None, ref_x=None, ref_y=None):
        size = size // 2
        # coreset_select(self, model, X, y, task_id, topk, out_loss=None, ref_x=None, ref_y=None):
        proxy_model=copy.deepcopy(model)
        for param in proxy_model.parameters():
            param.requires_grad = True

        BCSR_Coreset_selector = BCSR_Coreset(proxy_model,
                          lr_proxy_model=args.bcsr_lr_proxy_model,
                          beta=args.bcsr_beta,
                          out_dim=args.bcsr_out_dim,
                          max_outer_it=args.bcsr_max_outer_it,
                          max_inner_it=args.bcsr_max_inner_it,
                          weight_lr=args.bcsr_weight_lr,
                          candidate_batch_size=args.bcsr_candidate_batch_size,
                          logging_period=args.bcsr_logging_period)
        pick, outer_loss = BCSR_Coreset_selector.coreset_select(proxy_model, slide.cpu().numpy(), label.cpu().numpy(), task_id=task_id,
                                                 topk=args.buffer_size, out_loss=None,seen_classes=seen_classes)

        # top_p_ids = torch.topk(attn, size)[1][-1]
        # top_n_ids = torch.topk(-attn, size, dim=1)[1][-1]
        # idx = torch.cat((top_p_ids, top_n_ids))



    else:
        raise NotImplementedError

    return slide[idx]

def one_fold(args, fold=0):
    # ====== 1. 初始化设置块 ======
    # 设置分布式训练框架和随机种子
    fabric = Fabric(devices=1, accelerator="auto")
    seed_everything(args.seed)

    # 加载并初始化模型
    model = load_model(args)
    model = fabric.to_device(model)

    # 准备所有任务的测试数据加载器，用于最终评估
    test_dataloaders = [datamodule_gen(args, fold=fold, task=task)['test_loader'] for task in range(args.n_tasks)]
    # test_dataloaders = fabric.setup_dataloaders(*test_dataloaders)

    # 设置持续学习的内存缓冲区（用于存储历史样本进行回放）
    if hasattr(args, 'cl_method') and args.cl_method != 'joint':
        if hasattr(args, 'buffer_size') and args.buffer_size > 0:
            if args.buffer_balance:
                # 类别平衡的缓冲区：每个类别分配相等的存储空间
                buffer = ClsssIncrementalBuffer(buffer_size=args.buffer_size)
            else:
                # 简单缓冲区：不考虑类别平衡，随机存储样本
                buffer = SimpleBuffer(buffer_size=args.buffer_size)

    # 初始化结果存储和已见类别跟踪
    results = []
    seen_classes = np.empty(0, dtype=int)

    # ====== 2. 任务序列训练主循环 ======
    for task in range(args.n_tasks):

        # ====== 2.1 当前任务初始化块 ======
        # 清理上一个任务的旧模型，释放GPU内存
        if 'old_model' in locals():
            del old_model
        torch.cuda.empty_cache()

        # 清理之前的日志文件，设置当前任务的日志记录器
        if os.path.exists(f'{args.log_dir}/{args.exp_name}/fold_{fold}_task_{task}/metrics.csv'):
            os.remove(f'{args.log_dir}/{args.exp_name}/fold_{fold}_task_{task}/metrics.csv')
        logger = CSVLogger(root_dir=f'{args.log_dir}', name=f'{args.exp_name}', version=f'fold_{fold}_task_{task}')

        # ====== 2.2 数据准备和类别管理块 ======
        # 获取当前任务的数据模块（包含训练、验证、测试数据）
        datamodule = datamodule_gen(args, fold=fold, task=task)
        # 提取当前任务包含的类别
        cur_classes = np.asarray(np.unique(datamodule['train_dataset'].targets), dtype=int)
        # 保存之前见过的类别（用于知识蒸馏）
        old_seen_classes = seen_classes
        # 更新已见类别集合
        seen_classes = np.append(seen_classes, cur_classes)
        # 计算未见过的类别（用于掩码未见类别的输出）
        unseen_classes = list(set(np.arange(args.n_classes)) - set(seen_classes))

        # ====== 2.3 优化器配置块 ======
        # 根据配置选择学习率（支持每个任务不同的学习率）
        if isinstance(args.lr, list):
            lr = float(args.lr[task])
        else:
            lr = float(args.lr)

        # 创建优化器（支持Adam和SGD）
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        elif args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)

        # 设置数据加载器
        # model, optimizer = fabric.setup(model, optimizer)
        # train_loader, val_loader = fabric.setup_dataloaders(datamodule['train_loader'], datamodule['val_loader'])
        train_loader, val_loader = datamodule['train_loader'], datamodule['val_loader']

        # ====== 2.4 持续学习方法配置块 ======
        # 配置持续学习相关设置（知识蒸馏、权重归一化等）
        if hasattr(args, 'cl_method') and args.cl_method != 'joint':
            # 为知识蒸馏方法保存旧模型（LwF: Learning without Forgetting, MICIL, prev）
            if args.cl_method in ['LwF', 'MICIL', 'prev'] and task > 0:
                old_model = copy.deepcopy(model)
                # 冻结旧模型参数，仅用于推理
                for param in old_model.parameters():
                    param.requires_grad = False
                old_model.eval()
                old_model = fabric.to_device(old_model)

            # 权重归一化（Weight Normalization）- 防止表示空间漂移
            if args.wn and task > 0:
                with torch.no_grad():
                    if args.net == 'transmil':
                        model._fc2.weight.data = F.normalize(model._fc2.weight.data)
                    elif args.net in ['clam_sb', 'clam_mb']:
                        model.classifiers.weight.data = F.normalize(model.classifiers.weight.data)
                    else:
                        raise NotImplementedError
        
        # ====== 3. 模型训练主循环块 ======
        # 确定当前任务的训练轮数
        if isinstance(args.epochs, list):
            epochs = args.epochs[task]
        else:
            epochs = args.epochs

        # 配置早停机制
        if args.early_stop:
            early_stop = EarlyStopping(patience=args.patience, stop_epoch=epochs//2)

        # ====== 3.1 训练轮次循环 ======
        for i in range(epochs):
            # ====== 3.1.1 训练指标初始化 ======
            # 初始化训练和验证损失记录器
            train_loss_metric = MeanMetric()
            val_loss_metric = MeanMetric()

            # 根据分类任务类型初始化准确率计算器
            if args.n_classes == 2:
                val_acc_metric = fabric.to_device(Accuracy(task='binary'))
                # val_auc_metric = fabric.to_device(AUROC(task='binary'))
            else:
                val_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=args.n_classes, average='micro'))
                # val_auc_metric = fabric.to_device(AUROC(task='multiclass', num_classes=average='weighted'))

            # ====== 3.1.2 训练批次循环 ======
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                # 调试模式下限制批次数量
                if args.n_batches > 0 and batch_idx > args.n_batches:
                    break

                # 准备日志记录和梯度清零
                logger_batch = {'epoch': i, 'batch': batch_idx}
                print(batch['features'].shape)
                optimizer.zero_grad()
                batch = fabric.to_device(batch)

                # ====== 3.1.2.1 TransMIL模型前向传播和损失计算 ======
                if args.net == 'transmil':
                    # 前向传播
                    out = model(batch['features'])
                    # 掩码未见类别的输出（防止模型对未见类别产生高置信度预测）
                    out['logits'][:, unseen_classes] = -100
                    # 计算分类损失
                    loss = F.cross_entropy(out['logits'], batch['label'])

                    # ====== 持续学习损失计算（当前任务数据） ======
                    if hasattr(args, 'cl_method') and args.cl_on_current:
                        if args.cl_method == 'LwF' and task > 0:
                            # Learning without Forgetting: 知识蒸馏损失
                            old_out = old_model(batch['features'])
                            kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            loss = loss + kd_loss
                            logger_batch.update({'kd_loss': kd_loss.item()})
                        elif args.cl_method == 'MICIL' and task > 0:
                            # MICIL: 知识蒸馏 + 特征蒸馏
                            old_out = old_model(batch['features'])
                            kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            em_loss = F.mse_loss(out['features'], old_out['features'])
                            loss = loss + 10 * kd_loss + em_loss
                            logger_batch.update({'kd_loss': 10 * kd_loss.item(), 'em_loss': em_loss.item()})
                        elif args.cl_method == 'prev' and task > 0:
                            # 注意力保持方法：保持注意力模式不变
                            old_out = old_model(batch['features'], return_attn=True)
                            # logits_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            attn_loss = kd_loss_fn(out['attn1'], old_out['attn1']) + kd_loss_fn(out['attn2'], old_out['attn2']) + \
                                        F.mse_loss(out['h1'], old_out['h1']) + F.mse_loss(out['h2'], old_out['h2'])
                            # em_loss = F.mse_loss(out['features'], old_out['features'])
                            loss = loss + attn_loss
                            logger_batch.update({'attn_loss': attn_loss.item()})

                    # 更新训练损失并记录
                    train_loss_metric.update(loss.item())
                    logger_batch.update({'loss': loss.item()})
                    logger.log_metrics(logger_batch)

                # ====== 3.1.2.2 CLAM模型前向传播和损失计算 ======
                elif args.net in ['clam_sb', 'clam_mb']:
                    # print(batch['features'].device)
                    # print(batch['label'])

                    # CLAM前向传播：包含包级别和实例级别预测
                    out = model(batch['features'], batch['label'], instance_eval=True, return_features=True, seen_classes=seen_classes)
                    bag_loss = F.cross_entropy(out['logits'], batch['label'])  # 包级别损失
                    inst_loss = out['instance_loss']  # 实例级别损失
                    loss = 0.7*bag_loss + 0.3*inst_loss  # 加权组合

                    # ====== 持续学习损失计算（当前任务数据） ======
                    if hasattr(args, 'cl_method') and args.cl_on_current:
                        if args.cl_method == 'LwF' and task > 0:
                            # Learning without Forgetting: 知识蒸馏损失
                            old_out = old_model(batch['features'], batch['label'], instance_eval=True, seen_classes=old_seen_classes)
                            kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            loss = loss + kd_loss
                            logger_batch.update({'kd_loss': kd_loss.item()})
                        elif args.cl_method == 'MICIL' and task > 0:
                            # MICIL: 知识蒸馏 + 特征蒸馏
                            old_out = old_model(batch['features'], batch['label'], instance_eval=True, return_features=True, seen_classes=old_seen_classes)
                            kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            em_loss = F.mse_loss(out['features'], old_out['features'])
                            loss = loss + 10 * kd_loss + em_loss
                            logger_batch.update({'kd_loss': 10 * kd_loss.item(), 'em_loss': em_loss.item()})
                        elif args.cl_method == 'prev' and task > 0:
                            # 注意力保持方法：保持注意力权重不变
                            old_out = old_model(batch['features'], batch['label'], instance_eval=True, return_features=True, seen_classes=old_seen_classes)
                            # logits_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                            attn_loss = kd_loss_fn(out['A'], old_out['A'])  # 注意力权重蒸馏
                            # em_loss = F.mse_loss(out['features'], old_out['z'])
                            loss = loss + attn_loss
                            logger_batch.update({'attn_loss': attn_loss.item()})

                    # 更新训练损失并记录
                    train_loss_metric.update(loss.item())
                    logger_batch.update({'bag_loss': bag_loss.item(), 'inst_loss': inst_loss.item(), 'loss': loss.item()})
                    logger.log_metrics(logger_batch)
                else:
                    raise NotImplementedError

                # ====== 3.1.2.3 反向传播和参数更新 ======
                fabric.backward(loss)
                optimizer.step()

                # ====== 3.1.2.4 权重归一化（防止表示漂移） ======
                if args.wn and task > 0 and hasattr(args, 'cl_method'):
                    with torch.no_grad():
                        if args.net == 'transmil':
                            model._fc2.weight.data = F.normalize(model._fc2.weight.data)
                        elif args.net in ['clam_sb', 'clam_mb']:
                            model.classifiers.weight.data = F.normalize(model.classifiers.weight.data)
                        else:
                            raise NotImplementedError

                # ====== 3.1.2.5 内存缓冲区回放训练 ======
                # 从第二个任务开始，使用缓冲区中的历史样本进行回放训练
                if task > 0 and hasattr(args, 'cl_method') and hasattr(args, 'buffer_size') and args.buffer_size > 0:
                    # 每个epoch开始时重置缓冲区采样顺序
                    if batch_idx == 0: buffer.start_epoch()

                    # 确保缓冲区样本和当前任务样本的训练比例平衡
                    # 当前批次进度 >= 缓冲区样本输出进度时，从缓冲区采样
                    while batch_idx / len(train_loader) >= buffer.get_samples_output_count() / len(buffer):
                        old_batch = buffer.get_next_batch()
                        if old_batch is None: raise StopIteration  # 缓冲区样本已全部遍历完
                        old_batch = old_batch[0]
                        logger_batch = {'epoch': i, 'batch': batch_idx}
                        old_batch = fabric.to_device(old_batch)
                        optimizer.zero_grad()

                        # ====== 3.1.2.5.1 TransMIL缓冲区样本训练 ======
                        if args.net == 'transmil':
                            out = model(old_batch['features'], return_attn=True)
                            out['logits'][:, unseen_classes] = -100
                            loss = F.cross_entropy(out['logits'], old_batch['label'])

                            if hasattr(args, 'cl_method'):
                                if args.cl_method == 'LwF' and task > 0:
                                    old_out = old_model(old_batch['features'])
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    loss = loss + kd_loss
                                    logger_batch.update({'logits_loss': kd_loss.item()})
                                elif args.cl_method == 'derpp' and task > 0:
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_batch['logits'][:, old_seen_classes])
                                    loss = loss + kd_loss
                                    logger_batch.update({'logits_loss': kd_loss.item()})
                                elif args.cl_method == 'MICIL' and task > 0:
                                    old_out = old_model(old_batch['features'])
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    em_loss = F.mse_loss(out['features'], old_out['features'])
                                    loss = loss + 10 * kd_loss + em_loss
                                    logger_batch.update({'kd_loss': 10 * kd_loss.item(), 'em_loss': em_loss.item()})
                                elif args.cl_method == 'prev' and task > 0:
                                    old_out = old_model(old_batch['features'], return_attn=True)
                                    p_loss = 0.
                                    logits_loss = 10 * kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    attn_loss = 10 * (kd_loss_fn(out['attn1'], old_out['attn1'], ta=1) + kd_loss_fn(out['attn2'], old_out['attn2'], ta=1))
                                    h_loss = 10 * (F.mse_loss(out['h1'], old_out['h1']) + F.mse_loss(out['h2'], old_out['h2']))
                                    # p_loss +=  kd_loss_fn(out['vv1'], old_out['vv1']) + kd_loss_fn(out['vv2'], old_out['vv2'])
                                    # p_loss += F.mse_loss(out['features'], old_out['features'])
                                    loss = loss + logits_loss + attn_loss + h_loss
                                    logger_batch.update({'logits_loss': logits_loss.item(), 'attn_loss': attn_loss.item(), 'h_loss': h_loss.item()})
                                elif args.cl_method == 'ER':
                                    pass
                                else:
                                    raise NotImplementedError

                            train_loss_metric.update(loss.item())
                        elif args.net in ['clam_sb', 'clam_mb']:
                            out = model(old_batch['features'], old_batch['label'], instance_eval=True, return_features=True, seen_classes=seen_classes)
                            bag_loss = F.cross_entropy(out['logits'], old_batch['label'])
                            inst_loss = out['instance_loss']
                            loss = 0.7*bag_loss + 0.3*inst_loss
                            logger_batch.update({'bag_loss': bag_loss.item(), 'inst_loss': inst_loss.item()})

                            if hasattr(args, 'cl_method'):
                                if args.cl_method == 'LwF' and task > 0:
                                    old_out = old_model(old_batch['features'], old_batch['label'], instance_eval=True, return_features=True, seen_classes=old_seen_classes)
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    loss = loss + kd_loss
                                    logger_batch.update({'logits_loss': kd_loss.item()})
                                elif args.cl_method == 'derpp' and task > 0:
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_batch['logits'][:, old_seen_classes])
                                    loss = loss + kd_loss
                                    logger_batch.update({'logits_loss': kd_loss.item()})
                                elif args.cl_method == 'MICIL' and task > 0:
                                    old_out = old_model(old_batch['features'], old_batch['label'], instance_eval=True, return_features=True, seen_classes=old_seen_classes)
                                    kd_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    em_loss = F.mse_loss(out['features'], old_out['features'])
                                    loss = loss + 10 * kd_loss + em_loss
                                    logger_batch.update({'logits_loss': 10 * kd_loss.item(), 'features_loss': em_loss.item()})
                                elif args.cl_method == 'prev' and task > 0:
                                    old_out = old_model(old_batch['features'], old_batch['label'], instance_eval=True, return_features=True, seen_classes=old_seen_classes)
                                    logits_loss = kd_loss_fn(out['logits'][:, old_seen_classes], old_out['logits'][:, old_seen_classes])
                                    attn_loss = kd_loss_fn(out['A'], old_out['A'])
                                    loss = loss + attn_loss + 10 * logits_loss
                                    logger_batch.update({'attn_loss': attn_loss.item(), 'logits_loss': 10 * logits_loss.item()})
                                elif args.cl_method == 'ER':
                                    pass
                                else:
                                    raise NotImplementedError

                            train_loss_metric.update(loss.item())
                        else:
                            raise NotImplementedError
                        
                        logger_batch.update({'loss': loss.item()})
                        logger.log_metrics(logger_batch)

                        fabric.backward(loss)
                        optimizer.step()

                        if args.wn and task > 0 and hasattr(args, 'cl_method'):
                            with torch.no_grad():
                                if args.net == 'transmil':
                                    model._fc2.weight.data = F.normalize(model._fc2.weight.data)
                                elif args.net in ['clam_sb', 'clam_mb']:
                                    model.classifiers.weight.data = F.normalize(model.classifiers.weight.data)
                                else:
                                    raise NotImplementedError
                                    

            # ====== 3.2 验证阶段 ======
            model.eval()
            for batch_idx, batch in enumerate(val_loader):
                # 调试模式下限制验证批次数量
                if args.n_batches > 0 and batch_idx > args.n_batches:
                    break

                batch = fabric.to_device(batch)

                # ====== 3.2.1 CLAM模型验证 ======
                if args.net in ['clam_sb', 'clam_mb']:
                    with torch.no_grad():
                        out = model(batch['features'], batch['label'], instance_eval=True, seen_classes=seen_classes)
                    bag_loss = F.cross_entropy(out['logits'], batch['label'])
                    inst_loss = out['instance_loss']
                    loss = 0.7*bag_loss + 0.3*inst_loss
                    y_prob = F.softmax(out['logits'], dim=-1)  # 预测概率
                    y_hat = out['logits'].argmax(dim=-1)  # 预测类别

                    val_loss_metric.update(loss.item())
                    val_acc_metric.update(y_hat, batch['label'])
                    # val_auc_metric.update(y_prob, batch['label'])

                # ====== 3.2.2 TransMIL模型验证 ======
                elif args.net == 'transmil':
                    with torch.no_grad():
                        out = model(batch['features'])
                    out['logits'][:, unseen_classes] = -100  # 掩码未见类别
                    loss = F.cross_entropy(out['logits'], batch['label'])
                    y_prob = F.softmax(out['logits'], dim=-1)  # 预测概率
                    y_hat = out['logits'].argmax(dim=-1)  # 预测类别

                    val_loss_metric.update(loss.item())
                    val_acc_metric.update(y_hat, batch['label'])
                    # val_auc_metric.update(y_prob, batch['label'])
                else:
                    raise NotImplementedError
                

            # ====== 3.3 epoch结束后的日志记录和早停检查 ======
            if args.net in ['clam_sb', 'clam_mb', 'transmil']:
                # 计算并记录训练和验证指标
                train_loss = train_loss_metric.compute().item()
                val_loss = val_loss_metric.compute().item()
                val_acc = val_acc_metric.compute().item()
                # val_auc = val_auc_metric.compute().item()
                print(f'Epoch {i}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, val_acc={val_acc:.5f}\n')

                # 重置指标记录器，为下一个epoch做准备
                train_loss_metric.reset()
                val_loss_metric.reset()
                val_acc_metric.reset()
                # val_auc_metric.reset()
                logger.log_metrics({'epoch': i, 'train_loss_epoch': train_loss, 'val_loss_epoch': val_loss, 'val_acc_epoch': val_acc})
            else:
                train_loss = train_loss_metric.compute().item()
                val_loss = val_loss_metric.compute().item()
                print(f'Epoch {i}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}\n')

                train_loss_metric.reset()
                val_loss_metric.reset()
                logger.log_metrics({'epoch': i, 'train_loss_epoch': train_loss, 'val_loss_epoch': val_loss})

            # ====== 3.4 早停机制检查 ======
            if args.early_stop:
                early_stop(i, val_loss)
                # 如果当前验证损失最小，保存模型
                if early_stop.val_loss_min == val_loss:
                    torch.save(model.state_dict(), f'{args.log_dir}/{args.exp_name}/fold_{fold}_task_{task}.pt')
                # 检查是否满足早停条件
                if early_stop.early_stop:
                    print("Early stopping")
                    break

        # ====== 4. 模型保存和加载最佳权重 ======
        # 如果没有使用早停，在训练结束后保存模型
        if not args.early_stop:
            torch.save(model.state_dict(), f'{args.log_dir}/{args.exp_name}/fold_{fold}_task_{task}.pt')

        # 加载最佳模型权重并切换到评估模式
        model.load_state_dict(torch.load(f'{args.log_dir}/{args.exp_name}/fold_{fold}_task_{task}.pt'))
        model.eval()
        torch.cuda.empty_cache()

        # todo : 选择 example

        # ====== 5. 内存缓冲区数据添加块 ======
        # 将当前任务的代表性样本添加到缓冲区，用于未来任务的回放训练
        if hasattr(args, 'cl_method') and hasattr(args, 'buffer_size') and args.buffer_size > 0:
            seed_everything(args.seed)
            # 根据新增类别数量调整缓冲区大小
            buffer.adjust_buffer_size_by_new_classes(len(cur_classes))
            n_seen_samples_per_task = min(args.buffer_size, len(train_loader))

            # ====== 5.1 遍历训练数据并选择性添加到缓冲区 ======
            for batch_idx, batch in enumerate(train_loader):
                # seed_everything(args.seed + batch_idx)
                # 通过储蓄池采样判断是否将当前样本添加到缓冲区
                buffer_idx = buffer.add_judge(batch['label'].item())
                if buffer_idx == -1: continue  # 不添加当前样本
                # ====== 5.2 样本特征压缩（可选） ======
                # 如果配置了缓冲区样本大小限制，则对样本进行特征压缩
                if args.buffer_slide_size > 0:
                    # with torch.no_grad():
                    # 计算压缩后的特征大小
                    buffer_slide_size = args.buffer_slide_size * batch['features'].size(0) if isinstance(args.buffer_slide_size, float) else args.buffer_slide_size
                    batch = fabric.to_device(batch)

                    # ====== 5.2.1 CLAM模型的特征蒸馏 ======
                    if args.net in ['clam_sb', 'clam_mb']:
                        with torch.no_grad():
                            out = model(batch['features'], batch['label'], seen_classes=seen_classes)
                            attn = out['A'].mean(dim=0, keepdim=True).view(1, -1)  # 获取注意力权重
                        # 基于注意力权重选择代表性特征
                        slide = distill_slide(batch['features'].cpu(), attn.cpu(), size=buffer_slide_size, method=args.distill_method,model=model,label=batch['label'].cpu(),task_id=task,seen_classes=seen_classes)
                        batch['features'] = slide

                    # ====== 5.2.2 TransMIL模型的特征蒸馏 ======
                    elif args.net == 'transmil':
                        out = model(batch['features'], return_attn=True)
                        # 归一化注意力权重
                        attn1, attn2 = out['attn1']/(out['attn1'].max(dim=1, keepdim=True)[0]), out['attn2']/(out['attn2'].max(dim=1, keepdim=True)[0])
                        attn = torch.ones(attn1.shape).to(attn1.device)
                        attn = (attn2 + 1) * (attn1 + 1) / 4  # 组合两层注意力
                        attn = attn.mean(dim=0, keepdim=True).view(1, -1)
                        # 基于注意力权重选择代表性特征
                        slide = distill_slide(batch['features'].cpu(), attn.cpu(), size=buffer_slide_size, method=args.distill_method,model=model,label=batch['label'].cpu())
                        batch['features'] = slide
                    else:
                        raise NotImplementedError

                # ====== 5.3 DER++方法的logits存储 ======
                # 如果使用DER++方法，需要存储当前模型的输出logits
                if args.cl_method in ['derpp']:
                    with torch.no_grad():
                        batch = fabric.to_device(batch)
                        if args.net in ['clam_sb', 'clam_mb']:
                            out = model(batch['features'], batch['label'], seen_classes=seen_classes)
                        elif args.net == 'transmil':
                            out = model(batch['features'])
                    batch['logits'] = out['logits']  # 保存logits用于未来的蒸馏

                # ====== 5.4 将样本添加到缓冲区 ======
                buffer.add(batch, buffer_idx)
                # if batch_idx >= n_seen_samples_per_task:
                #     break

            # ====== 5.5 缓冲区状态记录 ======
            print(f'Buffer size: {len(buffer)}')
            print(f'Number of patches in buffer: {buffer.n_patches_total}')
            print(f'Labels in buffer: {buffer.labels}')
            # 保存缓冲区类别分布到CSV文件
            with open(f'{args.log_dir}/{args.exp_name}/fold_{fold}_task_{task}/buffer_labels.csv', 'a') as f:
                for key in buffer.labels.keys():
                    f.write("%s,%s\n"%(key,buffer.labels[key]))
            logger.log_metrics({'buffer_size': len(buffer), 'n_patches_in_buffer': buffer.n_patches_total})

        # ====== 6. 清理内存 ======
        del train_loader, val_loader, optimizer, datamodule

        # ====== 7. 模型测试评估块 ======
        # 在所有任务的测试集上评估当前模型的性能
        model.eval()
        # 根据分类任务类型初始化测试准确率计算器
        if args.n_classes == 2:
            test_acc_metric = fabric.to_device(Accuracy(task='binary'))
            # test_auc_metric = fabric.to_device(AUROC(task='binary'))
        else:
            test_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=args.n_classes, average='micro'))
            # test_auc_metric = fabric.to_device(AUROC(task='multiclass', num_classes=args.n_classes, average='weighted'))

        # ====== 7.1 在所有任务测试集上评估 ======
        result = {'fold': fold, 'task': task}
        for idx, test_loader in enumerate(test_dataloaders):
            # 遍历每个任务的测试集（测试前向传输能力和后向传输能力）
            for batch_idx, batch in enumerate(test_loader):
                # 调试模式下限制测试批次数量
                if args.n_batches > 0 and batch_idx > args.n_batches:
                    break

                batch = fabric.to_device(batch)

                # ====== 7.1.1 CLAM模型测试推理 ======
                if args.net in ['clam_sb', 'clam_mb']:
                    with torch.no_grad():
                        out = model(batch['features'], seen_classes=seen_classes)
                    # bag_loss = F.cross_entropy(out['logits'], batch['label'])
                    y_hat = out['logits'].argmax(dim=-1)  # 预测类别
                    y_prob = F.softmax(out['logits'], dim=-1)  # 预测概率

                    test_acc_metric.update(y_hat, batch['label'])
                    # test_auc_metric.update(y_prob, batch['label'])

                # ====== 7.1.2 TransMIL模型测试推理 ======
                elif args.net == 'transmil':
                    with torch.no_grad():
                        out = model(batch['features'])
                    out['logits'][:, unseen_classes] = -100  # 掩码未见类别
                    y_hat = out['logits'].argmax(dim=-1)  # 预测类别
                    y_prob = F.softmax(out['logits'], dim=-1)  # 预测概率

                    test_acc_metric.update(y_hat, batch['label'])
                    # test_auc_metric.update(y_prob, batch['label'])
                else:
                    raise NotImplementedError

            # ====== 7.1.3 记录当前任务测试结果 ======
            result.update({f'{idx}_acc': test_acc_metric.compute().item()})
            test_acc_metric.reset()
            # test_auc_metric.reset()

        # ====== 7.2 保存和输出结果 ======
        print(result)
        results.append(result)
        # 将结果保存到CSV文件
        df = pd.DataFrame(results)
        df.to_csv(f'{args.log_dir}/{args.exp_name}/fold_{fold}_results.csv', index=False)

        # 记录任务完成状态
        logger.finalize(f"Success on fold {fold} task {task}!")

    # ====== 8. 返回所有任务的测试结果 ======
    return results # [{'fold': 0, 'task': 0, '0_auc': 0.9, '0_acc': 0.8}, {...}, ...]


def one_fold_jt(args, fold=0):
    fabric = Fabric(devices=1, accelerator="auto")
    seed_everything(args.seed)

    model = load_model(args)
    model = fabric.to_device(model)

    # Prepare test dataloaders from all tasks
    datamodule = datamodule_gen(args, fold=fold, task=args.n_tasks)

    # optimizer
    if isinstance(args.lr, list):
        lr = float(args.lr[0])
    else:
        lr = float(args.lr)
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)

    # model, optimizer = fabric.setup(model, optimizer)
    # train_loader, val_loader = fabric.setup_dataloaders(datamodule['train_loader'], datamodule['val_loader'])
    train_loader, val_loader = datamodule['train_loader'], datamodule['val_loader']

    logger = CSVLogger(root_dir=f'{args.log_dir}', name=f'{args.exp_name}', version=f'fold_{fold}_JT')

    # Fitting
    if isinstance(args.epochs, list):
        epochs = args.epochs[0]
    else:
        epochs = args.epochs
    if args.early_stop:
        early_stop = EarlyStopping(patience=args.patience, stop_epoch=epochs//2)

    for i in range(epochs):
        # init logging
        train_loss_metric = MeanMetric()
        val_loss_metric = MeanMetric()
        if args.n_classes == 2:
            val_acc_metric = fabric.to_device(Accuracy(task='binary'))
            # val_auc_metric = fabric.to_device(AUROC(task='binary'))
        else:
            val_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=args.n_classes, average='micro'))
            # val_auc_metric = fabric.to_device(AUROC(task='multiclass', num_classes=args.n_classes, average='weighted'))

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            if args.n_batches > 0 and batch_idx > args.n_batches:
                break
            
            logger_batch = {'epoch': i, 'batch': batch_idx}

            optimizer.zero_grad()
            batch = fabric.to_device(batch)

            if args.net == 'transmil':
                out = model(batch['features'])
                loss = F.cross_entropy(out['logits'], batch['label'])

                train_loss_metric.update(loss.item())
                logger_batch.update({'loss': loss.item()})
                logger.log_metrics(logger_batch)

            elif args.net in ['clam_sb', 'clam_mb']:
                out = model(batch['features'], batch['label'], instance_eval=True, return_features=True)
                bag_loss = F.cross_entropy(out['logits'], batch['label'])
                inst_loss = out['instance_loss']
                loss = 0.7*bag_loss + 0.3*inst_loss

                train_loss_metric.update(loss.item())
                logger_batch.update({'bag_loss': bag_loss.item(), 'inst_loss': inst_loss.item(), 'loss': loss.item()})
                logger.log_metrics(logger_batch)
            else:
                raise NotImplementedError

            fabric.backward(loss)
            optimizer.step()

        # validation
        model.eval()
        for batch_idx, batch in enumerate(val_loader):
            if args.n_batches > 0 and batch_idx > args.n_batches:
                break

            batch = fabric.to_device(batch)

            if args.net in ['clam_sb', 'clam_mb']:
                with torch.no_grad():
                    out = model(batch['features'], batch['label'], instance_eval=True)
                bag_loss = F.cross_entropy(out['logits'], batch['label'])
                inst_loss = out['instance_loss']
                loss = 0.7*bag_loss + 0.3*inst_loss
                y_prob = F.softmax(out['logits'], dim=-1)
                y_hat = out['logits'].argmax(dim=-1)

                val_loss_metric.update(loss.item())
                val_acc_metric.update(y_hat, batch['label'])
                # val_auc_metric.update(y_prob, batch['label'])
            elif args.net == 'transmil':
                with torch.no_grad():
                    out = model(batch['features'])
                loss = F.cross_entropy(out['logits'], batch['label'])
                y_prob = F.softmax(out['logits'], dim=-1)
                y_hat = out['logits'].argmax(dim=-1)

                val_loss_metric.update(loss.item())
                val_acc_metric.update(y_hat, batch['label'])
                # val_auc_metric.update(y_prob, batch['label'])
            else:
                raise NotImplementedError
            

        # logging each epoch
        if args.net in ['clam_sb', 'clam_mb', 'transmil']:
            train_loss = train_loss_metric.compute().item()
            val_loss = val_loss_metric.compute().item()
            val_acc = val_acc_metric.compute().item()
            # val_auc = val_auc_metric.compute().item()
            print(f'Epoch {i}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, val_acc={val_acc:.5f}\n')

            train_loss_metric.reset()
            val_loss_metric.reset()
            val_acc_metric.reset()
            # val_auc_metric.reset()
            logger.log_metrics({'epoch': i, 'train_loss_epoch': train_loss, 'val_loss_epoch': val_loss, 'val_acc_epoch': val_acc})
        else:
            train_loss = train_loss_metric.compute().item()
            val_loss = val_loss_metric.compute().item()
            print(f'Epoch {i}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}\n')

            train_loss_metric.reset()
            val_loss_metric.reset()
            logger.log_metrics({'epoch': i, 'train_loss_epoch': train_loss, 'val_loss_epoch': val_loss})

        if args.early_stop:
            early_stop(i, val_loss)
            if early_stop.val_loss_min == val_loss:
                torch.save(model.state_dict(), f'{args.log_dir}/{args.exp_name}/fold_{fold}_JT.pt')
            if early_stop.early_stop:
                print("Early stopping")
                break

    del train_loader, val_loader, optimizer
    # testing, log acc and auc
    model.load_state_dict(torch.load(f'{args.log_dir}/{args.exp_name}/fold_{fold}_JT.pt'))
    model.eval()

    if args.n_classes == 2:
        test_acc_metric = fabric.to_device(Accuracy(task='binary'))
        test_acc_metric_task = fabric.to_device(Accuracy(task='binary'))
    else:
        test_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=args.n_classes, average='micro'))
        test_acc_metric_task = fabric.to_device(Accuracy(task='multiclass', num_classes=args.n_classes, average='micro'))
    
    test_dataloaders = fabric.setup_dataloaders(*datamodule['test_loader'])
    result = {'fold': fold}
    for idx, test_loader in enumerate(test_dataloaders):
        for batch_idx, batch in enumerate(test_loader):
            if args.n_batches > 0 and batch_idx > args.n_batches:
                break

            if args.net in ['clam_sb', 'clam_mb']:
                with torch.no_grad():
                    out = model(batch['features'])
                # bag_loss = F.cross_entropy(out['logits'], batch['label'])
                y_hat = out['logits'].argmax(dim=-1)
                # y_prob = F.softmax(out['logits'], dim=-1)

                test_acc_metric.update(y_hat, batch['label'])
                test_acc_metric_task.update(y_hat, batch['label'])
                # test_auc_metric.update(y_prob, batch['label'])
            elif args.net == 'transmil':
                with torch.no_grad():
                    out = model(batch['features'])
                y_hat = out['logits'].argmax(dim=-1)
                # y_prob = F.softmax(out['logits'], dim=-1)

                test_acc_metric.update(y_hat, batch['label'])
                test_acc_metric_task.update(y_hat, batch['label'])
                # test_auc_metric.update(y_prob, batch['label'])
            else:
                raise NotImplementedError
        result.update({f'{idx}_acc': test_acc_metric_task.compute().item()})

        test_acc_metric_task.reset()
        # test_auc_metric.reset()

    result.update({'acc': test_acc_metric.compute().item()})

    print(result)
    df = pd.DataFrame(result, index=[0])
    df.to_csv(f'{args.log_dir}/{args.exp_name}/fold_{fold}_results_JT.csv', index=False)
    logger.finalize(f"Success on fold {fold}!")
    return result

def test_on_one_fold(args, fold=0):
    fabric = Fabric(devices=1, accelerator="auto")
    model = load_model(args)
    test_dataloaders = [datamodule_gen(args, fold=fold, task=task)['test_loader'] for task in range(args.n_tasks)]
    test_dataloaders = fabric.setup_dataloaders(*test_dataloaders)

    results = []
    seen_classes = np.empty(0, dtype=int)
    for model_idx in range(args.n_tasks):
        cur_classes = np.asarray(np.unique(test_dataloaders[model_idx].dataset.targets), dtype=int)
        seen_classes = np.append(seen_classes, cur_classes)

        if args.load is not None:
            model.load_state_dict(torch.load(args.load))
        else:
            model.load_state_dict(torch.load(f'{args.log_dir}/{args.exp_name}/fold_{fold}_task_{model_idx}.pt'))
        model = fabric.to_device(model)
        model.eval()

        result = {'fold': fold, 'model': model_idx}

        for idx, test_loader in enumerate(test_dataloaders[:model_idx+1]):
            cur_classes = np.asarray(np.unique(test_loader.dataset.targets), dtype=int)
            if len(seen_classes) <= 2:
                ci_acc_metric = fabric.to_device(Accuracy(task='binary'))
                ci_auc_metric = fabric.to_device(AUROC(task='binary'))
            else:
                ci_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=len(seen_classes), average='micro'))
                ci_auc_metric = fabric.to_device(AUROC(task='multiclass', num_classes=len(seen_classes), average='weighted'))
            if len(cur_classes) <= 2:
                ti_acc_metric = fabric.to_device(Accuracy(task='binary'))
                ti_auc_metric = fabric.to_device(AUROC(task='binary'))
            else:
                ti_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=len(cur_classes), average='micro'))
                ti_auc_metric = fabric.to_device(AUROC(task='multiclass', num_classes=len(cur_classes), average='weighted'))

            for batch_idx, batch in enumerate(test_loader):
                if args.n_batches > 0 and batch_idx > args.n_batches:
                    break

                if args.net in ['clam_sb', 'clam_mb', 'transmil']:
                    with torch.no_grad():
                        out = model(batch['features'])

                    ci_logit = out['logits'][:, seen_classes]
                    ci_label = F.one_hot(batch['label'], num_classes=args.n_classes)[:, seen_classes]
                    ci_label = ci_label.argmax(dim=-1) 
                    y_hat = ci_logit.argmax(dim=-1)
                    y_prob = F.softmax(ci_logit, dim=-1)
                    if len(seen_classes) <= 2: y_prob = y_prob[:, -1]
                    ci_acc_metric.update(y_hat, ci_label)
                    ci_auc_metric.update(y_prob, ci_label)

                    ti_logit = out['logits'][:, cur_classes]
                    ti_label = F.one_hot(batch['label'], num_classes=args.n_classes)[:, cur_classes]
                    ti_label = ti_label.argmax(dim=-1)
                    y_hat = ti_logit.argmax(dim=-1)
                    y_prob = F.softmax(ti_logit, dim=-1)
                    if len(cur_classes) <= 2: y_prob = y_prob[:, -1]
                    ti_acc_metric.update(y_hat, ti_label)
                    ti_auc_metric.update(y_prob, ti_label)


            result.update({f'{idx}_ci_auc': ci_auc_metric.compute().item(), 
                           f'{idx}_ci_acc': ci_acc_metric.compute().item(),
                           f'{idx}_ti_auc': ti_auc_metric.compute().item(), 
                           f'{idx}_ti_acc': ti_acc_metric.compute().item()})
                
        print(result)
        results.append(result)

    return results

def test(args):
    seed_everything(args.seed)

    if args.load is None:
        results = []
        # Test models from all tasks
        for fold in range(args.folds_start, args.folds_end):
            result = test_on_one_fold(args, fold=fold)
            results.extend(result)
    else:
        fold = max(args.folds_start, 0)
        # model_fold = args.load.split('_')[-2]
        # assert fold == int(model_fold), f'Fold mismatch: {fold} vs {model_fold}'
        results = test_on_one_fold(args, fold=fold)

    # convert to csv
    df = pd.DataFrame(results)
    log_path = f'{args.log_dir}/{args.exp_name}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    df.to_csv(f'{log_path}/test_results.csv', index=False)

def main(args):
    log_path = f'{args.log_dir}/{args.exp_name}'
    if not os.path.exists(log_path): # log文件
        os.makedirs(log_path)
    # save args to yaml
    with open(f'{log_path}/args.yaml', 'w') as f: # 写入参数
        yaml.dump(vars(args), f) #

    results = []
    for fold in range(args.folds_start, args.folds_end):
        if 'jt' in args.dataset or (hasattr(args, 'cl_method') and args.cl_method == 'joint'):
            result = one_fold_jt(args, fold=fold)
        else:
            result = one_fold(args, fold=fold)

        results.extend(result)
    
    # convert to csv
    df = pd.DataFrame(results)
    df.to_csv(f'{log_path}/results.csv', index=False)

if __name__ == '__main__':
    args = init_args()

    if args.testing:
        test(args)
    else:
        main(args)
