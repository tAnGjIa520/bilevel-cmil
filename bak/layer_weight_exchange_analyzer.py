
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
from collections import OrderedDict

# use pure pytorch instead of pytorch-lightning

    
def add_argument(parser, name, **kwargs):
    """Helper function to add an argument to the parser if it doesn't already exist."""
    if not any(arg.dest == name for arg in parser._actions):
        parser.add_argument(f'--{name}', **kwargs)

def load_config_from_yaml(file_path):
    """Loads configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file) or {}

def init_args():
    parser = argparse.ArgumentParser(description='MIL-CL')
    parser.add_argument('--preset', type=str, default="configs/csc_clam_cl.yaml", help='preset config')

    args, remaining_argv = parser.parse_known_args()

    # Load configuration from YAML if preset is provided
    config = load_config_from_yaml(args.preset) if args.preset else {}

    # Update parser with options from YAML configuration
    for k, v in config.items():
        add_argument(parser, k, type=type(v), default=v)

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

def replace_layer_weights(weights, target_weights, layers_to_replace):
    new_state_dict = OrderedDict()
    
    replace_count = 0
    for k, v in weights.items():
        if any(layer in k for layer in layers_to_replace):
            new_state_dict[k] = target_weights[k]
            replace_count += 1
        else:
            new_state_dict[k] = v
    
    print(f'Replaced {replace_count} layers')
    return new_state_dict

def test_swap_model_on_one_fold(args, fold=0):
    fabric = Fabric(devices=1, accelerator="auto")
    model = load_model(args)
    target_weight = torch.load(args.target_weight)

    test_dataloaders = [datamodule_gen(args, fold=fold, task=task)['test_loader'] for task in range(args.n_tasks)]
    test_dataloaders = fabric.setup_dataloaders(*test_dataloaders)

    results = []
    seen_classes = np.empty(0, dtype=int)
    for task in range(args.n_tasks):
        cur_classes = np.asarray(np.unique(test_dataloaders[task].dataset.targets), dtype=int)
        seen_classes = np.append(seen_classes, cur_classes)

        # replace the layer weights with the target model at the split layer
        weight = torch.load(f'logs/{args.exp_name}/fold_{fold}_task_{task}.pt')
        weight = replace_layer_weights(weight, target_weight, args.swap_layers)
        model.load_state_dict(weight)
        model = fabric.to_device(model)
        model.eval()

        result = {'fold': fold, 'model': task}
        for idx, test_loader in enumerate(test_dataloaders[:task+1]):
            cur_classes = np.asarray(np.unique(test_loader.dataset.targets), dtype=int)
            seen_classes = np.append(seen_classes, cur_classes)
            
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
                
            ci_acc_metric.reset()
            ci_auc_metric.reset()
            ti_acc_metric.reset()
            ti_auc_metric.reset()

        print(result)
        results.append(result)

    return results

def test_two_models_on_one_fold(args, fold=0):
    fabric = Fabric(devices=1, accelerator="auto")
    tgt_model = load_model(args)
    tgt_weight = torch.load(args.target_weight)
    tgt_model.load_state_dict(tgt_weight)
    tgt_model = fabric.to_device(tgt_model)
    tgt_model.eval()

    test_dataloaders = [datamodule_gen(args, fold=fold, task=task)['test_loader'] for task in range(args.n_tasks)]
    test_dataloaders = fabric.setup_dataloaders(*test_dataloaders)

    results = []
    seen_classes = np.empty(0, dtype=int)
    for task in range(args.n_tasks):
        cur_classes = np.asarray(np.unique(test_dataloaders[task].dataset.targets), dtype=int)
        seen_classes = np.append(seen_classes, cur_classes)

        # replace the layer weights with the target model at the split layer
        src_model = load_model(args)
        weight = torch.load(f'logs/{args.exp_name}/fold_{fold}_task_{task}.pt')
        src_model.load_state_dict(weight)
        src_model = fabric.to_device(src_model)
        src_model.eval()

        result = {'fold': fold, 'task': task}

        for idx, test_loader in enumerate(test_dataloaders[:task+1]):
            cur_classes = np.asarray(np.unique(test_loader.dataset.targets), dtype=int)
            if len(seen_classes) <= 2:
                src_acc_metric = fabric.to_device(Accuracy(task='binary'))
                tgt_acc_metric = fabric.to_device(Accuracy(task='binary'))
            else:
                src_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=len(seen_classes), average='micro'))
                tgt_acc_metric = fabric.to_device(Accuracy(task='multiclass', num_classes=len(seen_classes), average='micro'))

            for batch_idx, batch in enumerate(test_loader):
                if args.n_batches > 0 and batch_idx > args.n_batches:
                    break

                if args.net in ['clam_sb', 'clam_mb', 'transmil']:
                    with torch.no_grad():
                        src_out = src_model(batch['features'])
                        tgt_out = tgt_model(batch['features'])

                    attn1_loss = F.mse_loss(src_out['attn1'], tgt_out['attn1'])
                    h1_loss = F.mse_loss(src_out['h1'], tgt_out['h1'])
                    hp_loss = F.mse_loss(src_out['hp'], tgt_out['hp'])
                    attn2_loss = F.mse_loss(src_out['attn2'], tgt_out['attn2'])
                    h2_loss = F.mse_loss(src_out['h2'], tgt_out['h2'])
                    z_loss = F.mse_loss(src_out['features'], tgt_out['features'])
                    logits_loss = F.mse_loss(src_out['logits'][:, seen_classes], tgt_out['logits'][:, seen_classes])


                    label = F.one_hot(batch['label'], num_classes=args.n_classes)[:, seen_classes]
                    label = label.argmax(dim=-1)
                    src_logit = src_out['logits'][:, seen_classes]
                    src_y_hat = src_logit.argmax(dim=-1)
                    src_acc_metric.update(src_y_hat, label)
                    tgt_logit = tgt_out['logits'][:, seen_classes]
                    tgt_y_hat = tgt_logit.argmax(dim=-1)
                    tgt_acc_metric.update(tgt_y_hat, label)

            result.update({f'{idx}_src_acc': src_acc_metric.compute().item(),
                           f'{idx}_tgt_acc': tgt_acc_metric.compute().item(),
                           f'{idx}_attn1_loss': attn1_loss.item(),
                            f'{idx}_h1_loss': h1_loss.item(),
                            f'{idx}_hp_loss': hp_loss.item(),
                            f'{idx}_attn2_loss': attn2_loss.item(),
                            f'{idx}_h2_loss': h2_loss.item(),
                            f'{idx}_z_loss': z_loss.item(),
                            f'{idx}_logits_loss': logits_loss.item()})
                
            src_acc_metric.reset()
            tgt_acc_metric.reset()

        print(result)
        results.append(result)

    return results

def test(args):
    seed_everything(args.seed)

    args.target_weight = 'logs/csc_transmil_cl/fold_0_task_2.pt'
    args.swap_layers = ['layer1', 'layer2'] # attention_net, classifiers | layer1, layer2, _fc2
    args.mode = 'swap' # 'swap' or 'both'

    if args.load is None:
        results = []
        # Test models from all tasks
        for fold in range(args.folds_start, args.folds_end):
            if args.mode == 'swap':
                result = test_swap_model_on_one_fold(args, fold=fold)
            elif args.mode == 'both':
                result = test_two_models_on_one_fold(args, fold=fold)
            results.extend(result)
    else:
        fold = max(args.folds_start, 0)
        model_fold = args.load.split('_')[-2]
        assert fold == int(model_fold), f'Fold mismatch: {fold} vs {model_fold}'
        if args.mode == 'swap':
            results = test_swap_model_on_one_fold(args, fold=fold)
        elif args.mode == 'both':
            results = test_two_models_on_one_fold(args, fold=fold)

    # convert to csv
    df = pd.DataFrame(results)
    log_path = f'logs/{args.exp_name}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    target_weight = args.target_weight.split('/')[-1].split('.')[0]
    df.to_csv(f'{log_path}/test_swap_{target_weight}_results.csv', index=False)

if __name__ == '__main__':
    args = init_args()
    test(args)

