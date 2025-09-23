import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets.camelyon import MIL_Dataset_Plus
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import os
import random

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    # weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) if len(dataset.slide_cls_ids[c]) > 0 else 1 for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)

def mil_collate_fn(batch):
    # concat each item in the batch
    collate_batch = {}
    item = batch[0]
    for key in item.keys():
        if isinstance(item[key], str):
            collate_batch[key] = [d[key] for d in batch]
        elif isinstance(item[key], (int, np.int64)) or (isinstance(item[key], torch.Tensor) and item[key].dim() == 0):
            collate_batch[key] = torch.LongTensor([d[key] for d in batch])
        else:
            collate_batch[key] = torch.cat([torch.as_tensor(d[key]) for d in batch], dim=0)
    return collate_batch

def get_mil_loader(split_dataset, training=False, weighted=False, batch_size=1):
    """
        return either the validation loader or training loader
    """
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(1)
    kwargs = {'num_workers': 4, 'generator': g, 'collate_fn': mil_collate_fn, 'worker_init_fn': seed_worker, 'batch_size': batch_size}
    # kwargs = {'num_workers': 4, 'collate_fn': mil_collate_fn, 'batch_size': batch_size}
    if training:
        if weighted:
            try:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, sampler=WeightedRandomSampler(weights, len(weights)), **kwargs)
            except Exception as e:
                print("Meet exception when using weighted sampler, use normal sampler instead: ", e)
                loader = DataLoader(split_dataset, shuffle=True, **kwargs)
        else:
            loader = DataLoader(split_dataset, shuffle=True, **kwargs)
    else:
        loader = DataLoader(split_dataset, **kwargs)

    return loader

def datamodule_gen(args, **kwargs):
    if args.dataset in ['camelyon16', 'camelyon17', 'camelyon', 'lung', 'rcc']:
        dataset = MIL_Dataset_Plus(csv_path = os.path.join(args.data_root, args.csv_file),
                            data_dir= os.path.join(args.data_root, args.feat_dir),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = args.label_dict,
                            patient_strat=False,
                            ignore=[])
        fold = kwargs.get('fold', 0)
        datasets = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, fold))
        train_bag, val_bag, test_bag = datasets['train'], datasets['val'], datasets['test']
        train_loader = get_mil_loader(train_bag, training=True, batch_size=args.batch_size, weighted=args.weighted_sample)
        val_loader = get_mil_loader(val_bag, training=False, batch_size=args.batch_size)
        test_loader = get_mil_loader(test_bag, training=False, batch_size=args.batch_size)

    # continual learning
    elif (args.dataset in ['camelyon_cl', 'camelyon_tcga_cl', 'c16_lung_rcc_cl']) and (not hasattr(args, 'cl_method') or args.cl_method != 'joint'):
        dataset = MIL_Dataset_Plus(csv_path = os.path.join(args.data_root, args.csv_file),
                            data_dir= os.path.join(args.data_root, args.feat_dir),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = args.label_dict,
                            patient_strat=False,
                            ignore=[])
        fold = kwargs.get('fold', 0)
        task = kwargs.get('task', 0)
        datasets = dataset.return_splits(from_id=False, csv_path='{}/fold_{}/splits_{}.csv'.format(args.split_dir, fold, task))
        train_bag, val_bag, test_bag = datasets['train'], datasets['val'], datasets['test']
        train_loader = get_mil_loader(train_bag, training=True, batch_size=args.batch_size, weighted=args.weighted_sample)
        val_loader = get_mil_loader(val_bag, training=False, batch_size=args.batch_size)
        test_loader = get_mil_loader(test_bag, training=False, batch_size=args.batch_size)

    # joint learning
    elif (args.dataset in ['camelyon_cl', 'camelyon_tcga_cl', 'c16_lung_rcc_cl']) and  args.cl_method == 'joint':
        from datasets.camelyon import ConcatDataset_MIL
        dataset = MIL_Dataset_Plus(csv_path = os.path.join(args.data_root, args.csv_file),
                                   data_dir= os.path.join(args.data_root, args.feat_dir),
                                   shuffle = False,
                                   seed = args.seed,
                                   print_info = True,
                                   label_dict = args.label_dict,
                                   patient_strat=False,
                                   ignore=[])
        fold = kwargs.get('fold', 0)
        task = kwargs.get('task', 0)
        train_bag = []
        val_bag = []
        test_bag = []
        for task_idx in range(task):
            datasets = dataset.return_splits(from_id=False, csv_path='{}/fold_{}/splits_{}.csv'.format(args.split_dir, fold, task_idx))
            train_bag_sub, val_bag_sub, test_bag_sub = datasets['train'], datasets['val'], datasets['test']
            train_bag.append(train_bag_sub)
            val_bag.append(val_bag_sub)
            test_bag.append(test_bag_sub)

        train_bag = ConcatDataset_MIL(train_bag)
        val_bag = ConcatDataset_MIL(val_bag)
        train_loader = get_mil_loader(train_bag, training=True, batch_size=args.batch_size, weighted=args.weighted_sample)
        val_loader = get_mil_loader(val_bag, training=False, batch_size=args.batch_size)
        test_loader = [get_mil_loader(test_bag_sub, batch_size=args.batch_size) for test_bag_sub in test_bag]

    elif args.dataset == 'csc_cl' and args.cl_method == 'joint':
        from datasets.csc import MILDataset
        import pandas as pd

        file_train = 'joint/' + '2_4_0_5_3_1_train'
        file_val = 'joint/' + '2_4_0_5_3_1_val'
        if args.n_tasks == 2:
            files_test = ['2_4_0_test', '5_3_1_test']
            files_test = ["sce_E2/" + sets for sets in files_test]
        elif args.n_tasks == 3:
            files_test = ['2_4_test', '0_5_test', '3_1_test']
            files_test = ["sce_E3/" + sets for sets in files_test]
        elif args.n_tasks == 5:
            files_test = ['2_4_test', '0_test', '5_test', '3_test', '1_test']
            files_test = ["sce_E5/" + sets for sets in files_test]

        df  = pd.read_csv(os.path.join(args.data_root, 'csv', file_train + '.csv'), dtype=str, delimiter=',')
        train_bag = MILDataset(data_frame=df, data_root=args.data_root)
        df = pd.read_csv(os.path.join(args.data_root, 'csv', file_val + '.csv'), dtype=str, delimiter=',')
        val_bag = MILDataset(data_frame=df, data_root=args.data_root)
        test_bag = []
        for task_idx in range(args.n_tasks):
            df = pd.read_csv(os.path.join(args.data_root, 'csv', files_test[task_idx] + '.csv'), dtype=str, delimiter=',')
            test_bag.append(MILDataset(data_frame=df, data_root=args.data_root))
        train_loader = get_mil_loader(train_bag, training=True, batch_size=args.batch_size)
        val_loader = get_mil_loader(val_bag, training=False, batch_size=args.batch_size)
        test_loader = [get_mil_loader(test_bag_sub, batch_size=args.batch_size) for test_bag_sub in test_bag]

    elif args.dataset == 'csc_cl':
        from datasets.csc import MILDataset
        import pandas as pd

        if args.n_tasks == 2:
            files_train = ['2_4_0_train', '5_3_1_train']
            files_valid = ['2_4_0_val', '5_3_1_val']
            files_test = ['2_4_0_test', '5_3_1_test']
            files_train = ["sce_E2/" + sets for sets in files_train]
            files_valid = ["sce_E2/" + sets for sets in files_valid]
            files_test = ["sce_E2/" + sets for sets in files_test]
        elif args.n_tasks == 3:
            files_train = ['2_4_train', '0_5_train', '3_1_train']
            files_valid = ['2_4_val', '0_5_val', '3_1_val']
            files_test = ['2_4_test', '0_5_test', '3_1_test']
            files_train = ["sce_E3/" + sets for sets in files_train]
            files_valid = ["sce_E3/" + sets for sets in files_valid]
            files_test = ["sce_E3/" + sets for sets in files_test]
        elif args.n_tasks == 5:
            files_train = ['2_4_train', '0_train', '5_train', '3_train', '1_train']
            files_valid = ['2_4_val', '0_val', '5_val', '3_val', '1_val']
            files_test = ['2_4_test', '0_test', '5_test', '3_test', '1_test']
            files_train = ["sce_E5/" + sets for sets in files_train]
            files_valid = ["sce_E5/" + sets for sets in files_valid]
            files_test = ["sce_E5/" + sets for sets in files_test]

        
        fold = kwargs.get('fold', 0)
        task = kwargs.get('task', 0)
        df = pd.read_csv(os.path.join(args.data_root, 'csv', files_train[task] + '.csv'), dtype=str, delimiter=',')
        train_bag = MILDataset(data_frame=df, data_root=args.data_root)
        df = pd.read_csv(os.path.join(args.data_root, 'csv', files_valid[task] + '.csv'), dtype=str, delimiter=',')
        val_bag = MILDataset(data_frame=df, data_root=args.data_root)
        df = pd.read_csv(os.path.join(args.data_root, 'csv', files_test[task] + '.csv'), dtype=str, delimiter=',')
        test_bag = MILDataset(data_frame=df, data_root=args.data_root)
        train_loader = get_mil_loader(train_bag, training=True, batch_size=args.batch_size)
        val_loader = get_mil_loader(val_bag, training=False, batch_size=args.batch_size)
        test_loader = get_mil_loader(test_bag, training=False, batch_size=args.batch_size)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dm = {'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
          'train_dataset': train_bag, 'val_dataset': val_bag, 'test_dataset': test_bag}

    return dm    
