"""
Multiple Instance Learning - Harder Tasks Collection
支持多种实例选择方法：Attention, KIBO双层优化, Random
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import argparse
import copy


class AttentionMIL(nn.Module):
    """基于Attention的多实例学习网络"""

    def __init__(self, input_dim=10, hidden_dim=128, attention_dim=64):
        super(AttentionMIL, self).__init__()

        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Attention机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [batch_size, num_instances, input_dim]
        """
        batch_size, num_instances, _ = x.shape

        # 提取特征
        x_flat = x.view(-1, x.shape[-1])
        h = self.feature_extractor(x_flat)
        h = h.view(batch_size, num_instances, -1)

        # 计算attention权重
        a = self.attention(h)
        a = torch.softmax(a, dim=1)

        # 加权聚合
        z = torch.sum(a * h, dim=1)

        # 分类
        y_prob = self.classifier(z)

        return y_prob.squeeze(), a.squeeze()


def collate_fn(batch):
    """自定义collate函数处理不同长度的包"""
    bags, labels = zip(*batch)

    # 找到最大长度
    max_len = max([bag.shape[0] for bag in bags])

    # Padding
    padded_bags = []
    for bag in bags:
        pad_size = max_len - bag.shape[0]
        if pad_size > 0:
            padding = torch.zeros(pad_size, bag.shape[1])
            padded_bag = torch.cat([bag, padding], dim=0)
        else:
            padded_bag = bag
        padded_bags.append(padded_bag)

    bags_tensor = torch.stack(padded_bags)
    labels_tensor = torch.stack(list(labels))

    return bags_tensor, labels_tensor


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for bags, labels in dataloader:
        bags = bags.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        y_prob, attention = model(bags)
        loss = criterion(y_prob, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (y_prob > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for bags, labels in dataloader:
            bags = bags.to(device)
            labels = labels.to(device)

            y_prob, attention = model(bags)
            loss = criterion(y_prob, labels)
            total_loss += loss.item()

            preds = (y_prob > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(y_prob.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    return avg_loss, acc, auc, precision, recall, f1


# ============================================================
# KIBO双层优化实例选择
# ============================================================

class BilevelInstanceSelection:
    """
    基于双层优化的实例选择（KIBO方法）

    内层：在加权实例上训练代理模型
    外层：优化实例权重使验证损失最小
    """

    def __init__(self, model, lr_model=0.01, lr_weight=0.1,
                 max_outer_it=20, max_inner_it=3, device='cuda'):
        self.proxy_model = copy.deepcopy(model)
        self.origin_model = copy.deepcopy(model)
        self.lr_model = lr_model
        self.lr_weight = lr_weight
        self.max_outer_it = max_outer_it
        self.max_inner_it = max_inner_it
        self.device = device

    def train_inner(self, data, labels, weights):
        """内层训练：在加权数据上训练代理模型"""
        self.proxy_model.train()
        optimizer = torch.optim.SGD(self.proxy_model.parameters(), lr=self.lr_model)

        total_loss = 0
        for _ in range(self.max_inner_it):
            optimizer.zero_grad()

            # 加权数据
            weighted_data = data * weights.unsqueeze(1)
            y_prob, _ = self.proxy_model(weighted_data.unsqueeze(0))

            # 二分类损失
            # 确保labels是tensor且形状正确
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=self.device)
            labels = labels.view(-1).float()
            loss = F.binary_cross_entropy(y_prob.view(-1), labels)
            loss.backward()
            optimizer.step()
            total_loss = loss.item()

        return total_loss

    def train_outer(self, data, labels, weights, topk):
        """外层训练：优化实例权重"""
        # self.proxy_model.eval()

        # 前向传播计算验证损失
        y_prob, _ = self.proxy_model(data.unsqueeze(0))
        
        y_prob = y_prob.view(-1)
        labels = labels.view(-1)
        
        outer_loss = F.binary_cross_entropy(y_prob, labels.float())

        # 选择top-k权重的正则化
        topk_weights, _ = weights.topk(topk)
        reg_term = -topk_weights.sum()  # 鼓励选择高权重

        total_loss = outer_loss + 1 * reg_term

        # 计算权重梯度（简化版，不使用完整隐式梯度）
        if weights.requires_grad:
            grad = torch.autograd.grad(total_loss, weights, retain_graph=True)[0]
        else:
            grad = torch.zeros_like(weights)

        # 更新权重
        with torch.no_grad():
            weights -= self.lr_weight * grad
            weights = torch.clamp(weights, min=0, max=1)

        return weights, total_loss.item()

    def select_instances(self, data, label, topk):
        """
        对单个包进行实例选择

        Args:
            data: [num_instances, input_dim] 包中的实例
            label: 包标签
            topk: 选择的实例数量

        Returns:
            selected_indices: 选中的实例索引
        """
        data = data.to(self.device)
        label = torch.tensor([label], dtype=torch.float32).to(self.device)

        n = data.shape[0]
        topk = min(topk, n)

        # 初始化实例权重
        weights = torch.rand(n, requires_grad=True, device=self.device)

        # 加载初始模型状态
        self.proxy_model.load_state_dict(self.origin_model.state_dict())

        # 双层优化
        for outer_it in range(self.max_outer_it):
            # 内层：训练代理模型
            inner_loss = self.train_inner(data, label, weights.detach())

            # 外层：优化权重
            weights, outer_loss = self.train_outer(data, label, weights, topk)

        # 选择top-k实例
        _, selected_indices = weights.topk(topk)

        return selected_indices.cpu().numpy()


def extract_instances(model, dataset, device, check_label_fn, selection_method='random',
                      top_k=3, bottom_k=2, bilevel_params=None):
    """
    统一的实例选择接口，支持多种选择方法

    Args:
        model: 训练好的模型
        dataset: 原始数据集
        device: 设备
        check_label_fn: 检查标签的函数
        selection_method: 选择方法 ('attention', 'kibo', 'random')
        top_k: 选择的实例数量（或attention方法的top-k）
        bottom_k: attention方法的bottom-k
        bilevel_params: KIBO方法的参数字典

    Returns:
        new_bags, new_labels, mismatch_stats
    """
    if selection_method == 'attention':
        return extract_attention_based_instances(model, dataset, device, check_label_fn, top_k, bottom_k)
    elif selection_method == 'kibo':
        return extract_kibo_based_instances(model, dataset, device, check_label_fn, top_k, bilevel_params)
    elif selection_method == 'random':
        return extract_random_instances(model, dataset, device, check_label_fn, top_k)
    
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    

def extract_random_instances(model, dataset, device, check_label_fn, num_select=5):
    """随机选择实例（baseline方法）"""
    new_bags = []
    new_labels = []

    total_bags = 0
    mismatch_count = 0
    pos_to_neg = 0
    neg_to_pos = 0

    print(f"\n从{len(dataset)}个包中随机选择实例...")
    print(f"策略: 每个包随机选择{num_select}个实例")

    np.random.seed(42)

    for i in range(len(dataset)):
        bag, label = dataset[i]
        items = dataset.bags[i]

        num_instances = len(items)
        num_select_actual = min(num_select, num_instances)

        # 随机选择
        selected_indices = np.random.choice(num_instances, num_select_actual, replace=False)
        new_bag = [items[idx] for idx in selected_indices]

        new_bags.append(new_bag)
        new_labels.append(label)

        # 检查标签一致性
        original_label = int(label)
        true_label_for_new_bag = 1 if check_label_fn(new_bag) else 0

        total_bags += 1
        if original_label != true_label_for_new_bag:
            mismatch_count += 1
            if original_label == 1 and true_label_for_new_bag == 0:
                pos_to_neg += 1
            elif original_label == 0 and true_label_for_new_bag == 1:
                neg_to_pos += 1

    avg_size = np.mean([len(bag) for bag in new_bags])

    print(f"完成! 创建了{len(new_bags)}个新包")
    print(f"平均包大小: {avg_size:.2f} (原始: {np.mean([len(dataset.bags[i]) for i in range(len(dataset))]):.2f})")
    print(f"\n标签一致性检查:")
    print(f"  总包数: {total_bags}")
    print(f"  标签不一致: {mismatch_count} ({mismatch_count/total_bags*100:.2f}%)")
    print(f"    - 正包→负包: {pos_to_neg} ({pos_to_neg/total_bags*100:.2f}%)")
    print(f"    - 负包→正包: {neg_to_pos} ({neg_to_pos/total_bags*100:.2f}%)")
    print(f"  标签一致: {total_bags - mismatch_count} ({(total_bags-mismatch_count)/total_bags*100:.2f}%)")

    mismatch_stats = {
        'total': total_bags,
        'mismatch': mismatch_count,
        'pos_to_neg': pos_to_neg,
        'neg_to_pos': neg_to_pos,
        'mismatch_rate': mismatch_count / total_bags
    }
    exit(0)
    return new_bags, new_labels, mismatch_stats


def extract_kibo_based_instances(model, dataset, device, check_label_fn, topk=5, bilevel_params=None):
    """基于KIBO双层优化的实例选择"""
    if bilevel_params is None:
        bilevel_params = {'max_outer_it': 10, 'max_inner_it': 2, 'lr_weight': 0.1}

    new_bags = []
    new_labels = []

    total_bags = 0
    mismatch_count = 0
    pos_to_neg = 0
    neg_to_pos = 0

    print(f"\n从{len(dataset)}个包中使用KIBO双层优化选择实例...")
    print(f"策略: 每个包选择top-{topk}个实例")
    print(f"参数: outer_it={bilevel_params.get('max_outer_it', 10)}, "
          f"inner_it={bilevel_params.get('max_inner_it', 2)}")

    # 创建双层优化选择器
    selector = BilevelInstanceSelection(
        model,
        max_outer_it=bilevel_params.get('max_outer_it', 10),
        max_inner_it=bilevel_params.get('max_inner_it', 2),
        lr_weight=bilevel_params.get('lr_weight', 0.1),
        device=device
    )

    for i in range(len(dataset)):
        bag, label = dataset[i]
        items = dataset.bags[i]

        # 使用KIBO选择实例
        selected_indices = selector.select_instances(bag, label, topk)
        new_bag = [items[idx] for idx in selected_indices]

        new_bags.append(new_bag)
        new_labels.append(label)

        # 检查标签一致性
        original_label = int(label)
        true_label_for_new_bag = 1 if check_label_fn(new_bag) else 0

        total_bags += 1
        if original_label != true_label_for_new_bag:
            mismatch_count += 1
            if original_label == 1 and true_label_for_new_bag == 0:
                pos_to_neg += 1
            elif original_label == 0 and true_label_for_new_bag == 1:
                neg_to_pos += 1

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i+1}/{len(dataset)} 个包...")

    avg_size = np.mean([len(bag) for bag in new_bags])

    print(f"完成! 创建了{len(new_bags)}个新包")
    print(f"平均包大小: {avg_size:.2f} (原始: {np.mean([len(dataset.bags[i]) for i in range(len(dataset))]):.2f})")
    print(f"\n标签一致性检查:")
    print(f"  总包数: {total_bags}")
    print(f"  标签不一致: {mismatch_count} ({mismatch_count/total_bags*100:.2f}%)")
    print(f"    - 正包→负包: {pos_to_neg} ({pos_to_neg/total_bags*100:.2f}%)")
    print(f"    - 负包→正包: {neg_to_pos} ({neg_to_pos/total_bags*100:.2f}%)")
    print(f"  标签一致: {total_bags - mismatch_count} ({(total_bags-mismatch_count)/total_bags*100:.2f}%)")

    mismatch_stats = {
        'total': total_bags,
        'mismatch': mismatch_count,
        'pos_to_neg': pos_to_neg,
        'neg_to_pos': neg_to_pos,
        'mismatch_rate': mismatch_count / total_bags
    }
    
    return new_bags, new_labels, mismatch_stats


def extract_attention_based_instances(model, dataset, device, check_label_fn, top_k=3, bottom_k=2):
    """
    从训练集中提取基于attention的实例，构建新的包

    Args:
        model: 训练好的模型
        dataset: 原始数据集
        device: 设备
        check_label_fn: 检查标签的函数
        top_k: 每个包选择top-k个最高attention的实例
        bottom_k: 每个包选择bottom-k个最低attention的实例

    Returns:
        new_bags: 新构建的包列表
        new_labels: 对应的标签列表
        mismatch_stats: 标签不一致统计信息
    """
    model.eval()
    new_bags = []
    new_labels = []

    total_bags = 0
    mismatch_count = 0
    pos_to_neg = 0
    neg_to_pos = 0

    print(f"\n从{len(dataset)}个包中提取attention-based实例...")
    print(f"策略: 每个包选择top-{top_k}和bottom-{bottom_k}个实例")

    with torch.no_grad():
        for i in range(len(dataset)):
            bag, label = dataset[i]
            bag_tensor = bag.unsqueeze(0).to(device)

            _, attention = model(bag_tensor)

            if attention.dim() == 0:
                attention = attention.unsqueeze(0)

            attention_weights = attention.cpu().numpy()
            items = dataset.bags[i]

            num_instances = len(items)
            attention_weights = attention_weights[:num_instances]

            sorted_indices = np.argsort(attention_weights)[::-1]

            selected_indices = []
            selected_indices.extend(sorted_indices[:min(top_k, len(sorted_indices))])
            if bottom_k > 0:
                selected_indices.extend(sorted_indices[-min(bottom_k, len(sorted_indices)):])

            selected_indices = list(set(selected_indices))
            new_bag = [items[idx] for idx in selected_indices]

            new_bags.append(new_bag)
            new_labels.append(label)

            # 检查标签一致性
            original_label = int(label)
            true_label_for_new_bag = 1 if check_label_fn(new_bag) else 0

            total_bags += 1
            if original_label != true_label_for_new_bag:
                mismatch_count += 1
                if original_label == 1 and true_label_for_new_bag == 0:
                    pos_to_neg += 1
                elif original_label == 0 and true_label_for_new_bag == 1:
                    neg_to_pos += 1

    avg_size = np.mean([len(bag) for bag in new_bags])

    print(f"完成! 创建了{len(new_bags)}个新包")
    print(f"平均包大小: {avg_size:.2f} (原始: {np.mean([len(dataset.bags[i]) for i in range(len(dataset))]):.2f})")
    print(f"\n标签一致性检查:")
    print(f"  总包数: {total_bags}")
    print(f"  标签不一致: {mismatch_count} ({mismatch_count/total_bags*100:.2f}%)")
    print(f"    - 正包→负包: {pos_to_neg} ({pos_to_neg/total_bags*100:.2f}%)")
    print(f"    - 负包→正包: {neg_to_pos} ({neg_to_pos/total_bags*100:.2f}%)")
    print(f"  标签一致: {total_bags - mismatch_count} ({(total_bags-mismatch_count)/total_bags*100:.2f}%)")

    mismatch_stats = {
        'total': total_bags,
        'mismatch': mismatch_count,
        'pos_to_neg': pos_to_neg,
        'neg_to_pos': neg_to_pos,
        'mismatch_rate': mismatch_count / total_bags
    }

    return new_bags, new_labels, mismatch_stats


class FilteredDataset(Dataset):
    """基于attention过滤后的数据集"""

    def __init__(self, bags, labels, encoding_fn):
        """
        Args:
            bags: 包列表
            labels: 标签列表
            encoding_fn: 编码函数
        """
        self.bags = bags
        self.labels = labels
        self.encoding_fn = encoding_fn

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        label = self.labels[idx]
        bag_tensor = self.encoding_fn(bag)
        return bag_tensor, torch.tensor(label, dtype=torch.float32)


# ============================================================
# 任务1: 扩展数字范围 (0-99)
# ============================================================

class LargeRangeDataset(Dataset):
    """大范围数字的连续对检测数据集 (0-99)

    正包: 存在连续数字对 (如 23,24 或 67,68)
    负包: 不存在连续数字对
    """

    def __init__(self, num_bags=1000, bag_size_range=(5, 20), pos_ratio=0.5, seed=42):
        self.num_bags = num_bags
        self.bag_size_range = bag_size_range
        self.bags = []
        self.labels = []

        np.random.seed(seed)

        for i in range(num_bags):
            bag_size = np.random.randint(bag_size_range[0], bag_size_range[1] + 1)
            is_positive = np.random.rand() < pos_ratio

            if is_positive:
                # 正包: 随机选择一对连续数字
                start = np.random.randint(0, 99)
                bag = [start, start + 1]
                # 填充其他数字
                bag += list(np.random.randint(0, 100, bag_size - 2))
                label = 1
            else:
                # 负包: 只使用偶数或间隔>1的数字
                if np.random.rand() < 0.5:
                    bag = list(np.random.choice(range(0, 100, 2), bag_size))
                else:
                    bag = list(np.random.choice(range(0, 100, 3), bag_size))
                label = 0

            np.random.shuffle(bag)
            self.bags.append(bag)
            self.labels.append(label)

    def __len__(self):
        return self.num_bags

    def __getitem__(self, idx):
        bag = self.bags[idx]
        label = self.labels[idx]

        # One-hot编码 (100维)
        bag_tensor = torch.zeros(len(bag), 100)
        for i, num in enumerate(bag):
            bag_tensor[i, num] = 1.0

        return bag_tensor, torch.tensor(label, dtype=torch.float32)


def check_large_range_consecutive(bag):
    """检查是否存在连续数字对"""
    nums = set(bag)
    for n in nums:
        if (n + 1) in nums:
            return True
    return False


def encode_large_range(bag):
    """编码函数"""
    bag_tensor = torch.zeros(len(bag), 100)
    for i, num in enumerate(bag):
        bag_tensor[i, num] = 1.0
    return bag_tensor


# ============================================================
# 任务2: 三元组检测 (0-9)
# ============================================================

class TripletDataset(Dataset):
    """连续三元组检测数据集

    正包: 存在连续三元组 (如 3,4,5 或 7,8,9)
    负包: 不存在连续三元组
    """

    def __init__(self, num_bags=1000, bag_size_range=(8, 25), pos_ratio=0.5, seed=42):
        self.num_bags = num_bags
        self.bag_size_range = bag_size_range
        self.bags = []
        self.labels = []

        np.random.seed(seed)

        for i in range(num_bags):
            bag_size = np.random.randint(bag_size_range[0], bag_size_range[1] + 1)
            is_positive = np.random.rand() < pos_ratio

            if is_positive:
                # 正包: 确保存在连续三元组
                start = np.random.randint(0, 8)  # 0-7, 这样start+2最多是9
                bag = [start, start + 1, start + 2]
                bag += list(np.random.randint(0, 10, bag_size - 3))
                label = 1
            else:
                # 负包: 使用间隔数字，避免三连续
                if np.random.rand() < 0.5:
                    # 只用{0, 2, 4, 6, 8}
                    bag = list(np.random.choice([0, 2, 4, 6, 8], bag_size))
                else:
                    # 只用{0, 3, 6, 9}
                    bag = list(np.random.choice([0, 3, 6, 9], bag_size))
                label = 0

            np.random.shuffle(bag)
            self.bags.append(bag)
            self.labels.append(label)

    def __len__(self):
        return self.num_bags

    def __getitem__(self, idx):
        bag = self.bags[idx]
        label = self.labels[idx]

        bag_tensor = torch.zeros(len(bag), 10)
        for i, digit in enumerate(bag):
            bag_tensor[i, digit] = 1.0

        return bag_tensor, torch.tensor(label, dtype=torch.float32)


def check_triplet(bag):
    """检查是否存在连续三元组"""
    nums = sorted(set(bag))
    for i in range(len(nums) - 2):
        if nums[i+1] == nums[i] + 1 and nums[i+2] == nums[i] + 2:
            return True
    return False


def encode_triplet(bag):
    """编码函数"""
    bag_tensor = torch.zeros(len(bag), 10)
    for i, digit in enumerate(bag):
        bag_tensor[i, digit] = 1.0
    return bag_tensor


# ============================================================
# 任务3: 和为特定值 (0-9)
# ============================================================

class SumToTargetDataset(Dataset):
    """和为目标值检测数据集

    正包: 存在两个数字和为10 (如 3+7, 4+6, 1+9)
    负包: 不存在这样的数字对
    """

    def __init__(self, num_bags=1000, bag_size_range=(5, 20), pos_ratio=0.5, target_sum=10, seed=42):
        self.num_bags = num_bags
        self.bag_size_range = bag_size_range
        self.target_sum = target_sum
        self.bags = []
        self.labels = []

        np.random.seed(seed)

        for i in range(num_bags):
            bag_size = np.random.randint(bag_size_range[0], bag_size_range[1] + 1)
            is_positive = np.random.rand() < pos_ratio

            if is_positive:
                # 正包: 确保存在和为target_sum的数字对
                # 例如 target_sum=10: (1,9), (2,8), (3,7), (4,6), (5,5)
                pairs = [(a, target_sum - a) for a in range(target_sum + 1) if 0 <= target_sum - a <= 9]
                pair = pairs[np.random.randint(0, len(pairs))]
                bag = [pair[0], pair[1]]
                bag += list(np.random.randint(0, 10, bag_size - 2))
                label = 1
            else:
                # 负包: 只使用不能组成target_sum的数字
                # 例如 target_sum=10: 只用0,1,2,3,4 (最大和为8)
                if np.random.rand() < 0.5:
                    bag = list(np.random.randint(0, min(5, target_sum // 2 + 1), bag_size))
                else:
                    # 或者只用大数字
                    bag = list(np.random.randint(max(6, target_sum - 4), 10, bag_size))
                label = 0

            np.random.shuffle(bag)
            self.bags.append(bag)
            self.labels.append(label)

    def __len__(self):
        return self.num_bags

    def __getitem__(self, idx):
        bag = self.bags[idx]
        label = self.labels[idx]

        bag_tensor = torch.zeros(len(bag), 10)
        for i, digit in enumerate(bag):
            bag_tensor[i, digit] = 1.0

        return bag_tensor, torch.tensor(label, dtype=torch.float32)


def check_sum_to_target(bag, target_sum=10):
    """检查是否存在和为target_sum的数字对"""
    nums = set(bag)
    for n in nums:
        if (target_sum - n) in nums:
            return True
    return False


def encode_sum(bag):
    """编码函数"""
    bag_tensor = torch.zeros(len(bag), 10)
    for i, digit in enumerate(bag):
        bag_tensor[i, digit] = 1.0
    return bag_tensor


# ============================================================
# 任务4: 递增序列检测 (0-9)
# ============================================================

class IncreasingSequenceDataset(Dataset):
    """递增序列检测数据集

    正包: 存在至少3个数字的严格递增序列 (如 1,3,7 或 2,5,8,9)
    负包: 不存在这样的递增序列
    """

    def __init__(self, num_bags=1000, bag_size_range=(5, 20), pos_ratio=0.5, min_seq_len=3, seed=42):
        self.num_bags = num_bags
        self.bag_size_range = bag_size_range
        self.min_seq_len = min_seq_len
        self.bags = []
        self.labels = []

        np.random.seed(seed)

        for i in range(num_bags):
            bag_size = np.random.randint(bag_size_range[0], bag_size_range[1] + 1)
            is_positive = np.random.rand() < pos_ratio

            if is_positive:
                # 正包: 构建递增序列
                seq_len = np.random.randint(min_seq_len, min(min_seq_len + 3, 8))
                seq = sorted(np.random.choice(10, seq_len, replace=False))
                bag = list(seq)
                bag += list(np.random.randint(0, 10, bag_size - seq_len))
                label = 1
            else:
                # 负包: 只使用2个不同的数字，或者使用递减
                if np.random.rand() < 0.5:
                    # 只用两个数字
                    nums = np.random.choice(10, 2, replace=False)
                    bag = list(np.random.choice(nums, bag_size))
                else:
                    # 递减序列
                    bag = list(np.random.randint(0, 10, bag_size))
                label = 0

            np.random.shuffle(bag)
            self.bags.append(bag)
            self.labels.append(label)

    def __len__(self):
        return self.num_bags

    def __getitem__(self, idx):
        bag = self.bags[idx]
        label = self.labels[idx]

        bag_tensor = torch.zeros(len(bag), 10)
        for i, digit in enumerate(bag):
            bag_tensor[i, digit] = 1.0

        return bag_tensor, torch.tensor(label, dtype=torch.float32)


def check_increasing_sequence(bag, min_len=3):
    """检查是否存在长度>=min_len的递增序列"""
    unique_sorted = sorted(set(bag))
    if len(unique_sorted) < min_len:
        return False

    # 检查是否有min_len个数字都在bag中
    for i in range(len(unique_sorted) - min_len + 1):
        subseq = unique_sorted[i:i+min_len]
        # 检查是否严格递增
        is_increasing = all(subseq[j+1] > subseq[j] for j in range(len(subseq)-1))
        if is_increasing:
            return True
    return False


def encode_increasing(bag):
    """编码函数"""
    bag_tensor = torch.zeros(len(bag), 10)
    for i, digit in enumerate(bag):
        bag_tensor[i, digit] = 1.0
    return bag_tensor


# ============================================================
# 主训练流程
# ============================================================

def run_experiment(task_name, dataset_class, check_fn, encode_fn, input_dim,
                   num_bags_train=2000, num_bags_val=500, num_bags_test=500,
                   num_epochs_stage1=50, num_epochs_stage2=50,
                   top_k=3, bottom_k=2, selection_method='attention',
                   bilevel_params=None, **dataset_kwargs):
    """
    运行两阶段MIL实验

    Args:
        task_name: 任务名称
        dataset_class: 数据集类
        check_fn: 检查标签的函数
        encode_fn: 编码函数
        input_dim: 输入维度
        num_bags_train: 训练集包数量
        num_bags_val: 验证集包数量
        num_bags_test: 测试集包数量
        num_epochs_stage1: 第一阶段训练轮数
        num_epochs_stage2: 第二阶段训练轮数
        top_k: 选择top-k个最高attention的实例
        bottom_k: 选择bottom-k个最低attention的实例
        selection_method: 实例选择方法 ('attention', 'kibo', 'random')
        bilevel_params: KIBO方法的参数字典
        **dataset_kwargs: 传递给数据集的额外参数
    """
    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = 32
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print(f"任务: {task_name}")
    print(f"两阶段训练: 原始数据 -> {selection_method.upper()}过滤 -> 重新训练")
    print("="*80)
    print(f"设备: {device}")
    print(f"输入维度: {input_dim}")
    print(f"包大小范围: {dataset_kwargs.get('bag_size_range', 'N/A')}")
    print(f"第一阶段训练轮数: {num_epochs_stage1}")
    print(f"第二阶段训练轮数: {num_epochs_stage2}")
    print(f"实例选择方法: {selection_method}")
    if selection_method == 'attention':
        print(f"选择策略: top-{top_k} + bottom-{bottom_k}")
    else:
        print(f"选择数量: {top_k}")

    # ========== 第一阶段: 在原始数据上训练 ==========
    print("\n" + "="*80)
    print("第一阶段: 在原始数据上训练")
    print("="*80)

    print("\n创建原始数据集...")
    train_dataset = dataset_class(num_bags=num_bags_train, seed=42, **dataset_kwargs)
    val_dataset = dataset_class(num_bags=num_bags_val, seed=43, **dataset_kwargs)
    test_dataset = dataset_class(num_bags=num_bags_test, seed=44, **dataset_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"训练集: {len(train_dataset)} 个包")
    print(f"验证集: {len(val_dataset)} 个包")
    print(f"测试集: {len(test_dataset)} 个包")

    # 创建模型
    model_stage1 = AttentionMIL(input_dim=input_dim, hidden_dim=128, attention_dim=64).to(device)
    criterion = nn.BCELoss()
    optimizer_stage1 = optim.Adam(model_stage1.parameters(), lr=learning_rate)

    # 训练第一阶段
    print("\n开始第一阶段训练...")
    best_val_auc_stage1 = 0
    best_model_state_stage1 = None

    for epoch in range(num_epochs_stage1):
        train_loss, train_acc = train_epoch(model_stage1, train_loader, criterion, optimizer_stage1, device)
        val_loss, val_acc, val_auc, val_precision, val_recall, val_f1 = evaluate(model_stage1, val_loader, criterion, device)

        if val_auc > best_val_auc_stage1:
            best_val_auc_stage1 = val_auc
            best_model_state_stage1 = model_stage1.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs_stage1}]")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

    model_stage1.load_state_dict(best_model_state_stage1)

    # 第一阶段测试
    print("\n" + "="*80)
    print("第一阶段测试集评估:")
    print("="*80)
    test_loss_s1, test_acc_s1, test_auc_s1, test_precision_s1, test_recall_s1, test_f1_s1 = evaluate(
        model_stage1, test_loader, criterion, device
    )
    print(f"Loss: {test_loss_s1:.4f}")
    print(f"Accuracy: {test_acc_s1:.4f}")
    print(f"AUC: {test_auc_s1:.4f}")
    print(f"Precision: {test_precision_s1:.4f}")
    print(f"Recall: {test_recall_s1:.4f}")
    print(f"F1 Score: {test_f1_s1:.4f}")

    # ========== 第二阶段: 基于Attention过滤并重新训练 ==========
    print("\n" + "="*80)
    print("第二阶段: 基于Attention过滤数据并重新训练")
    print("="*80)

    # 提取实例（使用指定的选择方法）
    filtered_train_bags, filtered_train_labels, train_mismatch_stats = extract_instances(
        model_stage1, train_dataset, device, check_fn,
        selection_method=selection_method, top_k=top_k, bottom_k=bottom_k,
        bilevel_params=bilevel_params
    )
    
    exit(0)

    
    filtered_val_bags, filtered_val_labels, val_mismatch_stats = extract_instances(
        model_stage1, val_dataset, device, check_fn,
        selection_method=selection_method, top_k=top_k, bottom_k=bottom_k,
        bilevel_params=bilevel_params
    )
    
    
    # 创建新数据集
    filtered_train_dataset = FilteredDataset(filtered_train_bags, filtered_train_labels, encode_fn)
    filtered_val_dataset = FilteredDataset(filtered_val_bags, filtered_val_labels, encode_fn)

    filtered_train_loader = DataLoader(filtered_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    filtered_val_loader = DataLoader(filtered_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 创建新模型（重新初始化）
    model_stage2 = AttentionMIL(input_dim=input_dim, hidden_dim=128, attention_dim=64).to(device)
    optimizer_stage2 = optim.Adam(model_stage2.parameters(), lr=learning_rate)

    # 训练第二阶段
    print("\n开始第二阶段训练...")
    best_val_auc_stage2 = 0
    best_model_state_stage2 = None

    for epoch in range(num_epochs_stage2):
        train_loss, train_acc = train_epoch(model_stage2, filtered_train_loader, criterion, optimizer_stage2, device)
        val_loss, val_acc, val_auc, val_precision, val_recall, val_f1 = evaluate(model_stage2, filtered_val_loader, criterion, device)

        if val_auc > best_val_auc_stage2:
            best_val_auc_stage2 = val_auc
            best_model_state_stage2 = model_stage2.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs_stage2}]")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

    model_stage2.load_state_dict(best_model_state_stage2)

    # 第二阶段测试（使用原始测试集）
    print("\n" + "="*80)
    print("第二阶段测试集评估 (在原始测试集上):")
    print("="*80)
    test_loss_s2, test_acc_s2, test_auc_s2, test_precision_s2, test_recall_s2, test_f1_s2 = evaluate(
        model_stage2, test_loader, criterion, device
    )
    print(f"Loss: {test_loss_s2:.4f}")
    print(f"Accuracy: {test_acc_s2:.4f}")
    print(f"AUC: {test_auc_s2:.4f}")
    print(f"Precision: {test_precision_s2:.4f}")
    print(f"Recall: {test_recall_s2:.4f}")
    print(f"F1 Score: {test_f1_s2:.4f}")

    # ========== 结果对比 ==========
    print("\n" + "="*80)
    print("结果对比:")
    print("="*80)
    print(f"第一阶段 (原始数据):")
    print(f"  Accuracy: {test_acc_s1:.4f}, AUC: {test_auc_s1:.4f}, F1: {test_f1_s1:.4f}")
    print(f"第二阶段 (Attention过滤):")
    print(f"  Accuracy: {test_acc_s2:.4f}, AUC: {test_auc_s2:.4f}, F1: {test_f1_s2:.4f}")
    print(f"性能变化:")
    print(f"  Accuracy: {test_acc_s2 - test_acc_s1:+.4f}")
    print(f"  AUC: {test_auc_s2 - test_auc_s1:+.4f}")
    print(f"  F1: {test_f1_s2 - test_f1_s1:+.4f}")

    print("\n" + "="*80)
    print("标签不一致总结:")
    print("="*80)
    print(f"训练集: {train_mismatch_stats['mismatch']}/{train_mismatch_stats['total']} "
          f"({train_mismatch_stats['mismatch_rate']*100:.2f}%) 包标签不一致")
    print(f"  正包→负包: {train_mismatch_stats['pos_to_neg']}")
    print(f"  负包→正包: {train_mismatch_stats['neg_to_pos']}")
    print(f"验证集: {val_mismatch_stats['mismatch']}/{val_mismatch_stats['total']} "
          f"({val_mismatch_stats['mismatch_rate']*100:.2f}%) 包标签不一致")
    print(f"  正包→负包: {val_mismatch_stats['pos_to_neg']}")
    print(f"  负包→正包: {val_mismatch_stats['neg_to_pos']}")

    print("\n" + "="*80)
    print(f"任务 {task_name} 完成!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='MIL - Harder Tasks with Multiple Selection Methods')
    parser.add_argument('--task', type=str, default='triplet',
                        choices=['large_range', 'triplet', 'sum', 'increasing', 'all'],
                        help='选择要运行的任务')
    parser.add_argument('--selection_method', type=str, default='attention',
                        choices=['attention', 'kibo', 'random', 'compare'],
                        help='实例选择方法: attention(基于注意力), kibo(双层优化), random(随机), compare(对比所有方法)')
    parser.add_argument('--epochs1', type=int, default=10, help='第一阶段训练轮数')
    parser.add_argument('--epochs2', type=int, default=10, help='第二阶段训练轮数')
    parser.add_argument('--new_size', type=int, default=10, help='新包的大小(选择的实例数量)')
    parser.add_argument('--num_bags', type=int, default=1000, help='训练集包数量')
    parser.add_argument('--kibo_outer_it', type=int, default=10, help='KIBO外层迭代次数')
    parser.add_argument('--kibo_inner_it', type=int, default=2, help='KIBO内层迭代次数')
    parser.add_argument('--kibo_lr_weight', type=float, default=0.1, help='KIBO权重学习率')
    parser.add_argument('--bag_size_min', type=int, default=20, help='包大小的最小值')
    parser.add_argument('--bag_size_max', type=int, default=50, help='包大小的最大值')

    args = parser.parse_args()

    # 根据 new_size 自动计算 top_k 和 bottom_k
    # attention 方法: 对半分，top_k 取一半，bottom_k 取另一半
    # 其他方法: top_k = new_size, bottom_k = 0
    top_k = args.new_size // 2
    bottom_k = args.new_size - top_k  # 确保总数正好是 new_size

    print(f"\n{'='*80}")
    print(f"实例选择配置:")
    print(f"  新包大小 (new_size): {args.new_size}")
    print(f"  Attention方法 -> top_k: {top_k}, bottom_k: {bottom_k}")
    print(f"  其他方法 (KIBO/Random) -> 选择数量: {args.new_size}")
    print(f"{'='*80}\n")

    # 构建KIBO参数
    bilevel_params = {
        'max_outer_it': args.kibo_outer_it,
        'max_inner_it': args.kibo_inner_it,
        'lr_weight': args.kibo_lr_weight
    }

    # 包大小范围 (统一由命令行参数控制)
    bag_size_range = (args.bag_size_min, args.bag_size_max)

    tasks = {
        'large_range': {
            'name': '扩展数字范围检测 (0-99连续对)',
            'dataset_class': LargeRangeDataset,
            'check_fn': check_large_range_consecutive,
            'encode_fn': encode_large_range,
            'input_dim': 100,
            'dataset_kwargs': {'bag_size_range': bag_size_range}
        },
        'triplet': {
            'name': '三元组检测 (连续三数字)',
            'dataset_class': TripletDataset,
            'check_fn': check_triplet,
            'encode_fn': encode_triplet,
            'input_dim': 10,
            'dataset_kwargs': {'bag_size_range': bag_size_range}
        },
        'sum': {
            'name': '和为10检测 (两数字和)',
            'dataset_class': SumToTargetDataset,
            'check_fn': lambda bag: check_sum_to_target(bag, 10),
            'encode_fn': encode_sum,
            'input_dim': 10,
            'dataset_kwargs': {'target_sum': 10, 'bag_size_range': bag_size_range}
        },
        'increasing': {
            'name': '递增序列检测 (3+个数字)',
            'dataset_class': IncreasingSequenceDataset,
            'check_fn': lambda bag: check_increasing_sequence(bag, 3),
            'encode_fn': encode_increasing,
            'input_dim': 10,
            'dataset_kwargs': {'min_seq_len': 3, 'bag_size_range': bag_size_range}
        }
    }

    if args.selection_method == 'compare':
        # 对比模式：运行三种方法
        print("\n" + "🔬"*40)
        print("对比实验模式：将依次运行 Attention, KIBO, Random 三种方法")
        print("🔬"*40 + "\n")

        methods = ['attention', 'kibo', 'random']
        results = {}

        for method in methods:
            print(f"\n{'='*80}")
            print(f"🚀 运行方法: {method.upper()}")
            print(f"{'='*80}\n")

            task_config = tasks[args.task] if args.task != 'all' else tasks['triplet']
            # 根据方法选择合适的 top_k 和 bottom_k
            if method == 'attention':
                method_top_k = top_k
                method_bottom_k = bottom_k
            else:  # kibo 或 random
                method_top_k = args.new_size
                method_bottom_k = 0

            run_experiment(
                task_name=f"{task_config['name']} [{method.upper()}]",
                dataset_class=task_config['dataset_class'],
                check_fn=task_config['check_fn'],
                encode_fn=task_config['encode_fn'],
                input_dim=task_config['input_dim'],
                num_bags_train=args.num_bags,
                num_bags_val=args.num_bags // 4,
                num_bags_test=500,
                num_epochs_stage1=args.epochs1,
                num_epochs_stage2=args.epochs2,
                top_k=method_top_k,
                bottom_k=method_bottom_k,
                selection_method=method,
                bilevel_params=bilevel_params if method == 'kibo' else None,
                **task_config['dataset_kwargs']
            )
            print("\n\n")

    elif args.task == 'all':
        # 运行所有任务
        for task_key in ['large_range', 'triplet', 'sum', 'increasing']:
            task_config = tasks[task_key]
            # 根据方法选择合适的 top_k 和 bottom_k
            if args.selection_method == 'attention':
                method_top_k = top_k
                method_bottom_k = bottom_k
            else:  # kibo 或 random
                method_top_k = args.new_size
                method_bottom_k = 0

            run_experiment(
                task_name=task_config['name'],
                dataset_class=task_config['dataset_class'],
                check_fn=task_config['check_fn'],
                encode_fn=task_config['encode_fn'],
                input_dim=task_config['input_dim'],
                num_bags_train=args.num_bags,
                num_bags_val=args.num_bags // 4,
                num_bags_test=500,
                num_epochs_stage1=args.epochs1,
                num_epochs_stage2=args.epochs2,
                top_k=method_top_k,
                bottom_k=method_bottom_k,
                selection_method=args.selection_method,
                bilevel_params=bilevel_params if args.selection_method == 'kibo' else None,
                **task_config['dataset_kwargs']
            )
            print("\n\n")
    else:
        # 运行单个任务
        task_config = tasks[args.task]
        # 根据方法选择合适的 top_k 和 bottom_k
        if args.selection_method == 'attention':
            method_top_k = top_k
            method_bottom_k = bottom_k
        else:  # kibo 或 random
            method_top_k = args.new_size
            method_bottom_k = 0

        run_experiment(
            task_name=task_config['name'],
            dataset_class=task_config['dataset_class'],
            check_fn=task_config['check_fn'],
            encode_fn=task_config['encode_fn'],
            input_dim=task_config['input_dim'],
            num_bags_train=args.num_bags,
            num_bags_val=args.num_bags // 4,
            num_bags_test=500,
            num_epochs_stage1=args.epochs1,
            num_epochs_stage2=args.epochs2,
            top_k=method_top_k,
            bottom_k=method_bottom_k,
            selection_method=args.selection_method,
            bilevel_params=bilevel_params if args.selection_method == 'kibo' else None,
            **task_config['dataset_kwargs']
        )


if __name__ == "__main__":
    main()
