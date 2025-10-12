"""
Attention-based Multiple Instance Learning for Consecutive Digit Pair Detection
检测包中是否存在连续数字对(0,1), (1,2), (2,3), (3,4)的多实例学习网络
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support


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

        # 提取特征: [batch_size * num_instances, hidden_dim]
        x_flat = x.view(-1, x.shape[-1])
        h = self.feature_extractor(x_flat)
        h = h.view(batch_size, num_instances, -1)  # [batch_size, num_instances, hidden_dim]

        # 计算attention权重
        a = self.attention(h)  # [batch_size, num_instances, 1]
        a = torch.softmax(a, dim=1)  # 在实例维度上做softmax

        # 加权聚合
        z = torch.sum(a * h, dim=1)  # [batch_size, hidden_dim]

        # 分类
        y_prob = self.classifier(z)  # [batch_size, 1]

        return y_prob.squeeze(), a.squeeze()


class Digit9Dataset(Dataset):
    """连续数字对检测数据集

    正包(label=1): 包中存在任意连续数字对(0,1), (1,2), (2,3), (3,4)
    负包(label=0): 包中不存在上述连续数字对
    """

    def __init__(self, num_bags=1000, bag_size_range=(5, 20), pos_ratio=0.5):
        """
        num_bags: 包的数量
        bag_size_range: 每个包中实例数量的范围
        pos_ratio: 正包的比例
        """
        self.num_bags = num_bags
        self.bag_size_range = bag_size_range
        self.bags = []
        self.labels = []

        np.random.seed(42)

        for i in range(num_bags):
            bag_size = np.random.randint(bag_size_range[0], bag_size_range[1] + 1)

            # 根据比例决定是正包还是负包
            is_positive = np.random.rand() < pos_ratio

            if is_positive:
                # 正包: 包含至少一对连续数字(0,1), (1,2), (2,3), (3,4)
                # 随机选择一对连续数字
                consecutive_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
                pair = consecutive_pairs[np.random.randint(0, len(consecutive_pairs))]

                # 确保这一对都出现在包中
                bag = [pair[0], pair[1]]

                # 填充其他数字(0-9)
                remaining_size = bag_size - 2
                bag += list(np.random.randint(0, 10, remaining_size))

                label = 1
            else:
                # 负包: 不包含任何连续数字对
                # 策略：只使用5-9的数字，或者只使用{0,2,4}这样间隔的数字
                if np.random.rand() < 0.5:
                    # 只使用5-9的数字（避免0-4的连续对）
                    bag = list(np.random.randint(5, 10, bag_size))
                else:
                    # 使用0,2,4的组合（没有连续关系）
                    bag = list(np.random.choice([0, 2, 4, 5, 6, 7, 8, 9], bag_size))

                label = 0

            # 随机打乱
            np.random.shuffle(bag)

            self.bags.append(bag)
            self.labels.append(label)

    def __len__(self):
        return self.num_bags

    def __getitem__(self, idx):
        bag = self.bags[idx]
        label = self.labels[idx]

        # One-hot编码
        bag_tensor = torch.zeros(len(bag), 10)
        for i, digit in enumerate(bag):
            bag_tensor[i, digit] = 1.0

        return bag_tensor, torch.tensor(label, dtype=torch.float32)


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

        # 前向传播
        y_prob, attention = model(bags)

        # 计算损失
        loss = criterion(y_prob, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 记录预测结果
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

            # 前向传播
            y_prob, attention = model(bags)

            # 计算损失
            loss = criterion(y_prob, labels)
            total_loss += loss.item()

            # 记录预测结果
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


def check_consecutive_pair(bag):
    """检查包中是否存在连续数字对(0,1), (1,2), (2,3), (3,4)"""
    digits_set = set(bag)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        if a in digits_set and b in digits_set:
            return True
    return False


def extract_attention_based_instances(model, dataset, device, top_k=3, bottom_k=2):
    """
    从训练集中提取基于attention的实例，构建新的包

    Args:
        model: 训练好的模型
        dataset: 原始数据集
        device: 设备
        top_k: 每个包选择top-k个最高attention的实例
        bottom_k: 每个包选择bottom-k个最低attention的实例

    Returns:
        new_bags: 新构建的包列表（每个包是数字列表）
        new_labels: 对应的标签列表
        mismatch_stats: 标签不一致统计信息
    """
    model.eval()
    new_bags = []
    new_labels = []

    # 统计标签不一致情况
    total_bags = 0
    mismatch_count = 0
    pos_to_neg = 0  # 原本是正包，过滤后变成负包
    neg_to_pos = 0  # 原本是负包，过滤后变成正包

    print(f"\n从{len(dataset)}个包中提取attention-based实例...")
    print(f"策略: 每个包选择top-{top_k}和bottom-{bottom_k}个实例")

    with torch.no_grad():
        for i in range(len(dataset)):
            bag, label = dataset[i]
            bag_tensor = bag.unsqueeze(0).to(device)

            # 获取attention权重
            _, attention = model(bag_tensor)

            # 处理attention维度
            if attention.dim() == 0:
                attention = attention.unsqueeze(0)

            attention_weights = attention.cpu().numpy()
            digits = dataset.bags[i]

            # 获取实际实例数量（排除padding）
            num_instances = len(digits)
            attention_weights = attention_weights[:num_instances]

            # 获取top-k和bottom-k的索引
            sorted_indices = np.argsort(attention_weights)[::-1]  # 降序

            # 选择实例
            selected_indices = []
            # Top-k
            selected_indices.extend(sorted_indices[:min(top_k, len(sorted_indices))])
            # Bottom-k
            if bottom_k > 0:
                selected_indices.extend(sorted_indices[-min(bottom_k, len(sorted_indices)):])

            # 去重并构建新包
            selected_indices = list(set(selected_indices))
            new_bag = [digits[idx] for idx in selected_indices]

            new_bags.append(new_bag)
            new_labels.append(label)

            # 检查新包的真实标签
            original_label = int(label)
            true_label_for_new_bag = 1 if check_consecutive_pair(new_bag) else 0

            total_bags += 1
            if original_label != true_label_for_new_bag:
                mismatch_count += 1
                if original_label == 1 and true_label_for_new_bag == 0:
                    pos_to_neg += 1
                elif original_label == 0 and true_label_for_new_bag == 1:
                    neg_to_pos += 1

    avg_size = np.mean([len(bag) for bag in new_bags])

    # 打印统计信息
    print(f"完成! 创建了{len(new_bags)}个新包")
    print(f"平均包大小: {avg_size:.2f} (原始: {np.mean([len(dataset.bags[i]) for i in range(len(dataset))]):.2f})")
    print(f"\n标签一致性检查:")
    print(f"  总包数: {total_bags}")
    print(f"  标签不一致: {mismatch_count} ({mismatch_count/total_bags*100:.2f}%)")
    print(f"    - 正包→负包 (丢失连续对): {pos_to_neg} ({pos_to_neg/total_bags*100:.2f}%)")
    print(f"    - 负包→正包 (意外产生连续对): {neg_to_pos} ({neg_to_pos/total_bags*100:.2f}%)")
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

    def __init__(self, bags, labels):
        """
        Args:
            bags: 包列表，每个包是数字列表
            labels: 标签列表
        """
        self.bags = bags
        self.labels = labels

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        label = self.labels[idx]

        # One-hot编码
        bag_tensor = torch.zeros(len(bag), 10)
        for i, digit in enumerate(bag):
            bag_tensor[i, digit] = 1.0

        return bag_tensor, torch.tensor(label, dtype=torch.float32)


def visualize_attention(model, dataset, device, num_samples=3):
    """可视化attention权重"""
    model.eval()
    print("\n" + "="*60)
    print("Attention权重可视化示例:")
    print("="*60)

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            bag, label = dataset[i]
            bag = bag.unsqueeze(0).to(device)

            y_prob, attention = model(bag)

            # 获取原始数字（直接从数据集中获取）
            digits = dataset.bags[i]

            # 检测是否存在连续数字对
            consecutive_pairs = []
            digits_set = set(digits)
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
                if a in digits_set and b in digits_set:
                    consecutive_pairs.append(f"({a},{b})")

            has_consecutive = len(consecutive_pairs) > 0
            pairs_str = ", ".join(consecutive_pairs) if consecutive_pairs else "无"

            print(f"\n样本 {i+1}:")
            print(f"  真实标签: {'正包(有连续对)' if label == 1 else '负包(无连续对)'}")
            print(f"  检测到的连续对: {pairs_str}")
            print(f"  预测概率: {y_prob.item():.4f}")
            print(f"  预测标签: {'正包' if y_prob.item() > 0.5 else '负包'}")
            print(f"  包内容: {digits}")
            print(f"  Attention权重:")

            if attention.dim() == 0:
                attention = attention.unsqueeze(0)

            for j, (digit, att) in enumerate(zip(digits, attention.cpu().numpy())):
                bar = "█" * int(att * 50)
                print(f"    实例{j+1} [数字={digit}]: {att:.4f} {bar}")


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 超参数
    num_epochs_stage1 = 10  # 第一阶段训练轮数
    num_epochs_stage2 = 10  # 第二阶段训练轮数
    batch_size = 32
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Attention-based实例选择参数
    top_k = 2  # 选择top-k个最高attention的实例
    bottom_k = 2  # 选择bottom-k个最低attention的实例

    print("="*60)
    print("基于Attention的多实例学习 - 连续数字对检测")
    print("两阶段训练: 原始数据 -> Attention过滤 -> 重新训练")
    print("="*60)
    print(f"设备: {device}")
    print(f"第一阶段训练轮数: {num_epochs_stage1}")
    print(f"第二阶段训练轮数: {num_epochs_stage2}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"实例选择策略: top-{top_k} + bottom-{bottom_k}")

    # ========== 第一阶段: 在原始数据上训练 ==========
    print("\n" + "="*60)
    print("第一阶段: 在原始数据上训练")
    print("="*60)

    # 创建数据集
    print("\n创建原始数据集...")
    train_dataset = Digit9Dataset(num_bags=2000, bag_size_range=(5, 20), pos_ratio=0.5)
    val_dataset = Digit9Dataset(num_bags=500, bag_size_range=(5, 20), pos_ratio=0.5)
    test_dataset = Digit9Dataset(num_bags=500, bag_size_range=(5, 20), pos_ratio=0.5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"训练集: {len(train_dataset)} 个包")
    print(f"验证集: {len(val_dataset)} 个包")
    print(f"测试集: {len(test_dataset)} 个包")

    # 创建模型
    model_stage1 = AttentionMIL(input_dim=10, hidden_dim=128, attention_dim=64).to(device)
    criterion = nn.BCELoss()
    optimizer_stage1 = optim.Adam(model_stage1.parameters(), lr=learning_rate)

    # 训练第一阶段
    print("\n开始第一阶段训练...")
    best_val_auc_stage1 = 0
    best_model_state_stage1 = None

    for epoch in range(num_epochs_stage1):
        train_loss, train_acc = train_epoch(model_stage1, train_loader, criterion, optimizer_stage1, device)
        val_loss, val_acc, val_auc, val_precision, val_recall, val_f1 = evaluate(model_stage1, val_loader, criterion, device)

        # 保存最佳模型
        if val_auc > best_val_auc_stage1:
            best_val_auc_stage1 = val_auc
            best_model_state_stage1 = model_stage1.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs_stage1}]")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

    # 加载最佳模型
    model_stage1.load_state_dict(best_model_state_stage1)

    # 第一阶段测试
    print("\n" + "="*60)
    print("第一阶段测试集评估:")
    print("="*60)
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
    print("\n" + "="*60)
    print("第二阶段: 基于Attention过滤数据并重新训练")
    print("="*60)

    # 提取attention-based实例
    filtered_train_bags, filtered_train_labels, train_mismatch_stats = extract_attention_based_instances(
        model_stage1, train_dataset, device, top_k=top_k, bottom_k=bottom_k
    )
    filtered_val_bags, filtered_val_labels, val_mismatch_stats = extract_attention_based_instances(
        model_stage1, val_dataset, device, top_k=top_k, bottom_k=bottom_k
    )

    # 创建新数据集
    filtered_train_dataset = FilteredDataset(filtered_train_bags, filtered_train_labels)
    filtered_val_dataset = FilteredDataset(filtered_val_bags, filtered_val_labels)

    filtered_train_loader = DataLoader(filtered_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    filtered_val_loader = DataLoader(filtered_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 创建新模型（重新初始化）
    model_stage2 = AttentionMIL(input_dim=10, hidden_dim=128, attention_dim=64).to(device)
    optimizer_stage2 = optim.Adam(model_stage2.parameters(), lr=learning_rate)

    # 训练第二阶段
    print("\n开始第二阶段训练...")
    best_val_auc_stage2 = 0
    best_model_state_stage2 = None

    for epoch in range(num_epochs_stage2):
        train_loss, train_acc = train_epoch(model_stage2, filtered_train_loader, criterion, optimizer_stage2, device)
        val_loss, val_acc, val_auc, val_precision, val_recall, val_f1 = evaluate(model_stage2, filtered_val_loader, criterion, device)

        # 保存最佳模型
        if val_auc > best_val_auc_stage2:
            best_val_auc_stage2 = val_auc
            best_model_state_stage2 = model_stage2.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs_stage2}]")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

    # 加载最佳模型
    model_stage2.load_state_dict(best_model_state_stage2)

    # 第二阶段测试（使用原始测试集）
    print("\n" + "="*60)
    print("第二阶段测试集评估 (在原始测试集上):")
    print("="*60)
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
    print("\n" + "="*60)
    print("结果对比:")
    print("="*60)
    print(f"第一阶段 (原始数据):")
    print(f"  Accuracy: {test_acc_s1:.4f}, AUC: {test_auc_s1:.4f}, F1: {test_f1_s1:.4f}")
    print(f"第二阶段 (Attention过滤):")
    print(f"  Accuracy: {test_acc_s2:.4f}, AUC: {test_auc_s2:.4f}, F1: {test_f1_s2:.4f}")
    print(f"性能变化:")
    print(f"  Accuracy: {test_acc_s2 - test_acc_s1:+.4f}")
    print(f"  AUC: {test_auc_s2 - test_auc_s1:+.4f}")
    print(f"  F1: {test_f1_s2 - test_f1_s1:+.4f}")

    print("\n" + "="*60)
    print("标签不一致总结:")
    print("="*60)
    print(f"训练集: {train_mismatch_stats['mismatch']}/{train_mismatch_stats['total']} "
          f"({train_mismatch_stats['mismatch_rate']*100:.2f}%) 包标签不一致")
    print(f"  正包→负包: {train_mismatch_stats['pos_to_neg']}")
    print(f"  负包→正包: {train_mismatch_stats['neg_to_pos']}")
    print(f"验证集: {val_mismatch_stats['mismatch']}/{val_mismatch_stats['total']} "
          f"({val_mismatch_stats['mismatch_rate']*100:.2f}%) 包标签不一致")
    print(f"  正包→负包: {val_mismatch_stats['pos_to_neg']}")
    print(f"  负包→正包: {val_mismatch_stats['neg_to_pos']}")

    # 可视化attention（使用第二阶段模型）
    visualize_attention(model_stage2, test_dataset, device, num_samples=5)

    print("\n" + "="*60)
    print("两阶段训练完成!")
    print("="*60)


if __name__ == "__main__":
    main()
