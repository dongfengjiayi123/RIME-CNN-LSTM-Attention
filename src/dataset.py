"""
数据集模块：用户行为序列数据加载与预处理
支持 Taobao User Behavior Dataset 等电商用户行为数据
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class UserBehaviorDataset(Dataset):
    """用户行为序列数据集

    每个样本包含：
    - behavior_seq: 用户历史行为类型序列 (padded to MAX_SEQ_LEN)
    - item_seq: 用户历史交互物品序列
    - time_seq: 行为时间间隔序列
    - label: 是否购买 (0/1)
    """

    def __init__(self, sequences, labels, max_seq_len=50):
        self.sequences = sequences
        self.labels = labels
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # 截断或填充到 max_seq_len
        behavior_seq = seq["behavior"][: self.max_seq_len]
        item_seq = seq["item"][: self.max_seq_len]
        time_seq = seq["time_interval"][: self.max_seq_len]

        pad_len = self.max_seq_len - len(behavior_seq)
        behavior_seq = behavior_seq + [0] * pad_len
        item_seq = item_seq + [0] * pad_len
        time_seq = time_seq + [0.0] * pad_len
        mask = [1] * (self.max_seq_len - pad_len) + [0] * pad_len

        return {
            "behavior_seq": torch.LongTensor(behavior_seq),
            "item_seq": torch.LongTensor(item_seq),
            "time_seq": torch.FloatTensor(time_seq),
            "mask": torch.FloatTensor(mask),
            "label": torch.FloatTensor([label]),
        }


def preprocess_raw_data(df, min_user_behaviors=5):
    """预处理原始用户行为数据

    Args:
        df: 包含 user_id, item_id, behavior_type, timestamp 的 DataFrame
        min_user_behaviors: 过滤行为次数过少的用户

    Returns:
        sequences: list of dict, 每个用户的行为序列
        labels: list of int, 购买标签
    """
    # 行为类型映射（如果原始数据是字符串）
    behavior_map = {"pv": 0, "fav": 1, "cart": 2, "buy": 3}
    if df["behavior_type"].dtype == object:
        df["behavior_type"] = df["behavior_type"].map(behavior_map).fillna(0).astype(int)

    # 按用户和时间排序
    df = df.sort_values(["user_id", "timestamp"])

    sequences = []
    labels = []

    for user_id, group in df.groupby("user_id"):
        if len(group) < min_user_behaviors:
            continue

        behaviors = group["behavior_type"].tolist()
        items = group["item_id"].astype("category").cat.codes.tolist()
        timestamps = group["timestamp"].tolist()

        # 计算时间间隔（归一化）
        time_intervals = [0.0]
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i - 1]) / 3600.0  # 小时
            time_intervals.append(min(delta, 24.0) / 24.0)  # 截断到1天并归一化

        # 标签：用户是否有购买行为
        label = 1 if 3 in behaviors else 0

        sequences.append(
            {
                "behavior": behaviors,
                "item": items,
                "time_interval": time_intervals,
            }
        )
        labels.append(label)

    return sequences, labels


def build_dataloader(sequences, labels, batch_size=256, max_seq_len=50, shuffle=True):
    """构建 DataLoader"""
    dataset = UserBehaviorDataset(sequences, labels, max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_test_split(sequences, labels, test_ratio=0.2, seed=42):
    """按用户划分训练集和测试集"""
    np.random.seed(seed)
    indices = np.random.permutation(len(labels))
    split = int(len(labels) * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_seq = [sequences[i] for i in train_idx]
    train_label = [labels[i] for i in train_idx]
    test_seq = [sequences[i] for i in test_idx]
    test_label = [labels[i] for i in test_idx]

    return train_seq, train_label, test_seq, test_label
