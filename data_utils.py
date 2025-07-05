"""
数据处理工具，包括数据读取、编码、数据集创建等
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

def getSequenceData(first_dir, file_name):
    """
    从文件读取序列数据和标签
    
    Args:
        first_dir: 目录路径
        file_name: 文件名
        
    Returns:
        Tuple[List[str], List[np.ndarray]]: 序列和标签列表
    """
    data, label = [], []
    path = os.path.join(first_dir, f"{file_name}.txt")

    with open(path) as f:
        for each in f:
            each = each.strip()
            if each[0] == '>':
                label.append(np.array(list(each[1:]), dtype=int))
            else:
                data.append(each)

    return data, label


def PadEncode(data, label, max_len):
    """
    编码氨基酸序列并填充到固定长度
    
    Args:
        data: 氨基酸序列列表
        label: 标签列表
        max_len: 最大序列长度
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 编码序列、标签和序列长度
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    data_e, label_e, seq_length = [], [], []

    for i, seq in enumerate(data):
        seq = seq.strip()
        if any(residue not in amino_acids for residue in seq):
            continue  # 跳过含有无效残基的序列

        if len(seq) > max_len:
            continue  # 跳过过长的序列

        encoded_seq = [amino_acids.index(residue) + 1 for residue in seq]  # 编码序列（加1以避免使用0，0用于填充）
        seq_length.append(len(encoded_seq))  # 记录实际长度

        # 使用0填充以保持统一长度
        encoded_seq += [0] * (max_len - len(encoded_seq))
        data_e.append(encoded_seq)
        label_e.append(label[i])

    return np.array(data_e), np.array(label_e), np.array(seq_length)


class PeptideDataset(Dataset):
    """肽序列数据集类，支持多种特征类型"""
    
    def __init__(self, sequences, labels, seq_lengths, features=None, feature_order=None):
        """
        初始化数据集
        
        Args:
            sequences: 编码的蛋白质序列
            labels: 目标标签
            seq_lengths: 序列长度
            features: 各种特征的字典 {'feature_name': tensor, ...}
            feature_order: 特征排序顺序，如果指定，则按此顺序排列特征
        """
        self.sequences = sequences
        self.labels = labels
        self.seq_lengths = seq_lengths
        self.features = features or {}
        
        # 使用指定的特征顺序（如果提供）
        if feature_order:
            # 只包含存在于features中的特征名称
            self.feature_names = [f for f in feature_order if f in self.features]
            # 添加任何可能不在feature_order中但存在于features中的特征
            for f in self.features:
                if f not in self.feature_names:
                    self.feature_names.append(f)
        else:
            # 否则按字母顺序排序
            self.feature_names = sorted(list(self.features.keys()))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # 基本数据
        item = [self.sequences[idx], self.labels[idx], self.seq_lengths[idx]]
        
        # 添加各种特征
        for feature_name in self.feature_names:
            feature_tensor = self.features.get(feature_name)
            if feature_tensor is not None:
                item.append(feature_tensor[idx])
            else:
                item.append(None)
        
        return tuple(item)


def create_dataloaders(x_train, y_train, train_length, 
                      x_val, y_val, val_length, 
                      x_test, y_test, test_length,
                      all_features=None, batch_size=32, feature_order=None):
    """
    创建支持多种特征的DataLoaders
    
    Args:
        x_train, y_train, train_length: 训练数据
        x_val, y_val, val_length: 验证数据
        x_test, y_test, test_length: 测试数据
        all_features: 所有特征的字典 {'feature_name': {'train': tensor, 'val': tensor, 'test': tensor}}
        batch_size: 批量大小
        feature_order: 特征排序顺序
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: 训练、验证和测试的DataLoaders
    """
    # 准备特征字典
    train_features = {}
    val_features = {}
    test_features = {}
    
    if all_features:
        for feature_name, feature_dict in all_features.items():
            if 'train' in feature_dict:
                train_features[feature_name] = feature_dict['train']
            if 'val' in feature_dict:
                val_features[feature_name] = feature_dict['val']
            if 'test' in feature_dict:
                test_features[feature_name] = feature_dict['test']
    
    # 创建数据集
    train_dataset = PeptideDataset(x_train, y_train, train_length, train_features, feature_order)
    val_dataset = PeptideDataset(x_val, y_val, val_length, val_features, feature_order)
    test_dataset = PeptideDataset(x_test, y_test, test_length, test_features, feature_order)
    
    # 创建DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def load_dataset(args):
    """
    加载和预处理数据集
    
    Args:
        args: 配置参数
        
    Returns:
        元组: 训练、验证和测试数据集及DataLoaders
    """
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')
    max_length = args.max_length

    # 获取序列和标签
    train_sequence_data, train_sequence_label = getSequenceData(train_dir, 'train')
    val_sequence_data, val_sequence_label = getSequenceData(val_dir, 'val')
    test_sequence_data, test_sequence_label = getSequenceData(test_dir, 'test')

    # 转换为numpy数组
    y_train = np.array(train_sequence_label)
    y_val = np.array(val_sequence_label)
    y_test = np.array(test_sequence_label)

    # 编码和填充序列
    x_train, y_train, train_length = PadEncode(train_sequence_data, y_train, max_length)
    x_val, y_val, val_length = PadEncode(val_sequence_data, y_val, max_length)
    x_test, y_test, test_length = PadEncode(test_sequence_data, y_test, max_length)

    # 转换为torch张量
    x_train_tensor = torch.LongTensor(x_train)
    x_val_tensor = torch.LongTensor(x_val)
    x_test_tensor = torch.LongTensor(x_test)
    
    train_length_tensor = torch.LongTensor(train_length)
    val_length_tensor = torch.LongTensor(val_length)
    test_length_tensor = torch.LongTensor(test_length)
    
    y_train_tensor = torch.Tensor(y_train)
    y_val_tensor = torch.Tensor(y_val)
    y_test_tensor = torch.Tensor(y_test)
    
    print(f"训练集: {x_train.shape[0]} 个样本")
    print(f"验证集: {x_val.shape[0]} 个样本")
    print(f"测试集: {x_test.shape[0]} 个样本")
    
    return (
        train_sequence_data, val_sequence_data, test_sequence_data,
        x_train_tensor, y_train_tensor, train_length_tensor,
        x_val_tensor, y_val_tensor, val_length_tensor,
        x_test_tensor, y_test_tensor, test_length_tensor
    )