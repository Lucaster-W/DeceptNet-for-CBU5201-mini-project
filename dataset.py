import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os

class DeceptionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def prepare_datasets(config):
    """准备训练集、验证集和测试集"""
    # 加载数据
    data = pd.read_csv(os.path.join(config.DATA_DIR, config.METADATA_FILE))
    
    # 将Story_type转换为数值标签
    data['label'] = (data['Story_type'] == 'Deceptive Story').astype(int)
    
    # 首先分离出测试集（20%的数据）
    train_val_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data[['label', 'Language']]
    )
    
    # 然后将剩余数据分为训练集和验证集（在剩余数据中验证集占20%，即总数据的16%）
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=0.2,
        random_state=42,
        stratify=train_val_data[['label', 'Language']]
    )
    
    # 打印原始数据集大小
    print(f"原始数据集大小:")
    print(f"训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    print(f"测试集: {len(test_data)} 样本")
    
    return train_data, val_data, test_data

def create_dataloaders(features, labels, config, shuffle=True):
    """创建数据加载器"""
    dataset = DeceptionDataset(features, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle
    )
    return dataloader 