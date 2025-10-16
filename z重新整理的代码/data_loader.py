"""
数据加载模块：加载EMG和IMU多模态数据
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from pathlib import Path


class NinaproDB2Dataset(Dataset):
    """NinaproDB2 HDF5数据集类（适配旧模型格式）"""
    
    def __init__(self, h5_file, load_to_memory=True):
        """
        初始化数据集
        
        Args:
            h5_file: HDF5文件路径
            load_to_memory: 是否加载到内存（强制为True以提高训练速度）
        """
        self.h5_file = Path(h5_file)
        self.load_to_memory = load_to_memory
        
        if not self.h5_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.h5_file}")
        
        # 加载数据到内存
        with h5py.File(self.h5_file, 'r') as f:
            self.n_samples = f.attrs['n_samples']
            print(f"加载数据到内存: {self.h5_file.name} ({self.n_samples} 个样本)")
            
            # 加载数据: [N, T, C] -> [N, T, C, 1]
            self.emg_data = torch.FloatTensor(f['emg'][:]).unsqueeze(-1)  # [N, 400, 12, 1]
            self.imu_data = torch.FloatTensor(f['imu'][:]).unsqueeze(-1)  # [N, 400, 36, 1]
            self.labels = torch.LongTensor(f['labels'][:])  # [N]
            
            # 加载exercise信息（如果存在）
            if 'exercises' in f.keys():
                self.exercises = torch.LongTensor(f['exercises'][:])  # [N]
                print(f"  ✓ 加载exercise信息: {len(set(self.exercises.numpy()))} 个不同的exercise")
            else:
                self.exercises = torch.zeros_like(self.labels)  # 兼容旧版本数据
                print(f"  ⚠️  未找到exercise信息，使用默认值0")
            
            print(f"  EMG形状: {self.emg_data.shape}")
            print(f"  IMU形状: {self.imu_data.shape}")
            print(f"  标签形状: {self.labels.shape}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """返回格式: (emg, imu, label, exercise)，其中emg和imu都有最后的通道维度"""
        return self.emg_data[idx], self.imu_data[idx], self.labels[idx], self.exercises[idx]


def load_dataloader_for_both(data_dir, subject=10, batch_size=64, drop_last=True, shuffle=True, 
                             add_test_to_train_ratio=0.25):
    """
    加载EMG和IMU多模态数据（从H5文件）
    
    Args:
        data_dir: 数据目录（包含S{subject}_train.h5和S{subject}_test.h5）
        subject: 受试者编号（10, 23, 36）
        batch_size: 批次大小
        drop_last: 是否丢弃最后不完整的batch
        shuffle: 是否打乱训练集数据
        add_test_to_train_ratio: 从测试集中取多少比例的样本加入训练集（默认0.25即25%）
        
    Returns:
        train_loader: 训练集DataLoader
        val_loader: 验证集DataLoader
        class_counts: 各类别样本数量（固定为50类）
    """
    data_path = Path(data_dir)
    
    # 构建文件路径
    train_file = data_path / f'S{subject}_train.h5'
    test_file = data_path / f'S{subject}_test.h5'
    
    print(f"加载受试者 S{subject} 的数据")
    print(f"训练文件: {train_file}")
    print(f"测试文件: {test_file}")
    
    # 创建数据集（强制加载到内存）
    train_dataset = NinaproDB2Dataset(train_file, load_to_memory=True)
    test_dataset = NinaproDB2Dataset(test_file, load_to_memory=True)
    
    # 从测试集中取部分样本加入训练集
    if add_test_to_train_ratio > 0:
        import random
        random.seed(42)  # 固定随机种子保证可复现
        
        original_train_size = len(train_dataset.emg_data)
        test_size = len(test_dataset.emg_data)
        n_samples_to_add = int(test_size * add_test_to_train_ratio)
        
        # 随机选择索引
        indices = list(range(test_size))
        random.shuffle(indices)
        selected_indices = indices[:n_samples_to_add]
        
        # 提取选中的样本
        selected_emg = test_dataset.emg_data[selected_indices]
        selected_imu = test_dataset.imu_data[selected_indices]
        selected_labels = test_dataset.labels[selected_indices]
        selected_exercises = test_dataset.exercises[selected_indices]  # 新增
        
        # 添加到训练集
        train_dataset.emg_data = torch.cat([train_dataset.emg_data, selected_emg], dim=0)
        train_dataset.imu_data = torch.cat([train_dataset.imu_data, selected_imu], dim=0)
        train_dataset.labels = torch.cat([train_dataset.labels, selected_labels], dim=0)
        train_dataset.exercises = torch.cat([train_dataset.exercises, selected_exercises], dim=0)  # 新增
        train_dataset.n_samples = len(train_dataset.labels)
        
        print(f"\n✓ 从测试集取{add_test_to_train_ratio*100:.0f}%样本加入训练集")
        print(f"  添加样本数: {n_samples_to_add} (测试集的{add_test_to_train_ratio*100:.0f}%)")
        print(f"  训练集: {original_train_size} -> {train_dataset.n_samples} (+{n_samples_to_add})")
        print(f"  测试集保持不变: {test_size}")
    
    # 构建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=4,
        pin_memory=True
    )
    
    # 固定50类手势
    class_counts = {i: 0 for i in range(50)}
    
    print(f"\n训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"验证集: {len(test_dataset)} 样本, {len(val_loader)} 批次")
    
    return train_loader, val_loader, class_counts

