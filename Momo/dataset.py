"""
PyTorch Dataset类用于加载NinaproDB2预处理后的HDF5数据
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path


class NinaproDB2Dataset(Dataset):
    """
    NinaproDB2 HDF5数据集类
    
    支持：
    - 懒加载（按需加载数据，节省内存）
    - 多模态数据（EMG + IMU）
    - 灵活的数据返回模式
    - 单受试者或合并数据加载
    """
    
    def __init__(self, h5_file, mode='both', load_to_memory=False, return_subject=False):
        """
        初始化数据集
        
        参数:
            h5_file: HDF5文件路径
            mode: 数据模式
                - 'both': 返回EMG和IMU（默认）
                - 'emg': 仅返回EMG
                - 'imu': 仅返回IMU
            load_to_memory: 是否一次性加载全部数据到内存
                - False: 懒加载（推荐，适合大数据集）
                - True: 全部加载到内存（适合小数据集，训练更快）
            return_subject: 是否返回受试者ID（仅对合并数据有效）
        """
        self.h5_file = Path(h5_file)
        self.mode = mode
        self.load_to_memory = load_to_memory
        self.return_subject = return_subject
        
        if not self.h5_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.h5_file}")
        
        # 读取数据集信息（临时打开文件）
        with h5py.File(self.h5_file, 'r') as f:
            self.n_samples = f.attrs['n_samples']
            self.window_size = f.attrs['window_size']
            self.n_emg_channels = f.attrs['n_emg_channels']
            self.n_imu_channels = f.attrs['n_imu_channels']
            # 检查是否有受试者信息（合并数据才有）
            self.has_subjects = 'subjects' in f
            
            # 如果选择加载到内存
            if self.load_to_memory:
                print(f"加载数据到内存: {self.h5_file.name}")
                if mode in ['both', 'emg']:
                    self.emg_data = f['emg'][:]
                if mode in ['both', 'imu']:
                    self.imu_data = f['imu'][:]
                self.labels = f['labels'][:]
                if self.has_subjects and return_subject:
                    self.subjects = f['subjects'][:]
                print(f"  已加载 {self.n_samples} 个样本")
            else:
                print(f"使用懒加载模式: {self.h5_file.name}")
                print(f"  样本数: {self.n_samples}")
        
        # 懒加载模式下不保持文件句柄（每次访问时重新打开）
        self.h5_handle = None
        
    def __len__(self):
        """返回数据集大小"""
        return self.n_samples
    
    def __del__(self):
        """析构函数：清理资源"""
        # 懒加载模式下不需要关闭文件句柄（每次访问时独立开关）
        pass
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        返回:
            如果mode='both': (emg, imu, label) 或 (emg, imu, label, subject_id)
            如果mode='emg': (emg, label) 或 (emg, label, subject_id)
            如果mode='imu': (imu, label) 或 (imu, label, subject_id)
        """
        if self.load_to_memory:
            # 从内存读取
            if self.mode == 'both':
                emg = torch.FloatTensor(self.emg_data[idx])
                imu = torch.FloatTensor(self.imu_data[idx])
                label = torch.LongTensor([self.labels[idx]])[0]
                if self.return_subject and self.has_subjects:
                    subject_id = torch.LongTensor([self.subjects[idx]])[0]
                    return emg, imu, label, subject_id
                return emg, imu, label
            elif self.mode == 'emg':
                emg = torch.FloatTensor(self.emg_data[idx])
                label = torch.LongTensor([self.labels[idx]])[0]
                if self.return_subject and self.has_subjects:
                    subject_id = torch.LongTensor([self.subjects[idx]])[0]
                    return emg, label, subject_id
                return emg, label
            elif self.mode == 'imu':
                imu = torch.FloatTensor(self.imu_data[idx])
                label = torch.LongTensor([self.labels[idx]])[0]
                if self.return_subject and self.has_subjects:
                    subject_id = torch.LongTensor([self.subjects[idx]])[0]
                    return imu, label, subject_id
                return imu, label
        else:
            # 懒加载：每次重新打开文件（避免多进程竞争）
            with h5py.File(self.h5_file, 'r') as f:
                if self.mode == 'both':
                    emg = torch.FloatTensor(f['emg'][idx])
                    imu = torch.FloatTensor(f['imu'][idx])
                    label = torch.LongTensor([f['labels'][idx]])[0]
                    if self.return_subject and self.has_subjects:
                        subject_id = torch.LongTensor([f['subjects'][idx]])[0]
                        return emg, imu, label, subject_id
                    return emg, imu, label
                elif self.mode == 'emg':
                    emg = torch.FloatTensor(f['emg'][idx])
                    label = torch.LongTensor([f['labels'][idx]])[0]
                    if self.return_subject and self.has_subjects:
                        subject_id = torch.LongTensor([f['subjects'][idx]])[0]
                        return emg, label, subject_id
                    return emg, label
                elif self.mode == 'imu':
                    imu = torch.FloatTensor(f['imu'][idx])
                    label = torch.LongTensor([f['labels'][idx]])[0]
                    if self.return_subject and self.has_subjects:
                        subject_id = torch.LongTensor([f['subjects'][idx]])[0]
                        return imu, label, subject_id
                    return imu, label
    
    def get_info(self):
        """获取数据集信息"""
        info = {
            'n_samples': self.n_samples,
            'window_size': self.window_size,
            'n_emg_channels': self.n_emg_channels,
            'n_imu_channels': self.n_imu_channels,
            'mode': self.mode,
            'load_to_memory': self.load_to_memory
        }
        return info


def create_dataloaders(data_dir, batch_size=32, num_workers=4, 
                       mode='both', load_to_memory=False, shuffle_train=True,
                       subject=None, return_subject=False):
    """
    创建训练和测试DataLoader
    
    参数:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        mode: 数据模式 ('both', 'emg', 'imu')
        load_to_memory: 是否加载到内存
        shuffle_train: 是否打乱训练集
        subject: 指定受试者编号（如10, 23, 36），None表示使用合并数据
        return_subject: 是否返回受试者ID（仅对合并数据有效）
        
    返回:
        train_loader: 训练集DataLoader
        test_loader: 测试集DataLoader
    """
    data_path = Path(data_dir)
    
    # 根据是否指定受试者选择文件
    if subject is not None:
        train_file = data_path / f'S{subject}_train.h5'
        test_file = data_path / f'S{subject}_test.h5'
        print(f"加载受试者 S{subject} 的数据")
    else:
        train_file = data_path / 'all_train.h5'
        test_file = data_path / 'all_test.h5'
        print(f"加载合并数据（所有受试者）")
    
    # 创建数据集
    train_dataset = NinaproDB2Dataset(
        h5_file=train_file,
        mode=mode,
        load_to_memory=load_to_memory,
        return_subject=return_subject
    )
    
    test_dataset = NinaproDB2Dataset(
        h5_file=test_file,
        mode=mode,
        load_to_memory=load_to_memory,
        return_subject=return_subject
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True  # 加速GPU训练
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n{'='*60}")
    print("DataLoader创建完成")
    print(f"{'='*60}")
    print(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")
    print(f"批次大小: {batch_size}")
    print(f"数据模式: {mode}")
    if subject:
        print(f"受试者: S{subject}")
    else:
        print(f"数据类型: 合并数据")
    print(f"{'='*60}\n")
    
    return train_loader, test_loader


if __name__ == "__main__":
    """
    使用示例
    """
    import pickle
    
    # 数据路径
    data_dir = "/home/mlsnrs/data/wrj/Momo/processed_data"
    
    print("="*60)
    print("NinaproDB2 PyTorch Dataset 使用示例")
    print("="*60)
    
    # 方式1: 直接使用Dataset
    print("\n【方式1】直接使用Dataset类")
    print("-" * 60)
    train_dataset = NinaproDB2Dataset(
        h5_file=f"{data_dir}/train_data.h5",
        mode='both',
        load_to_memory=False  # 懒加载模式
    )
    
    print(f"\n数据集信息:")
    info = train_dataset.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 获取单个样本
    print(f"\n获取第一个样本:")
    emg, imu, label = train_dataset[0]
    print(f"  EMG形状: {emg.shape}")
    print(f"  IMU形状: {imu.shape}")
    print(f"  标签: {label.item()}")
    
    # 方式2: 使用DataLoader（推荐用于训练）
    print("\n【方式2】使用DataLoader（推荐）")
    print("-" * 60)
    train_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=2,
        mode='both',
        load_to_memory=False
    )
    
    # 迭代一个批次
    print("迭代第一个批次:")
    for batch_emg, batch_imu, batch_labels in train_loader:
        print(f"  批次EMG形状: {batch_emg.shape}")  # (batch_size, window_size, n_channels)
        print(f"  批次IMU形状: {batch_imu.shape}")
        print(f"  批次标签形状: {batch_labels.shape}")
        print(f"  批次标签: {batch_labels[:5]}")  # 显示前5个标签
        break
    
    # 加载元数据
    print("\n【元数据信息】")
    print("-" * 60)
    with open(f"{data_dir}/metadata.pkl", 'rb') as f:
        metadata_info = pickle.load(f)
    
    metadata = metadata_info['metadata']
    print(f"受试者: {metadata['subjects']}")
    print(f"采样率: {metadata['fs']} Hz")
    print(f"类别数: {metadata['n_classes']}")
    print(f"窗口大小: {metadata['window_size_ms']} ms")
    
    print("\n" + "="*60)
    print("示例运行完成！您可以在训练代码中使用 create_dataloaders() 函数")
    print("="*60)

