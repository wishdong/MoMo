# NinaproDB2 EMG+IMU 多模态手势识别

用于NinaproDB2数据集的预处理和PyTorch训练数据加载。

## 📁 项目结构

```
Momo/
├── data/                      # 原始数据
│   ├── DB2_s10/              # 受试者10
│   ├── DB2_s23/              # 受试者23
│   └── DB2_s36/              # 受试者36
├── processed_data/            # 预处理后的数据（运行后生成）
│   ├── train_data.h5         # 训练集（HDF5格式）
│   ├── test_data.h5          # 测试集（HDF5格式）
│   └── metadata.pkl          # 元数据和标准化器
├── preprocess.py             # 数据预处理脚本
├── dataset.py                # PyTorch Dataset类
└── requirements.txt          # 依赖包
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /home/mlsnrs/data/wrj/Momo
pip3 install -r requirements.txt
```

### 2. 运行数据预处理

```bash
python3 preprocess.py
```

这将：
- 加载受试者10、23、36的数据
- 执行EMG和IMU预处理（滤波、整流、平滑等）
- 分割窗口（200ms窗口，25ms步长）
- 划分训练集和测试集
- 保存为HDF5格式到 `processed_data/` 目录

### 3. 在训练代码中使用数据

```python
from dataset import create_dataloaders

# 创建DataLoader
train_loader, test_loader = create_dataloaders(
    data_dir='processed_data',
    batch_size=32,
    num_workers=4,
    mode='both',          # 'both': EMG+IMU, 'emg': 仅EMG, 'imu': 仅IMU
    load_to_memory=False  # False: 懒加载（推荐）, True: 全部加载到内存
)

# 训练循环
for epoch in range(num_epochs):
    for emg, imu, labels in train_loader:
        # emg: (batch_size, 400, 12)  - EMG数据
        # imu: (batch_size, 400, 36)  - IMU数据
        # labels: (batch_size,)        - 标签 (0-49)
        
        # 您的训练代码...
        pass
```

## 📊 数据格式说明

### HDF5文件结构

每个HDF5文件包含：
- `emg`: EMG数据，形状 (N, 400, 12)
  - N: 样本数
  - 400: 窗口长度（200ms @ 2kHz）
  - 12: EMG通道数
  
- `imu`: IMU数据，形状 (N, 400, 36)
  - N: 样本数
  - 400: 窗口长度
  - 36: IMU通道数（12电极 × 3轴）
  
- `labels`: 标签，形状 (N,)
  - 类别: 0-49（0=休息，1-49=各种手势）

### 数据划分

- **训练集**: 重复 1, 3, 4, 6
- **测试集**: 重复 2, 5

## 🔧 预处理流程

### EMG预处理
1. 带通滤波 (10-500 Hz) - Butterworth 4阶
2. 全波整流
3. RMS平滑 (200ms窗口, 50ms步长)

### IMU预处理
1. 带通滤波 (10-500 Hz) - Butterworth 4阶
2. 高通滤波去趋势 (0.5 Hz)

### 分割与归一化
1. 滑动窗口分割（200ms窗口，25ms步长）
2. 排除运动边界前后100ms的过渡期
3. Z-score标准化（使用训练集统计量）

## 💡 使用技巧

### 懒加载 vs 内存加载

**懒加载（推荐）**：
```python
dataset = NinaproDB2Dataset(
    'processed_data/train_data.h5',
    load_to_memory=False  # 按需加载
)
```
- ✅ 内存占用小
- ✅ 适合大数据集
- ⚠️ 略慢于内存加载

**内存加载**：
```python
dataset = NinaproDB2Dataset(
    'processed_data/train_data.h5',
    load_to_memory=True  # 一次性加载
)
```
- ✅ 训练速度快
- ⚠️ 需要足够内存

### 单模态训练

仅使用EMG：
```python
train_loader, test_loader = create_dataloaders(
    data_dir='processed_data',
    mode='emg'  # 只返回EMG数据
)

for emg, labels in train_loader:
    # emg: (batch_size, 400, 12)
    # labels: (batch_size,)
    pass
```

## 📝 注意事项

1. **HDF5优势**：支持懒加载，配合PyTorch DataLoader效率高
2. **多线程加载**：`num_workers=4` 可加速数据加载
3. **GPU训练**：DataLoader已启用 `pin_memory=True` 加速GPU传输
4. **类别数**：50类（包括休息类）

## 🔍 查看数据集信息

运行预处理后会自动打印详细信息，也可以运行：

```bash
python3 dataset.py
```

查看数据集示例和使用方法。

