# EMG-IMU 多模态手势识别

基于EMG（肌电信号）和IMU（惯性测量单元）的多模态手势识别项目。

## 📌 项目结构

```
z重新整理的代码/
├── config.py          # 配置文件（EMG和IMU的模型配置）
├── model.py           # 模型定义（三路并行网络架构）
├── data_loader.py     # 数据加载模块
├── trainer.py         # 训练模块（分离式训练策略）
├── train.py           # 主训练脚本（训练入口）
├── test.py            # 测试/推理脚本
└── README.md          # 项目说明文档
```

## 🏗️ 模型架构

### 三路并行架构

```
输入:
  ├─ EMG: [batch, 400, 12, 1]   # 12通道肌电信号
  └─ IMU: [batch, 400, 36, 1]   # 36通道IMU信号（12传感器×3轴）

编码器:
  ├─ EMG编码器 → [batch, 64, 12, 1]
  ├─ IMU编码器 → [batch, 64, 36, 1]
  └─ 融合编码器 → [batch, 64, 72, 1]

分类器:
  ├─ EMG分类器 → [batch, 50]
  ├─ IMU分类器 → [batch, 50]
  └─ 融合分类器 → [batch, 50]

输出:
  └─ 三路加权求和 → [batch, 50] → 50类手势分类
```

### 编码器内部结构

每个编码器包含：
1. **时序层1**（Temporal_layer）：GLU门控机制 + BatchNorm
2. **空间层1**（Spatial_layer）：SEBlock注意力 + 自适应GNN（切比雪夫K=1 + 多头注意力）
3. **时序层2**（Temporal_layer）：+ BatchNorm

### 核心特性

- ✅ **自适应邻接矩阵**：通过多头注意力学习传感器间关系
- ✅ **分离式训练**：三个模块独立优化，避免模态间干扰
- ✅ **detach机制**：融合分支不影响encoder训练
- ✅ **时空解耦**：Temporal层和Spatial层交替提取特征

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch numpy
```

### 2. 准备数据

数据格式为`.pt`文件，包含以下字段：
- `train_examples_datasets`: 训练集EMG数据
- `train_acc_datasets`: 训练集IMU数据
- `train_labels_datasets`: 训练集标签
- `test_examples_datasets`: 验证集EMG数据
- `test_acc_datasets`: 验证集IMU数据
- `test_labels_datasets`: 验证集标签
- `class_counts`: 各类别样本数量

### 3. 训练模型

```bash
# 基础训练
python train.py --subject 24 --gpu 0

# 自定义参数
python train.py \
    --subject 24 \
    --data_path /path/to/DB2_s{subject}allseg40020mZsc_rest.pt \
    --batch_size 64 \
    --num_epochs 500 \
    --gpu 0 \
    --save_dir ./weights
```

### 4. 测试模型

```bash
python test.py \
    --subject 24 \
    --model_path ./weights/subject24_best.pt \
    --gpu 0
```

## ⚙️ 训练配置

### 核心超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 64 | 批次大小 |
| num_epochs | 500 | 最大训练轮数 |
| learning_rate | 0.01 | 学习率（SGD优化器） |
| patience | 15 | 早停初始耐心值 |
| patience_increase | 10 | 找到更好模型时增加的耐心值 |

### 优化策略

- **优化器**：SGD（所有优化器lr=0.01）
- **损失函数**：CrossEntropyLoss
- **学习率调度**：ReduceLROnPlateau（监控验证准确率，factor=0.2，patience=15）
- **早停策略**：验证准确率不再提升时停止训练

### 分离式训练

使用三个独立优化器：
```python
optimizer_imu = SGD(imu_encoder + imu_classifier, lr=0.01)
optimizer_emg = SGD(emg_encoder + emg_classifier, lr=0.01)
optimizer_fusion = SGD(fusion_encoder + fusion_classifier, lr=0.01)
```

反向传播顺序：
```python
imu_loss.backward(retain_graph=True)
optimizer_imu.step()

emg_loss.backward(retain_graph=True)
optimizer_emg.step()

fusion_loss.backward()
optimizer_fusion.step()
```

## 📊 实验结果

受试者24的训练日志示例：
```
训练集大小: EMG (8000, 400, 12, 1), IMU (8000, 400, 36, 1)
验证集大小: EMG (4000, 400, 12, 1), IMU (4000, 400, 36, 1)
模型参数总数: ~X,XXX,XXX
训练完成，耗时 XX分 XX秒
最佳验证准确率: XX.XX%
```

## 🔍 代码说明

### config.py
定义EMG和IMU两种模态的配置参数。

### model.py
包含所有模型组件：
- 基础层：ChebConv, GNN, Temporal_layer, SEBlock, Spatial_layer
- 编码器：Encoder
- 融合模块：CrossAttentionFusion, FusionEncoder
- 分类器：Classifier
- 完整模型：MultimodalGestureNet

### trainer.py
训练核心逻辑，实现分离式训练策略。

### train.py
训练入口脚本，处理参数解析和训练流程。

## 📝 注意事项

1. **数据路径**：请修改`train.py`中的`--data_path`参数指向你的数据文件
2. **GPU设置**：使用`--gpu`参数指定GPU编号
3. **随机种子**：默认seed=0，可通过`--seed`修改以复现实验

## 📄 引用

如果使用本代码，请引用相关论文。

## 📧 联系方式

如有问题请联系：[your-email@example.com]

