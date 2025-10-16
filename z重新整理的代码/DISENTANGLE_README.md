# 解纠缠模块使用说明

## 📌 概述

本项目实现了基于解纠缠表征学习的多模态手势识别系统。通过将EMG和IMU特征分解为**共享表征**和**独特表征**，模型能够更好地捕获跨模态的共性和模态特有的信息。

---

## 🏗️ 架构设计

### 数据流

```
输入:
  X1 (EMG): [batch, 400, 12, 1]
  X2 (IMU): [batch, 400, 36, 1]
        ↓
编码器 (原有):
  H1 (EMG特征): [batch, 32, 12, 1]
  H2 (IMU特征): [batch, 32, 36, 1]
        ↓
解纠缠编码器 (新增):
  H1 → Z1 (共享) [batch, 128] + U1 (独特) [batch, 64]
  H2 → Z2 (共享) [batch, 128] + U2 (独特) [batch, 64]
        ↓
融合分类:
  concat([Z1, U1, Z2, U2]) → [batch, 384] → 分类器 → [batch, 50]
```

### 损失函数

#### 独特损失 (L_private)
```
L_private = -λ1*(I(U1;Y) + I(U2;Y))        # 最大化任务相关性
          + λ2*(I(U1;Z1) + I(U2;Z2))       # 独特-共享独立
```

#### 共享损失 (L_shared)
```
L_shared = λ3*d(Z1,Z2)                     # 共享表征对齐
         - λ4*(I(Z1;Y) + I(Z2;Y))          # 最大化任务相关性
         + λ5*R_diversity(Z1,Z2)           # 多样性正则（防坍缩）
```

#### 总损失
```
L_total = L_cls + α*L_private + β*L_shared
```

---

## 🚀 快速开始

### 1. 基础训练（不使用解纠缠）

```bash
python train.py --s 10 --gpu 0
```

### 2. 使用解纠缠训练

```bash
python train.py --s 10 --gpu 0 --use-disentangle
```

### 3. 自定义解纠缠参数

```bash
python train.py --s 10 --gpu 0 \
    --use-disentangle \
    --d-shared 128 \
    --d-private 64 \
    --alpha 0.5 \
    --beta 0.5
```

---

## ⚙️ 参数说明

### 解纠缠相关参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--use-disentangle` | flag | False | 是否启用解纠缠损失 |
| `--d-shared` | int | 128 | 共享表征维度 |
| `--d-private` | int | 64 | 独特表征维度 |
| `--alpha` | float | 0.5 | L_private总权重 |
| `--beta` | float | 0.5 | L_shared总权重 |

### 配置文件参数 (config.py)

在 `DisentangleConfigs` 类中可以调整更细粒度的参数：

```python
class DisentangleConfigs:
    # 编码器维度
    d_shared = 128              # 共享表征维度
    d_private = 64              # 独特表征维度
    dropout = 0.3               # Dropout率
    
    # 损失权重
    lambda1 = 0.1               # I(Ui;Y) 任务相关性
    lambda2 = 0.5               # I(Ui;Zi) 独立性
    lambda3 = 1.0               # d(Z1,Z2) 对齐
    lambda4 = 0.1               # I(Zi;Y) 任务相关性
    lambda5 = 0.05              # R_diversity 多样性
    
    # 总权重
    alpha = 0.5                 # L_private总权重
    beta = 0.5                  # L_shared总权重
    
    # InfoNCE参数
    temperature = 0.07          # 温度参数
    
    # 训练策略
    warmup_epochs = 0           # 预热轮数
```

---

## 📊 实验对比

### 消融实验

```bash
# 1. 基础模型（无解纠缠）
python train.py --s 10 --gpu 0

# 2. 完整解纠缠模型
python train.py --s 10 --gpu 0 --use-disentangle

# 3. 只使用对齐损失
python train.py --s 10 --gpu 0 --use-disentangle --alpha 0 --beta 1

# 4. 只使用独立性损失
python train.py --s 10 --gpu 0 --use-disentangle --alpha 1 --beta 0
```

### 超参数搜索

```bash
# 不同的α和β组合
for alpha in 0.1 0.5 1.0; do
  for beta in 0.1 0.5 1.0; do
    python train.py --s 10 --gpu 0 \
      --use-disentangle \
      --alpha $alpha \
      --beta $beta
  done
done
```

---

## 📁 新增文件说明

### 1. `models/disentangle_modules.py`
- `SharedEncoder`: 共享编码器（提取跨模态共享信息）
- `PrivateEncoder`: 独特编码器（提取模态特有信息）
- `ProjectionHead`: 投影头（可选，用于对比学习）

### 2. `models/mutual_info_estimators.py`
- `InfoNCE`: 互信息下界估计（用于最大化）
- `CLUB`: 互信息上界估计（用于最小化）
- `HSIC`: 核独立性度量
- `BarlowTwins`: 去冗余正则
- `CovarianceRegularizer`: 协方差正则（轻量级）

### 3. `models/disentangle_loss.py`
- `DisentangleLoss`: 完整的解纠缠损失函数
- `SimplifiedDisentangleLoss`: 简化版（用于快速实验）

### 4. `models/model.py` (修改)
- 新增 `MultimodalGestureNetWithDisentangle` 类
- 原有 `MultimodalGestureNet` 完全保留

### 5. `config.py` (修改)
- 新增 `DisentangleConfigs` 配置类

### 6. `trainer.py` (修改)
- 添加 `use_disentangle` 和 `disentangle_config` 参数
- 添加解纠缠损失计算和监控
- 向后兼容原有训练流程

### 7. `train.py` (修改)
- 添加解纠缠相关命令行参数
- 根据参数选择模型类型

---

## 🔍 监控指标

使用SwanLab可以监控以下指标：

### 基础指标
- `train_loss` / `val_loss`: 总损失
- `train_acc` / `val_acc`: 总体准确率
- `train_acc_emg` / `val_acc_emg`: EMG分支准确率
- `train_acc_imu` / `val_acc_imu`: IMU分支准确率

### 解纠缠指标（启用解纠缠时）
- `train_loss_private` / `val_loss_private`: 独特损失
- `train_loss_shared` / `val_loss_shared`: 共享损失

---

## 💡 使用建议

### 1. 超参数调优顺序

1. **先确定编码器维度**：
   - 从默认值开始：`d_shared=128`, `d_private=64`
   - 如果模型容量不足，可以增大维度

2. **调整总权重α和β**：
   - 从 `α=0.5, β=0.5` 开始
   - 如果分类性能下降，降低α和β
   - 如果解纠缠效果不明显，增大α和β

3. **微调细粒度权重λ1-λ5**：
   - 在 `config.py` 中调整
   - 建议先固定大部分权重，每次只调一个

### 2. 训练策略

- **Warmup**: 如果训练不稳定，可以设置 `warmup_epochs > 0`，前几轮只用分类损失
- **学习率**: 解纠缠模型可能需要更小的学习率（如0.0003）
- **Batch size**: 对比学习需要较大的batch size（建议≥64）

### 3. 调试技巧

- 使用 `SimplifiedDisentangleLoss` 快速验证流程
- 检查SwanLab中的损失曲线，确保各项损失都在合理范围
- 如果某项损失为NaN，检查对应的互信息估计器

---

## 🐛 常见问题

### Q1: 训练时出现NaN
**A**: 可能原因：
- InfoNCE温度参数过小，导致exp溢出
- CLUB网络输出的log_var范围过大
- 学习率过大

**解决方案**：
- 增大 `temperature`（如0.1）
- 降低学习率
- 添加梯度裁剪

### Q2: 解纠缠效果不明显
**A**: 可能原因：
- α和β权重过小
- 独立性约束不够强

**解决方案**：
- 增大 `alpha` 和 `beta`
- 增大 `lambda2`（独立性权重）
- 检查Z和U的相关性（可视化）

### Q3: 分类性能下降
**A**: 可能原因：
- 解纠缠约束过强，损失了任务相关信息
- 融合分类器容量不足

**解决方案**：
- 降低 `alpha` 和 `beta`
- 增大 `lambda1` 和 `lambda4`（任务相关性）
- 增大 `d_shared` 和 `d_private`

---

## 📚 参考文献

1. **Contrastive Modality-Disentangled Learning for Multimodal Recommendation**
   - 对比模态解纠缠学习
   - 互信息上界近似

2. **Factorized Contrastive Learning: Going Beyond Multi-view Redundancy**
   - 因子化对比学习
   - 去冗余约束

3. **InfoMAE: Pairing-Efficient Cross-Modal Alignment**
   - 信息掩码自编码器
   - 跨模态对齐

---

## 📧 联系方式

如有问题，请提Issue或联系开发者。

---

## 📝 更新日志

### v1.0 (2025-10-13)
- ✅ 实现完整的解纠缠框架
- ✅ 支持多种互信息估计方法
- ✅ 向后兼容原有代码
- ✅ 添加SwanLab监控
- ✅ 提供丰富的配置选项

