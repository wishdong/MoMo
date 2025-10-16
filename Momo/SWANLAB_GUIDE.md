# SwanLab监控配置指南

## 📊 SwanLab简介

SwanLab是一个优秀的机器学习实验管理平台，可以帮助您：
- 跟踪实验参数和指标
- 可视化训练过程
- 对比不同实验结果
- 管理模型版本

## 🚀 快速开始

### 1. 环境准备

SwanLab已经安装在Momo环境中：
```bash
conda activate Momo
```

### 2. 注册SwanLab账号

访问 [SwanLab官网](https://swanlab.cn) 注册账号（可选，本地模式无需注册）

### 3. 运行训练（SwanLab版本）

使用集成SwanLab的训练脚本：
```bash
# 激活环境
conda activate Momo

# 进入项目目录
cd /home/mlsnrs/data/wrj/MoMo/Momo

# 运行训练
bash run_train_swanlab.sh
```

或者手动运行：
```bash
python3 train_swanlab.py \
    --data_dir processed_data \
    --subject 10 \
    --use_swanlab \
    --swanlab_project "Momo-Gesture-Recognition"
```

## 📋 SwanLab功能特性

### 🎯 自动记录的指标

**训练指标**：
- `train_loss` - 训练总损失
- `train_loss_emg` - EMG分支损失
- `train_loss_imu` - IMU分支损失 
- `train_loss_fusion` - 融合分支损失
- `train_acc_emg` - EMG分支准确率
- `train_acc_imu` - IMU分支准确率
- `train_acc_fusion` - 融合分支准确率
- `train_acc_final` - 最终准确率
- `learning_rate` - 学习率
- `epoch_time` - 每轮训练时间

**测试指标**：
- `test_loss` - 测试总损失
- `test_loss_emg/imu/fusion` - 各分支测试损失
- `test_acc_emg/imu/fusion/final` - 各分支测试准确率

**最佳记录**：
- `best_accuracy` - 历史最佳准确率
- `best_epoch` - 最佳准确率对应的轮次

### ⚙️ 自动记录的配置

**模型配置**：
- 模型架构参数
- 各通道数量
- 特征维度设置

**训练配置**：
- 学习率、批次大小
- 优化器、调度器设置
- 损失函数权重

**数据配置**：
- 受试者信息
- 窗口大小
- 设备信息

## 🎮 使用参数

### SwanLab相关参数

```bash
# 启用/禁用SwanLab
--use_swanlab              # 启用SwanLab监控
--no-use_swanlab          # 禁用SwanLab监控

# SwanLab项目配置
--swanlab_project "项目名"  # 设置SwanLab项目名称
--swanlab_logdir "路径"    # 本地日志目录（可选）
```

### 同时使用TensorBoard和SwanLab

```bash
python3 train_swanlab.py \
    --use_tensorboard \    # 启用TensorBoard
    --use_swanlab \        # 启用SwanLab
    --swanlab_project "Momo-MultiModal-Gesture"
```

## 📁 输出文件结构

使用SwanLab后，会生成以下目录结构：

```
Momo/
├── checkpoints/S10_swanlab/     # 模型检查点
│   ├── best_checkpoint.pth      # 最佳模型
│   ├── latest_checkpoint.pth    # 最新模型  
│   └── checkpoint_epoch_*.pth   # 定期保存
├── logs/S10_swanlab/           # TensorBoard日志
└── swanlab/                    # SwanLab本地日志（如果指定）
```

## 💡 使用技巧

### 1. 实验命名策略

实验名称自动生成格式：
```
RMSCM_S{受试者}_{优化器}_lr{学习率}_{时间戳}
```

例如：`RMSCM_S10_adam_lr0.001_20241011_143025`

### 2. 多受试者对比实验

```bash
# 受试者10
python3 train_swanlab.py --subject 10 --swanlab_project "Momo-LOSO"

# 受试者23  
python3 train_swanlab.py --subject 23 --swanlab_project "Momo-LOSO"

# 受试者36
python3 train_swanlab.py --subject 36 --swanlab_project "Momo-LOSO"
```

### 3. 超参数对比实验

```bash
# 不同学习率
python3 train_swanlab.py --lr 0.001 --swanlab_project "Momo-Hyperparams"
python3 train_swanlab.py --lr 0.01 --swanlab_project "Momo-Hyperparams" 
python3 train_swanlab.py --lr 0.0001 --swanlab_project "Momo-Hyperparams"

# 不同批次大小
python3 train_swanlab.py --batch_size 16 --swanlab_project "Momo-BatchSize"
python3 train_swanlab.py --batch_size 32 --swanlab_project "Momo-BatchSize"
python3 train_swanlab.py --batch_size 64 --swanlab_project "Momo-BatchSize"
```

### 4. 本地模式运行

如果不想上传到云端，可以使用本地模式：
```bash
# 方式1：设置环境变量
export SWANLAB_MODE=disabled
python3 train_swanlab.py --use_swanlab

# 方式2：修改代码中的mode参数
# swanlab.init(..., mode='disabled')
```

## 🔍 监控界面

SwanLab会提供以下监控功能：

### 实时图表
- 损失曲线（训练/测试）
- 准确率曲线（各模态）
- 学习率变化
- 训练时间统计

### 实验对比
- 多个实验的指标对比
- 参数配置对比
- 最佳结果汇总

### 系统监控
- GPU/CPU使用率
- 内存占用
- 训练进度

## 🛠 故障排除

### 常见问题

1. **SwanLab初始化失败**
   ```bash
   # 检查网络连接
   ping swanlab.cn
   
   # 使用本地模式
   export SWANLAB_MODE=disabled
   ```

2. **日志上传缓慢**
   ```bash
   # 使用本地目录保存
   python3 train_swanlab.py --swanlab_logdir "./swanlab_logs"
   ```

3. **实验名称冲突**
   - SwanLab会自动处理同名实验
   - 或手动指定唯一的实验名称

### 调试模式

启用详细日志：
```bash
export SWANLAB_DEBUG=1
python3 train_swanlab.py --use_swanlab
```

## 📚 更多资源

- [SwanLab官方文档](https://docs.swanlab.cn)
- [SwanLab GitHub](https://github.com/SwanHubX/SwanLab)
- [PyTorch集成示例](https://docs.swanlab.cn/zh/guide_cloud/integration/integration-pytorch.html)

## 🎊 总结

通过SwanLab监控，您可以：

✅ **实时监控**训练过程中的各项指标
✅ **可视化对比**不同实验的效果 
✅ **自动记录**实验配置和结果
✅ **团队协作**共享实验结果
✅ **版本管理**跟踪模型演进过程

开始您的SwanLab监控之旅吧！🚀
