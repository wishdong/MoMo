#!/bin/bash

# ============================================================
# 主控脚本：依次运行所有阶段
# ============================================================

LOG_DIR="./experiment_logs"
mkdir -p $LOG_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/main_${TIMESTAMP}.log"

echo "============================================================" | tee $MAIN_LOG
echo "🚀 批量实验主控脚本" | tee -a $MAIN_LOG
echo "📅 开始时间: $(date)" | tee -a $MAIN_LOG
echo "🖥️  使用GPU: 2, 3, 4, 6, 7" | tee -a $MAIN_LOG
echo "============================================================" | tee -a $MAIN_LOG

# 给脚本添加执行权限
chmod +x run_stage1_ablation.sh
chmod +x run_stage2_full_model.sh
chmod +x run_stage3_hyperparam_grid.sh
chmod +x run_stage4_hyperparam_lambda.sh
chmod +x run_aggregate.sh

# ============================================================
# 阶段1：消融实验
# ============================================================
echo "" | tee -a $MAIN_LOG
echo "▶▶▶ 开始阶段1：消融实验" | tee -a $MAIN_LOG
./run_stage1_ablation.sh

if [ $? -eq 0 ]; then
    echo "✅ 阶段1完成" | tee -a $MAIN_LOG
else
    echo "❌ 阶段1失败" | tee -a $MAIN_LOG
    echo "是否继续下一阶段？(y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# 检查进度
python check_progress.py --stage stage1

# ============================================================
# 阶段2：完整模型全受试者
# ============================================================
echo "" | tee -a $MAIN_LOG
echo "▶▶▶ 开始阶段2：完整模型全受试者" | tee -a $MAIN_LOG
./run_stage2_full_model.sh

if [ $? -eq 0 ]; then
    echo "✅ 阶段2完成" | tee -a $MAIN_LOG
else
    echo "❌ 阶段2失败" | tee -a $MAIN_LOG
fi

# 检查进度
python check_progress.py --stage stage2

# ============================================================
# 阶段3：超参数网格搜索
# ============================================================
echo "" | tee -a $MAIN_LOG
echo "▶▶▶ 开始阶段3：超参数网格搜索（α和β）" | tee -a $MAIN_LOG
./run_stage3_hyperparam_grid.sh

if [ $? -eq 0 ]; then
    echo "✅ 阶段3完成" | tee -a $MAIN_LOG
else
    echo "❌ 阶段3失败" | tee -a $MAIN_LOG
fi

# 检查进度
python check_progress.py --stage stage3

# ============================================================
# 阶段4：超参数搜索（λ）
# ============================================================
echo "" | tee -a $MAIN_LOG
echo "▶▶▶ 开始阶段4：超参数搜索（λ）" | tee -a $MAIN_LOG
./run_stage4_hyperparam_lambda.sh

if [ $? -eq 0 ]; then
    echo "✅ 阶段4完成" | tee -a $MAIN_LOG
else
    echo "❌ 阶段4失败" | tee -a $MAIN_LOG
fi

# 检查进度
python check_progress.py --stage stage4

# ============================================================
# 阶段5：聚合评估
# ============================================================
echo "" | tee -a $MAIN_LOG
echo "▶▶▶ 开始阶段5：聚合评估" | tee -a $MAIN_LOG
./run_aggregate.sh

if [ $? -eq 0 ]; then
    echo "✅ 阶段5完成" | tee -a $MAIN_LOG
else
    echo "❌ 阶段5失败" | tee -a $MAIN_LOG
fi

# ============================================================
# 最终统计
# ============================================================
echo "" | tee -a $MAIN_LOG
echo "============================================================" | tee -a $MAIN_LOG
echo "🎉 所有阶段完成！" | tee -a $MAIN_LOG
echo "📅 结束时间: $(date)" | tee -a $MAIN_LOG
echo "============================================================" | tee -a $MAIN_LOG

# 最终进度检查
python check_progress.py --stage all

echo "" | tee -a $MAIN_LOG
echo "📁 结果位置：" | tee -a $MAIN_LOG
echo "  - 单个实验: ./results/{dataset}/subject{id}/{experiment_id}/" | tee -a $MAIN_LOG
echo "  - 聚合结果: ./results/{dataset}/aggregated/{experiment_id}/" | tee -a $MAIN_LOG
echo "  - 模型权重: ./weights/{dataset}/subject{id}/{experiment_id}.pt" | tee -a $MAIN_LOG
echo "  - 实验日志: ./experiment_logs/" | tee -a $MAIN_LOG

