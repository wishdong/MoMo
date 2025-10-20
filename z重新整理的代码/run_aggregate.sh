#!/bin/bash

# ============================================================
# 阶段5：聚合评估
# 为每个数据集的每个实验类型生成聚合混淆矩阵
# ============================================================

LOG_DIR="./experiment_logs"
SUMMARY_DIR="$LOG_DIR/stage_summaries"
AGGREGATE_LOG_DIR="$LOG_DIR/aggregate"
mkdir -p $LOG_DIR
mkdir -p $SUMMARY_DIR
mkdir -p $AGGREGATE_LOG_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$SUMMARY_DIR/stage5_aggregate_${TIMESTAMP}.log"

echo "============================================================" | tee $LOG_FILE
echo "📈 阶段5：聚合评估" | tee -a $LOG_FILE
echo "📅 开始时间: $(date)" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE

# ============================================================
# 聚合主要实验的结果
# ============================================================

echo "" | tee -a $LOG_FILE
echo "🔄 聚合消融实验结果..." | tee -a $LOG_FILE

# DB5消融实验（受试者1,5,10）
for exp_id in M0_base M1_disentangle M3_full D1_private_only D2_shared_only FA1_no_constraint FA2_align_only FA3_balance_only; do
    echo "  聚合: DB5 - ${exp_id}" | tee -a $LOG_FILE
    python train.py --dataset DB5 --aggregate-results --use-disentangle \
        --experiment_id $exp_id --aggregate-subjects "1,5,10" \
        2>&1 | tee $AGGREGATE_LOG_DIR/DB5_${exp_id}.log
done

# DB7消融实验（受试者3,7,11）
for exp_id in M0_base M1_disentangle M3_full D1_private_only D2_shared_only FA1_no_constraint FA2_align_only FA3_balance_only; do
    echo "  聚合: DB7 - ${exp_id}" | tee -a $LOG_FILE
    python train.py --dataset DB7 --aggregate-results --use-disentangle \
        --experiment_id $exp_id --aggregate-subjects "3,7,11" \
        2>&1 | tee $AGGREGATE_LOG_DIR/DB7_${exp_id}.log
done

# ============================================================
# 聚合完整模型（全受试者）
# ============================================================

echo "" | tee -a $LOG_FILE
echo "🔄 聚合完整模型结果（全受试者）..." | tee -a $LOG_FILE

# DB2全受试者（1-40）
echo "  聚合: DB2 - M3_full (40个受试者)" | tee -a $LOG_FILE
python train.py --dataset DB2 --aggregate-results --use-adaptive-fusion \
    --experiment_id M3_full --aggregate-subjects "all" \
    2>&1 | tee $AGGREGATE_LOG_DIR/DB2_M3_full.log

# DB3全受试者（1-11）
echo "  聚合: DB3 - M3_full (11个受试者)" | tee -a $LOG_FILE
python train.py --dataset DB3 --aggregate-results --use-adaptive-fusion \
    --experiment_id M3_full --aggregate-subjects "all" \
    2>&1 | tee $AGGREGATE_LOG_DIR/DB3_M3_full.log

# DB5全受试者（1-10）
echo "  聚合: DB5 - M3_full (10个受试者)" | tee -a $LOG_FILE
python train.py --dataset DB5 --aggregate-results --use-adaptive-fusion \
    --experiment_id M3_full --aggregate-subjects "all" \
    2>&1 | tee $AGGREGATE_LOG_DIR/DB5_M3_full.log

# DB7全受试者（1-22）
echo "  聚合: DB7 - M3_full (22个受试者)" | tee -a $LOG_FILE
python train.py --dataset DB7 --aggregate-results --use-adaptive-fusion \
    --experiment_id M3_full --aggregate-subjects "all" \
    2>&1 | tee $AGGREGATE_LOG_DIR/DB7_M3_full.log

# ============================================================
# 聚合超参数搜索结果（可选，数据量大）
# ============================================================

echo "" | tee -a $LOG_FILE
echo "🔄 聚合超参数搜索结果..." | tee -a $LOG_FILE

# 只聚合几个关键的超参数组合（避免生成太多图）
KEY_HP_COMBOS=(
    "HP_a0.0_b0.0"
    "HP_a0.3_b0.3"
    "HP_a0.5_b0.5"
    "HP_a0.7_b0.7"
    "HP_a1.0_b1.0"
)

for exp_id in "${KEY_HP_COMBOS[@]}"; do
    echo "  聚合: DB3 - ${exp_id}" | tee -a $LOG_FILE
    python train.py --dataset DB3 --aggregate-results --use-disentangle \
        --experiment_id $exp_id --aggregate-subjects "2,6,11" \
        2>&1 | tee $AGGREGATE_LOG_DIR/DB3_${exp_id}.log
    
    echo "  聚合: DB5 - ${exp_id}" | tee -a $LOG_FILE
    python train.py --dataset DB5 --aggregate-results --use-disentangle \
        --experiment_id $exp_id --aggregate-subjects "1,5,10" \
        2>&1 | tee $AGGREGATE_LOG_DIR/DB5_${exp_id}.log
done

echo "" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "🎉 阶段5完成：聚合评估" | tee -a $LOG_FILE
echo "📅 结束时间: $(date)" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE

# 列出生成的聚合结果
echo "" | tee -a $LOG_FILE
echo "📁 生成的聚合结果：" | tee -a $LOG_FILE
echo "  DB2: $(ls -d results/DB2/aggregated/*/ 2>/dev/null | wc -l) 个" | tee -a $LOG_FILE
echo "  DB3: $(ls -d results/DB3/aggregated/*/ 2>/dev/null | wc -l) 个" | tee -a $LOG_FILE
echo "  DB5: $(ls -d results/DB5/aggregated/*/ 2>/dev/null | wc -l) 个" | tee -a $LOG_FILE
echo "  DB7: $(ls -d results/DB7/aggregated/*/ 2>/dev/null | wc -l) 个" | tee -a $LOG_FILE

