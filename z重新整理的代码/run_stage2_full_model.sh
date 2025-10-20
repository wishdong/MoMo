#!/bin/bash

# ============================================================
# 阶段2：完整模型全受试者（DB2, DB3, DB5, DB7）
# GPU 2, 3, 4, 6, 7 并行运行
# 总任务：67个（DB2:40, DB3:6, DB5:10, DB7:11）
# 预计时间：~8.4小时（最慢的GPU决定）
# ============================================================

set +e
set -o pipefail
trap 'echo ""; echo "⚠️  检测到中断信号，正在停止所有后台任务..."; kill $(jobs -p) 2>/dev/null; wait; echo "✅ 所有任务已停止"; exit 130' INT TERM

BASE_ARGS="--batch_size 64 --num_epochs 20 --save-predictions"
LOG_DIR="./experiment_logs"
SUMMARY_DIR="$LOG_DIR/stage_summaries"
mkdir -p $LOG_DIR
mkdir -p $SUMMARY_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$SUMMARY_DIR/stage2_full_model_${TIMESTAMP}.log"
ERROR_LOG="$SUMMARY_DIR/stage2_errors_${TIMESTAMP}.log"

SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

echo "============================================================" | tee $LOG_FILE
echo "📊 阶段2：完整模型全受试者（所有数据集）" | tee -a $LOG_FILE
echo "📅 开始时间: $(date)" | tee -a $LOG_FILE
echo "🖥️  使用GPU: 2, 3, 4, 6, 7" | tee -a $LOG_FILE
echo "💡 提示: Ctrl+C 可以安全停止所有任务" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE

# 检查实验是否已完成
check_completed() {
    local dataset=$1
    local subject=$2
    local exp_id=$3
    
    if [ -f "results/${dataset}/subject${subject}/${exp_id}/metrics.json" ] && \
       [ -f "results/${dataset}/subject${subject}/${exp_id}/predictions.pkl" ]; then
        return 0
    else
        return 1
    fi
}

# 运行单个实验
run_experiment() {
    local dataset=$1
    local subject=$2
    local gpu=$3
    
    local exp_id="M3_full"
    
    if check_completed $dataset $subject $exp_id; then
        echo "  ✓ 跳过: ${dataset} S${subject}" | tee -a $LOG_FILE
        SKIP_COUNT=$((SKIP_COUNT+1))
        return 0
    fi
    
    # 创建数据集和受试者的日志目录
    local exp_log_dir="$LOG_DIR/${dataset}/subject${subject}"
    mkdir -p $exp_log_dir
    
    local exp_log_file="$exp_log_dir/${exp_id}.log"
    
    echo "  ▶ [GPU $gpu] ${dataset} S${subject}" | tee -a $LOG_FILE
    
    python train.py --dataset $dataset --s $subject --gpu $gpu \
        $BASE_ARGS --use-adaptive-fusion --experiment_id $exp_id \
        2>&1 | tee $exp_log_file
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "  ✅ 完成: ${dataset} S${subject}" | tee -a $LOG_FILE
        SUCCESS_COUNT=$((SUCCESS_COUNT+1))
        return 0
    else
        # 醒目的错误提示
        echo "" | tee -a $LOG_FILE
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a $LOG_FILE
        echo "❌❌❌ 实验失败！ ❌❌❌" | tee -a $LOG_FILE
        echo "  数据集: ${dataset}" | tee -a $LOG_FILE
        echo "  受试者: S${subject}" | tee -a $LOG_FILE
        echo "  实验ID: ${exp_id}" | tee -a $LOG_FILE
        echo "  错误码: ${exit_code}" | tee -a $LOG_FILE
        echo "  日志文件: ${exp_log_file}" | tee -a $LOG_FILE
        echo "  查看错误: tail -50 ${exp_log_file}" | tee -a $LOG_FILE
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE
        
        echo "[$(date)] FAILED: ${dataset} S${subject} M3_full (exit_code: ${exit_code})" >> $ERROR_LOG
        echo "  Log: ${exp_log_file}" >> $ERROR_LOG
        
        FAIL_COUNT=$((FAIL_COUNT+1))
        return 1
    fi
}

# ============================================================
# GPU 2: DB2[1-16] (16任务)
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 2] DB2[1-16] 开始..." | tee -a $LOG_FILE
    
    for subject in {1..16}; do
        echo "[GPU 2] 进度: $((subject))/16" | tee -a $LOG_FILE
        run_experiment DB2 $subject 2
    done
    
    echo "[GPU 2] ✅ DB2[1-16] 完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 3: DB2[17-32] (16任务)
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 3] DB2[17-32] 开始..." | tee -a $LOG_FILE
    
    for subject in {17..32}; do
        echo "[GPU 3] 进度: $((subject-16))/16" | tee -a $LOG_FILE
        run_experiment DB2 $subject 3
    done
    
    echo "[GPU 3] ✅ DB2[17-32] 完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 4: DB2[33-40] + DB7全部 (8+11=19任务)
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 4] DB2[33-40] + DB7全部 开始..." | tee -a $LOG_FILE
    
    # DB2[33-40]
    for subject in {33..40}; do
        echo "[GPU 4] 进度: $((subject-32))/19" | tee -a $LOG_FILE
        run_experiment DB2 $subject 4
    done
    
    # DB7全部（S2-S12）
    for subject in {2..12}; do
        echo "[GPU 4] 进度: $((subject+7))/19" | tee -a $LOG_FILE
        run_experiment DB7 $subject 4
    done
    
    echo "[GPU 4] ✅ DB2[33-40] + DB7全部 完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 6: DB3全部 (6任务)
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 6] DB3全部 开始..." | tee -a $LOG_FILE
    
    # DB3全部（S2, S4, S5, S6, S9, S11）
    COUNT=0
    for subject in 2 4 5 6 9 11; do
        COUNT=$((COUNT+1))
        echo "[GPU 6] 进度: ${COUNT}/6" | tee -a $LOG_FILE
        run_experiment DB3 $subject 6
    done
    
    echo "[GPU 6] ✅ DB3全部 完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 7: DB5[1-10] (10任务)
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 7] DB5[1-10] 开始..." | tee -a $LOG_FILE
    
    for subject in {1..10}; do
        echo "[GPU 7] 进度: ${subject}/10" | tee -a $LOG_FILE
        run_experiment DB5 $subject 7
    done
    
    echo "[GPU 7] ✅ DB5[1-10] 完成！" | tee -a $LOG_FILE
} &

# 等待所有GPU完成
wait

echo "" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "🎉 阶段2完成：完整模型全受试者" | tee -a $LOG_FILE
echo "📅 结束时间: $(date)" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE

# 统计完成情况
echo "" | tee -a $LOG_FILE
echo "📊 执行统计：" | tee -a $LOG_FILE
echo "  ✅ 成功: ${SUCCESS_COUNT}" | tee -a $LOG_FILE
echo "  ❌ 失败: ${FAIL_COUNT}" | tee -a $LOG_FILE
echo "  ⏭️  跳过: ${SKIP_COUNT}" | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "📂 结果统计：" | tee -a $LOG_FILE
DB2_COUNT=$(ls -1 results/DB2/subject*/M3_full/metrics.json 2>/dev/null | wc -l)
DB3_COUNT=$(ls -1 results/DB3/subject*/M3_full/metrics.json 2>/dev/null | wc -l)
DB5_COUNT=$(ls -1 results/DB5/subject*/M3_full/metrics.json 2>/dev/null | wc -l)
DB7_COUNT=$(ls -1 results/DB7/subject*/M3_full/metrics.json 2>/dev/null | wc -l)
echo "  DB2: ${DB2_COUNT} / 40" | tee -a $LOG_FILE
echo "  DB3: ${DB3_COUNT} / 6" | tee -a $LOG_FILE
echo "  DB5: ${DB5_COUNT} / 10" | tee -a $LOG_FILE
echo "  DB7: ${DB7_COUNT} / 11" | tee -a $LOG_FILE
echo "  总计: $((DB2_COUNT + DB3_COUNT + DB5_COUNT + DB7_COUNT)) / 67" | tee -a $LOG_FILE

if [ $FAIL_COUNT -gt 0 ]; then
    echo "" | tee -a $LOG_FILE
    echo "⚠️  发现 ${FAIL_COUNT} 个失败的实验，详见: $ERROR_LOG" | tee -a $LOG_FILE
fi

