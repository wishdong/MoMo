#!/bin/bash

# ============================================================
# 阶段4：超参数搜索（λ_align和λ_balance）
# GPU 0, 1, 2, 3, 4, 6 并行运行
# 总任务：72个（DB2, DB3, DB5各3个代表性受试者）
# 预计时间：~6-12小时
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
LOG_FILE="$SUMMARY_DIR/stage4_hyperparam_lambda_${TIMESTAMP}.log"
ERROR_LOG="$SUMMARY_DIR/stage4_errors_${TIMESTAMP}.log"

SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

echo "============================================================" | tee $LOG_FILE
echo "🔬 阶段4：超参数搜索（λ_align和λ_balance）" | tee -a $LOG_FILE
echo "📅 开始时间: $(date)" | tee -a $LOG_FILE
echo "🖥️  使用GPU: 0, 1, 2, 3, 4, 6" | tee -a $LOG_FILE
echo "📊 数据集: DB2, DB3, DB5" | tee -a $LOG_FILE
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

# 检查是否可以复用
can_reuse_lambda() {
    local param_name=$1  # "align" 或 "balance"
    local param_value=$2
    local dataset=$3
    local subject=$4
    
    if [ "$param_name" == "align" ]; then
        if [ "$param_value" == "0.0" ]; then
            # λ_align=0 → 复用FA3_balance_only
            check_completed $dataset $subject "FA3_balance_only"
            return $?
        elif [ "$param_value" == "0.1" ]; then
            # λ_align=0.1, λ_balance=0.05 → 复用M3_full
            check_completed $dataset $subject "M3_full"
            return $?
        fi
    elif [ "$param_name" == "balance" ]; then
        if [ "$param_value" == "0.0" ]; then
            # λ_balance=0 → 复用FA2_align_only
            check_completed $dataset $subject "FA2_align_only"
            return $?
        elif [ "$param_value" == "0.05" ]; then
            # λ_align=0.1, λ_balance=0.05 → 复用M3_full
            check_completed $dataset $subject "M3_full"
            return $?
        fi
    fi
    
    return 1  # 不能复用
}

# 运行λ_align实验
run_lambda_align() {
    local dataset=$1
    local subject=$2
    local lambda_align=$3
    local gpu=$4
    
    local exp_id="HP_align${lambda_align}"
    
    if check_completed $dataset $subject $exp_id; then
        SKIP_COUNT=$((SKIP_COUNT+1))
        return 0
    fi
    
    if can_reuse_lambda "align" $lambda_align $dataset $subject; then
        echo "  ✓ 复用: ${dataset} S${subject} λ_align=${lambda_align}" | tee -a $LOG_FILE
        SKIP_COUNT=$((SKIP_COUNT+1))
        return 0
    fi
    
    # 创建数据集和受试者的日志目录
    local exp_log_dir="$LOG_DIR/${dataset}/subject${subject}"
    mkdir -p $exp_log_dir
    
    local exp_log_file="$exp_log_dir/${exp_id}.log"
    
    echo "  ▶ [GPU $gpu] ${dataset} S${subject} λ_align=${lambda_align}" | tee -a $LOG_FILE
    
    python train.py --dataset $dataset --s $subject --gpu $gpu \
        $BASE_ARGS --use-adaptive-fusion \
        --lambda-align $lambda_align --lambda-balance 0.05 \
        --experiment_id $exp_id \
        2>&1 | tee $exp_log_file
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "  ✅ 完成: ${dataset} S${subject} λ_align=${lambda_align}" | tee -a $LOG_FILE
        SUCCESS_COUNT=$((SUCCESS_COUNT+1))
        return 0
    else
        echo "" | tee -a $LOG_FILE
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a $LOG_FILE
        echo "❌❌❌ 实验失败！ ❌❌❌" | tee -a $LOG_FILE
        echo "  数据集: ${dataset}, 受试者: S${subject}" | tee -a $LOG_FILE
        echo "  参数: λ_align=${lambda_align}" | tee -a $LOG_FILE
        echo "  错误码: ${exit_code}" | tee -a $LOG_FILE
        echo "  日志: tail -50 ${exp_log_file}" | tee -a $LOG_FILE
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE
        
        echo "[$(date)] FAILED: ${dataset} S${subject} HP_align${lambda_align} (exit_code: ${exit_code})" >> $ERROR_LOG
        echo "  Log: ${exp_log_file}" >> $ERROR_LOG
        
        FAIL_COUNT=$((FAIL_COUNT+1))
        return 1
    fi
}

# 运行λ_balance实验
run_lambda_balance() {
    local dataset=$1
    local subject=$2
    local lambda_balance=$3
    local gpu=$4
    
    local exp_id="HP_balance${lambda_balance}"
    
    if check_completed $dataset $subject $exp_id; then
        SKIP_COUNT=$((SKIP_COUNT+1))
        return 0
    fi
    
    if can_reuse_lambda "balance" $lambda_balance $dataset $subject; then
        echo "  ✓ 复用: ${dataset} S${subject} λ_balance=${lambda_balance}" | tee -a $LOG_FILE
        SKIP_COUNT=$((SKIP_COUNT+1))
        return 0
    fi
    
    # 创建数据集和受试者的日志目录
    local exp_log_dir="$LOG_DIR/${dataset}/subject${subject}"
    mkdir -p $exp_log_dir
    
    local exp_log_file="$exp_log_dir/${exp_id}.log"
    
    echo "  ▶ [GPU $gpu] ${dataset} S${subject} λ_balance=${lambda_balance}" | tee -a $LOG_FILE
    
    python train.py --dataset $dataset --s $subject --gpu $gpu \
        $BASE_ARGS --use-adaptive-fusion \
        --lambda-align 0.1 --lambda-balance $lambda_balance \
        --experiment_id $exp_id \
        2>&1 | tee $exp_log_file
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "  ✅ 完成: ${dataset} S${subject} λ_balance=${lambda_balance}" | tee -a $LOG_FILE
        SUCCESS_COUNT=$((SUCCESS_COUNT+1))
        return 0
    else
        echo "" | tee -a $LOG_FILE
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a $LOG_FILE
        echo "❌❌❌ 实验失败！ ❌❌❌" | tee -a $LOG_FILE
        echo "  数据集: ${dataset}, 受试者: S${subject}" | tee -a $LOG_FILE
        echo "  参数: λ_balance=${lambda_balance}" | tee -a $LOG_FILE
        echo "  错误码: ${exit_code}" | tee -a $LOG_FILE
        echo "  日志: tail -50 ${exp_log_file}" | tee -a $LOG_FILE
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE
        
        echo "[$(date)] FAILED: ${dataset} S${subject} HP_balance${lambda_balance} (exit_code: ${exit_code})" >> $ERROR_LOG
        echo "  Log: ${exp_log_file}" >> $ERROR_LOG
        
        FAIL_COUNT=$((FAIL_COUNT+1))
        return 1
    fi
}

# ============================================================
# GPU 2: DB2 λ_align搜索 (3受试者×6值=18任务，去除2个复用=16任务)
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 2] DB2 λ_align搜索开始..." | tee -a $LOG_FILE
    
    TASK_NUM=0
    
    for subject in 10 20 30; do
        for lambda_align in 0.0 0.01 0.05 0.1 0.2 0.5; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 2] 进度: ${TASK_NUM}/18" | tee -a $LOG_FILE
            run_lambda_align DB2 $subject $lambda_align 2
        done
    done
    
    echo "[GPU 2] ✅ DB2 λ_align搜索完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 3: DB3 λ_align搜索 (3受试者×6值=18任务)
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 3] DB3 λ_align搜索开始..." | tee -a $LOG_FILE
    
    TASK_NUM=0
    
    for subject in 2 6 11; do
        for lambda_align in 0.0 0.01 0.05 0.1 0.2 0.5; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 3] 进度: ${TASK_NUM}/18" | tee -a $LOG_FILE
            run_lambda_align DB3 $subject $lambda_align 3
        done
    done
    
    echo "[GPU 3] ✅ DB3 λ_align搜索完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 4: DB2 λ_balance搜索 (3受试者×6值=18任务)
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 4] DB2 λ_balance搜索开始..." | tee -a $LOG_FILE
    
    TASK_NUM=0
    
    for subject in 10 20 30; do
        for lambda_balance in 0.0 0.01 0.05 0.1 0.2 0.5; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 4] 进度: ${TASK_NUM}/18" | tee -a $LOG_FILE
            run_lambda_balance DB2 $subject $lambda_balance 4
        done
    done
    
    echo "[GPU 4] ✅ DB2 λ_balance搜索完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 6: DB3 λ_balance搜索 (3受试者×6值=18任务)
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 6] DB3 λ_balance搜索开始..." | tee -a $LOG_FILE
    
    TASK_NUM=0
    
    for subject in 2 6 11; do
        for lambda_balance in 0.0 0.01 0.05 0.1 0.2 0.5; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 6] 进度: ${TASK_NUM}/18" | tee -a $LOG_FILE
            run_lambda_balance DB3 $subject $lambda_balance 6
        done
    done
    
    echo "[GPU 6] ✅ DB3 λ_balance搜索完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 0: DB5 λ_align搜索 (3受试者×6值=18任务) - 新增
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 0] DB5 λ_align搜索开始..." | tee -a $LOG_FILE
    
    TASK_NUM=0
    
    for subject in 1 5 10; do
        for lambda_align in 0.0 0.01 0.05 0.1 0.2 0.5; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 0] 进度: ${TASK_NUM}/18" | tee -a $LOG_FILE
            run_lambda_align DB5 $subject $lambda_align 0
        done
    done
    
    echo "[GPU 0] ✅ DB5 λ_align搜索完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 1: DB5 λ_balance搜索 (3受试者×6值=18任务) - 新增
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 1] DB5 λ_balance搜索开始..." | tee -a $LOG_FILE
    
    TASK_NUM=0
    
    for subject in 1 5 10; do
        for lambda_balance in 0.0 0.01 0.05 0.1 0.2 0.5; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 1] 进度: ${TASK_NUM}/18" | tee -a $LOG_FILE
            run_lambda_balance DB5 $subject $lambda_balance 1
        done
    done
    
    echo "[GPU 1] ✅ DB5 λ_balance搜索完成！" | tee -a $LOG_FILE
} &

# 等待所有GPU完成
wait

echo "" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "🎉 阶段4完成：超参数搜索（λ_align和λ_balance）" | tee -a $LOG_FILE
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
DB2_ALIGN_COUNT=$(ls -1 results/DB2/subject*/HP_align*/metrics.json 2>/dev/null | wc -l)
DB3_ALIGN_COUNT=$(ls -1 results/DB3/subject*/HP_align*/metrics.json 2>/dev/null | wc -l)
DB5_ALIGN_COUNT=$(ls -1 results/DB5/subject*/HP_align*/metrics.json 2>/dev/null | wc -l)
DB2_BALANCE_COUNT=$(ls -1 results/DB2/subject*/HP_balance*/metrics.json 2>/dev/null | wc -l)
DB3_BALANCE_COUNT=$(ls -1 results/DB3/subject*/HP_balance*/metrics.json 2>/dev/null | wc -l)
DB5_BALANCE_COUNT=$(ls -1 results/DB5/subject*/HP_balance*/metrics.json 2>/dev/null | wc -l)

echo "  DB2 λ_align:   ${DB2_ALIGN_COUNT} / 18" | tee -a $LOG_FILE
echo "  DB3 λ_align:   ${DB3_ALIGN_COUNT} / 18" | tee -a $LOG_FILE
echo "  DB5 λ_align:   ${DB5_ALIGN_COUNT} / 18" | tee -a $LOG_FILE
echo "  DB2 λ_balance: ${DB2_BALANCE_COUNT} / 18" | tee -a $LOG_FILE
echo "  DB3 λ_balance: ${DB3_BALANCE_COUNT} / 18" | tee -a $LOG_FILE
echo "  DB5 λ_balance: ${DB5_BALANCE_COUNT} / 18" | tee -a $LOG_FILE
echo "  总计: $((DB2_ALIGN_COUNT + DB3_ALIGN_COUNT + DB5_ALIGN_COUNT + DB2_BALANCE_COUNT + DB3_BALANCE_COUNT + DB5_BALANCE_COUNT)) / 108" | tee -a $LOG_FILE

if [ $FAIL_COUNT -gt 0 ]; then
    echo "" | tee -a $LOG_FILE
    echo "⚠️  发现 ${FAIL_COUNT} 个失败的实验，详见: $ERROR_LOG" | tee -a $LOG_FILE
fi

