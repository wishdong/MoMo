#!/bin/bash

# ============================================================
# 阶段1：消融实验（DB5, DB7）
# GPU 6, 7 并行运行
# 总任务：48个（DB5: 24, DB7: 24）
# 预计时间：~12小时
# ============================================================

# 错误处理：遇到错误时继续，但记录错误
set +e  # 不要在错误时立即退出
set -o pipefail  # 管道中任何命令失败都会导致整个管道失败（捕获python的真实退出码）

# 信号处理：Ctrl+C时杀死所有子进程
trap 'echo ""; echo "⚠️  检测到中断信号，正在停止所有后台任务..."; kill $(jobs -p) 2>/dev/null; wait; echo "✅ 所有任务已停止"; exit 130' INT TERM

BASE_ARGS="--batch_size 64 --num_epochs 20 --save-predictions"
LOG_DIR="./experiment_logs"
SUMMARY_DIR="$LOG_DIR/stage_summaries"
mkdir -p $LOG_DIR
mkdir -p $SUMMARY_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$SUMMARY_DIR/stage1_ablation_${TIMESTAMP}.log"
ERROR_LOG="$SUMMARY_DIR/stage1_errors_${TIMESTAMP}.log"
WARNING_LOG="$SUMMARY_DIR/stage1_warnings_${TIMESTAMP}.log"

# 错误计数器
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
WARNING_COUNT=0

echo "============================================================" | tee $LOG_FILE
echo "📊 阶段1：消融实验（DB5, DB7）" | tee -a $LOG_FILE
echo "📅 开始时间: $(date)" | tee -a $LOG_FILE
echo "🖥️  使用GPU: 6, 7" | tee -a $LOG_FILE
echo "💡 提示: Ctrl+C 可以安全停止所有任务" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE

# 检查实验是否已完成的函数
check_completed() {
    local dataset=$1
    local subject=$2
    local exp_id=$3
    
    if [ -f "results/${dataset}/subject${subject}/${exp_id}/metrics.json" ] && \
       [ -f "results/${dataset}/subject${subject}/${exp_id}/predictions.pkl" ]; then
        return 0  # 已完成
    else
        return 1  # 未完成
    fi
}

# 运行单个实验的函数
run_experiment() {
    local dataset=$1
    local subject=$2
    local exp_id=$3
    local args=$4
    local gpu=$5
    
    if check_completed $dataset $subject $exp_id; then
        echo "  ✓ 跳过已完成: ${dataset} S${subject} ${exp_id}" | tee -a $LOG_FILE
        SKIP_COUNT=$((SKIP_COUNT+1))
        return 0
    fi
    
    # 创建数据集和受试者的日志目录
    local exp_log_dir="$LOG_DIR/${dataset}/subject${subject}"
    mkdir -p $exp_log_dir
    
    local exp_log_file="$exp_log_dir/${exp_id}.log"
    
    echo "  ▶ 运行: ${dataset} S${subject} ${exp_id}" | tee -a $LOG_FILE
    echo "    日志: ${exp_log_file}" | tee -a $LOG_FILE
    
    # 运行训练（显示输出到控制台和日志文件）
    python train.py --dataset $dataset --s $subject --gpu $gpu \
        $BASE_ARGS $args --experiment_id $exp_id \
        2>&1 | tee $exp_log_file
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # 检查日志中是否有警告
        if grep -q "⚠️" $exp_log_file; then
            echo "  ⚠️  完成但有警告: ${dataset} S${subject} ${exp_id}" | tee -a $LOG_FILE
            echo "[$(date)] WARNING: ${dataset} S${subject} ${exp_id}" >> $WARNING_LOG
            echo "  Log: ${exp_log_file}" >> $WARNING_LOG
            grep "⚠️" $exp_log_file >> $WARNING_LOG
            echo "" >> $WARNING_LOG
            WARNING_COUNT=$((WARNING_COUNT+1))
        else
            echo "  ✅ 完成: ${dataset} S${subject} ${exp_id}" | tee -a $LOG_FILE
        fi
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
        
        # 记录到错误日志
        echo "[$(date)] FAILED: ${dataset} S${subject} ${exp_id} (exit_code: ${exit_code})" >> $ERROR_LOG
        echo "  Log: ${exp_log_file}" >> $ERROR_LOG
        
        FAIL_COUNT=$((FAIL_COUNT+1))
        return 1
    fi
}

# ============================================================
# GPU 6: DB5的消融实验（受试者1, 5, 10）
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 6] DB5 消融实验开始..." | tee -a $LOG_FILE
    TASK_COUNT=0
    TOTAL_TASKS=24
    
    for subject in 1 5 10; do
        echo "" | tee -a $LOG_FILE
        echo "[GPU 6] === DB5 受试者 ${subject} ===" | tee -a $LOG_FILE
        
        # M0: 基础模型
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 6] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB5 $subject M0_base "" 6
        
        # M1: 解纠缠模型
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 6] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB5 $subject M1_disentangle "--use-disentangle" 6
        
        # M3: 完整模型
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 6] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB5 $subject M3_full "--use-adaptive-fusion" 6
        
        # D1: 只有L_private
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 6] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB5 $subject D1_private_only "--use-disentangle --alpha 0.5 --beta 0" 6
        
        # D2: 只有L_shared
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 6] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB5 $subject D2_shared_only "--use-disentangle --alpha 0 --beta 0.5" 6
        
        # FA1: 无约束
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 6] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB5 $subject FA1_no_constraint "--use-adaptive-fusion --lambda-align 0 --lambda-balance 0" 6
        
        # FA2: 只有L_align
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 6] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB5 $subject FA2_align_only "--use-adaptive-fusion --lambda-align 0.1 --lambda-balance 0" 6
        
        # FA3: 只有L_balance
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 6] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB5 $subject FA3_balance_only "--use-adaptive-fusion --lambda-align 0 --lambda-balance 0.05" 6
    done
    
    echo "[GPU 6] ✅ DB5 消融实验完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 7: DB7的消融实验（受试者3, 7, 11）
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 7] DB7 消融实验开始..." | tee -a $LOG_FILE
    TASK_COUNT=0
    TOTAL_TASKS=24
    
    for subject in 3 7 11; do
        echo "" | tee -a $LOG_FILE
        echo "[GPU 7] === DB7 受试者 ${subject} ===" | tee -a $LOG_FILE
        
        # M0: 基础模型
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 7] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB7 $subject M0_base "" 7
        
        # M1: 解纠缠模型
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 7] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB7 $subject M1_disentangle "--use-disentangle" 7
        
        # M3: 完整模型
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 7] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB7 $subject M3_full "--use-adaptive-fusion" 7
        
        # D1: 只有L_private
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 7] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB7 $subject D1_private_only "--use-disentangle --alpha 0.5 --beta 0" 7
        
        # D2: 只有L_shared
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 7] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB7 $subject D2_shared_only "--use-disentangle --alpha 0 --beta 0.5" 7
        
        # FA1: 无约束
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 7] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB7 $subject FA1_no_constraint "--use-adaptive-fusion --lambda-align 0 --lambda-balance 0" 7
        
        # FA2: 只有L_align
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 7] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB7 $subject FA2_align_only "--use-adaptive-fusion --lambda-align 0.1 --lambda-balance 0" 7
        
        # FA3: 只有L_balance
        TASK_COUNT=$((TASK_COUNT+1))
        echo "[GPU 7] 进度: ${TASK_COUNT}/${TOTAL_TASKS}" | tee -a $LOG_FILE
        run_experiment DB7 $subject FA3_balance_only "--use-adaptive-fusion --lambda-align 0 --lambda-balance 0.05" 7
    done
    
    echo "[GPU 7] ✅ DB7 消融实验完成！" | tee -a $LOG_FILE
} &

# 等待所有GPU完成
wait

echo "" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "🎉 阶段1完成：消融实验" | tee -a $LOG_FILE
echo "📅 结束时间: $(date)" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE

# 统计完成情况
echo "" | tee -a $LOG_FILE
echo "📊 执行统计：" | tee -a $LOG_FILE
echo "  ✅ 成功: ${SUCCESS_COUNT}" | tee -a $LOG_FILE
echo "  ⚠️  警告: ${WARNING_COUNT} (成功但有警告)" | tee -a $LOG_FILE
echo "  ❌ 失败: ${FAIL_COUNT}" | tee -a $LOG_FILE
echo "  ⏭️  跳过: ${SKIP_COUNT}" | tee -a $LOG_FILE

echo "" | tee -a $LOG_FILE
echo "📂 结果统计：" | tee -a $LOG_FILE
DB5_COUNT=$(ls -1 results/DB5/subject*/*/metrics.json 2>/dev/null | wc -l)
DB7_COUNT=$(ls -1 results/DB7/subject*/*/metrics.json 2>/dev/null | wc -l)
echo "  DB5: ${DB5_COUNT} 个实验结果" | tee -a $LOG_FILE
echo "  DB7: ${DB7_COUNT} 个实验结果" | tee -a $LOG_FILE

# 如果有失败，显示错误日志
if [ $FAIL_COUNT -gt 0 ]; then
    echo "" | tee -a $LOG_FILE
    echo "❌ 发现 ${FAIL_COUNT} 个失败的实验，详见: $ERROR_LOG" | tee -a $LOG_FILE
fi

# 如果有警告，显示警告日志
if [ $WARNING_COUNT -gt 0 ]; then
    echo "" | tee -a $LOG_FILE
    echo "⚠️  发现 ${WARNING_COUNT} 个警告，详见: $WARNING_LOG" | tee -a $LOG_FILE
    echo "   (这些实验已完成，但有非致命警告，如FLOPs计算失败)" | tee -a $LOG_FILE
fi

