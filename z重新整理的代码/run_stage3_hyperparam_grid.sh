#!/bin/bash

# ============================================================
# 阶段3：超参数网格搜索（α和β）
# GPU 0, 1, 2, 3, 4, 6, 7 并行运行（7个GPU）
# 总任务：216个（DB3和DB5，各3个受试者）
# 6×6=36个组合，2个数据集，各3个受试者
# 预计时间：~15-20小时（7个GPU并行）
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
LOG_FILE="$SUMMARY_DIR/stage3_hyperparam_grid_${TIMESTAMP}.log"
ERROR_LOG="$SUMMARY_DIR/stage3_errors_${TIMESTAMP}.log"

SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

echo "============================================================" | tee $LOG_FILE
echo "🔬 阶段3：超参数网格搜索（α和β）" | tee -a $LOG_FILE
echo "📅 开始时间: $(date)" | tee -a $LOG_FILE
echo "🖥️  使用GPU: 0, 1, 2, 3, 4, 6, 7" | tee -a $LOG_FILE
echo "📊 数据集: DB3, DB5" | tee -a $LOG_FILE
echo "📊 参数范围: α,β ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}" | tee -a $LOG_FILE
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

# 检查是否可以复用其他实验
can_reuse() {
    local alpha=$1
    local beta=$2
    local dataset=$3
    local subject=$4
    
    # 可以复用的组合
    if [ "$alpha" == "0.0" ] && [ "$beta" == "0.0" ]; then
        # 复用M0_base
        check_completed $dataset $subject "M0_base"
        return $?
    elif [ "$alpha" == "0.5" ] && [ "$beta" == "0.0" ]; then
        # 复用D1_private_only
        check_completed $dataset $subject "D1_private_only"
        return $?
    elif [ "$alpha" == "0.0" ] && [ "$beta" == "0.5" ]; then
        # 复用D2_shared_only
        check_completed $dataset $subject "D2_shared_only"
        return $?
    elif [ "$alpha" == "0.5" ] && [ "$beta" == "0.5" ]; then
        # 复用M1_disentangle
        check_completed $dataset $subject "M1_disentangle"
        return $?
    else
        return 1  # 不能复用
    fi
}

# 运行单个实验
run_experiment() {
    local dataset=$1
    local subject=$2
    local alpha=$3
    local beta=$4
    local gpu=$5
    
    local exp_id="HP_a${alpha}_b${beta}"
    
    # 检查是否已完成
    if check_completed $dataset $subject $exp_id; then
        SKIP_COUNT=$((SKIP_COUNT+1))
        return 0
    fi
    
    # 检查是否可以复用
    if can_reuse $alpha $beta $dataset $subject; then
        echo "  ✓ 复用: ${dataset} S${subject} α=${alpha} β=${beta}" | tee -a $LOG_FILE
        SKIP_COUNT=$((SKIP_COUNT+1))
        return 0
    fi
    
    # 创建数据集和受试者的日志目录
    local exp_log_dir="$LOG_DIR/${dataset}/subject${subject}"
    mkdir -p $exp_log_dir
    
    local exp_log_file="$exp_log_dir/${exp_id}.log"
    
    echo "  ▶ [GPU $gpu] ${dataset} S${subject} α=${alpha} β=${beta}" | tee -a $LOG_FILE
    
    python train.py --dataset $dataset --s $subject --gpu $gpu \
        $BASE_ARGS --use-disentangle --alpha $alpha --beta $beta \
        --experiment_id $exp_id \
        2>&1 | tee $exp_log_file
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "  ✅ 完成: ${dataset} S${subject} α=${alpha} β=${beta}" | tee -a $LOG_FILE
        SUCCESS_COUNT=$((SUCCESS_COUNT+1))
        return 0
    else
        echo "" | tee -a $LOG_FILE
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a $LOG_FILE
        echo "❌❌❌ 实验失败！ ❌❌❌" | tee -a $LOG_FILE
        echo "  数据集: ${dataset}, 受试者: S${subject}" | tee -a $LOG_FILE
        echo "  参数: α=${alpha}, β=${beta}" | tee -a $LOG_FILE
        echo "  错误码: ${exit_code}" | tee -a $LOG_FILE
        echo "  日志: tail -50 ${exp_log_file}" | tee -a $LOG_FILE
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE
        
        echo "[$(date)] FAILED: ${dataset} S${subject} HP_a${alpha}_b${beta} (exit_code: ${exit_code})" >> $ERROR_LOG
        echo "  Log: ${exp_log_file}" >> $ERROR_LOG
        
        FAIL_COUNT=$((FAIL_COUNT+1))
        return 1
    fi
}

# ============================================================
# 将36个组合分成7组，分配到7个GPU
# ============================================================

# 所有α和β的组合
ALPHA_VALUES=(0.0 0.1 0.3 0.5 0.7 1.0)
BETA_VALUES=(0.0 0.1 0.3 0.5 0.7 1.0)

# 生成所有组合（36个）
COMBINATIONS=()
for alpha in "${ALPHA_VALUES[@]}"; do
    for beta in "${BETA_VALUES[@]}"; do
        COMBINATIONS+=("${alpha},${beta}")
    done
done

# 分成7组（尽量均匀）
GROUP1=("${COMBINATIONS[@]:0:5}")    # 组合0-4   (5个)
GROUP2=("${COMBINATIONS[@]:5:5}")    # 组合5-9   (5个)
GROUP3=("${COMBINATIONS[@]:10:5}")   # 组合10-14 (5个)
GROUP4=("${COMBINATIONS[@]:15:5}")   # 组合15-19 (5个)
GROUP5=("${COMBINATIONS[@]:20:5}")   # 组合20-24 (5个)
GROUP6=("${COMBINATIONS[@]:25:6}")   # 组合25-30 (6个)
GROUP7=("${COMBINATIONS[@]:31:5}")   # 组合31-35 (5个)

# ============================================================
# GPU 0: 组合1-5 × 6受试者 = 30任务
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 0] 组合1-5开始..." | tee -a $LOG_FILE
    
    TASK_NUM=0
    TOTAL=30
    
    for combo in "${GROUP1[@]}"; do
        IFS=',' read -r alpha beta <<< "$combo"
        
        # DB3: 受试者2, 6, 11
        for subject in 2 6 11; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 0] 进度: ${TASK_NUM}/${TOTAL}" | tee -a $LOG_FILE
            run_experiment DB3 $subject $alpha $beta 0
        done
        
        # DB5: 受试者1, 5, 10
        for subject in 1 5 10; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 0] 进度: ${TASK_NUM}/${TOTAL}" | tee -a $LOG_FILE
            run_experiment DB5 $subject $alpha $beta 0
        done
    done
    
    echo "[GPU 0] ✅ 完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 1: 组合6-10 × 6受试者 = 30任务
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 1] 组合6-10开始..." | tee -a $LOG_FILE
    
    TASK_NUM=0
    TOTAL=30
    
    for combo in "${GROUP2[@]}"; do
        IFS=',' read -r alpha beta <<< "$combo"
        
        for subject in 2 6 11; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 1] 进度: ${TASK_NUM}/${TOTAL}" | tee -a $LOG_FILE
            run_experiment DB3 $subject $alpha $beta 1
        done
        
        for subject in 1 5 10; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 1] 进度: ${TASK_NUM}/${TOTAL}" | tee -a $LOG_FILE
            run_experiment DB5 $subject $alpha $beta 1
        done
    done
    
    echo "[GPU 1] ✅ 完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 2: 组合11-15 × 6受试者 = 30任务
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 2] 组合11-15开始..." | tee -a $LOG_FILE
    
    TASK_NUM=0
    TOTAL=30
    
    for combo in "${GROUP3[@]}"; do
        IFS=',' read -r alpha beta <<< "$combo"
        
        for subject in 2 6 11; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 2] 进度: ${TASK_NUM}/${TOTAL}" | tee -a $LOG_FILE
            run_experiment DB3 $subject $alpha $beta 2
        done
        
        for subject in 1 5 10; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 2] 进度: ${TASK_NUM}/${TOTAL}" | tee -a $LOG_FILE
            run_experiment DB5 $subject $alpha $beta 2
        done
    done
    
    echo "[GPU 2] ✅ 完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 3: 组合16-20 × 6受试者 = 30任务
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 3] 组合16-20开始..." | tee -a $LOG_FILE
    
    TASK_NUM=0
    TOTAL=30
    
    for combo in "${GROUP4[@]}"; do
        IFS=',' read -r alpha beta <<< "$combo"
        
        for subject in 2 6 11; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 3] 进度: ${TASK_NUM}/${TOTAL}" | tee -a $LOG_FILE
            run_experiment DB3 $subject $alpha $beta 3
        done
        
        for subject in 1 5 10; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 3] 进度: ${TASK_NUM}/${TOTAL}" | tee -a $LOG_FILE
            run_experiment DB5 $subject $alpha $beta 3
        done
    done
    
    echo "[GPU 3] ✅ 完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 4: 组合21-25 × 6受试者 = 30任务
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 4] 组合21-25开始..." | tee -a $LOG_FILE
    
    TASK_NUM=0
    TOTAL=30
    
    for combo in "${GROUP5[@]}"; do
        IFS=',' read -r alpha beta <<< "$combo"
        
        for subject in 2 6 11; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 4] 进度: ${TASK_NUM}/${TOTAL}" | tee -a $LOG_FILE
            run_experiment DB3 $subject $alpha $beta 4
        done
        
        for subject in 1 5 10; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 4] 进度: ${TASK_NUM}/${TOTAL}" | tee -a $LOG_FILE
            run_experiment DB5 $subject $alpha $beta 4
        done
    done
    
    echo "[GPU 4] ✅ 完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 6: 组合26-31 × 6受试者 = 36任务
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 6] 组合26-31开始..." | tee -a $LOG_FILE
    
    TASK_NUM=0
    TOTAL=36
    
    for combo in "${GROUP6[@]}"; do
        IFS=',' read -r alpha beta <<< "$combo"
        
        for subject in 2 6 11; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 6] 进度: ${TASK_NUM}/${TOTAL}" | tee -a $LOG_FILE
            run_experiment DB3 $subject $alpha $beta 6
        done
        
        for subject in 1 5 10; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 6] 进度: ${TASK_NUM}/${TOTAL}" | tee -a $LOG_FILE
            run_experiment DB5 $subject $alpha $beta 6
        done
    done
    
    echo "[GPU 6] ✅ 完成！" | tee -a $LOG_FILE
} &

# ============================================================
# GPU 7: 组合32-36 × 6受试者 = 30任务
# ============================================================
{
    echo "" | tee -a $LOG_FILE
    echo "[GPU 7] 组合32-36开始..." | tee -a $LOG_FILE
    
    TASK_NUM=0
    TOTAL=30
    
    for combo in "${GROUP7[@]}"; do
        IFS=',' read -r alpha beta <<< "$combo"
        
        for subject in 2 6 11; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 7] 进度: ${TASK_NUM}/${TOTAL}" | tee -a $LOG_FILE
            run_experiment DB3 $subject $alpha $beta 7
        done
        
        for subject in 1 5 10; do
            TASK_NUM=$((TASK_NUM+1))
            echo "[GPU 7] 进度: ${TASK_NUM}/${TOTAL}" | tee -a $LOG_FILE
            run_experiment DB5 $subject $alpha $beta 7
        done
    done
    
    echo "[GPU 7] ✅ 完成！" | tee -a $LOG_FILE
} &

# 等待所有GPU完成
wait

echo "" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "🎉 阶段3完成：超参数网格搜索（α和β）" | tee -a $LOG_FILE
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
DB3_HP_COUNT=$(ls -1 results/DB3/subject*/HP_a*_b*/metrics.json 2>/dev/null | wc -l)
DB5_HP_COUNT=$(ls -1 results/DB5/subject*/HP_a*_b*/metrics.json 2>/dev/null | wc -l)
echo "  DB3: ${DB3_HP_COUNT} / 108 (36组合×3受试者)" | tee -a $LOG_FILE
echo "  DB5: ${DB5_HP_COUNT} / 108" | tee -a $LOG_FILE
echo "  总计: $((DB3_HP_COUNT + DB5_HP_COUNT)) / 216" | tee -a $LOG_FILE

if [ $FAIL_COUNT -gt 0 ]; then
    echo "" | tee -a $LOG_FILE
    echo "⚠️  发现 ${FAIL_COUNT} 个失败的实验，详见: $ERROR_LOG" | tee -a $LOG_FILE
fi

