#!/bin/bash
# Ninapro数据集批量预处理脚本
# 支持DB2, DB3, DB5, DB7四个数据集

# 设置路径
DATA_ROOT="/home/xuweishi/KBS25/MoMo/Momo/data"
OUTPUT_DIR="/home/xuweishi/KBS25/MoMo/Momo/processed_data"
SCRIPT_PATH="/home/xuweishi/KBS25/MoMo/Momo/preprocess.py"

# 日志目录
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================"
echo "Ninapro数据集批量预处理"
echo "开始时间: $(date)"
echo "========================================"
echo ""

# ========== 处理DB2数据集 ==========
echo ">>> 开始处理DB2数据集..."
DB2_LOG="$LOG_DIR/DB2_${TIMESTAMP}.log"

# DB2受试者列表（1-40，排除11和已处理的：10,13,14,15,16,17,18,23,36）
# 剩余需要处理：1,2,3,4,5,6,7,8,9,12,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35,37,38,39,40
DB2_SUBJECTS="1,2,3,4,5,6,7,8,9,12,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35,37,38,39,40"

echo "  已处理受试者: 10,13,14,15,16,17,18,23,36"
echo "  待处理受试者: $DB2_SUBJECTS"
echo ""

python "$SCRIPT_PATH" \
    --dataset DB2 \
    --subjects "$DB2_SUBJECTS" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --fs 2000 \
    2>&1 | tee "$DB2_LOG"

echo "DB2处理完成！日志: $DB2_LOG"
echo ""

# ========== 处理DB3数据集 ==========
echo ">>> 开始处理DB3数据集..."
DB3_LOG="$LOG_DIR/DB3_${TIMESTAMP}.log"

# DB3受试者列表（2,4,5,6,9,11）
DB3_SUBJECTS="2,4,5,6,9,11"

python "$SCRIPT_PATH" \
    --dataset DB3 \
    --subjects "$DB3_SUBJECTS" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --fs 2000 \
    2>&1 | tee "$DB3_LOG"

echo "DB3处理完成！日志: $DB3_LOG"
echo ""

# ========== 处理DB5数据集 ==========
echo ">>> 开始处理DB5数据集..."
DB5_LOG="$LOG_DIR/DB5_${TIMESTAMP}.log"

# DB5受试者列表（1-10）
DB5_SUBJECTS="1-10"

python "$SCRIPT_PATH" \
    --dataset DB5 \
    --subjects "$DB5_SUBJECTS" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --fs 2000 \
    2>&1 | tee "$DB5_LOG"

echo "DB5处理完成！日志: $DB5_LOG"
echo ""

# ========== 处理DB7数据集 ==========
echo ">>> 开始处理DB7数据集..."
DB7_LOG="$LOG_DIR/DB7_${TIMESTAMP}.log"

# DB7受试者列表（2-13）
DB7_SUBJECTS="2-13"

python "$SCRIPT_PATH" \
    --dataset DB7 \
    --subjects "$DB7_SUBJECTS" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --fs 2000 \
    2>&1 | tee "$DB7_LOG"

echo "DB7处理完成！日志: $DB7_LOG"
echo ""

# ========== 总结 ==========
echo "========================================"
echo "所有数据集处理完成！"
echo "结束时间: $(date)"
echo "========================================"
echo ""
echo "输出目录结构:"
echo "  $OUTPUT_DIR/"
echo "    ├── DB2/         # DB2数据集 (39个受试者)"
echo "    ├── DB3/         # DB3数据集 (6个受试者)"
echo "    ├── DB5/         # DB5数据集 (10个受试者)"
echo "    ├── DB7/         # DB7数据集 (12个受试者)"
echo "    └── logs/        # 处理日志"
echo ""
echo "每个数据集目录包含:"
echo "  - S{subject}_train.h5  # 训练数据"
echo "  - S{subject}_test.h5   # 测试数据"
echo "  - metadata.pkl         # 元数据和scalers"
echo ""

