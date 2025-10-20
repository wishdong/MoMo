#!/bin/bash
# Ninapro单数据集预处理脚本
# 用法: ./process_dataset.sh <dataset> [subjects]
# 示例:
#   ./process_dataset.sh DB2              # 处理DB2的所有受试者
#   ./process_dataset.sh DB2 10           # 只处理DB2的受试者10
#   ./process_dataset.sh DB2 10,23,36     # 处理DB2的受试者10,23,36
#   ./process_dataset.sh DB5 1-5          # 处理DB5的受试者1到5

# 设置路径
DATA_ROOT="/home/xuweishi/KBS25/MoMo/Momo/data"
OUTPUT_DIR="/home/xuweishi/KBS25/MoMo/Momo/processed_data"
SCRIPT_PATH="/home/xuweishi/KBS25/MoMo/Momo/preprocess.py"

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <dataset> [subjects]"
    echo ""
    echo "参数:"
    echo "  dataset   - 数据集名称 (DB2, DB3, DB5, DB7)"
    echo "  subjects  - 受试者列表（可选）"
    echo "              如果不指定，将处理该数据集的所有受试者"
    echo ""
    echo "示例:"
    echo "  $0 DB2              # 处理DB2的所有受试者"
    echo "  $0 DB2 10           # 只处理DB2的受试者10"
    echo "  $0 DB2 10,23,36     # 处理DB2的受试者10,23,36"
    echo "  $0 DB5 1-5          # 处理DB5的受试者1到5"
    echo ""
    exit 1
fi

DATASET=$1

# 验证数据集名称
if [[ ! "$DATASET" =~ ^(DB2|DB3|DB5|DB7)$ ]]; then
    echo "错误: 不支持的数据集 '$DATASET'"
    echo "支持的数据集: DB2, DB3, DB5, DB7"
    exit 1
fi

# 设置默认受试者列表
if [ $# -eq 2 ]; then
    SUBJECTS=$2
else
    # 默认受试者列表
    case $DATASET in
        DB2)
            SUBJECTS="1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40"
            echo "使用默认受试者列表: DB2所有受试者 (1-40，排除11)"
            ;;
        DB3)
            SUBJECTS="2,4,5,6,9,11"
            echo "使用默认受试者列表: DB3所有受试者 (2,4,5,6,9,11)"
            ;;
        DB5)
            SUBJECTS="1-10"
            echo "使用默认受试者列表: DB5所有受试者 (1-10)"
            ;;
        DB7)
            SUBJECTS="2-13"
            echo "使用默认受试者列表: DB7所有受试者 (2-13)"
            ;;
    esac
fi

# 创建日志目录
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# 获取时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/${DATASET}_${TIMESTAMP}.log"

# 显示信息
echo "========================================"
echo "Ninapro数据预处理"
echo "========================================"
echo "数据集: $DATASET"
echo "受试者: $SUBJECTS"
echo "数据根目录: $DATA_ROOT"
echo "输出目录: $OUTPUT_DIR/$DATASET"
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"
echo "========================================"
echo ""

# 运行预处理
python "$SCRIPT_PATH" \
    --dataset "$DATASET" \
    --subjects "$SUBJECTS" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --fs 2000 \
    2>&1 | tee "$LOG_FILE"

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "处理完成！"
    echo "结束时间: $(date)"
    echo "========================================"
    echo "输出位置: $OUTPUT_DIR/$DATASET/"
    echo "日志文件: $LOG_FILE"
else
    echo ""
    echo "========================================"
    echo "处理失败！请查看日志文件:"
    echo "$LOG_FILE"
    echo "========================================"
    exit 1
fi

