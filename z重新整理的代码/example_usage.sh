#!/bin/bash
# 批量实验运行示例脚本

# ============================================================
# 示例1：运行单个实验
# ============================================================
echo "示例1：在DB5上训练单个受试者的完整模型"
python train.py --dataset DB5 --s 1 --gpu 0 --use-adaptive-fusion --save-predictions --num_epochs 20

# ============================================================
# 示例2：运行主消融实验（3个模型）
# ============================================================
echo "示例2：DB5数据集，受试者1的主消融实验"
python train.py --dataset DB5 --s 1 --gpu 0 --save-predictions --experiment_id M0_base
python train.py --dataset DB5 --s 1 --gpu 0 --use-disentangle --save-predictions --experiment_id M1_disentangle
python train.py --dataset DB5 --s 1 --gpu 0 --use-adaptive-fusion --save-predictions --experiment_id M3_full

# ============================================================
# 示例3：超参数搜索
# ============================================================
echo "示例3：超参数搜索 - alpha和beta的影响"
for alpha in 0.0 0.3 0.5 0.7 1.0; do
    for beta in 0.0 0.3 0.5 0.7 1.0; do
        python train.py --dataset DB2 --s 10 \
            --use-disentangle \
            --alpha $alpha --beta $beta \
            --save-predictions \
            --experiment_id HP_a${alpha}_b${beta} \
            --gpu 0
    done
done

# ============================================================
# 示例4：使用批量运行脚本（推荐）
# ============================================================
echo "示例4：使用run_experiments.py批量运行"

# 4.1 预览要执行的任务（dry-run模式）
python run_experiments.py --group main_ablation --subjects "1,5,10" --dry-run

# 4.2 实际执行主消融实验（代表性受试者）
python run_experiments.py --group main_ablation --subjects representative --gpu 0 --yes

# 4.3 执行所有实验组
python run_experiments.py --group all --subjects all --gpu 0

# ============================================================
# 示例5：聚合评估
# ============================================================
echo "示例5：聚合所有受试者的结果"
python train.py --dataset DB5 --aggregate-results --use-adaptive-fusion --aggregate-subjects "all"

# ============================================================
# 不同数据集的示例
# ============================================================
# DB2 (50类, 12 EMG, 36 IMU)
python train.py --dataset DB2 --s 10 --gpu 0 --use-adaptive-fusion --save-predictions

# DB3 (50类, 12 EMG, 36 IMU, 截肢患者)
python train.py --dataset DB3 --s 1 --gpu 0 --use-adaptive-fusion --save-predictions

# DB5 (53类, 16 EMG, 3 IMU)
python train.py --dataset DB5 --s 1 --gpu 0 --use-adaptive-fusion --save-predictions

# DB7 (41类, 12 EMG, 36 IMU)
python train.py --dataset DB7 --s 1 --gpu 0 --use-adaptive-fusion --save-predictions

