#!/bin/bash

set -euo pipefail

# ============================================================
# MTEAN Black Carp
# End-to-end supervised training from scratch
# 3 seeds × 5-fold CV
# NO pretraining
# ============================================================

cd /home/data/biofish/yjn/workspace/MTEAN

SEEDS=(1 2 3)
FOLDS=(0 1 2 3 4)

MODEL_CONFIG="config/model_config_blackcarp.json"
TRAINING_CONFIG="training/config/training_config.yml"

# 单独用一个 done 标志判断某个 fold 是否完整跑完
STATUS_DIR="output/run_status/blackcarp"
mkdir -p "$STATUS_DIR"

echo "============================================================"
echo " MTEAN Black Carp"
echo " Training from scratch"
echo " Seeds : ${SEEDS[*]}"
echo " Folds : ${FOLDS[*]}"
echo "============================================================"

for seed in "${SEEDS[@]}"; do

    echo
    echo "############################################################"
    echo "                     SEED ${seed}"
    echo "############################################################"

    # ========================================================
    # 1. 检查该 seed 的 CV 划分
    # ========================================================
    CV_FILE="data/blackcarp499/cv_splits_${seed}.csv"

    if [ ! -f "$CV_FILE" ]; then

        echo ">>> 生成 Seed ${seed} 的 5-fold CV 划分"

        python3 training/data/generate_cv_splits.py "${seed}"

        if [ ! -f "$CV_FILE" ]; then
            echo "❌ CV 文件生成失败："
            echo "   $CV_FILE"
            exit 1
        fi

    else

        echo "✅ CV 文件已存在：$CV_FILE"

    fi


    # ========================================================
    # 2. 依次跑 Fold 0-4
    # ========================================================
    for fold in "${FOLDS[@]}"; do

        DONE_FILE="${STATUS_DIR}/seed_${seed}_fold_${fold}.done"

        echo
        echo "============================================================"
        echo " Seed ${seed} | Fold ${fold}"
        echo "============================================================"

        # ----------------------------------------------------
        # 已完整完成则跳过
        # ----------------------------------------------------
        if [ -f "$DONE_FILE" ]; then

            echo "✅ Seed ${seed} Fold ${fold} 已完整完成"
            echo ">>> 跳过"

            continue
        fi


        # ----------------------------------------------------
        # 从随机初始化开始训练
        # ----------------------------------------------------
        echo ">>> 开始训练"
        echo ">>> 初始化方式：random initialization"
        echo ">>> 不使用 pretraining"
        echo ">>> 不加载 --checkpoint"

        python3 training/train.py \
            --model-config "$MODEL_CONFIG" \
            --training-config "$TRAINING_CONFIG" \
            --fold "${fold}" \
            --seed "${seed}"


        # ----------------------------------------------------
        # 只有 train.py 正常完整结束，才建立 done 标志
        # ----------------------------------------------------
        touch "$DONE_FILE"

        echo
        echo "✅ Seed ${seed} Fold ${fold} 完整训练结束"
        echo "✅ 完成标志：$DONE_FILE"

    done


    echo
    echo "############################################################"
    echo "✅ Seed ${seed} 的 5 folds 全部完成"
    echo "############################################################"

done


echo
echo "============================================================"
echo "✅ Black Carp 所有 Seed × Fold 已全部完成"
echo "============================================================"
