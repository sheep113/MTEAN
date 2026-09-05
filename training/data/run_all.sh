#!/bin/bash
cd "/home/data/biofish/yjn/workspace/deep GS/Whisperer_of_DNA-master"

SEEDS=(1 2 3 4 5 )

for seed in "${SEEDS[@]}"; do
    echo "============================================"
    echo "        处理种子: $seed"
    echo "============================================"

    # 1. 生成该种子的交叉验证划分文件（如果不存在）
    if [ ! -f "data/blackcarp/cv_splits_${seed}.csv" ]; then
        echo ">>> 生成种子 ${seed} 的交叉验证划分..."
        python3 training/data/generate_cv_splits.py ${seed}
    else
        echo ">>> 种子 ${seed} 的 cv_splits 已存在，跳过"
    fi

    # 2. 预训练（如果检查点不存在）
    PRETRAIN_CKPT="output/blackcarp/blackcarp_pretrain/blackcarp_${seed}/last.ckpt"
    if [ ! -f "$PRETRAIN_CKPT" ]; then
        echo ">>> 预训练 (种子 ${seed})..."
        python3 training/train.py \
            --model-config config/model_config_blackcarp_pretrain.json \
            --training-config training/config/training_config_pretrain.yml \
            --seed ${seed}
    else
        echo ">>> 预训练检查点已存在，跳过"
    fi

    # 3. 微调（5折）
    for fold in 0 1 2 3 4; do
        FINETUNE_DIR="output/blackcarp/blackcarp_finetune_cv/blackcarp_${seed}/fold_${fold}"
        if [ -z "$(ls -A $FINETUNE_DIR 2>/dev/null)" ]; then
            echo ">>> 微调 种子${seed} Fold${fold}..."
            python3 training/train.py \
                --model-config config/model_config_blackcarp.json \
                --training-config training/config/training_config.yml \
                --fold ${fold} \
                --seed ${seed} \
                --checkpoint "$PRETRAIN_CKPT"
        else
            echo ">>> 种子${seed} Fold${fold} 已有检查点，跳过"
        fi
    done
done

echo "全部种子处理完成！"