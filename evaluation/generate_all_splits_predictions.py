#!/usr/bin/env python3
"""
生成预训练和非预训练模型在训练集、验证集、独立测试集上的预测值。
输出：
  evaluation/blackcarp/test_results/pretrain_all_splits.csv
  evaluation/blackcarp/test_results/nopretrain_all_splits.csv
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import h5py
import torch
from pathlib import Path
import logging
import warnings

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False

ROOT = Path.cwd()
os.chdir(ROOT)

# ========== 配置 ==========
TRAINVAL_H5 = "output/blackcarp/blackcarp_preprocessed_fixed_trainval.h5"
TEST_H5 = "output/blackcarp/blackcarp_preprocessed_fixed_test.h5"
MODEL_CONFIG = "config/model_config_blackcarp.json"
PHENO_FILE = "data/blackcarp499/phongraph_new.tsv"
TRAINVAL_SAMPLES = "data/blackcarp499/trainval_samples.txt"
OUT_DIR = "evaluation/blackcarp/test_results"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [1, 2, 3]
FOLDS = [0, 1, 2, 3, 4]

sys.path.append(str(ROOT))
from training.models.DNAWhisper import DNAWhisper


def get_norm_stats():
    """计算训练验证集表型的均值和标准差（用于反标准化预测值）"""
    pheno = pd.read_csv(PHENO_FILE, sep='\t').set_index('sample_id')
    with open(TRAINVAL_SAMPLES) as f:
        tv_ids = [line.strip() for line in f if line.strip()]
    pheno_tv = pheno.loc[tv_ids, ['BW', 'LE']]
    mean = pheno_tv.mean().values.astype(np.float32)
    std = pheno_tv.std().values.astype(np.float32)
    std = np.where(std > 1e-8, std, 1.0)
    return mean, std


def load_h5(h5_path):
    """加载 H5 文件，返回基因型、表型和样本ID"""
    with h5py.File(h5_path, 'r') as f:
        geno = f['features/genotype_features'][:]  # [SNP, samples, 10]
        pheno = f['phenotypes'][:]                # [samples, 2]
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['sample_ids'][:]]
    return geno, pheno, ids


def subset_geno(geno_full, seed):
    """根据种子复现训练时的随机丢弃 SNP"""
    n_snps = geno_full.shape[0]
    drop = n_snps % 32
    random.seed(seed)
    drop_idx = set(random.sample(range(n_snps), drop))
    keep_idx = [i for i in range(n_snps) if i not in drop_idx]
    return geno_full[keep_idx, :, :]


def find_checkpoint(checkpoint_dir, fold):
    """查找最佳或最后一个检查点"""
    ckpt = checkpoint_dir / "last.ckpt"
    if ckpt.exists():
        return ckpt
    # 否则查找最佳检查点（文件名包含 val_pearson_corr_epoch）
    best_ckpts = list(checkpoint_dir.glob(f"fold{fold}-epoch=*.ckpt"))
    if best_ckpts:
        # 按 val_pearson_corr_epoch 值排序，取最大值
        ckpt = max(best_ckpts, key=lambda p: float(p.name.split('val_pearson_corr_epoch=')[-1].split('.ckpt')[0]))
        return ckpt
    return None


def generate_predictions(checkpoint_base, out_name):
    """对指定模型（预训练或非预训练）生成所有数据集的预测"""
    train_mean, train_std = get_norm_stats()
    geno_tv, pheno_tv, tv_ids = load_h5(TRAINVAL_H5)
    geno_test, pheno_test, test_ids = load_h5(TEST_H5)

    records = []

    for seed in SEEDS:
        # 子集基因型（按种子丢弃）
        geno_tv_seed = subset_geno(geno_tv, seed)
        geno_test_seed = subset_geno(geno_test, seed)

        # 转置为 [samples, SNP, 10]
        x_tv = torch.from_numpy(np.transpose(geno_tv_seed, (1, 0, 2))).float()
        x_test = torch.from_numpy(np.transpose(geno_test_seed, (1, 0, 2))).float()

        cv = pd.read_csv(f"data/blackcarp499/cv_splits_{seed}.csv")
        tv_id_to_idx = {sid: i for i, sid in enumerate(tv_ids)}

        for fold in FOLDS:
            ckpt_dir = Path(checkpoint_base.format(seed=seed, fold=fold))
            ckpt = find_checkpoint(ckpt_dir, fold)
            if ckpt is None:
                print(f"种子{seed} 折{fold} 无检查点，跳过")
                continue

            print(f"处理 种子{seed} 折{fold}: {ckpt.name}")
            model = DNAWhisper.load_from_checkpoint(str(ckpt), config=json.load(open(MODEL_CONFIG)), strict=False)
            model.cpu().eval()

            train_ids = cv[(cv['fold'] == fold) & (cv['split'] == 'train')]['sample_id'].tolist()
            val_ids = cv[(cv['fold'] == fold) & (cv['split'] == 'val')]['sample_id'].tolist()
            train_idx = [tv_id_to_idx[sid] for sid in train_ids if sid in tv_id_to_idx]
            val_idx = [tv_id_to_idx[sid] for sid in val_ids if sid in tv_id_to_idx]

            # 处理三个数据集
            for split_name, idx_list, x_all, true_pheno_all, ids_all in [
                ('train', train_idx, x_tv, pheno_tv, tv_ids),
                ('val', val_idx, x_tv, pheno_tv, tv_ids),
                ('test', list(range(len(test_ids))), x_test, pheno_test, test_ids)
            ]:
                if not idx_list:
                    continue

                x_sub = x_all[idx_list]
                with torch.no_grad():
                    preds = model(x_sub.cpu(), None)['final_pred'].cpu().numpy()
                # 反标准化预测值
                preds_orig = preds * train_std + train_mean

                if split_name == 'test':
                    true_ids = ids_all
                    true_pheno = true_pheno_all
                else:
                    true_ids = [tv_ids[i] for i in idx_list]
                    true_pheno = true_pheno_all[idx_list]

                for j, sid in enumerate(true_ids):
                    records.append({
                        'split': split_name,
                        'sample_id': sid,
                        'BW_true': true_pheno[j, 0],
                        'BW_pred': preds_orig[j, 0],
                        'LE_true': true_pheno[j, 1],
                        'LE_pred': preds_orig[j, 1]
                    })

    df = pd.DataFrame(records)
    out_path = os.path.join(OUT_DIR, out_name)
    df.to_csv(out_path, index=False)
    print(f"已保存预测结果: {out_path}")


if __name__ == "__main__":
    # 预训练模型
    generate_predictions(
        checkpoint_base="output/blackcarp/blackcarp_finetune_cv/blackcarp_{seed}/fold_{fold}",
        out_name="pretrain_all_splits.csv"
    )

    # 非预训练模型
    generate_predictions(
        checkpoint_base="output/blackcarp/blackcarp_finetune_cv_nopretrain/blackcarp_{seed}/fold_{fold}",
        out_name="nopretrain_all_splits.csv"
    )