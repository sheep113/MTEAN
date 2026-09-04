#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
深度学习模型独立测试集评估脚本
功能：
  1. 加载固定测试集 H5 数据
  2. 对每个种子、每个折的微调模型（预训练或非预训练）进行预测
  3. 反标准化预测值到原始表型量纲
  4. 计算 Pearson 相关系数、MSE、R²
  5. 保存各折详细结果与汇总统计

用法：
  python3 evaluation/evaluate_test_performance.py
输出目录：
  evaluation/blackcarp/test_results/
"""

import os, sys, json, random
import numpy as np
import pandas as pd
import h5py
import torch
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import logging, warnings

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False

ROOT = Path.cwd()
os.chdir(ROOT)

TEST_H5 = "output/blackcarp/blackcarp_preprocessed_fixed_test.h5"
MODEL_CONFIG = "config/model_config_blackcarp.json"
OUTPUT_DIR = "evaluation/blackcarp/test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [1, 2, 3]
FOLDS = [0, 1, 2, 3, 4]

sys.path.append(str(ROOT))
from training.models.DNAWhisper import DNAWhisper

def load_test_data():
    with h5py.File(TEST_H5, 'r') as f:
        geno_orig = f['features/genotype_features'][:]  # [SNP, samples, 10]
        pheno = f['phenotypes'][:]                      # [samples, 2]
        sample_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['sample_ids'][:]]
    return geno_orig, pheno, sample_ids

def get_norm_stats():
    """计算训练验证集表型的均值和标准差（与训练时标准化一致）"""
    pheno = pd.read_csv("data/blackcarp/phongraph_new.tsv", sep='\t').set_index('sample_id')
    with open("data/blackcarp/trainval_samples.txt") as f:
        trainval_ids = [line.strip() for line in f if line.strip()]
    pheno_tv = pheno.loc[trainval_ids, ['BW','LE']]
    mean = pheno_tv.mean().values.astype(np.float32)
    std = pheno_tv.std().values.astype(np.float32)
    std = np.where(std > 1e-8, std, 1.0)
    return mean, std

def main():
    # 获取标准化参数
    train_mean, train_std = get_norm_stats()
    print(f"训练验证集均值: {train_mean}, 标准差: {train_std}")

    geno_orig, pheno, sample_ids = load_test_data()
    n_snps_orig = geno_orig.shape[0]
    num_drop = n_snps_orig % 32
    print(f"测试集 SNP 总数: {n_snps_orig}, 每个种子丢弃 {num_drop} 个 SNP")
    print(f"测试集样本数: {len(sample_ids)}")

    all_results = []
    prediction_records = []

    for seed in SEEDS:
        random.seed(seed)
        drop_indices = set(random.sample(range(n_snps_orig), num_drop))
        keep_indices = [i for i in range(n_snps_orig) if i not in drop_indices]
        geno = geno_orig[keep_indices, :, :]
        geno = np.transpose(geno, (1, 0, 2))
        geno_tensor = torch.from_numpy(geno).float()

        for fold in FOLDS:
            fold_dir = Path(f"output/blackcarp/blackcarp_finetune_cv/blackcarp_{seed}/fold_{fold}")
            ckpt = fold_dir / "last.ckpt"
            if not ckpt.exists():
                best_ckpts = list(fold_dir.glob(f"fold{fold}-epoch=*.ckpt"))
                if best_ckpts:
                    ckpt = max(best_ckpts, key=lambda p: float(p.name.split('val_pearson_corr_epoch=')[-1].split('.ckpt')[0]))
                else:
                    print(f"种子{seed} 折{fold} 无检查点，跳过")
                    continue

            print(f"处理 种子{seed} 折{fold}: {ckpt.name}")
            model = DNAWhisper.load_from_checkpoint(str(ckpt), config=json.load(open(MODEL_CONFIG)), strict=False)
            model = model.cpu()
            model.eval()
            model.freeze()

            with torch.no_grad():
                preds = model(geno_tensor.cpu(), None)['final_pred'].cpu().numpy()

            # 反标准化预测值到原始量纲
            preds_orig = preds * train_std + train_mean

            # 保存预测记录
            for sample_idx, sid in enumerate(sample_ids):
                prediction_records.append({
                    'seed': seed,
                    'fold': fold,
                    'sample_id': sid,
                    'BW_true': pheno[sample_idx, 0],
                    'BW_pred': preds_orig[sample_idx, 0],
                    'LE_true': pheno[sample_idx, 1],
                    'LE_pred': preds_orig[sample_idx, 1]
                })

            # 计算指标（使用反标准化后的预测值）
            for idx, trait in enumerate(['BW', 'LE']):
                y_true = pheno[:, idx]
                y_pred = preds_orig[:, idx]
                r, pval = pearsonr(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                all_results.append({
                    'seed': seed,
                    'fold': fold,
                    'trait': trait,
                    'pearson_r': r,
                    'pearson_p': pval,
                    'mse': mse,
                    'r2': r2
                })

    # 保存预测记录
    pred_df = pd.DataFrame(prediction_records)
    pred_csv = os.path.join(OUTPUT_DIR, "deep_learning_predictions_test_seed1_3.csv")
    pred_df.to_csv(pred_csv, index=False)
    print(f"\n预测值已保存到: {pred_csv}")

    # 保存各折结果
    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(OUTPUT_DIR, "deep_learning_test_results_seed1_3.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"各折结果已保存到: {results_csv}")

    # 汇总
    summary = results_df.groupby('trait').agg(
        mean_pearson=('pearson_r', 'mean'),
        std_pearson=('pearson_r', 'std'),
        mean_mse=('mse', 'mean'),
        std_mse=('mse', 'std'),
        mean_r2=('r2', 'mean'),
        std_r2=('r2', 'std')
    ).reset_index()
    summary_csv = os.path.join(OUTPUT_DIR, "deep_learning_test_summary_seed1_3.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"\n深度学习模型测试集平均性能：")
    print(summary)

if __name__ == "__main__":
    main()
