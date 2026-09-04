#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
多种子平均预测散点图绘制脚本
功能：
  1. 读取测试集预测 CSV（deep_learning_predictions_test_seed1_3.csv）
  2. 对每个测试样本的多种子、多折预测取平均
  3. 将真实值与预测值统一缩放到 0-1 范围
  4. 绘制包含 train/val/test 散点、密度云、对角线、拟合线和95%置信区间的综合散点图
  5. 每个性状输出一张图（BW、LE）

用法：
  python3 evaluation/plot_multi_seed_average.py
输出：
  evaluation/blackcarp/test_results/BW_multi_seed_average.png
  evaluation/blackcarp/test_results/LE_multi_seed_average.png
"""

import os, sys, json, random
import numpy as np
import pandas as pd
import h5py
import torch
from pathlib import Path
from scipy.stats import pearsonr, t
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import logging, warnings

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False

ROOT = Path.cwd()
os.chdir(ROOT)

TRAINVAL_H5 = "output/blackcarp/blackcarp_preprocessed_fixed_trainval.h5"
TEST_H5 = "output/blackcarp/blackcarp_preprocessed_fixed_test.h5"
MODEL_CONFIG = "config/model_config_blackcarp.json"
PHENO_FILE = "data/blackcarp/phongraph_new.tsv"
TRAINVAL_SAMPLES = "data/blackcarp/trainval_samples.txt"
OUT_DIR = "evaluation/blackcarp/test_results"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [1, 2, 3]
FOLDS = [0, 1, 2, 3, 4]

sys.path.append(str(ROOT))
from training.models.DNAWhisper import DNAWhisper

def get_norm_stats():
    pheno = pd.read_csv(PHENO_FILE, sep='\t').set_index('sample_id')
    with open(TRAINVAL_SAMPLES) as f:
        tv_ids = [line.strip() for line in f if line.strip()]
    pheno_tv = pheno.loc[tv_ids, ['BW','LE']]
    mean = pheno_tv.mean().values.astype(np.float32)
    std = pheno_tv.std().values.astype(np.float32)
    std = np.where(std > 1e-8, std, 1.0)
    return mean, std

def load_h5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        geno = f['features/genotype_features'][:]
        pheno = f['phenotypes'][:]
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['sample_ids'][:]]
    return geno, pheno, ids

def subset_geno(geno_full, seed):
    n_snps = geno_full.shape[0]
    drop = n_snps % 32
    random.seed(seed)
    drop_idx = set(random.sample(range(n_snps), drop))
    keep_idx = [i for i in range(n_snps) if i not in drop_idx]
    return geno_full[keep_idx, :, :]

def main():
    train_mean, train_std = get_norm_stats()
    geno_tv, pheno_tv, tv_ids = load_h5(TRAINVAL_H5)
    geno_test, pheno_test, test_ids = load_h5(TEST_H5)

    # 存储所有预测记录
    all_records = []

    for seed in SEEDS:
        print(f"处理种子 {seed} ...")
        geno_tv_seed = subset_geno(geno_tv, seed)
        geno_test_seed = subset_geno(geno_test, seed)
        x_tv = torch.from_numpy(np.transpose(geno_tv_seed, (1,0,2))).float()
        x_test = torch.from_numpy(np.transpose(geno_test_seed, (1,0,2))).float()

        cv = pd.read_csv(f"data/blackcarp/cv_splits_{seed}.csv")
        tv_id_to_idx = {sid:i for i,sid in enumerate(tv_ids)}

        for fold in FOLDS:
            ckpt_dir = Path(f"output/blackcarp/blackcarp_finetune_cv/blackcarp_{seed}/fold_{fold}")
            ckpt = ckpt_dir / "last.ckpt"
            if not ckpt.exists():
                best = list(ckpt_dir.glob(f"fold{fold}-epoch=*.ckpt"))
                if best:
                    ckpt = max(best, key=lambda p: float(p.name.split('val_pearson_corr_epoch=')[-1].split('.ckpt')[0]))
                else:
                    continue
            model = DNAWhisper.load_from_checkpoint(str(ckpt), config=json.load(open(MODEL_CONFIG)), strict=False)
            model.cpu().eval()

            train_ids = cv[(cv['fold']==fold) & (cv['split']=='train')]['sample_id'].tolist()
            val_ids = cv[(cv['fold']==fold) & (cv['split']=='val')]['sample_id'].tolist()
            train_idx = [tv_id_to_idx[sid] for sid in train_ids if sid in tv_id_to_idx]
            val_idx = [tv_id_to_idx[sid] for sid in val_ids if sid in tv_id_to_idx]

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
                preds_orig = preds * train_std + train_mean

                if split_name == 'test':
                    true_ids = ids_all
                    true_pheno = true_pheno_all
                else:
                    true_ids = [tv_ids[i] for i in idx_list]
                    true_pheno = true_pheno_all[idx_list]

                for j, sid in enumerate(true_ids):
                    all_records.append({
                        'seed': seed,
                        'fold': fold,
                        'split': split_name,
                        'sample_id': sid,
                        'BW_true': true_pheno[j,0],
                        'BW_pred': preds_orig[j,0],
                        'LE_true': true_pheno[j,1],
                        'LE_pred': preds_orig[j,1]
                    })

    # 构建 DataFrame
    df = pd.DataFrame(all_records)
    # 保存原始记录
    df.to_csv(os.path.join(OUT_DIR, "all_predictions_multi_seed.csv"), index=False)

    # 多种子多折平均：按样本ID和split分组，对预测和真实取平均（真实不变）
    avg_df = df.groupby(['split', 'sample_id']).agg(
        BW_true=('BW_true', 'mean'),
        BW_pred=('BW_pred', 'mean'),
        LE_true=('LE_true', 'mean'),
        LE_pred=('LE_pred', 'mean')
    ).reset_index()

    # 绘图
    split_colors = {
        'train': sns.color_palette("Set2")[0],
        'val': sns.color_palette("Set2")[1],
        'test': sns.color_palette("Set2")[2]
    }

    for pheno in ['BW', 'LE']:
        label_col = f"{pheno}_true"
        pred_col = f"{pheno}_pred"

        # 对全部样本的标签和预测统一 MinMax 缩放到 0-1
        scaler = MinMaxScaler()
        all_values = np.concatenate([avg_df[label_col].values.reshape(-1,1), avg_df[pred_col].values.reshape(-1,1)], axis=0)
        scaler.fit(all_values)
        avg_df['label_scaled'] = scaler.transform(avg_df[label_col].values.reshape(-1,1))
        avg_df['pred_scaled'] = scaler.transform(avg_df[pred_col].values.reshape(-1,1))

        plt.figure(figsize=(8, 8))
        all_labels = []
        all_preds = []

        for split in ['train', 'val', 'test']:
            sub = avg_df[avg_df['split'] == split]
            if sub.empty:
                continue
            l = sub['label_scaled'].values
            p = sub['pred_scaled'].values
            all_labels.extend(l)
            all_preds.extend(p)

            if l.std() > 0 and p.std() > 0:
                r, _ = pearsonr(l, p)
                lab = f'{split} (r={r:.2f})'
            else:
                lab = f'{split} (r=N/A)'

            plt.scatter(l, p, color=split_colors[split], alpha=0.6, s=20, label=lab, zorder=2)

        if all_labels:
            np_labels = np.array(all_labels)
            np_preds = np.array(all_preds)

            # 密度云
            if len(np_labels) > 2:
                try:
                    sns.kdeplot(x=np_labels, y=np_preds, fill=True, cmap="RdBu_r",
                                alpha=0.3, levels=7, thresh=0.05, zorder=0)
                except Exception as e:
                    print(f"KDE 失败: {e}")

            # 对角线
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='y=x', zorder=1)

            # 拟合线和置信区间
            if len(np_labels) > 2 and np_labels.std() > 0 and np_preds.std() > 0:
                coeffs, cov_matrix = np.polyfit(np_labels, np_preds, 1, cov=True)
                poly = np.poly1d(coeffs)
                x_fit = np.linspace(0, 1, 200)
                y_fit = poly(x_fit)

                n = len(np_labels)
                dof = n - 2
                t_crit = t.ppf(1 - 0.05 / 2, dof)
                ci_half = np.zeros_like(x_fit)
                for i, xv in enumerate(x_fit):
                    design = np.array([xv, 1.0])
                    se = np.sqrt(design @ cov_matrix @ design.T)
                    ci_half[i] = t_crit * se

                plt.plot(x_fit, y_fit, 'r-', lw=2.5,
                         label=f'Fit (y={coeffs[0]:.2f}x+{coeffs[1]:.2f})', zorder=3)
                plt.fill_between(x_fit, y_fit - ci_half, y_fit + ci_half,
                                 color='red', alpha=0.2, label='95% CI for Fit Line', zorder=2.5)

            plt.title(f'{pheno}: Prediction vs Label (Multi-seed Average)', fontsize=16)
            plt.xlabel(f'{pheno} Label', fontsize=14)
            plt.ylabel(f'{pheno} Prediction', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.tight_layout()
            out_png = os.path.join(OUT_DIR, f"{pheno}_multi_seed_average.png")
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"已保存: {out_png}")

    print("完成！")

if __name__ == "__main__":
    main()
