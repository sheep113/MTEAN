#!/usr/bin/env python3
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
PHENO_FILE = "data/blackcarp499/phongraph_new.tsv"
TRAINVAL_SAMPLES = "data/blackcarp499/trainval_samples.txt"
OUTPUT_DIR = "evaluation/blackcarp/test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def load_test_data():
    with h5py.File(TEST_H5, 'r') as f:
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
    geno_orig, pheno, sample_ids = load_test_data()
    n_snps_orig = geno_orig.shape[0]
    num_drop = n_snps_orig % 32
    print(f"测试集 SNP 总数: {n_snps_orig}, 每个种子丢弃 {num_drop} 个 SNP")

    all_results = []

    for seed in SEEDS:
        random.seed(seed)
        drop_indices = set(random.sample(range(n_snps_orig), num_drop))
        keep_indices = [i for i in range(n_snps_orig) if i not in drop_indices]
        geno = geno_orig[keep_indices, :, :]
        geno = np.transpose(geno, (1, 0, 2))
        geno_tensor = torch.from_numpy(geno).float()

        for fold in FOLDS:
            ckpt_dir = Path(f"output/blackcarp/blackcarp_finetune_cv_nopretrain/blackcarp_{seed}/fold_{fold}")
            ckpt = ckpt_dir / "last.ckpt"
            if not ckpt.exists():
                best = list(ckpt_dir.glob(f"fold{fold}-epoch=*.ckpt"))
                if best:
                    ckpt = max(best, key=lambda p: float(p.name.split('val_pearson_corr_epoch=')[-1].split('.ckpt')[0]))
                else:
                    print(f"种子{seed} 折{fold} 无检查点，跳过")
                    continue

            print(f"处理 种子{seed} 折{fold}: {ckpt.name}")
            model = DNAWhisper.load_from_checkpoint(str(ckpt), config=json.load(open(MODEL_CONFIG)), strict=False)
            model.cpu().eval()

            with torch.no_grad():
                preds = model(geno_tensor.cpu(), None)['final_pred'].cpu().numpy()
            preds_orig = preds * train_std + train_mean

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

    results_df = pd.DataFrame(all_results)
    print("\n非预训练模型各折测试集性能：")
    print(results_df)

    summary = results_df.groupby('trait').agg(
        mean_pearson=('pearson_r', 'mean'),
        std_pearson=('pearson_r', 'std'),
        mean_mse=('mse', 'mean'),
        std_mse=('mse', 'std'),
        mean_r2=('r2', 'mean'),
        std_r2=('r2', 'std')
    ).reset_index()
    print("\n非预训练模型测试集平均性能：")
    print(summary)

    results_df.to_csv(os.path.join(OUTPUT_DIR, "nopretrain_test_results.csv"), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "nopretrain_test_summary.csv"), index=False)
    print(f"结果已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
