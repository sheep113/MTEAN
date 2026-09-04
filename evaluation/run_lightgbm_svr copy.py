#!/usr/bin/env python3
"""
LightGBM 和 SVR 在独立测试集上的评估脚本（种子1-3）
数据源：PLINK 二进制文件 + fixed_candidates.txt + trainval/test 划分 + cv_splits
输出：evaluation/blackcarp/ml_models/test_results/
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

# 项目根目录
ROOT = Path.cwd()
os.chdir(ROOT)

# 配置
PLINK_PREFIX = "/home/data/biofish/yjn/workspace/deep GS/Whisperer_of_DNA-master/data/blackcarp/filtered_snp_keep"
FIXED_CAND = "data/blackcarp/fixed_candidates.txt"
PHENO_FILE = "data/blackcarp/phongraph_new.tsv"
TRAINVAL_SAMPLES = "data/blackcarp/trainval_samples.txt"
TEST_SAMPLES = "data/blackcarp/test_samples.txt"
OUT_DIR = "evaluation/blackcarp/ml_models/test_results"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [1, 2, 3]
FOLDS = [0, 1, 2, 3, 4]
TRAITS = ['BW', 'LE']

def read_bed_subset(bed_path, n_samples, snp_indices):
    """从 .bed 读取指定 SNP 索引的基因型（0/1/2/NaN），返回 (n_samples, n_snps)"""
    bytes_per_snp = (n_samples + 3) // 4
    geno = np.empty((n_samples, len(snp_indices)), dtype=np.float32)
    with open(bed_path, 'rb') as f:
        f.seek(3)
        for i, idx in enumerate(snp_indices):
            f.seek(3 + idx * bytes_per_snp)
            raw = f.read(bytes_per_snp)
            for s in range(n_samples):
                byte = s // 4
                shift = (s % 4) * 2
                code = (raw[byte] >> shift) & 0x03
                if code == 0b00:
                    geno[s, i] = 0.0
                elif code == 0b01:
                    geno[s, i] = 1.0
                elif code == 0b10:
                    geno[s, i] = 2.0
                else:
                    geno[s, i] = np.nan
    return geno

def main():
    # 读取 FAM / BIM
    fam = pd.read_csv(f"{PLINK_PREFIX}.fam", sep=r'\s+', header=None,
                      names=['fid','iid','pid','mid','sex','pheno'])
    bim = pd.read_csv(f"{PLINK_PREFIX}.bim", sep=r'\s+', header=None,
                      names=['chr','snp','cm','pos','a1','a2'])
    sample_ids = (fam['iid'] + '_' + fam['iid']).tolist()
    n_samples = len(sample_ids)
    print(f"样本数: {n_samples}")

    # 读取候选 SNP
    with open(FIXED_CAND) as f:
        fixed_snps = [line.strip() for line in f if line.strip()]
    print(f"候选 SNP 数: {len(fixed_snps)}")

    bim['snp_id'] = bim['chr'].astype(str) + ':' + bim['pos'].astype(str)
    snp_to_idx = {sid: i for i, sid in enumerate(bim['snp_id'])}
    valid_snps = [sid for sid in fixed_snps if sid in snp_to_idx]
    print(f"实际匹配 SNP 数: {len(valid_snps)}")
    snp_indices = np.sort([snp_to_idx[sid] for sid in valid_snps])

    # 读取基因型矩阵
    print("读取基因型...")
    geno = read_bed_subset(f"{PLINK_PREFIX}.bed", n_samples, snp_indices)  # [n_samples, n_snps]
    print(f"基因型矩阵形状: {geno.shape}")

    # 读取表型，对齐样本顺序
    pheno = pd.read_csv(PHENO_FILE, sep='\t').set_index('sample_id')
    pheno = pheno.reindex(sample_ids)
    # 提取 BW 和 LE
    phenotypes = pheno[['BW', 'LE']].values.astype(np.float64)
    print(f"表型矩阵形状: {phenotypes.shape}")

    # 读取样本划分
    with open(TRAINVAL_SAMPLES) as f:
        trainval_ids = [line.strip() for line in f if line.strip()]
    with open(TEST_SAMPLES) as f:
        test_ids = [line.strip() for line in f if line.strip()]

    sid2idx = {sid: i for i, sid in enumerate(sample_ids)}
    trainval_idx = [sid2idx[x] for x in trainval_ids if x in sid2idx]
    test_idx = [sid2idx[x] for x in test_ids if x in sid2idx]
    print(f"训练验证样本数: {len(trainval_idx)}, 测试样本数: {len(test_idx)}")

    # 准备结果容器
    all_results = []

    # 循环种子
    for seed in SEEDS:
        cv_file = f"data/blackcarp/cv_splits_{seed}.csv"
        cv = pd.read_csv(cv_file)
        print(f"\n处理种子 {seed} ...")

        for trait in TRAITS:
            print(f"  表型 {trait}")
            y_all = phenotypes[:, TRAITS.index(trait)]

            for fold in FOLDS:
                train_ids = cv[(cv['fold'] == fold) & (cv['split'] == 'train')]['sample_id'].tolist()
                # 训练集索引（在 trainval 内）
                train_local_idx = [sid2idx[x] for x in train_ids if x in sid2idx]
                # 测试集索引（独立测试集）
                test_local_idx = test_idx

                X_train = geno[train_local_idx, :]
                y_train = y_all[train_local_idx]
                X_test = geno[test_local_idx, :]
                y_test = y_all[test_local_idx]

                # 处理缺失值：用训练集中位数填充
                train_medians = np.nanmedian(X_train, axis=0)
                # 防止所有值为NaN
                train_medians = np.where(np.isnan(train_medians), 0, train_medians)
                X_train = np.where(np.isnan(X_train), train_medians, X_train)
                X_test = np.where(np.isnan(X_test), train_medians, X_test)

                # LightGBM
                lgb = LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=15,
                    max_depth=5,
                    random_state=seed,
                    verbose=-1
                )
                lgb.fit(X_train, y_train)
                y_pred_lgb = lgb.predict(X_test)
                r_lgb, _ = pearsonr(y_test, y_pred_lgb)
                mse_lgb = mean_squared_error(y_test, y_pred_lgb)
                r2_lgb = r2_score(y_test, y_pred_lgb)

                # SVR
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                svr.fit(X_train_scaled, y_train)
                y_pred_svr = svr.predict(X_test_scaled)
                r_svr, _ = pearsonr(y_test, y_pred_svr)
                mse_svr = mean_squared_error(y_test, y_pred_svr)
                r2_svr = r2_score(y_test, y_pred_svr)

                all_results.append({
                    'seed': seed,
                    'fold': fold,
                    'trait': trait,
                    'model': 'LightGBM',
                    'pearson_r': r_lgb,
                    'mse': mse_lgb,
                    'r2': r2_lgb
                })
                all_results.append({
                    'seed': seed,
                    'fold': fold,
                    'trait': trait,
                    'model': 'SVR',
                    'pearson_r': r_svr,
                    'mse': mse_svr,
                    'r2': r2_svr
                })

                print(f"    Fold {fold}: LightGBM r={r_lgb:.4f}, SVR r={r_svr:.4f}")

    # 汇总
    results_df = pd.DataFrame(all_results)
    print("\n各折结果预览：")
    print(results_df.head(10))

    # 按模型和性状汇总
    summary = results_df.groupby(['model', 'trait']).agg(
        mean_pearson=('pearson_r', 'mean'),
        std_pearson=('pearson_r', 'std'),
        mean_mse=('mse', 'mean'),
        std_mse=('mse', 'std'),
        mean_r2=('r2', 'mean'),
        std_r2=('r2', 'std')
    ).reset_index()
    print("\nLightGBM 与 SVR 测试集平均性能：")
    print(summary)

    # 保存
    results_df.to_csv(os.path.join(OUT_DIR, "lightgbm_svr_test_results_seed1_3.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "lightgbm_svr_test_summary_seed1_3.csv"), index=False)
    print(f"\n结果已保存到 {OUT_DIR}")

if __name__ == "__main__":
    main()