#!/usr/bin/env python3
"""
LightGBM 和 SVR 在独立测试集上的评估（使用 geno_fixed CSV，原始模型参数）
数据源：geno_fixed_for_python.csv + 分性状候选SNP + trainval/test 划分 + cv_splits
输出：evaluation/blackcarp/ml_models/test_results/
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

# 配置
GENO_CSV = "data/blackcarp/geno_fixed_for_python.csv"          # 基因型矩阵 CSV（行名样本ID，列名SNP ID）
FIXED_CAND_BW = "data/blackcarp/fixed_candidates_BW.txt"
FIXED_CAND_LE = "data/blackcarp/fixed_candidates_LE.txt"
PHENO_FILE = "data/blackcarp/phongraph_new.tsv"
TRAINVAL_SAMPLES = "data/blackcarp/trainval_samples.txt"
TEST_SAMPLES = "data/blackcarp/test_samples.txt"
OUT_DIR = "evaluation/blackcarp/ml_models/test_results"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [1, 2, 3]
FOLDS = [0, 1, 2, 3, 4]
TRAITS = ['BW', 'LE']

def main():
    # 读取基因型 CSV
    print("读取基因型 CSV ...")
    geno_df = pd.read_csv(GENO_CSV, index_col=0)   # 行索引 = 样本ID，列 = SNP ID
    sample_ids = geno_df.index.tolist()
    all_snp_ids = geno_df.columns.tolist()
    print(f"样本数: {len(sample_ids)}, SNP总数: {len(all_snp_ids)}")
    print(f"基因型矩阵总缺失率: {geno_df.isna().mean().mean():.4f}")

    # 读取表型
    pheno = pd.read_csv(PHENO_FILE, sep='\t').set_index('sample_id')
    pheno = pheno.reindex(sample_ids)   # 按基因型样本顺序对齐
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

    # 读取候选 SNP
    cand_files = {'BW': FIXED_CAND_BW, 'LE': FIXED_CAND_LE}
    snp_lists = {}
    for trait in TRAITS:
        with open(cand_files[trait]) as f:
            snps = [line.strip() for line in f if line.strip()]
        valid_snps = [s for s in snps if s in all_snp_ids]
        print(f"性状 {trait}: 候选SNP {len(snps)} 个，匹配 {len(valid_snps)} 个")
        snp_lists[trait] = valid_snps

    # 模型评估
    all_results = []

    for seed in SEEDS:
        cv_file = f"data/blackcarp/cv_splits_{seed}.csv"
        cv = pd.read_csv(cv_file)
        print(f"\n处理种子 {seed} ...")

        for trait in TRAITS:
            print(f"  表型 {trait}")
            y_all = phenotypes[:, TRAITS.index(trait)]
            X_all = geno_df[snp_lists[trait]].values  # 提取该性状的特征矩阵

            for fold in FOLDS:
                train_ids = cv[(cv['fold'] == fold) & (cv['split'] == 'train')]['sample_id'].tolist()
                train_local_idx = [sid2idx[x] for x in train_ids if x in sid2idx]
                test_local_idx = test_idx

                X_train = X_all[train_local_idx, :]
                y_train = y_all[train_local_idx]
                X_test = X_all[test_local_idx, :]
                y_test = y_all[test_local_idx]

                # 缺失值处理：用训练集中位数填充
                train_medians = np.nanmedian(X_train, axis=0)
                train_medians = np.where(np.isnan(train_medians), 0, train_medians)
                X_train = np.where(np.isnan(X_train), train_medians, X_train)
                X_test = np.where(np.isnan(X_test), train_medians, X_test)

                # LightGBM（使用原始自定义参数）
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

                # SVR（使用原始参数）
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

    results_df.to_csv(os.path.join(OUT_DIR, "lightgbm_svr_test_results_seed1_3.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "lightgbm_svr_test_summary_seed1_3.csv"), index=False)
    print(f"\n结果已保存到 {OUT_DIR}")

if __name__ == "__main__":
    main()