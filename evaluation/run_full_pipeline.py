#!/usr/bin/env python3
"""
完整的 SNP 重要性分析流程：
权重提取 → 对齐合并 → 元数据/表型生成 → 综合评分 → Top SNP & 曼哈顿图
输出目录：evaluation/blackcarp/enrichment_results/
"""

import os
import sys
import glob
import random
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ========== 配置 ==========
SEEDS = [1, 2, 3]
FOLDS = [0, 1, 2, 3, 4]
TRAINVAL_H5 = "output/blackcarp/blackcarp_preprocessed_fixed_trainval.h5"
PHENO_FILE = "data/blackcarp/phongraph_new.tsv"
TRAINVAL_SAMPLES_FILE = "data/blackcarp/trainval_samples.txt"
OUT_BASE = "evaluation/blackcarp"
ENRICH_DIR = os.path.join(OUT_BASE, "enrichment_results")
os.makedirs(ENRICH_DIR, exist_ok=True)

# 分析参数
WINDOW_SIZE = 25
SIG_THRESHOLD = 3.0
TOP_K = 1000   # 输出前1000个重要SNP

# ========== 工具函数 ==========
def parse_chromosome(c):
    s = str(c)
    if s.lower().startswith('chr'):
        s = s[3:]
    try:
        return int(s)
    except:
        return 1

def minmax_scale(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + eps)

def empirical_p_from_score(score):
    S = len(score)
    rank_asc = rankdata(score, method='average')
    rank_desc = S + 1 - rank_asc
    p = rank_desc / (S + 1)
    neg_log10_p = -np.log10(p + 1e-300)
    return p, neg_log10_p

def compute_snp_importance(W, Y, meta_df, expert_idx, window_size=25):
    """Utility + Correlation + 局部富集 综合评分"""
    B, E, S = W.shape
    W_e = W[:, expert_idx, :]
    y = Y[:, expert_idx]
    eps = 1e-12

    # Utility
    mu = np.mean(W_e, axis=0)
    med = np.median(W_e, axis=0)
    utility_raw = np.log10((mu + eps) / (med + eps))
    utility_exp = np.exp(utility_raw)
    utility = minmax_scale(utility_exp)

    # Correlation
    W_centered = W_e - mu[None, :]
    y_centered = y - y.mean()
    cov = W_centered.T @ y_centered
    denom = np.sqrt(np.sum(W_centered**2, axis=0) * np.sum(y_centered**2)) + eps
    correlation = np.abs(cov / denom)
    correlation[np.std(W_e, axis=0) < eps] = 0.0

    # 局部富集
    df = meta_df.copy()
    df['_orig_idx'] = np.arange(S)
    df['_chr_num'] = df['Chromosome'].apply(parse_chromosome)
    df = df.sort_values(['_chr_num', 'Position']).reset_index(drop=True)
    sorted_idx = df['_orig_idx'].values

    u_sorted = utility[sorted_idx]
    c_sorted = correlation[sorted_idx]

    enrichment_sorted = np.zeros(S)
    smoothed_sorted = np.zeros(S)
    enriched_sorted = np.zeros(S)

    # smoothed correlation
    for chr_val, group in df.groupby('_chr_num'):
        idxs = group.index.values
        chr_c = c_sorted[idxs]
        m = len(idxs)
        for i_local, pos in enumerate(idxs):
            start = max(0, i_local - window_size)
            end = min(m, i_local + window_size + 1)
            win_c = chr_c[start:end]
            med_abs_c = np.median(np.abs(win_c))
            smoothed_sorted[pos] = chr_c[i_local] / (med_abs_c + eps)

    # enrichment & enriched correlation
    for chr_val, group in df.groupby('_chr_num'):
        idxs = group.index.values
        chr_u = u_sorted[idxs]
        chr_s = smoothed_sorted[idxs]
        m = len(idxs)
        for i_local, pos in enumerate(idxs):
            start = max(0, i_local - window_size)
            end = min(m, i_local + window_size + 1)
            win_u = chr_u[start:end]
            top_k_u = max(1, len(win_u) // 2)
            top_u = np.partition(win_u, -top_k_u)[-top_k_u:]
            local_density_u = np.mean(top_u)
            enrichment_sorted[pos] = chr_u[i_local] * (1 + local_density_u)

            win_s = chr_s[start:end]
            top_k_s = max(1, len(win_s) // 2)
            top_s = np.partition(np.abs(win_s), -top_k_s)[-top_k_s:]
            local_density_s = np.mean(top_s)
            enriched_sorted[pos] = chr_s[i_local] * (1 + local_density_s)

    # 映射回原顺序
    enrichment_score = np.zeros(S)
    smoothed_correlation = np.zeros(S)
    enriched_correlation = np.zeros(S)
    enrichment_score[sorted_idx] = enrichment_sorted
    smoothed_correlation[sorted_idx] = smoothed_sorted
    enriched_correlation[sorted_idx] = enriched_sorted

    if enrichment_score.max() > 0:
        enrichment_score = enrichment_score / enrichment_score.max()
    max_abs_enriched = np.max(np.abs(enriched_correlation))
    if max_abs_enriched > 0:
        enriched_correlation = enriched_correlation / max_abs_enriched

    combined_v2 = enrichment_score * (1 + np.abs(enriched_correlation))
    combined_v2 = minmax_scale(combined_v2)

    p, neg_log10_p = empirical_p_from_score(combined_v2)

    return {'combined_v2': combined_v2, 'empirical_p': p, 'neg_log10_p': neg_log10_p}

def plot_manhattan(meta_df, neg_log10_p, title, output_path, sig_threshold=3.0):
    df = meta_df.copy()
    df['neg_log10_p'] = neg_log10_p
    df['chr_num'] = df['Chromosome'].apply(parse_chromosome)
    df = df.sort_values(['chr_num', 'Position']).reset_index(drop=True)

    colors = ["#00008B", "#FF8C00"]
    chroms = sorted(df['chr_num'].unique())
    color_map = {ch: colors[i % len(colors)] for i, ch in enumerate(chroms)}

    x_positions, x_ticks, x_labels = [], [], []
    for i, ch in enumerate(chroms):
        chr_df = df[df['chr_num'] == ch]
        n_points = len(chr_df)
        x = np.linspace(i, i+1, n_points, endpoint=False)
        x_positions.extend(x)
        x_ticks.append(i + 0.5)
        x_labels.append(str(int(ch)))
    df['x'] = x_positions
    df['color'] = df['chr_num'].map(color_map)

    plt.figure(figsize=(14, 6))
    plt.scatter(df['x'], df['neg_log10_p'], c=df['color'], s=15, alpha=0.85, edgecolors='none')
    plt.axhline(y=sig_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    plt.xticks(x_ticks, x_labels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Chromosome', fontsize=14, fontweight='bold')
    plt.ylabel('-log10(empirical P-value)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.2)
    plt.gca().spines['bottom'].set_linewidth(1.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ========== 主流程 ==========
def main():
    print("=" * 60)
    print("开始完整 SNP 重要性分析流程")
    print("=" * 60)

    # 1. 获取 H5 完整 SNP 数和位置信息
    print("\n[1] 读取 H5 获取 SNP 总数...")
    with h5py.File(TRAINVAL_H5, 'r') as f:
        n_snps_full = f['features/genotype_features'].shape[0]
        position_features = f['features/position_features'][:]
    print(f"    H5 中 SNP 总数: {n_snps_full}")

    # 2. 读取训练验证池样本顺序
    print("\n[2] 读取样本顺序...")
    with open(TRAINVAL_SAMPLES_FILE) as f:
        sample_order = [line.strip() for line in f if line.strip()]
    sample_to_idx = {s: i for i, s in enumerate(sample_order)}
    n_samples = len(sample_order)
    print(f"    训练验证池样本数: {n_samples}")

    # 3. 检查权重文件是否存在，缺失则提取
    print("\n[3] 检查权重文件...")
    for seed in SEEDS:
        for fold in FOLDS:
            out_dir = os.path.join(OUT_BASE, f"seed{seed}_fold{fold}")
            if not os.path.exists(out_dir) or not glob.glob(os.path.join(out_dir, "snp_weights_batch_*.npy")):
                print(f"    权重缺失: {out_dir}，开始提取...")
                h5_weights = f"logs0/DNAWhisper_finetune_cv/blackcarp_{seed}/fold_{fold}/val_pooling_weights.h5"
                if not os.path.exists(h5_weights):
                    print(f"    警告: {h5_weights} 不存在，跳过")
                    continue
                os.makedirs(out_dir, exist_ok=True)
                cmd = (f"python evaluation/calSNPweights.py --h5_path {h5_weights} "
                       f"--output_dir {out_dir} --batch_size 32 --normalize --experts 0,1")
                os.system(cmd)
                if not glob.glob(os.path.join(out_dir, "snp_weights_batch_*.npy")):
                    print(f"    错误: 提取失败 {out_dir}")

    # 4. 对齐并合并权重
    print("\n[4] 对齐并合并权重...")
    drop_count = n_snps_full % 32
    print(f"    每个种子随机丢弃 SNP 数: {drop_count}")

    sum_weights = np.zeros((n_samples, 2, n_snps_full))
    count = np.zeros(n_samples)

    for seed in SEEDS:
        random.seed(seed)
        drop_indices = set(random.sample(range(n_snps_full), drop_count))
        keep_indices = [i for i in range(n_snps_full) if i not in drop_indices]
        print(f"    种子 {seed}: 保留 {len(keep_indices)} 个 SNP")

        for fold in FOLDS:
            out_dir = os.path.join(OUT_BASE, f"seed{seed}_fold{fold}")
            files = sorted(glob.glob(os.path.join(out_dir, "snp_weights_batch_*.npy")))
            if not files:
                continue
            weights = np.concatenate([np.load(f) for f in files], axis=0)  # [N_val,2,keep]
            cv_file = f"data/blackcarp/cv_splits_{seed}.csv"
            cv = pd.read_csv(cv_file)
            val_ids = cv[(cv['fold'] == fold) & (cv['split'] == 'val')]['sample_id'].tolist()
            if weights.shape[0] != len(val_ids):
                print(f"    警告: 种子{seed}折{fold} 权重样本数 {weights.shape[0]} 与验证集样本数 {len(val_ids)} 不一致")
                continue
            full_weights = np.zeros((weights.shape[0], 2, n_snps_full))
            full_weights[:, :, keep_indices] = weights
            for i, sid in enumerate(val_ids):
                if sid in sample_to_idx:
                    idx = sample_to_idx[sid]
                    sum_weights[idx] += full_weights[i]
                    count[idx] += 1

    count[count == 0] = 1
    mean_weights = sum_weights / count[:, None, None]
    print(f"    合并后平均权重形状: {mean_weights.shape}")

    # 5. 保存权重、表型、元数据
    print("\n[5] 保存权重、表型、元数据...")
    np.save(os.path.join(OUT_BASE, "snp_weights_train.npy"), mean_weights)

    pheno = pd.read_csv(PHENO_FILE, sep='\t').set_index('sample_id')
    pheno = pheno.reindex(sample_order)
    labels = pheno[['BW', 'LE']].values.astype(np.float32)
    np.save(os.path.join(OUT_BASE, "predictions_phenotype_labels_train.npy"), labels)

    chrom = position_features[:, 0].astype(int)
    position = position_features[:, 1].astype(int)
    snp_ids = [f"{c}:{p}" for c, p in zip(chrom, position)]
    meta_df = pd.DataFrame({'SNP_ID': snp_ids, 'Chromosome': chrom, 'Position': position})
    meta_df.to_csv(os.path.join(OUT_BASE, "snp_metadata.csv"), index=False)
    print(f"    权重、表型、元数据已保存至 {OUT_BASE}")

    # 6. 运行重要性分析
    print("\n[6] 开始重要性分析...")
    W = np.load(os.path.join(OUT_BASE, "snp_weights_train.npy"))
    Y = np.load(os.path.join(OUT_BASE, "predictions_phenotype_labels_train.npy"))
    meta_df = pd.read_csv(os.path.join(OUT_BASE, "snp_metadata.csv"))

    assert W.shape[0] == Y.shape[0], "样本数不一致"
    assert W.shape[2] == len(meta_df), "SNP数不一致"
    assert 'Chromosome' in meta_df.columns and 'Position' in meta_df.columns, "元数据列名错误"

    experts = {0: 'BW', 1: 'LE'}
    for exp_idx in [0, 1]:
        print(f"\n    处理专家: {experts[exp_idx]}")
        res = compute_snp_importance(W, Y, meta_df, exp_idx, WINDOW_SIZE)
        top_df = meta_df.copy()
        top_df['neg_log10_p'] = res['neg_log10_p']
        top_df['empirical_p'] = res['empirical_p']
        top_df['combined_v2'] = res['combined_v2']
        top_df = top_df.sort_values('neg_log10_p', ascending=False).head(TOP_K)
        out_csv = os.path.join(ENRICH_DIR, f"top_snps_{experts[exp_idx]}.csv")
        top_df.to_csv(out_csv, index=False)
        print(f"    Top SNP 保存至: {out_csv}")
        out_png = os.path.join(ENRICH_DIR, f"manhattan_{experts[exp_idx]}.png")
        plot_manhattan(meta_df, res['neg_log10_p'],
                       title=f"SNP Importance Plot - {experts[exp_idx]}",
                       output_path=out_png,
                       sig_threshold=SIG_THRESHOLD)
        print(f"    图保存至: {out_png}")

    print("\n✅ 全部完成！结果保存在:", ENRICH_DIR)
    # 验证 Top SNP 是否都在固定候选集中
    print("\n[7] 验证 Top SNP 与固定候选集的匹配情况...")
    with open("data/blackcarp/fixed_candidates.txt") as f:
        fixed_set = set(line.strip() for line in f if line.strip())

    for exp_idx in [0, 1]:
        trait = experts[exp_idx]
        top_csv = os.path.join(ENRICH_DIR, f"top_snps_{trait}.csv")
        if not os.path.exists(top_csv):
            print(f"    {trait}: 没有 Top SNP 文件，跳过")
            continue
        top_df = pd.read_csv(top_csv)
        missing = set(top_df['SNP_ID']) - fixed_set
        print(f"    {trait}: Top SNP 总数 {len(top_df)}, 缺失 {len(missing)} 个")
        if missing:
            print(f"    缺失示例: {list(missing)[:5]}")


if __name__ == "__main__":
    main()