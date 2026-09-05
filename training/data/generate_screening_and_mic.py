#!/usr/bin/env python3
"""
方案3：无独立筛选集，直接在训练验证池上做MIC筛选。
步骤：
1. 分层抽取独立测试集 (默认15%)
2. 剩余为训练验证池
3. 在训练验证池上计算 MIC，分别输出 BW 和 LE 的 top 15000 SNP，同时输出并集
输出文件：
  test_samples.txt
  trainval_samples.txt
  fixed_candidates.txt        # 并集，用于多性状深度学习模型
  fixed_candidates_BW.txt     # BW 单独的 top 15000，用于单性状模型
  fixed_candidates_LE.txt     # LE 单独的 top 15000，用于单性状模型
"""

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from minepy import MINE
import h5py
from sklearn.model_selection import StratifiedShuffleSplit

# ========== 配置 ==========
H5_FILE = 'output/blackcarp/blackcarp_preprocessed 398w.h5'
PHENO_FILE = "data/blackcarp/phongraph_new.tsv"
BIM_PREFIX = "/home/data/biofish/yjn/workspace/deep GS/Whisperer_of_DNA-master/data/blackcarp/filtered_snp_keep"
OUT_DIR = "data/blackcarp"

TOP_K = 15000
ALPHA = 0.6
C = 15
MIN_SAMPLES = 10
N_THREADS = 128
CHUNK_SIZE = 5000
TEST_RATIO = 0.15        # 独立测试集比例
# =========================

def read_bed_as_dosage(bed_path, n_samples, n_snps):
    """读取 PLINK .bed 为 0/1/2 矩阵"""
    with open(bed_path, 'rb') as f:
        magic = f.read(3)
        if magic != b'\x6c\x1b\x01':
            raise ValueError("非标准 .bed 文件")
        bytes_per_snp = (n_samples + 3) // 4
        dosage = np.empty((n_samples, n_snps), dtype=np.float32)
        for snp_idx in range(n_snps):
            raw_bytes = f.read(bytes_per_snp)
            for sample_idx in range(n_samples):
                byte_idx = sample_idx // 4
                bit_shift = (sample_idx % 4) * 2
                code = (raw_bytes[byte_idx] >> bit_shift) & 0x03
                if code == 0b00:
                    dosage[sample_idx, snp_idx] = 0.0
                elif code == 0b01:
                    dosage[sample_idx, snp_idx] = 1.0
                elif code == 0b10:
                    dosage[sample_idx, snp_idx] = 2.0
                else:
                    dosage[sample_idx, snp_idx] = np.nan
        return dosage

def process_chunk(chunk_data):
    snp_chunk, pheno_vals, alpha, c, min_samples = chunk_data
    results = []
    for snp in snp_chunk:
        mask = ~(np.isnan(snp) | np.isnan(pheno_vals))
        x_clean = snp[mask]
        y_clean = pheno_vals[mask]
        if len(x_clean) < min_samples:
            results.append((np.nan, len(x_clean)))
            continue
        if len(np.unique(x_clean)) <= 1 or len(np.unique(y_clean)) <= 1:
            results.append((0.0, len(x_clean)))
            continue
        try:
            mine = MINE(alpha=alpha, c=c)
            mine.compute_score(x_clean, y_clean)
            results.append((mine.mic(), len(x_clean)))
        except:
            results.append((np.nan, len(x_clean)))
    return results

def load_sample_ids_from_h5():
    with h5py.File(H5_FILE, 'r') as f:
        all_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['sample_ids'][:]]
    return all_ids

def stratified_split(pheno, strata, test_size, seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_idx, test_idx in sss.split(pheno, strata):
        return list(train_idx), list(test_idx)

def main():
    if len(sys.argv) < 2:
        print("用法: python generate_screening_and_mic.py <seed>")
        sys.exit(1)
    seed = int(sys.argv[1])

    all_ids = load_sample_ids_from_h5()
    print(f"H5 样本数: {len(all_ids)}")

    # 读取表型
    pheno = pd.read_csv(PHENO_FILE, sep='\t' if PHENO_FILE.endswith('.tsv') else ',')
    pheno = pheno.set_index('sample_id')
    pheno = pheno.reindex(all_ids).dropna(subset=['BW', 'LE'])
    valid_ids = pheno.index.tolist()
    print(f"有效样本数: {len(valid_ids)}")

    # 分层变量
    bw_bins = pd.qcut(pheno['BW'], q=2, labels=False, duplicates='drop')
    le_bins = pd.qcut(pheno['LE'], q=2, labels=False, duplicates='drop')
    strata = bw_bins.astype(str) + '_' + le_bins.astype(str)

    # 划分独立测试集
    try:
        trainval_idx, test_idx = stratified_split(pheno, strata, TEST_RATIO, seed)
    except ValueError:
        indices = np.arange(len(valid_ids))
        np.random.seed(seed)
        np.random.shuffle(indices)
        n_test = int(len(valid_ids) * TEST_RATIO)
        test_idx = indices[:n_test]
        trainval_idx = indices[n_test:]

    test_samples = [valid_ids[i] for i in test_idx]
    trainval_samples = [valid_ids[i] for i in trainval_idx]
    print(f"独立测试集: {len(test_samples)}")
    print(f"训练验证池: {len(trainval_samples)}")

    # 保存样本列表
    with open(os.path.join(OUT_DIR, "test_samples.txt"), 'w') as f:
        f.write('\n'.join(test_samples))
    with open(os.path.join(OUT_DIR, "trainval_samples.txt"), 'w') as f:
        f.write('\n'.join(trainval_samples))

    # 读取基因型数据
    fam = pd.read_csv(f"{BIM_PREFIX}.fam", sep=r'\s+', header=None,
                      names=['fid','iid','pid','mid','sex','pheno'])
    fam['sample_iid'] = fam.apply(lambda r: f"{r['iid']}_{r['iid']}" if r['fid']==r['iid'] else r['iid'], axis=1)
    iids = list(fam['sample_iid'])
    n_samples_total = len(iids)

    bim = pd.read_csv(f"{BIM_PREFIX}.bim", sep=r'\s+', header=None,
                      names=['chr','snp','cm','pos','a1','a2'])
    snp_ids = [f"{row['chr']}:{row['pos']}" for _, row in bim.iterrows()]
    n_snps = len(snp_ids)
    print(f"基因型文件样本数: {n_samples_total}, SNP总数: {n_snps}")

    # 只取训练验证池样本
    trainval_idx_in_geno = [iids.index(sid) for sid in trainval_samples if sid in iids]
    trainval_geno = read_bed_as_dosage(f"{BIM_PREFIX}.bed", n_samples_total, n_snps)[trainval_idx_in_geno, :]
    trainval_iids = [iids[i] for i in trainval_idx_in_geno]

    # 对应表型
    pheno_tv = pheno.reindex(trainval_iids)
    bw_tv = pheno_tv['BW'].values.astype(np.float64)
    le_tv = pheno_tv['LE'].values.astype(np.float64)

    # 定义 MIC 计算函数
    def calc_mic(pheno_vals):
        mic = np.full(n_snps, np.nan, dtype=np.float32)
        chunks = []
        for s in range(0, n_snps, CHUNK_SIZE):
            e = min(s + CHUNK_SIZE, n_snps)
            chunk = trainval_geno[:, s:e].T   # [chunk_snps, n_trainval]
            chunks.append((chunk, pheno_vals, ALPHA, C, MIN_SAMPLES))
        with ProcessPoolExecutor(max_workers=N_THREADS) as ex:
            results = list(tqdm(ex.map(process_chunk, chunks), total=len(chunks), desc="MIC"))
        idx = 0
        for chunk_res in results:
            for mic_val, _ in chunk_res:
                mic[idx] = mic_val
                idx += 1
        return mic

    print("在训练验证池上计算 BW MIC...")
    mic_bw = calc_mic(bw_tv)
    print("在训练验证池上计算 LE MIC...")
    mic_le = calc_mic(le_tv)

    # 取 top 15000 并集，过滤 NaN
    def get_top_valid_indices(mic_values, top_k):
        valid = ~np.isnan(mic_values)
        valid_indices = np.where(valid)[0]
        if len(valid_indices) <= top_k:
            return valid_indices
        sorted_valid = valid_indices[np.argsort(mic_values[valid_indices])[::-1]]
        return sorted_valid[:top_k]

    top_bw = get_top_valid_indices(mic_bw, TOP_K)
    top_le = get_top_valid_indices(mic_le, TOP_K)
    union_idx = np.union1d(top_bw, top_le)

    # 转换成 SNP ID 列表
    bw_snps = [snp_ids[i] for i in top_bw]
    le_snps = [snp_ids[i] for i in top_le]
    union_snps = [snp_ids[i] for i in union_idx]

    # 保存并集（用于多性状深度学习模型）
    fixed_file = os.path.join(OUT_DIR, "fixed_candidates.txt")
    with open(fixed_file, 'w') as f:
        f.write('\n'.join(union_snps))
    print(f"已保存并集候选 SNP: {fixed_file}，共 {len(union_snps)} 个")

    # 保存 BW 单独候选（用于单性状模型）
    bw_file = os.path.join(OUT_DIR, "fixed_candidates_BW.txt")
    with open(bw_file, 'w') as f:
        f.write('\n'.join(bw_snps))
    print(f"已保存 BW 候选 SNP: {bw_file}，共 {len(bw_snps)} 个")

    # 保存 LE 单独候选（用于单性状模型）
    le_file = os.path.join(OUT_DIR, "fixed_candidates_LE.txt")
    with open(le_file, 'w') as f:
        f.write('\n'.join(le_snps))
    print(f"已保存 LE 候选 SNP: {le_file}，共 {len(le_snps)} 个")

if __name__ == "__main__":
    main()