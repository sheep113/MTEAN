#!/usr/bin/env python3
"""调试版：根据固定候选 SNP 列表生成训练验证/测试 H5，并检查写入数量"""
import os, sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

PLINK_PREFIX = "/home/data/biofish/yjn/workspace/deep GS/Whisperer_of_DNA-master/data/blackcarp/filtered_snp_keep"
FIXED_CAND = "data/blackcarp/fixed_candidates.txt"
PHENO_FILE = "data/blackcarp/phongraph_new.tsv"
TRAINVAL_SAMPLES = "data/blackcarp/trainval_samples.txt"
TEST_SAMPLES = "data/blackcarp/test_samples.txt"
OUT_DIR = "output/blackcarp"
OUT_TRAINVAL_H5 = os.path.join(OUT_DIR, "blackcarp_preprocessed_fixed_trainval.h5")
OUT_TEST_H5 = os.path.join(OUT_DIR, "blackcarp_preprocessed_fixed_test.h5")

def read_bed_subset(bed_path, n_samples, snp_indices):
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
                if code == 0b00: geno[s, i] = 0.0
                elif code == 0b01: geno[s, i] = 1.0
                elif code == 0b10: geno[s, i] = 2.0
                else: geno[s, i] = np.nan
    return geno

def one_hot_encode(geno):
    n, m = geno.shape
    out = np.zeros((n, m, 10), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            v = geno[i, j]
            if not np.isnan(v): out[i, j, int(v)] = 1.0
    return out

def main():
    # 读取 FAM/BIM
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
    missing = len(fixed_snps) - len(valid_snps)
    print(f"在 BIM 中未找到的候选 SNP: {missing}")
    snp_indices = np.sort([snp_to_idx[sid] for sid in valid_snps])
    print(f"实际要提取的 SNP 索引数: {len(snp_indices)}")

    # 读取基因型
    geno_raw = read_bed_subset(f"{PLINK_PREFIX}.bed", n_samples, snp_indices)
    print(f"读取基因型后形状: {geno_raw.shape}")

    geno_onehot = one_hot_encode(geno_raw)
    print(f"独热编码后形状: {geno_onehot.shape}")

    geno_snp_first = np.transpose(geno_onehot, (1, 0, 2))
    print(f"转置后 [SNP,样本,10] 形状: {geno_snp_first.shape}")

    # 位置特征
    bim_sel = bim.iloc[snp_indices].reset_index(drop=True)
    pos_feat = np.zeros((len(snp_indices), 6), dtype=np.float64)
    pos_feat[:, 0] = bim_sel['chr'].astype(np.float64)
    pos_feat[:, 1] = bim_sel['pos'].astype(np.float64)
    pos_feat[:, 2] = bim_sel['pos'].astype(np.float64)
    pos_feat[:, 3:] = [1000.0, 0.001, 0.5]
    print(f"位置特征行数: {pos_feat.shape[0]}")

    # 表型
    pheno = pd.read_csv(PHENO_FILE, sep='\t').set_index('sample_id')
    pheno = pheno.reindex(sample_ids)
    phenotypes = np.stack([pheno['BW'].values, pheno['LE'].values], axis=1).astype(np.float32)
    pheno_names = np.array(['BW', 'LE'], dtype='S')

    # 样本分组
    with open(TRAINVAL_SAMPLES) as f:
        tv_ids = [x.strip() for x in f if x.strip()]
    with open(TEST_SAMPLES) as f:
        test_ids = [x.strip() for x in f if x.strip()]

    sid2idx = {sid: i for i, sid in enumerate(sample_ids)}
    tv_idx = [sid2idx[x] for x in tv_ids if x in sid2idx]
    test_idx = [sid2idx[x] for x in test_ids if x in sid2idx]
    print(f"训练验证样本数: {len(tv_idx)}, 测试样本数: {len(test_idx)}")

    def write_h5(out_path, sample_idx_list):
        # 删除旧文件（确保覆盖）
        if os.path.exists(out_path):
            os.remove(out_path)
        with h5py.File(out_path, 'w') as h5:
            h5.create_dataset('features/genotype_features', data=geno_snp_first[:, sample_idx_list, :])
            h5.create_dataset('features/position_features', data=pos_feat)
            h5.create_dataset('phenotypes', data=phenotypes[sample_idx_list, :])
            h5.create_dataset('phenotype_names', data=pheno_names)
            h5.create_dataset('sample_ids', data=np.array([sample_ids[i] for i in sample_idx_list], dtype='S'))
            h5.create_dataset('phenotypes_na_mask', data=np.zeros((len(sample_idx_list), 2), dtype=np.uint8))
        print(f"已写入 {out_path}")

    write_h5(OUT_TRAINVAL_H5, tv_idx)
    write_h5(OUT_TEST_H5, test_idx)

    # 立即读回验证
    with h5py.File(OUT_TRAINVAL_H5, 'r') as f:
        print(f"验证：trainval H5 genotype_features 行数: {f['features/genotype_features'].shape[0]}")
        print(f"验证：trainval H5 position_features 行数: {f['features/position_features'].shape[0]}")

if __name__ == "__main__":
    main()
