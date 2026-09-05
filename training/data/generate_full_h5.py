#!/usr/bin/env python3
"""根据SNP列表生成包含所有样本的HDF5文件"""
import sys
import h5py, pandas as pd, numpy as np
from pathlib import Path

FULL_H5 = "output/blackcarp/blackcarp_preprocessed 398w.h5"
SNP_INFO = "output/blackcarp/blackcarp_preprocessed 398w"
OUT_DIR = "output/blackcarp"
SNP_LIST = "data/blackcarp/chip_candidates_seed43.txt"   # 修改为你的列表路径

def main():
    with open(SNP_LIST) as f:
        selected_snps = [line.strip() for line in f if line.strip()]
    print(f"目标SNP数量: {len(selected_snps)}")

    snp_df = pd.read_csv(SNP_INFO, sep='\t')
    all_snp_ids = snp_df['snp_id'].tolist()
    idx = [all_snp_ids.index(sid) for sid in selected_snps if sid in all_snp_ids]
    print(f"实际匹配数量: {len(idx)}")

    with h5py.File(FULL_H5, 'r') as src:
        geno = src['features/genotype_features'][:][idx, :, :]
        pos  = src['features/position_features'][:][idx, :]
        with h5py.File(f"{OUT_DIR}/blackcarp_preprocessed_full.h5", 'w') as dst:
            dst.create_dataset('features/genotype_features', data=geno)
            dst.create_dataset('features/position_features', data=pos)
            dst.create_dataset('phenotypes', data=src['phenotypes'][:])
            dst.create_dataset('phenotype_names', data=src['phenotype_names'][:])
            dst.create_dataset('sample_ids', data=src['sample_ids'][:])
            if 'phenotypes_na_mask' in src:
                dst.create_dataset('phenotypes_na_mask', data=src['phenotypes_na_mask'][:])
    print("全量HDF5文件已生成: output/blackcarp/blackcarp_preprocessed_full.h5")

if __name__ == "__main__":
    main()