import numpy as np
import pandas as pd

# 配置路径
PLINK_PREFIX = "/home/data/biofish/yjn/workspace/MTEAN/data/blackcarp499/filtered_snp_keep"
CAND_BW = "data/blackcarp499/fixed_candidates_BW.txt"
CAND_LE = "data/blackcarp499/fixed_candidates_LE.txt"

def read_bed_snps(bed_path, n_samples, snp_indices):
    """读取指定索引的 SNP 基因型，返回 (n_samples, len(snp_indices))"""
    bytes_per_snp = (n_samples + 3) // 4
    geno = np.empty((n_samples, len(snp_indices)), dtype=np.float32)
    with open(bed_path, 'rb') as f:
        f.seek(3)  # 跳过魔数
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
    # 读取 fam / bim
    fam = pd.read_csv(f"{PLINK_PREFIX}.fam", sep=r'\s+', header=None,
                      names=['fid','iid','pid','mid','sex','pheno'])
    bim = pd.read_csv(f"{PLINK_PREFIX}.bim", sep=r'\s+', header=None,
                      names=['chr','snp','cm','pos','a1','a2'])
    n_samples = len(fam)
    bim['snp_id'] = bim['chr'].astype(str) + ':' + bim['pos'].astype(str)
    snp_to_idx = {sid: i for i, sid in enumerate(bim['snp_id'])}

    # 处理每个性状
    for trait, cand_file in [('BW', CAND_BW), ('LE', CAND_LE)]:
        with open(cand_file) as f:
            cand_snps = [line.strip() for line in f if line.strip()]
        print(f"\n===== 性状 {trait} =====")
        print(f"候选 SNP 总数: {len(cand_snps)}")

        # 找到在 BIM 中存在的 SNP 及其索引
        valid_snps = [s for s in cand_snps if s in snp_to_idx]
        print(f"在 BIM 中匹配的 SNP 数: {len(valid_snps)}")
        if not valid_snps:
            print("没有匹配，跳过。")
            continue
        indices = [snp_to_idx[s] for s in valid_snps]
        indices = np.sort(indices)

        # 读取这些 SNP 的基因型
        geno = read_bed_snps(f"{PLINK_PREFIX}.bed", n_samples, indices)
        missing_rate = np.isnan(geno).mean(axis=0)
        total_missing = np.isnan(geno).mean()

        print(f"这些 SNP 的总缺失率: {total_missing:.4f} ({total_missing*100:.2f}%)")
        print(f"每个 SNP 缺失率分布:")
        print(f"  最小值: {missing_rate.min():.4f}")
        print(f"  最大值: {missing_rate.max():.4f}")
        print(f"  中位数: {np.median(missing_rate):.4f}")
        print(f"  缺失率 >5% 的 SNP 数: {(missing_rate>0.05).sum()}")
        print(f"  缺失率 >10% 的 SNP 数: {(missing_rate>0.10).sum()}")
        print(f"  缺失率 >50% 的 SNP 数: {(missing_rate>0.50).sum()}")

if __name__ == "__main__":
    main()