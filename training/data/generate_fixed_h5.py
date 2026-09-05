#!/usr/bin/env python3
"""
根据 fixed_candidates.txt
从 PLINK 提取候选 SNP。

使用项目统一的 10 维碱基组合编码，
生成：

1. blackcarp_preprocessed_fixed_trainval.h5
2. blackcarp_preprocessed_fixed_test.h5
"""

import os
import sys

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from utils.snp_utils import batch_encode_genotypes


PLINK_PREFIX = '/home/data/biofish/yjn/workspace/MTEAN/data/blackcarp499/filtered_snp_keep'

FIXED_CAND = (
    "data/blackcarp499/"
    "fixed_candidates.txt"
)

PHENO_FILE = (
    "data/blackcarp499/"
    "phongraph_new.tsv"
)

TRAINVAL_SAMPLES = (
    "data/blackcarp499/"
    "trainval_samples.txt"
)

TEST_SAMPLES = (
    "data/blackcarp499/"
    "test_samples.txt"
)

OUT_DIR = (
    "output/blackcarp"
)

OUT_TRAINVAL_H5 = os.path.join(
    OUT_DIR,
    "blackcarp_preprocessed_fixed_trainval.h5"
)

OUT_TEST_H5 = os.path.join(
    OUT_DIR,
    "blackcarp_preprocessed_fixed_test.h5"
)



def normalize_chr(x):
    """
    统一染色体表示：
      1     -> "1"
      1.0   -> "1"
      "1.0" -> "1"
      X     -> "X"
    """
    s = str(x).strip()

    try:
        v = float(s)
        if v.is_integer():
            return str(int(v))
    except (ValueError, TypeError):
        pass

    return s


def normalize_pos(x):
    """
    SNP 物理位置必须按整数处理：
      123456
      123456.0
      "123456.0"
    全部统一成：
      "123456"
    """
    s = str(x).strip()

    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        raise ValueError(f"无法解析 SNP position: {x!r}")


def make_chr_pos_id(chrom, pos):
    """统一 SNP ID：chr:position"""
    return f"{normalize_chr(chrom)}:{normalize_pos(pos)}"


def read_bed_subset(
    bed_path,
    n_samples,
    snp_indices
):
    """
    正确读取指定 SNP 的
    SNP-major PLINK .bed。

    返回：
        [n_samples, n_selected_snps]

    编码：
        00 -> 0
        01 -> missing
        10 -> 1
        11 -> 2
    """

    bytes_per_snp = (
        n_samples + 3
    ) // 4

    geno = np.empty(
        (
            n_samples,
            len(snp_indices)
        ),
        dtype=np.float32
    )

    with open(
        bed_path,
        "rb"
    ) as f:

        magic = f.read(3)

        if magic != b"\x6c\x1b\x01":

            raise ValueError(
                "非标准 SNP-major "
                "PLINK .bed 文件"
            )

        for out_idx, snp_idx in enumerate(
            snp_indices
        ):

            f.seek(
                3
                +
                int(snp_idx)
                *
                bytes_per_snp
            )

            raw = f.read(
                bytes_per_snp
            )

            if len(raw) != bytes_per_snp:

                raise IOError(
                    f"BED 文件读取不完整："
                    f"SNP {snp_idx}"
                )

            for sample_idx in range(
                n_samples
            ):

                byte_idx = (
                    sample_idx // 4
                )

                shift = (
                    sample_idx % 4
                ) * 2

                code = (
                    raw[byte_idx]
                    >> shift
                ) & 0x03

                if code == 0b00:

                    geno[
                        sample_idx,
                        out_idx
                    ] = 0.0

                elif code == 0b01:

                    geno[
                        sample_idx,
                        out_idx
                    ] = np.nan

                elif code == 0b10:

                    geno[
                        sample_idx,
                        out_idx
                    ] = 1.0

                else:

                    geno[
                        sample_idx,
                        out_idx
                    ] = 2.0

    return geno


def main():

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    fam = pd.read_csv(
        f"{PLINK_PREFIX}.fam",
        sep=r"\s+",
        header=None,
        names=[
            "fid",
            "iid",
            "pid",
            "mid",
            "sex",
            "pheno"
        ],
        dtype={
            "fid": str,
            "iid": str
        }
    )

    bim = pd.read_csv(
        f"{PLINK_PREFIX}.bim",
        sep=r"\s+",
        header=None,
        names=[
            "chr",
            "snp",
            "cm",
            "pos",
            "a1",
            "a2"
        ]
    )

    fam["sample_id"] = (
        fam.apply(
            lambda r:
            f"{r['iid']}_{r['iid']}"
            if r["fid"] == r["iid"]
            else r["iid"],
            axis=1
        )
    )

    sample_ids = (
        fam["sample_id"]
        .tolist()
    )

    n_samples = (
        len(sample_ids)
    )

    print(
        f"PLINK 样本数: "
        f"{n_samples}"
    )

    with open(
        FIXED_CAND
    ) as f:

        fixed_snps = []

        for line in f:
            line = line.strip()

            if not line:
                continue

            if ":" not in line:
                raise ValueError(
                    f"候选 SNP ID 格式错误，应为 chr:pos，实际为: {line}"
                )

            chrom, pos = line.split(":", 1)

            fixed_snps.append(
                make_chr_pos_id(chrom, pos)
            )

    print(
        f"候选 SNP 数: "
        f"{len(fixed_snps)}"
    )

    bim["snp_id"] = [
        make_chr_pos_id(chrom, pos)
        for chrom, pos in zip(
            bim["chr"],
            bim["pos"]
        )
    ]

    # chr:pos 是 BED <-> BIM 的唯一匹配键。
    # .bed 没有 SNP ID，其第 i 个 SNP 必须对应 .bim 第 i 行。
    duplicated = bim["snp_id"].duplicated(keep=False)

    if duplicated.any():
        dup = bim.loc[
            duplicated,
            ["chr", "pos", "snp", "snp_id"]
        ]

        print(
            f"警告：BIM 中发现 {len(dup)} 行重复 chr:pos"
        )
        print(dup.head(20).to_string(index=False))

        raise RuntimeError(
            "BIM 中存在重复的 chr:position，"
            "不能仅靠 chr:pos 唯一匹配。"
        )

    snp_to_idx = {
        sid: i
        for i, sid in enumerate(
            bim["snp_id"]
        )
    }

    valid_snps = [

        sid

        for sid
        in fixed_snps

        if sid in snp_to_idx

    ]

    missing_snps = [

        sid

        for sid
        in fixed_snps

        if sid not in snp_to_idx

    ]

    if missing_snps:

        print(
            f"警告：BIM 中未找到 "
            f"{len(missing_snps)} "
            f"个候选 SNP"
        )

        print(
            "前10个：",
            missing_snps[:10]
        )

    snp_indices = np.array(
        sorted(
            snp_to_idx[sid]
            for sid
            in valid_snps
        ),
        dtype=np.int64
    )

    print(
        f"实际提取 SNP 数: "
        f"{len(snp_indices)}"
    )

    if len(snp_indices) == 0:

        raise RuntimeError(
            "没有可提取的候选 SNP"
        )

    bim_sel = (
        bim
        .iloc[snp_indices]
        .reset_index(drop=True)
    )

    print(
        "开始正确读取候选 SNP "
        "的 BED 基因型 ..."
    )

    geno_raw = (
        read_bed_subset(
            f"{PLINK_PREFIX}.bed",
            n_samples,
            snp_indices
        )
    )

    print(
        "Dosage 形状 "
        "[样本,SNP]:",
        geno_raw.shape
    )

    print(
        "缺失基因型比例:",
        float(
            np.isnan(
                geno_raw
            ).mean()
        )
    )

    # 项目内部统一约定：
    # missing genotype = 3
    geno_codes = np.where(
        np.isnan(geno_raw),
        3.0,
        geno_raw
    ).astype(
        np.float32
    )

    print(
        "开始转换为项目统一的 "
        "10维碱基组合编码 ..."
    )

    geno_snp_first = (
        batch_encode_genotypes(
            raw_snps=geno_codes.T,
            allele1=(
                bim_sel["a1"]
                .astype(str)
                .values
            ),
            allele2=(
                bim_sel["a2"]
                .astype(str)
                .values
            ),
            n_jobs=min(
                16,
                os.cpu_count() or 1
            ),
            chunk_size=5000
        )
    )

    print(
        "10维编码形状 "
        "[SNP,样本,10]:",
        geno_snp_first.shape
    )

    channel_counts = np.count_nonzero(
        geno_snp_first,
        axis=(0, 1)
    )

    print(
        "10个编码通道非零计数:",
        channel_counts.tolist()
    )

    pos_feat = np.zeros(
        (
            len(snp_indices),
            6
        ),
        dtype=np.float64
    )

    pos_feat[:, 0] = (
        pd.to_numeric(
            bim_sel["chr"],
            errors="coerce"
        )
        .fillna(0)
        .values
    )

    pos_feat[:, 1] = (
        pd.to_numeric(
            bim_sel["pos"],
            errors="coerce"
        )
        .fillna(0)
        .values
    )

    pos_feat[:, 2] = (
        pos_feat[:, 1]
    )

    pos_feat[:, 3:] = [
        1000.0,
        0.001,
        0.5
    ]

    pheno = pd.read_csv(
        PHENO_FILE,
        sep="\t"
    )

    pheno = (
        pheno
        .set_index("sample_id")
        .reindex(sample_ids)
    )

    phenotypes = np.stack(
        [
            pheno["BW"].values,
            pheno["LE"].values
        ],
        axis=1
    ).astype(
        np.float32
    )

    pheno_names = np.array(
        [
            "BW",
            "LE"
        ],
        dtype="S"
    )

    with open(
        TRAINVAL_SAMPLES
    ) as f:

        tv_ids = [

            x.strip()

            for x in f

            if x.strip()

        ]

    with open(
        TEST_SAMPLES
    ) as f:

        test_ids = [

            x.strip()

            for x in f

            if x.strip()

        ]

    sid2idx = {

        sid: i

        for i, sid
        in enumerate(
            sample_ids
        )

    }

    missing_tv = [

        sid

        for sid
        in tv_ids

        if sid not in sid2idx

    ]

    missing_test = [

        sid

        for sid
        in test_ids

        if sid not in sid2idx

    ]

    if missing_tv:

        print(
            f"警告：训练验证集 "
            f"{len(missing_tv)} "
            f"个样本未找到"
        )

        print(
            "前10个：",
            missing_tv[:10]
        )

    if missing_test:

        print(
            f"警告：测试集 "
            f"{len(missing_test)} "
            f"个样本未找到"
        )

        print(
            "前10个：",
            missing_test[:10]
        )

    tv_idx = [

        sid2idx[sid]

        for sid
        in tv_ids

        if sid in sid2idx

    ]

    test_idx = [

        sid2idx[sid]

        for sid
        in test_ids

        if sid in sid2idx

    ]

    overlap = (
        set(tv_ids)
        &
        set(test_ids)
    )

    if overlap:

        raise RuntimeError(
            f"训练验证集和独立测试集"
            f"发生重叠："
            f"{len(overlap)} 个样本"
        )

    print(
        f"训练验证样本数: "
        f"{len(tv_idx)}"
    )

    print(
        f"独立测试样本数: "
        f"{len(test_idx)}"
    )

    def write_h5(
        out_path,
        sample_idx_list
    ):

        if os.path.exists(
            out_path
        ):

            os.remove(
                out_path
            )

        idx = np.asarray(
            sample_idx_list,
            dtype=np.int64
        )

        selected_pheno = (
            phenotypes[
                idx,
                :
            ]
        )

        na_mask = (
            np.isnan(
                selected_pheno
            )
            .astype(np.uint8)
        )

        with h5py.File(
            out_path,
            "w"
        ) as h5:

            h5.create_dataset(
                "features/genotype_features",
                data=geno_snp_first[
                    :,
                    idx,
                    :
                ],
                compression="gzip",
                compression_opts=4
            )

            h5.create_dataset(
                "features/position_features",
                data=pos_feat
            )

            h5.create_dataset(
                "phenotypes",
                data=selected_pheno
            )

            h5.create_dataset(
                "phenotype_names",
                data=pheno_names
            )

            h5.create_dataset(
                "sample_ids",
                data=np.array(
                    [
                        sample_ids[i]
                        for i in idx
                    ],
                    dtype="S"
                )
            )

            h5.create_dataset(
                "phenotypes_na_mask",
                data=na_mask
            )

        print(
            f"已写入："
            f"{out_path}"
        )

    write_h5(
        OUT_TRAINVAL_H5,
        tv_idx
    )

    write_h5(
        OUT_TEST_H5,
        test_idx
    )

    for path in [
        OUT_TRAINVAL_H5,
        OUT_TEST_H5
    ]:

        with h5py.File(
            path,
            "r"
        ) as f:

            print(
                f"{path}"
            )

            print(
                " genotype:",
                f[
                    "features/"
                    "genotype_features"
                ].shape
            )

            print(
                " phenotype:",
                f[
                    "phenotypes"
                ].shape
            )

            print(
                " samples:",
                len(
                    f[
                        "sample_ids"
                    ]
                )
            )

    print(
        "完成。"
    )

    print(
        "训练验证集和独立测试集"
        "使用完全相同的 SNP "
        "和编码方式。"
    )


if __name__ == "__main__":
    main()
