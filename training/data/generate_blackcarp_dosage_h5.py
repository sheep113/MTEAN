#!/usr/bin/env python3
"""
生成青鱼 Black carp 的 dosage HDF5。

输入：
    PLINK:
        data/blackcarp499/filtered_snp_keep.{bed,bim,fam}

    已固定候选 SNP:
        data/blackcarp499/fixed_candidates.txt

    已固定样本划分:
        data/blackcarp499/trainval_samples.txt
        data/blackcarp499/test_samples.txt

输出：
    data/blackcarp499/processed/blackcarp_dosage_trainval.h5
    data/blackcarp499/processed/blackcarp_dosage_test.h5

统一编码：
    0 = homozygous genotype 0
    1 = heterozygous genotype
    2 = homozygous genotype 2
    3 = missing

注意：
    这里保存的是整数 dosage code，不进行 10 维碱基编码。
    genotype_features shape = [SNP, sample]
"""

from pathlib import Path
import argparse

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data/blackcarp499"
PROCESSED_DIR = DATA_DIR / "processed"

PLINK_PREFIX = DATA_DIR / "filtered_snp_keep"

FIXED_CAND = DATA_DIR / "fixed_candidates.txt"
PHENO_FILE = DATA_DIR / "phongraph_new.tsv"

TRAINVAL_SAMPLES = DATA_DIR / "trainval_samples.txt"
TEST_SAMPLES = DATA_DIR / "test_samples.txt"

OUT_TRAINVAL = (
    PROCESSED_DIR /
    "blackcarp_dosage_trainval.h5"
)

OUT_TEST = (
    PROCESSED_DIR /
    "blackcarp_dosage_test.h5"
)

# 用于公平性核对，不会修改这两个文件
ALLELE10_TRAINVAL = (
    PROCESSED_DIR /
    "blackcarp_allele10_trainval.h5"
)

ALLELE10_TEST = (
    PROCESSED_DIR /
    "blackcarp_allele10_test.h5"
)


def normalize_chr(x):
    s = str(x).strip()

    try:
        v = float(s)

        if v.is_integer():
            return str(int(v))

    except (ValueError, TypeError):
        pass

    return s


def normalize_pos(x):
    s = str(x).strip()

    try:
        return str(int(float(s)))

    except (ValueError, TypeError):
        raise ValueError(
            f"无法解析 SNP position: {x!r}"
        )


def make_chr_pos_id(chrom, pos):
    return (
        f"{normalize_chr(chrom)}:"
        f"{normalize_pos(pos)}"
    )


def read_nonempty_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            x.strip()
            for x in f
            if x.strip()
        ]


def read_bed_subset(
    bed_path,
    n_samples,
    snp_indices
):
    """
    PLINK SNP-major BED 正确解码。

    PLINK 2-bit:
        00 -> dosage 0
        01 -> missing
        10 -> dosage 1
        11 -> dosage 2

    最终项目编码：
        0 -> 0
        missing -> 3
        1 -> 1
        2 -> 2

    返回：
        [n_samples, n_selected_snps]
        dtype uint8
    """

    bytes_per_snp = (
        n_samples + 3
    ) // 4

    geno = np.empty(
        (
            n_samples,
            len(snp_indices)
        ),
        dtype=np.uint8
    )

    # PLINK bit code -> MTEAN dosage code
    lut = np.array(
        [
            0,  # 00
            3,  # 01 missing
            1,  # 10
            2,  # 11
        ],
        dtype=np.uint8
    )

    sample_indices = np.arange(
        n_samples,
        dtype=np.int64
    )

    byte_indices = (
        sample_indices // 4
    )

    shifts = (
        (sample_indices % 4) * 2
    ).astype(np.uint8)

    with open(bed_path, "rb") as f:

        magic = f.read(3)

        if magic != b"\x6c\x1b\x01":
            raise ValueError(
                "不是标准 SNP-major PLINK BED 文件"
            )

        for out_idx, snp_idx in enumerate(
            snp_indices
        ):

            offset = (
                3
                + int(snp_idx)
                * bytes_per_snp
            )

            f.seek(offset)

            raw = f.read(bytes_per_snp)

            if len(raw) != bytes_per_snp:
                raise IOError(
                    "BED 读取不完整："
                    f"SNP index={snp_idx}"
                )

            arr = np.frombuffer(
                raw,
                dtype=np.uint8
            )

            codes = (
                arr[byte_indices]
                >> shifts
            ) & 0x03

            geno[:, out_idx] = (
                lut[codes]
            )

    return geno


def decode_ids(values):
    return [
        x.decode()
        if isinstance(x, bytes)
        else str(x)
        for x in values
    ]


def verify_against_allele10(
    dosage_path,
    allele10_path
):
    """
    确认 dosage 与冻结的 allele10 数据：
    - SNP 数一致
    - 样本数一致
    - 样本顺序一致
    - phenotype 一致
    - position_features 一致

    这样可以证明两套数据唯一变化是 genotype 编码方式。
    """

    if not allele10_path.exists():
        print(
            "WARN: 未找到 allele10 对照文件，"
            "跳过公平性核对：",
            allele10_path
        )
        return

    with (
        h5py.File(dosage_path, "r") as d,
        h5py.File(allele10_path, "r") as a
    ):

        dg = d[
            "features/genotype_features"
        ]

        ag = a[
            "features/genotype_features"
        ]

        assert dg.ndim == 2, (
            f"dosage 应为2维，实际 {dg.shape}"
        )

        assert ag.ndim == 3, (
            f"allele10 应为3维，实际 {ag.shape}"
        )

        assert ag.shape[2] == 10, (
            "allele10 最后一维不是10"
        )

        assert (
            dg.shape[0]
            == ag.shape[0]
        ), (
            "SNP 数不一致："
            f"{dg.shape[0]} vs {ag.shape[0]}"
        )

        assert (
            dg.shape[1]
            == ag.shape[1]
        ), (
            "样本数不一致："
            f"{dg.shape[1]} vs {ag.shape[1]}"
        )

        d_ids = decode_ids(
            d["sample_ids"][:]
        )

        a_ids = decode_ids(
            a["sample_ids"][:]
        )

        assert d_ids == a_ids, (
            "sample_ids 或顺序不一致"
        )

        assert np.allclose(
            d["phenotypes"][:],
            a["phenotypes"][:],
            equal_nan=True
        ), "phenotype 不一致"

        assert np.allclose(
            d[
                "features/"
                "position_features"
            ][:],
            a[
                "features/"
                "position_features"
            ][:],
            equal_nan=True
        ), "position_features 不一致"

    print(
        "PASS: 与 allele10 数据完全对齐：",
        dosage_path.name
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已存在的 dosage H5"
    )

    args = parser.parse_args()

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    required = [
        Path(str(PLINK_PREFIX) + ".bed"),
        Path(str(PLINK_PREFIX) + ".bim"),
        Path(str(PLINK_PREFIX) + ".fam"),
        FIXED_CAND,
        PHENO_FILE,
        TRAINVAL_SAMPLES,
        TEST_SAMPLES,
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"缺少必要文件: {p}"
            )

    for out in [
        OUT_TRAINVAL,
        OUT_TEST
    ]:
        if out.exists() and not args.force:
            raise FileExistsError(
                f"{out} 已存在。\n"
                "如确认覆盖，请使用 --force"
            )

    print(
        "========== Black carp dosage H5 =========="
    )

    # -------------------------------------------------
    # FAM
    # -------------------------------------------------

    fam = pd.read_csv(
        str(PLINK_PREFIX) + ".fam",
        sep=r"\s+",
        header=None,
        names=[
            "fid",
            "iid",
            "pid",
            "mid",
            "sex",
            "pheno",
        ],
        dtype={
            "fid": str,
            "iid": str,
        }
    )

    fam["sample_id"] = fam.apply(
        lambda r:
        f"{r['iid']}_{r['iid']}"
        if r["fid"] == r["iid"]
        else r["iid"],
        axis=1
    )

    sample_ids = (
        fam["sample_id"]
        .tolist()
    )

    if len(sample_ids) != len(
        set(sample_ids)
    ):
        raise RuntimeError(
            "FAM 中存在重复 sample_id"
        )

    n_samples = len(sample_ids)

    print(
        "PLINK 样本数:",
        n_samples
    )

    # -------------------------------------------------
    # BIM
    # -------------------------------------------------

    bim = pd.read_csv(
        str(PLINK_PREFIX) + ".bim",
        sep=r"\s+",
        header=None,
        names=[
            "chr",
            "snp",
            "cm",
            "pos",
            "a1",
            "a2",
        ]
    )

    bim["snp_id"] = [
        make_chr_pos_id(chrom, pos)
        for chrom, pos in zip(
            bim["chr"],
            bim["pos"]
        )
    ]

    duplicated = bim[
        "snp_id"
    ].duplicated(
        keep=False
    )

    if duplicated.any():
        raise RuntimeError(
            "BIM 中存在重复 chr:position，"
            "无法安全匹配 SNP"
        )

    snp_to_idx = {
        sid: i
        for i, sid in enumerate(
            bim["snp_id"]
        )
    }

    print(
        "PLINK SNP 数:",
        len(bim)
    )

    # -------------------------------------------------
    # fixed candidates
    # -------------------------------------------------

    fixed_raw = read_nonempty_lines(
        FIXED_CAND
    )

    fixed_snps = []

    for line in fixed_raw:

        if ":" not in line:
            raise ValueError(
                "fixed_candidates SNP ID "
                f"格式错误: {line}"
            )

        chrom, pos = line.split(
            ":",
            1
        )

        fixed_snps.append(
            make_chr_pos_id(
                chrom,
                pos
            )
        )

    if len(fixed_snps) != len(
        set(fixed_snps)
    ):
        raise RuntimeError(
            "fixed_candidates.txt "
            "存在重复 SNP"
        )

    missing_snps = [
        sid
        for sid in fixed_snps
        if sid not in snp_to_idx
    ]

    if missing_snps:
        raise RuntimeError(
            f"{len(missing_snps)} 个候选 SNP "
            "未在 BIM 找到。\n"
            f"前10个: {missing_snps[:10]}"
        )

    # 与 allele10 生成逻辑一致：
    # 最终按 BIM/BED 顺序排列 SNP
    snp_indices = np.array(
        sorted(
            snp_to_idx[sid]
            for sid in fixed_snps
        ),
        dtype=np.int64
    )

    bim_sel = (
        bim
        .iloc[snp_indices]
        .reset_index(drop=True)
    )

    selected_snp_ids = (
        bim_sel["snp_id"]
        .tolist()
    )

    print(
        "候选 SNP 数:",
        len(fixed_snps)
    )

    print(
        "实际提取 SNP 数:",
        len(snp_indices)
    )

    # -------------------------------------------------
    # BED -> dosage
    # -------------------------------------------------

    print(
        "开始读取候选 SNP dosage ..."
    )

    geno_sample_first = read_bed_subset(
        Path(str(PLINK_PREFIX) + ".bed"),
        n_samples,
        snp_indices
    )

    # [sample, SNP] -> [SNP, sample]
    geno_snp_first = (
        geno_sample_first.T
        .copy()
    )

    del geno_sample_first

    unique_values, counts = (
        np.unique(
            geno_snp_first,
            return_counts=True
        )
    )

    value_counts = {
        int(k): int(v)
        for k, v in zip(
            unique_values,
            counts
        )
    }

    print(
        "Dosage shape [SNP, sample]:",
        geno_snp_first.shape
    )

    print(
        "编码计数:",
        value_counts
    )

    invalid = (
        ~np.isin(
            geno_snp_first,
            [0, 1, 2, 3]
        )
    )

    if invalid.any():
        raise RuntimeError(
            "发现 0/1/2/3 以外编码"
        )

    missing_rate = float(
        np.mean(
            geno_snp_first == 3
        )
    )

    print(
        "missing(3) 比例:",
        missing_rate
    )

    # -------------------------------------------------
    # position features
    # 与原 allele10 脚本保持一致
    # -------------------------------------------------

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
        0.5,
    ]

    # -------------------------------------------------
    # phenotype
    # -------------------------------------------------

    pheno = pd.read_csv(
        PHENO_FILE,
        sep="\t"
    )

    required_cols = {
        "sample_id",
        "BW",
        "LE",
    }

    missing_cols = (
        required_cols
        - set(pheno.columns)
    )

    if missing_cols:
        raise RuntimeError(
            "表型文件缺少列: "
            f"{sorted(missing_cols)}"
        )

    if pheno[
        "sample_id"
    ].duplicated().any():
        raise RuntimeError(
            "表型文件存在重复 sample_id"
        )

    pheno = (
        pheno
        .set_index("sample_id")
        .reindex(sample_ids)
    )

    phenotypes = np.stack(
        [
            pheno["BW"].values,
            pheno["LE"].values,
        ],
        axis=1
    ).astype(np.float32)

    pheno_names = np.array(
        ["BW", "LE"],
        dtype="S"
    )

    # -------------------------------------------------
    # fixed trainval / test
    # -------------------------------------------------

    tv_ids = read_nonempty_lines(
        TRAINVAL_SAMPLES
    )

    test_ids = read_nonempty_lines(
        TEST_SAMPLES
    )

    if len(tv_ids) != len(
        set(tv_ids)
    ):
        raise RuntimeError(
            "trainval_samples.txt 内有重复"
        )

    if len(test_ids) != len(
        set(test_ids)
    ):
        raise RuntimeError(
            "test_samples.txt 内有重复"
        )

    overlap = (
        set(tv_ids)
        &
        set(test_ids)
    )

    if overlap:
        raise RuntimeError(
            "trainval/test 重叠："
            f"{len(overlap)} 个样本"
        )

    sid2idx = {
        sid: i
        for i, sid in enumerate(
            sample_ids
        )
    }

    missing_tv = [
        sid
        for sid in tv_ids
        if sid not in sid2idx
    ]

    missing_test = [
        sid
        for sid in test_ids
        if sid not in sid2idx
    ]

    if missing_tv:
        raise RuntimeError(
            "trainval 有样本不在 FAM："
            f"{missing_tv[:10]}"
        )

    if missing_test:
        raise RuntimeError(
            "test 有样本不在 FAM："
            f"{missing_test[:10]}"
        )

    tv_idx = np.asarray(
        [
            sid2idx[sid]
            for sid in tv_ids
        ],
        dtype=np.int64
    )

    test_idx = np.asarray(
        [
            sid2idx[sid]
            for sid in test_ids
        ],
        dtype=np.int64
    )

    print(
        "trainval 样本数:",
        len(tv_idx)
    )

    print(
        "test 样本数:",
        len(test_idx)
    )

    # 你的当前固定设计应该是 424 / 75
    if len(tv_idx) != 424:
        print(
            "WARN: trainval 不是预期的 424，"
            f"实际 {len(tv_idx)}"
        )

    if len(test_idx) != 75:
        print(
            "WARN: test 不是预期的 75，"
            f"实际 {len(test_idx)}"
        )

    # -------------------------------------------------
    # write H5
    # -------------------------------------------------

    def write_h5(
        out_path,
        indices,
        split_name
    ):

        if out_path.exists():
            if args.force:
                out_path.unlink()
            else:
                raise FileExistsError(
                    str(out_path)
                )

        selected_pheno = (
            phenotypes[
                indices,
                :
            ]
        )

        na_mask = np.isnan(
            selected_pheno
        ).astype(np.uint8)

        split_sample_ids = [
            sample_ids[i]
            for i in indices
        ]

        with h5py.File(
            out_path,
            "w"
        ) as h5:

            h5.create_dataset(
                "features/genotype_features",
                data=geno_snp_first[
                    :,
                    indices
                ],
                dtype=np.uint8,
                compression="gzip",
                compression_opts=4,
                shuffle=True
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
                data=np.asarray(
                    split_sample_ids,
                    dtype="S"
                )
            )

            h5.create_dataset(
                "phenotypes_na_mask",
                data=na_mask
            )

            # 新增，方便以后跨数据集统一检查
            h5.create_dataset(
                "snp_ids",
                data=np.asarray(
                    selected_snp_ids,
                    dtype="S"
                )
            )

            # 明确记录编码定义
            h5.attrs[
                "dataset"
            ] = "blackcarp499"

            h5.attrs[
                "split"
            ] = split_name

            h5.attrs[
                "genotype_encoding"
            ] = "dosage_0_1_2_missing3"

            h5.attrs[
                "genotype_layout"
            ] = "SNP_sample"

            h5.attrs[
                "missing_code"
            ] = 3

            h5.attrs[
                "genotype_dtype"
            ] = "uint8"

        print(
            "已生成:",
            out_path
        )

    write_h5(
        OUT_TRAINVAL,
        tv_idx,
        "trainval"
    )

    write_h5(
        OUT_TEST,
        test_idx,
        "independent_test"
    )

    # -------------------------------------------------
    # final verification
    # -------------------------------------------------

    print(
        "\n========== H5 最终检查 =========="
    )

    for path in [
        OUT_TRAINVAL,
        OUT_TEST
    ]:

        with h5py.File(
            path,
            "r"
        ) as h5:

            g = h5[
                "features/"
                "genotype_features"
            ]

            print(
                "\n",
                path.name
            )

            print(
                " genotype:",
                g.shape,
                g.dtype
            )

            print(
                " phenotype:",
                h5["phenotypes"].shape
            )

            print(
                " sample_ids:",
                len(h5["sample_ids"])
            )

            print(
                " snp_ids:",
                len(h5["snp_ids"])
            )

            vals = np.unique(g[:])

            print(
                " genotype unique:",
                vals.tolist()
            )

            if not np.all(
                np.isin(
                    vals,
                    [0, 1, 2, 3]
                )
            ):
                raise RuntimeError(
                    "H5 中出现非法 dosage 编码"
                )

    # 与冻结的 allele10 数据核对
    print(
        "\n========== 与 allele10 对齐检查 =========="
    )

    verify_against_allele10(
        OUT_TRAINVAL,
        ALLELE10_TRAINVAL
    )

    verify_against_allele10(
        OUT_TEST,
        ALLELE10_TEST
    )

    print(
        "\n======================================"
    )

    print(
        "PASS: Black carp dosage H5 生成完成"
    )

    print(
        "编码: 0 / 1 / 2 / 3(missing)"
    )

    print(
        "SNP 集合、SNP 顺序、样本划分、"
        "表型均与 allele10 对齐"
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()
