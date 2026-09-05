#!/usr/bin/env python3
"""
方案3：无独立筛选集，直接在训练验证池上做 MIC 筛选。

步骤：
1. 分层抽取独立测试集（默认15%）
2. 剩余为训练验证池
3. 在训练验证池上计算 MIC
4. 分别输出 BW、LE top SNP 以及两者并集

注意：
独立测试集不会参与 MIC。
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from minepy import MINE
from sklearn.model_selection import StratifiedShuffleSplit


H5_FILE = "output/blackcarp/blackcarp_preprocessed 398w.h5"

PHENO_FILE = "data/blackcarp499/phongraph_new.tsv"

BIM_PREFIX = '/home/data/biofish/yjn/workspace/MTEAN/data/blackcarp499/filtered_snp_keep'

OUT_DIR = "data/blackcarp499"

TOP_K = 15000
ALPHA = 0.6
C = 15
MIN_SAMPLES = 10

N_THREADS = 128
CHUNK_SIZE = 5000

TEST_RATIO = 0.15



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


def read_bed_as_dosage(
    bed_path,
    n_samples,
    n_snps
):
    """
    正确读取 SNP-major PLINK .bed。

    返回：
        [n_samples, n_snps]

    PLINK BED 2-bit 编码：

        00 -> 0
        01 -> missing
        10 -> 1
        11 -> 2
    """

    bytes_per_snp = (n_samples + 3) // 4

    dosage = np.empty(
        (n_samples, n_snps),
        dtype=np.float32
    )

    with open(bed_path, "rb") as f:

        magic = f.read(3)

        if magic != b"\x6c\x1b\x01":
            raise ValueError(
                "非标准 SNP-major PLINK .bed 文件"
            )

        for snp_idx in range(n_snps):

            raw = f.read(bytes_per_snp)

            if len(raw) != bytes_per_snp:
                raise IOError(
                    f"BED 文件读取不完整："
                    f"SNP {snp_idx} "
                    f"期望 {bytes_per_snp} bytes，"
                    f"实际 {len(raw)} bytes"
                )

            for sample_idx in range(n_samples):

                byte_idx = sample_idx // 4
                shift = (sample_idx % 4) * 2

                code = (
                    raw[byte_idx] >> shift
                ) & 0x03

                if code == 0b00:

                    dosage[
                        sample_idx,
                        snp_idx
                    ] = 0.0

                elif code == 0b01:

                    dosage[
                        sample_idx,
                        snp_idx
                    ] = np.nan

                elif code == 0b10:

                    dosage[
                        sample_idx,
                        snp_idx
                    ] = 1.0

                else:

                    dosage[
                        sample_idx,
                        snp_idx
                    ] = 2.0

    return dosage


def process_chunk(
    chunk_data
):

    (
        snp_chunk,
        pheno_vals,
        alpha,
        c,
        min_samples
    ) = chunk_data

    results = []

    for snp in snp_chunk:

        mask = ~(
            np.isnan(snp)
            |
            np.isnan(pheno_vals)
        )

        x_clean = snp[mask]
        y_clean = pheno_vals[mask]

        if len(x_clean) < min_samples:

            results.append(
                (np.nan, len(x_clean))
            )

            continue

        if (
            len(np.unique(x_clean)) <= 1
            or
            len(np.unique(y_clean)) <= 1
        ):

            results.append(
                (0.0, len(x_clean))
            )

            continue

        try:

            mine = MINE(
                alpha=alpha,
                c=c
            )

            mine.compute_score(
                x_clean,
                y_clean
            )

            results.append(
                (
                    mine.mic(),
                    len(x_clean)
                )
            )

        except Exception:

            results.append(
                (
                    np.nan,
                    len(x_clean)
                )
            )

    return results


def load_sample_ids_from_h5():

    with h5py.File(
        H5_FILE,
        "r"
    ) as f:

        sample_ids = [

            x.decode()
            if isinstance(x, bytes)
            else str(x)

            for x in f["sample_ids"][:]
        ]

    return sample_ids


def stratified_split(
    pheno,
    strata,
    test_size,
    seed
):

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed
    )

    for train_idx, test_idx in sss.split(
        pheno,
        strata
    ):

        return (
            list(train_idx),
            list(test_idx)
        )


def main():

    if len(sys.argv) < 2:

        print(
            "用法："
            "python "
            "training/data/"
            "generate_screening_and_mic.py "
            "<seed>"
        )

        sys.exit(1)

    seed = int(
        sys.argv[1]
    )

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    all_ids = (
        load_sample_ids_from_h5()
    )

    print(
        f"H5 样本数: "
        f"{len(all_ids)}"
    )

    pheno = pd.read_csv(
        PHENO_FILE,
        sep="\t"
    )

    pheno = (
        pheno
        .set_index("sample_id")
        .reindex(all_ids)
        .dropna(
            subset=[
                "BW",
                "LE"
            ]
        )
    )

    valid_ids = (
        pheno.index.tolist()
    )

    print(
        f"BW/LE 均有效样本数: "
        f"{len(valid_ids)}"
    )

    bw_bins = pd.qcut(
        pheno["BW"],
        q=2,
        labels=False,
        duplicates="drop"
    )

    le_bins = pd.qcut(
        pheno["LE"],
        q=2,
        labels=False,
        duplicates="drop"
    )

    strata = (
        bw_bins.astype(str)
        +
        "_"
        +
        le_bins.astype(str)
    )

    try:

        (
            trainval_idx,
            test_idx
        ) = stratified_split(
            pheno,
            strata,
            TEST_RATIO,
            seed
        )

    except ValueError:

        indices = np.arange(
            len(valid_ids)
        )

        rng = np.random.default_rng(
            seed
        )

        rng.shuffle(
            indices
        )

        n_test = int(
            len(valid_ids)
            *
            TEST_RATIO
        )

        test_idx = (
            indices[:n_test]
        )

        trainval_idx = (
            indices[n_test:]
        )

    test_samples = [

        valid_ids[i]

        for i in test_idx

    ]

    trainval_samples = [

        valid_ids[i]

        for i in trainval_idx

    ]

    print(
        f"独立测试集: "
        f"{len(test_samples)}"
    )

    print(
        f"训练验证池: "
        f"{len(trainval_samples)}"
    )

    with open(
        os.path.join(
            OUT_DIR,
            "test_samples.txt"
        ),
        "w"
    ) as f:

        f.write(
            "\n".join(
                test_samples
            )
        )

    with open(
        os.path.join(
            OUT_DIR,
            "trainval_samples.txt"
        ),
        "w"
    ) as f:

        f.write(
            "\n".join(
                trainval_samples
            )
        )

    fam = pd.read_csv(
        f"{BIM_PREFIX}.fam",
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

    fam["sample_iid"] = (
        fam.apply(
            lambda r:
            f"{r['iid']}_{r['iid']}"
            if r["fid"] == r["iid"]
            else r["iid"],
            axis=1
        )
    )

    iids = (
        fam["sample_iid"]
        .tolist()
    )

    n_samples_total = (
        len(iids)
    )

    bim = pd.read_csv(
        f"{BIM_PREFIX}.bim",
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

    snp_ids = [
        make_chr_pos_id(chrom, pos)
        for chrom, pos in zip(
            bim["chr"],
            bim["pos"]
        )
    ]

    n_snps = (
        len(snp_ids)
    )

    print(
        f"PLINK 样本数: "
        f"{n_samples_total}"
    )

    print(
        f"PLINK SNP 数: "
        f"{n_snps}"
    )

    iid_to_idx = {
        sid: i
        for i, sid
        in enumerate(iids)
    }

    missing_ids = [

        sid

        for sid
        in trainval_samples

        if sid not in iid_to_idx

    ]

    if missing_ids:

        print(
            f"警告："
            f"{len(missing_ids)} "
            f"个训练验证样本"
            f"未在 FAM 中找到"
        )

        print(
            "前10个：",
            missing_ids[:10]
        )

    trainval_idx_in_geno = [

        iid_to_idx[sid]

        for sid
        in trainval_samples

        if sid in iid_to_idx

    ]

    trainval_iids = [

        iids[i]

        for i
        in trainval_idx_in_geno

    ]

    print(
        "开始正确解码 PLINK BED ..."
    )

    all_geno = (
        read_bed_as_dosage(
            f"{BIM_PREFIX}.bed",
            n_samples_total,
            n_snps
        )
    )

    trainval_geno = (
        all_geno[
            trainval_idx_in_geno,
            :
        ]
    )

    del all_geno

    print(
        "训练验证基因型形状:",
        trainval_geno.shape
    )

    pheno_tv = (
        pheno.reindex(
            trainval_iids
        )
    )

    bw_tv = (
        pheno_tv["BW"]
        .values
        .astype(np.float64)
    )

    le_tv = (
        pheno_tv["LE"]
        .values
        .astype(np.float64)
    )

    def calc_mic(
        pheno_vals,
        trait_name
    ):

        mic = np.full(
            n_snps,
            np.nan,
            dtype=np.float32
        )

        chunks = []

        for s in range(
            0,
            n_snps,
            CHUNK_SIZE
        ):

            e = min(
                s + CHUNK_SIZE,
                n_snps
            )

            chunk = (
                trainval_geno[
                    :,
                    s:e
                ].T
            )

            chunks.append(
                (
                    chunk,
                    pheno_vals,
                    ALPHA,
                    C,
                    MIN_SAMPLES
                )
            )

        with ProcessPoolExecutor(
            max_workers=N_THREADS
        ) as ex:

            results = list(
                tqdm(
                    ex.map(
                        process_chunk,
                        chunks
                    ),
                    total=len(chunks),
                    desc=f"MIC-{trait_name}"
                )
            )

        idx = 0

        for chunk_res in results:

            for mic_val, _ in chunk_res:

                mic[idx] = (
                    mic_val
                )

                idx += 1

        return mic

    print(
        "在训练验证池上计算 "
        "BW MIC ..."
    )

    mic_bw = calc_mic(
        bw_tv,
        "BW"
    )

    print(
        "在训练验证池上计算 "
        "LE MIC ..."
    )

    mic_le = calc_mic(
        le_tv,
        "LE"
    )

    def get_top_valid_indices(
        mic_values,
        top_k
    ):

        valid_indices = np.where(
            ~np.isnan(
                mic_values
            )
        )[0]

        if len(
            valid_indices
        ) <= top_k:

            return (
                valid_indices
            )

        order = np.argsort(
            mic_values[
                valid_indices
            ]
        )[::-1]

        return (
            valid_indices[
                order[:top_k]
            ]
        )

    top_bw = (
        get_top_valid_indices(
            mic_bw,
            TOP_K
        )
    )

    top_le = (
        get_top_valid_indices(
            mic_le,
            TOP_K
        )
    )

    union_idx = (
        np.union1d(
            top_bw,
            top_le
        )
    )

    bw_snps = [

        snp_ids[i]
        for i
        in top_bw

    ]

    le_snps = [

        snp_ids[i]
        for i
        in top_le

    ]

    union_snps = [

        snp_ids[i]
        for i
        in union_idx

    ]

    outputs = {

        "fixed_candidates.txt":
            union_snps,

        "fixed_candidates_BW.txt":
            bw_snps,

        "fixed_candidates_LE.txt":
            le_snps
    }

    for filename, snps in (
        outputs.items()
    ):

        path = os.path.join(
            OUT_DIR,
            filename
        )

        with open(
            path,
            "w"
        ) as f:

            f.write(
                "\n".join(
                    snps
                )
            )

        print(
            f"已保存 {path}: "
            f"{len(snps)} SNP"
        )

    print(
        "MIC 筛选完成。"
    )

    print(
        "独立测试集没有参与 MIC。"
    )


if __name__ == "__main__":
    main()
