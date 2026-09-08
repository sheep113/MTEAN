#!/usr/bin/env python3

from pathlib import Path
import argparse
import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


CONFIG = {
    "Trout1935": {
        "geno": "myGD.txt",
        "pheno": "phongraph.tsv",
        "traits": ["Binary_survival", "Final_weight"],
        "n_snps": 27490,
        "trainval": 1567,
        "test": 277,
        "raw_samples": 1935,
        "missing_code": 5,
    },

    "Carp1259": {
        "geno": "geno.txt",
        "pheno": "phongraph.tsv",
        "traits": ["survival", "SL"],
        "n_snps": 15615,
        "trainval": 1070,
        "test": 189,
        "raw_samples": 1259,
        "missing_code": None,
    },
}


def read_ids(path):
    ids = [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{path} 存在重复 ID")

    return ids


def read_header(geno_path):
    with open(geno_path, "r") as f:
        line = f.readline().strip()

    cols = line.split()

    if len(cols) < 2:
        raise RuntimeError("genotype header 异常")

    return cols[0], cols[1:]


def make_h5(
    path,
    n_snps,
    sample_ids,
    snp_ids,
    phenotypes,
    phenotype_names,
    dataset,
    split
):
    f = h5py.File(path, "w")

    ds = f.create_dataset(
        "features/genotype_features",
        shape=(n_snps, len(sample_ids)),
        dtype="u1",
        chunks=(min(2048, n_snps), 1),
        compression="gzip",
        compression_opts=1,
        shuffle=True
    )

    f.create_dataset(
        "phenotypes",
        data=phenotypes.astype(np.float32)
    )

    f.create_dataset(
        "phenotype_names",
        data=np.asarray(phenotype_names, dtype="S")
    )

    f.create_dataset(
        "sample_ids",
        data=np.asarray(sample_ids, dtype="S")
    )

    f.create_dataset(
        "snp_ids",
        data=np.asarray(snp_ids, dtype="S")
    )

    f.create_dataset(
        "phenotypes_na_mask",
        data=np.zeros(
            phenotypes.shape,
            dtype=np.uint8
        )
    )

    f.attrs["dataset"] = dataset
    f.attrs["split"] = split
    f.attrs["genotype_encoding"] = "0_1_2_missing3"
    f.attrs["genotype_layout"] = "SNP_sample"
    f.attrs["snp_order"] = "original_file_column_order"
    f.attrs["physical_position_used"] = False

    return f, ds


def parse_genotype_line(
    line,
    expected_snps,
    missing_code
):
    # 第一段是 sample ID，其余都是 genotype
    parts = line.rstrip("\n").split(maxsplit=1)

    if len(parts) != 2:
        raise RuntimeError("genotype 行格式异常")

    sid = parts[0]
    values = parts[1]

    # Carp 中 NA 直接变成 token 3
    values = values.replace("NA", "3")
    values = values.replace("NaN", "3")
    values = values.replace("nan", "3")

    arr = np.fromstring(
        values,
        sep=" ",
        dtype=np.float32
    )

    if arr.size != expected_snps:
        raise RuntimeError(
            f"{sid}: SNP 数量 {arr.size} != {expected_snps}"
        )

    if missing_code is not None:
        arr[arr == missing_code] = 3

    # 理论上这里只允许 0/1/2/3
    bad = ~np.isin(arr, [0, 1, 2, 3])

    if bad.any():
        vals = np.unique(arr[bad])
        raise RuntimeError(
            f"{sid}: 非法 genotype {vals[:10]}"
        )

    return sid, arr.astype(np.uint8)


def process(dataset, force=False):

    cfg = CONFIG[dataset]

    d = ROOT / "data" / dataset
    processed = d / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    geno_path = d / cfg["geno"]
    pheno_path = d / cfg["pheno"]

    trainval_txt = d / "trainval_samples.txt"
    test_txt = d / "test_samples.txt"

    out_tv = processed / f"{dataset}_dosage_trainval.h5"
    out_te = processed / f"{dataset}_dosage_test.h5"

    if not force:
        for p in [out_tv, out_te]:
            if p.exists():
                raise RuntimeError(
                    f"{p} 已存在；如确认覆盖请加 --force"
                )

    if force:
        for p in [out_tv, out_te]:
            if p.exists():
                p.unlink()

    print("=" * 70)
    print(dataset)
    print("=" * 70)

    # --------------------------------------------------
    # SNP header
    # --------------------------------------------------
    id_col, snp_ids = read_header(geno_path)

    print("ID column:", id_col)
    print("SNP:", len(snp_ids))

    if len(snp_ids) != cfg["n_snps"]:
        raise RuntimeError(
            f"SNP 数错误: {len(snp_ids)}"
        )

    if len(snp_ids) != len(set(snp_ids)):
        raise RuntimeError("SNP ID 重复")

    # --------------------------------------------------
    # split
    # --------------------------------------------------
    tv_file_ids = read_ids(trainval_txt)
    te_file_ids = read_ids(test_txt)

    if len(tv_file_ids) != cfg["trainval"]:
        raise RuntimeError(
            f"trainval={len(tv_file_ids)}"
        )

    if len(te_file_ids) != cfg["test"]:
        raise RuntimeError(
            f"test={len(te_file_ids)}"
        )

    tv_set = set(tv_file_ids)
    te_set = set(te_file_ids)

    if tv_set & te_set:
        raise RuntimeError("trainval/test overlap")

    # --------------------------------------------------
    # phenotype
    # --------------------------------------------------
    ph = pd.read_csv(
        pheno_path,
        sep="\t",
        dtype={"sample_id": str}
    )

    ph["sample_id"] = (
        ph["sample_id"]
        .astype(str)
        .str.strip()
    )

    for trait in cfg["traits"]:
        ph[trait] = pd.to_numeric(
            ph[trait],
            errors="coerce"
        )

    ph = ph.set_index("sample_id")

    # --------------------------------------------------
    # 第一次极轻量扫描：
    # 只读取每行第一个 ID，恢复 genotype 原始行顺序
    # --------------------------------------------------
    print("\n扫描 genotype 样本顺序 ...")

    geno_order = []

    with open(geno_path, "r") as f:
        next(f)

        for line in f:
            if not line.strip():
                continue

            sid = line.split(None, 1)[0]
            geno_order.append(sid)

    print("raw samples:", len(geno_order))

    if len(geno_order) != cfg["raw_samples"]:
        raise RuntimeError(
            f"原始样本数 {len(geno_order)} "
            f"!= {cfg['raw_samples']}"
        )

    if len(geno_order) != len(set(geno_order)):
        raise RuntimeError(
            "genotype ID 重复"
        )

    # H5 中仍保持原始 genotype 行顺序
    tv_ids = [
        x for x in geno_order
        if x in tv_set
    ]

    te_ids = [
        x for x in geno_order
        if x in te_set
    ]

    if set(tv_ids) != tv_set:
        raise RuntimeError(
            "部分 trainval ID 不在 genotype"
        )

    if set(te_ids) != te_set:
        raise RuntimeError(
            "部分 test ID 不在 genotype"
        )

    tv_pheno = (
        ph.loc[
            tv_ids,
            cfg["traits"]
        ]
        .to_numpy(dtype=np.float32)
    )

    te_pheno = (
        ph.loc[
            te_ids,
            cfg["traits"]
        ]
        .to_numpy(dtype=np.float32)
    )

    if np.isnan(tv_pheno).any():
        raise RuntimeError(
            "trainval 中存在缺失 phenotype"
        )

    if np.isnan(te_pheno).any():
        raise RuntimeError(
            "test 中存在缺失 phenotype"
        )

    print("trainval:", len(tv_ids))
    print("test:", len(te_ids))

    # sample -> H5 column
    tv_col = {
        sid: i
        for i, sid in enumerate(tv_ids)
    }

    te_col = {
        sid: i
        for i, sid in enumerate(te_ids)
    }

    # --------------------------------------------------
    # H5
    # --------------------------------------------------
    tv_f, tv_ds = make_h5(
        out_tv,
        cfg["n_snps"],
        tv_ids,
        snp_ids,
        tv_pheno,
        cfg["traits"],
        dataset,
        "trainval"
    )

    te_f, te_ds = make_h5(
        out_te,
        cfg["n_snps"],
        te_ids,
        snp_ids,
        te_pheno,
        cfg["traits"],
        dataset,
        "independent_test"
    )

    counts = np.zeros(4, dtype=np.int64)

    n_rows = 0
    n_written = 0

    # --------------------------------------------------
    # 第二次逐行处理 genotype
    # --------------------------------------------------
    print("\n开始逐样本流式写 H5 ...")

    try:
        with open(geno_path, "r") as f:

            next(f)  # header

            for line in f:

                if not line.strip():
                    continue

                n_rows += 1

                sid = line.split(None, 1)[0]

                # Trout 的 91 个不完整 phenotype 样本：
                # 根本不解析其 27490 SNP，直接跳过
                if (
                    sid not in tv_set
                    and sid not in te_set
                ):
                    continue

                sid2, arr = parse_genotype_line(
                    line,
                    cfg["n_snps"],
                    cfg["missing_code"]
                )

                if sid != sid2:
                    raise RuntimeError(
                        "sample ID 解析异常"
                    )

                if sid in tv_col:
                    tv_ds[:, tv_col[sid]] = arr

                elif sid in te_col:
                    te_ds[:, te_col[sid]] = arr

                else:
                    raise RuntimeError(
                        f"未知 sample: {sid}"
                    )

                counts += np.bincount(
                    arr,
                    minlength=4
                )[:4]

                n_written += 1

                if (
                    n_written % 100 == 0
                    or n_written ==
                    len(tv_ids) + len(te_ids)
                ):
                    print(
                        f"written: "
                        f"{n_written}/"
                        f"{len(tv_ids)+len(te_ids)}"
                    )

    finally:
        tv_f.close()
        te_f.close()

    if n_rows != cfg["raw_samples"]:
        raise RuntimeError(
            f"读取行数 {n_rows} "
            f"!= {cfg['raw_samples']}"
        )

    expected_written = (
        len(tv_ids)
        +
        len(te_ids)
    )

    if n_written != expected_written:
        raise RuntimeError(
            f"写入 {n_written} "
            f"!= {expected_written}"
        )

    # --------------------------------------------------
    # final verify
    # --------------------------------------------------
    print("\n验证输出 ...")

    for path, ns in [
        (out_tv, len(tv_ids)),
        (out_te, len(te_ids)),
    ]:

        with h5py.File(path, "r") as h:

            g = h[
                "features/genotype_features"
            ]

            print(
                path.name,
                g.shape,
                g.dtype,
                h["phenotypes"].shape
            )

            assert g.shape == (
                cfg["n_snps"],
                ns
            )

            assert h["phenotypes"].shape == (
                ns,
                2
            )

            # 抽几列检查即可，
            # 不把整个 H5 重新读入内存
            cols = sorted(
                set([
                    0,
                    ns // 2,
                    ns - 1
                ])
            )

            for col in cols:
                u = np.unique(
                    g[:, col]
                )

                if not set(
                    u.tolist()
                ).issubset(
                    {0, 1, 2, 3}
                ):
                    raise RuntimeError(
                        f"非法 genotype: {u}"
                    )

    total = counts.sum()

    print("\ngenotype counts:")
    print(
        {
            i: int(counts[i])
            for i in range(4)
        }
    )

    print(
        "missing ratio:",
        float(counts[3] / total)
    )

    print("\nPASS:", dataset)
    print(out_tv)
    print(out_te)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        choices=[
            "Trout1935",
            "Carp1259",
            "all"
        ]
    )

    parser.add_argument(
        "--force",
        action="store_true"
    )

    args = parser.parse_args()

    names = (
        ["Trout1935", "Carp1259"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for name in names:
        process(
            name,
            force=args.force
        )


if __name__ == "__main__":
    main()
