#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def clean_id(x):
    return x.astype(str).str.strip()


def backup_if_exists(path):
    if path.exists():
        bak = path.with_name(path.name + ".bak")
        i = 1
        while bak.exists():
            bak = path.with_name(path.name + f".bak{i}")
            i += 1
        path.rename(bak)
        print(f"备份旧文件: {path} -> {bak}")


# ============================================================
# Trout1935
# ============================================================

def make_trout():

    print("\n" + "=" * 70)
    print("Trout1935")
    print("=" * 70)

    d = ROOT / "data/Trout2047"

    geno_file = d / "myGD.txt"
    pheno_file = d / "phenotype.csv"
    out_file = d / "phongraph.tsv"

    if not geno_file.exists():
        raise FileNotFoundError(geno_file)

    if not pheno_file.exists():
        raise FileNotFoundError(pheno_file)

    # --------------------------------------------------------
    # genotype 样本 ID
    # --------------------------------------------------------

    geno = pd.read_csv(
        geno_file,
        sep=r"\s+",
        usecols=[0],
        dtype=str
    )

    geno.columns = ["sample_id"]
    geno["sample_id"] = clean_id(geno["sample_id"])

    if geno["sample_id"].duplicated().any():
        raise RuntimeError("Trout genotype 存在重复 sample_id")

    # --------------------------------------------------------
    # phenotype
    # --------------------------------------------------------

    pheno = pd.read_csv(
        pheno_file,
        dtype=str,
        encoding="utf-8-sig"
    )

    required = [
        "Animal_id",
        "Binary_survival",
        "Final_weight"
    ]

    miss = [x for x in required if x not in pheno.columns]

    if miss:
        raise RuntimeError(
            f"Trout phenotype 缺少列: {miss}"
        )

    pheno = pheno[required].copy()

    pheno["Animal_id"] = clean_id(
        pheno["Animal_id"]
    )

    if pheno["Animal_id"].duplicated().any():
        raise RuntimeError(
            "Trout phenotype 存在重复 Animal_id"
        )

    pheno["Binary_survival"] = pd.to_numeric(
        pheno["Binary_survival"],
        errors="coerce"
    )

    pheno["Final_weight"] = pd.to_numeric(
        pheno["Final_weight"],
        errors="coerce"
    )

    pheno = pheno.rename(
        columns={"Animal_id": "sample_id"}
    )

    # --------------------------------------------------------
    # 严格按 genotype 中样本顺序对齐
    # --------------------------------------------------------

    out = geno.merge(
        pheno,
        on="sample_id",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # 检查
    # --------------------------------------------------------

    print(f"genotype 样本数: {len(geno)}")
    print(f"phenotype 原始样本数: {len(pheno)}")
    print(f"对齐后样本数: {len(out)}")

    print("\n缺失值:")
    print(
        out[
            ["Binary_survival", "Final_weight"]
        ].isna().sum()
    )

    valid = out.dropna(
        subset=[
            "Binary_survival",
            "Final_weight"
        ]
    ).copy()

    print(
        f"\n两个性状均完整: {len(valid)}/{len(out)}"
    )

    print("\nBinary_survival 分布:")
    print(
        valid["Binary_survival"]
        .value_counts()
        .sort_index()
    )

    print("\nFinal_weight 描述:")
    print(
        valid["Final_weight"].describe()
    )

    # survival 本身就是分类变量；
    # 连续性状按和青鱼类似的二分位划分
    weight_bin = pd.qcut(
        valid["Final_weight"],
        q=2,
        labels=False,
        duplicates="drop"
    )

    valid["strata"] = (
        valid["Binary_survival"]
        .astype(int)
        .astype(str)
        + "_"
        + weight_bin.astype(str)
    )

    print("\n联合分层: survival × Final_weight二分位")
    print(
        valid["strata"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # 输出，格式与青鱼一致
    # --------------------------------------------------------

    backup_if_exists(out_file)

    out.to_csv(
        out_file,
        sep="\t",
        index=False,
        na_rep="NA"
    )

    print(f"\n已生成: {out_file}")

    print("\n前5行:")
    print(out.head().to_string(index=False))


# ============================================================
# Carp1259
# ============================================================

def make_carp():

    print("\n" + "=" * 70)
    print("Carp1259")
    print("=" * 70)

    d = ROOT / "data/Carp1259"

    geno_file = d / "geno.txt"
    pheno_file = d / "phenotype.csv"
    out_file = d / "phongraph.tsv"

    if not geno_file.exists():
        raise FileNotFoundError(geno_file)

    if not pheno_file.exists():
        raise FileNotFoundError(pheno_file)

    # --------------------------------------------------------
    # genotype 样本 ID
    # --------------------------------------------------------

    geno = pd.read_csv(
        geno_file,
        sep=r"\s+",
        usecols=[0],
        dtype=str
    )

    geno.columns = ["sample_id"]
    geno["sample_id"] = clean_id(geno["sample_id"])

    if geno["sample_id"].duplicated().any():
        raise RuntimeError(
            "Carp genotype 存在重复 sample_id"
        )

    # --------------------------------------------------------
    # phenotype
    # --------------------------------------------------------

    pheno = pd.read_csv(
        pheno_file,
        dtype=str,
        encoding="utf-8-sig"
    )

    required = [
        "Id",
        "survival",
        "SL"
    ]

    miss = [x for x in required if x not in pheno.columns]

    if miss:
        raise RuntimeError(
            f"Carp phenotype 缺少列: {miss}"
        )

    pheno = pheno[required].copy()

    pheno["Id"] = clean_id(
        pheno["Id"]
    )

    if pheno["Id"].duplicated().any():
        raise RuntimeError(
            "Carp phenotype 存在重复 Id"
        )

    pheno["survival"] = pd.to_numeric(
        pheno["survival"],
        errors="coerce"
    )

    pheno["SL"] = pd.to_numeric(
        pheno["SL"],
        errors="coerce"
    )

    pheno = pheno.rename(
        columns={"Id": "sample_id"}
    )

    # --------------------------------------------------------
    # 按 geno.txt 原始样本顺序对齐
    # --------------------------------------------------------

    out = geno.merge(
        pheno,
        on="sample_id",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # 检查
    # --------------------------------------------------------

    print(f"genotype 样本数: {len(geno)}")
    print(f"phenotype 原始样本数: {len(pheno)}")
    print(f"对齐后样本数: {len(out)}")

    print("\n缺失值:")
    print(
        out[
            ["survival", "SL"]
        ].isna().sum()
    )

    valid = out.dropna(
        subset=[
            "survival",
            "SL"
        ]
    ).copy()

    print(
        f"\n两个性状均完整: {len(valid)}/{len(out)}"
    )

    print("\nsurvival 分布:")
    print(
        valid["survival"]
        .value_counts()
        .sort_index()
    )

    print("\nSL 描述:")
    print(
        valid["SL"].describe()
    )

    sl_bin = pd.qcut(
        valid["SL"],
        q=2,
        labels=False,
        duplicates="drop"
    )

    valid["strata"] = (
        valid["survival"]
        .astype(int)
        .astype(str)
        + "_"
        + sl_bin.astype(str)
    )

    print("\n联合分层: survival × SL二分位")
    print(
        valid["strata"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    backup_if_exists(out_file)

    out.to_csv(
        out_file,
        sep="\t",
        index=False,
        na_rep="NA"
    )

    print(f"\n已生成: {out_file}")

    print("\n前5行:")
    print(out.head().to_string(index=False))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    make_trout()
    make_carp()

    print("\n" + "=" * 70)
    print("PASS: 两个表型文件生成完成")
    print("=" * 70)

    print(
        "data/Trout2047/phongraph.tsv"
    )

    print(
        "data/Carp1259/phongraph.tsv"
    )

    print(
        "\n本步骤没有生成 trainval/test，"
        "没有修改 genotype，"
        "没有生成 H5。"
    )
