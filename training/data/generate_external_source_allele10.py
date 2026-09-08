#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import gc

import h5py
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedShuffleSplit,
    KFold,
)

try:
    from openpyxl import load_workbook
except ImportError:
    raise RuntimeError(
        "缺少 openpyxl，请先运行: pip install openpyxl"
    )


ROOT = Path("data")

TROUT_DIR = ROOT / "Trout2047"
CARP_DIR = ROOT / "Carp1259"

SEEDS = [1, 2, 3]
N_FOLDS = 5
TEST_SIZE = 0.15
OUTER_SEED = 1

# 和项目 utils.snp_utils 完全相同的 10D 通道顺序
PAIR_TO_CHANNEL = {
    "AA": 0,
    "AT": 1, "TA": 1,
    "AC": 2, "CA": 2,
    "AG": 3, "GA": 3,
    "TT": 4,
    "TC": 5, "CT": 5,
    "TG": 6, "GT": 6,
    "CC": 7,
    "CG": 8, "GC": 8,
    "GG": 9,
}

VALID_BASES = {"A", "C", "G", "T"}


# ============================================================
# 通用
# ============================================================

def make_channels(a1, a2):
    """
    每个 SNP 建立：
       genotype 0 -> channel
       genotype 1 -> channel
       genotype 2 -> channel
    """
    a1 = np.asarray(a1).astype(str)
    a2 = np.asarray(a2).astype(str)

    channels = np.empty(
        (len(a1), 3),
        dtype=np.int8
    )

    for i, (x, y) in enumerate(zip(a1, a2)):
        x = x.strip().upper()
        y = y.strip().upper()

        if x not in VALID_BASES or y not in VALID_BASES:
            raise RuntimeError(
                f"非法 allele: SNP index={i}, {x}/{y}"
            )

        if x == y:
            raise RuntimeError(
                f"非双等位 SNP: index={i}, {x}/{y}"
            )

        channels[i, 0] = PAIR_TO_CHANNEL[x + x]
        channels[i, 1] = PAIR_TO_CHANNEL[x + y]
        channels[i, 2] = PAIR_TO_CHANNEL[y + y]

    return channels


def make_position_features(chrom, pos, numeric_chr=True):
    chrom = pd.Series(chrom)
    pos = pd.to_numeric(
        pd.Series(pos),
        errors="coerce"
    ).fillna(0).to_numpy(dtype=np.float64)

    if numeric_chr:
        chr_num = pd.to_numeric(
            chrom,
            errors="coerce"
        ).fillna(0).to_numpy(dtype=np.float64)
    else:
        codes, _ = pd.factorize(
            chrom.astype(str),
            sort=False
        )

        chr_num = (
            codes.astype(np.float64)
            + 1.0
        )

    feat = np.zeros(
        (len(pos), 6),
        dtype=np.float64
    )

    feat[:, 0] = chr_num
    feat[:, 1] = pos
    feat[:, 2] = pos
    feat[:, 3] = 1000.0
    feat[:, 4] = 0.001
    feat[:, 5] = 0.5

    return feat


def joint_outer_split(
    pheno,
    binary_col,
    continuous_col,
):
    """
    85/15 fixed outer split:
      binary trait × continuous q=2
    """
    df = pheno.copy()

    if df[[binary_col, continuous_col]].isna().any().any():
        raise RuntimeError(
            "outer split 输入存在缺失 phenotype"
        )

    cont_bin = pd.qcut(
        df[continuous_col],
        q=2,
        labels=False,
        duplicates="drop",
    )

    if cont_bin.nunique() != 2:
        raise RuntimeError(
            f"{continuous_col} 无法划分为两个分位组"
        )

    strata = (
        df[binary_col].astype(str)
        + "_"
        + cont_bin.astype(str)
    )

    print("\n联合分层计数:")
    print(strata.value_counts().sort_index())

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=OUTER_SEED,
    )

    train_idx, test_idx = next(
        sss.split(
            np.zeros(len(df)),
            strata
        )
    )

    # 只改变 membership，不打乱最终原始样本顺序
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    if set(train_idx) & set(test_idx):
        raise RuntimeError("trainval/test overlap")

    return train_idx, test_idx


def write_splits_and_cv(
    base,
    pheno,
    train_idx,
    test_idx,
):
    tv_ids = (
        pheno.iloc[train_idx]["sample_id"]
        .astype(str)
        .tolist()
    )

    test_ids = (
        pheno.iloc[test_idx]["sample_id"]
        .astype(str)
        .tolist()
    )

    (base / "trainval_samples.txt").write_text(
        "\n".join(tv_ids) + "\n"
    )

    (base / "test_samples.txt").write_text(
        "\n".join(test_ids) + "\n"
    )

    for seed in SEEDS:
        kf = KFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=seed,
        )

        out = base / f"cv_splits_{seed}.csv"

        rows = []

        for fold, (tr, va) in enumerate(
            kf.split(tv_ids)
        ):
            rows.extend(
                [
                    (fold, tv_ids[i], "train")
                    for i in tr
                ]
            )

            rows.extend(
                [
                    (fold, tv_ids[i], "val")
                    for i in va
                ]
            )

        with out.open("w", newline="") as f:
            w = csv.writer(f)

            w.writerow([
                "fold",
                "sample_id",
                "split"
            ])

            w.writerows(rows)

    print(
        f"trainval={len(tv_ids)}, "
        f"test={len(test_ids)}"
    )


def write_h5_stream(
    out_path,
    geno,
    sample_ids,
    phenotype,
    phenotype_names,
    sample_idx,
    channels,
    pos_feat,
    snp_ids,
    allele1,
    allele2,
    orientation,
    snp_chunk=256,
):
    """
    geno: [sample, SNP], uint8
          0/1/2 genotype
          3 missing

    H5:
      genotype_features = [SNP, sample, 10]
    """
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if out_path.exists():
        out_path.unlink()

    idx = np.asarray(
        sample_idx,
        dtype=np.int64
    )

    n_snps = geno.shape[1]
    n_samples = len(idx)

    selected_pheno = phenotype[idx].astype(
        np.float32
    )

    na_mask = np.isnan(
        selected_pheno
    ).astype(np.uint8)

    sid_selected = [
        str(sample_ids[i])
        for i in idx
    ]

    sid_len = max(
        1,
        max(len(x) for x in sid_selected)
    )

    snp_len = max(
        1,
        max(len(str(x)) for x in snp_ids)
    )

    with h5py.File(
        out_path,
        "w"
    ) as h5:

        ds = h5.create_dataset(
            "features/genotype_features",
            shape=(
                n_snps,
                n_samples,
                10
            ),
            dtype=np.float32,
            chunks=(
                min(snp_chunk, n_snps),
                min(64, n_samples),
                10
            ),
            compression="gzip",
            compression_opts=4,
        )

        h5.create_dataset(
            "features/position_features",
            data=pos_feat,
        )

        h5.create_dataset(
            "phenotypes",
            data=selected_pheno,
        )

        h5.create_dataset(
            "phenotype_names",
            data=np.asarray(
                phenotype_names,
                dtype="S"
            ),
        )

        h5.create_dataset(
            "sample_ids",
            data=np.asarray(
                sid_selected,
                dtype=f"S{sid_len}"
            ),
        )

        h5.create_dataset(
            "phenotypes_na_mask",
            data=na_mask,
        )

        # 额外保存，模型不会受影响
        h5.create_dataset(
            "snp_ids",
            data=np.asarray(
                snp_ids,
                dtype=f"S{snp_len}"
            ),
        )

        h5.create_dataset(
            "allele1",
            data=np.asarray(
                allele1,
                dtype="S1"
            ),
        )

        h5.create_dataset(
            "allele2",
            data=np.asarray(
                allele2,
                dtype="S1"
            ),
        )

        h5.attrs[
            "genotype_orientation"
        ] = orientation

        # 低内存逐 SNP chunk 编码
        for s in range(
            0,
            n_snps,
            snp_chunk
        ):
            e = min(
                s + snp_chunk,
                n_snps
            )

            # [sample, chunk] -> [chunk, sample]
            g = geno[
                np.ix_(
                    idx,
                    np.arange(s, e)
                )
            ].T

            out = np.zeros(
                (
                    e - s,
                    n_samples,
                    10
                ),
                dtype=np.float32
            )

            for code in (0, 1, 2):
                rr, cc = np.where(
                    g == code
                )

                if len(rr):
                    ch = channels[
                        s + rr,
                        code
                    ]

                    out[
                        rr,
                        cc,
                        ch
                    ] = 1.0

            ds[s:e, :, :] = out

            if (
                s == 0
                or e == n_snps
                or (s // snp_chunk) % 20 == 0
            ):
                print(
                    f"  H5: {e}/{n_snps} SNP"
                )

    print("写入:", out_path)


def final_h5_check(paths):
    print("\n===== H5 FINAL CHECK =====")

    for path in paths:
        with h5py.File(
            path,
            "r"
        ) as f:

            g = f[
                "features/genotype_features"
            ]

            p = f["phenotypes"]

            print("\n", path)
            print(
                " genotype:",
                g.shape,
                g.dtype
            )
            print(
                " phenotype:",
                p.shape
            )
            print(
                " samples:",
                len(f["sample_ids"])
            )
            print(
                " SNP:",
                len(f["snp_ids"])
            )
            print(
                " orientation:",
                f.attrs[
                    "genotype_orientation"
                ]
            )


# ============================================================
# TROUT
# ============================================================

def read_trout_annotation(path):
    print(
        "读取虹鳟 57K annotation:",
        path
    )

    wb = load_workbook(
        path,
        read_only=True,
        data_only=True,
    )

    ws = wb[
        wb.sheetnames[0]
    ]

    rows = ws.iter_rows(
        values_only=True
    )

    header = [
        str(x).strip()
        if x is not None
        else ""
        for x in next(rows)
    ]

    required = [
        "Affy SNP ID",
        "Allele A",
        "Allele B",
    ]

    for c in required:
        if c not in header:
            raise RuntimeError(
                f"annotation 缺少字段: {c}\n"
                f"实际字段: {header}"
            )

    i_id = header.index(
        "Affy SNP ID"
    )

    i_a = header.index(
        "Allele A"
    )

    i_b = header.index(
        "Allele B"
    )

    ann = {}

    for row in rows:
        if row[i_id] is None:
            continue

        marker = str(
            row[i_id]
        ).strip()

        a = str(
            row[i_a]
        ).strip().upper()

        b = str(
            row[i_b]
        ).strip().upper()

        if marker:
            if marker in ann:
                raise RuntimeError(
                    f"annotation duplicate: {marker}"
                )

            ann[marker] = (
                a,
                b
            )

    wb.close()

    print(
        "57K annotation SNP:",
        len(ann)
    )

    return ann


def read_trout_genotype(
    path,
    n_snps,
):
    ids = []
    rows = []

    lut = np.full(
        256,
        255,
        dtype=np.uint8
    )

    lut[ord("0")] = 0
    lut[ord("1")] = 1
    lut[ord("2")] = 2

    # BLUPF90 missing
    lut[ord("5")] = 3

    with path.open() as f:
        for line_no, line in enumerate(
            f,
            1
        ):
            p = line.strip().split()

            if len(p) != 2:
                raise RuntimeError(
                    f"GenoSRS 第{line_no}行 "
                    f"不是2字段"
                )

            sid, text = p

            if len(text) != n_snps:
                raise RuntimeError(
                    f"{sid}: genotype长度 "
                    f"{len(text)} != {n_snps}"
                )

            raw = np.frombuffer(
                text.encode("ascii"),
                dtype=np.uint8
            )

            arr = lut[raw]

            if np.any(arr == 255):
                bad = np.unique(
                    raw[arr == 255]
                )

                raise RuntimeError(
                    f"未知 genotype字符: "
                    f"{[chr(x) for x in bad]}"
                )

            ids.append(sid)
            rows.append(arr)

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "虹鳟 genotype sample ID重复"
        )

    geno = np.stack(
        rows,
        axis=0
    )

    return ids, geno


def process_trout():
    print(
        "\n"
        + "=" * 76
        + "\nTROUT / Barria2019\n"
        + "=" * 76
    )

    base = TROUT_DIR

    geno_file = (
        base / "GenoSRS_blup.txt"
    )

    map_file = (
        base / "Map.txt"
    )

    pheno_file = (
        base / "Phenotype.txt"
    )

    for p in [
        geno_file,
        map_file,
        pheno_file,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    ann_files = [
        p
        for p in base.iterdir()
        if (
            p.is_file()
            and p.suffix.lower() == ".xlsx"
            and "appendixs2"
            in p.name.lower()
        )
    ]

    if len(ann_files) != 1:
        raise RuntimeError(
            "找不到唯一的 "
            "men12337...appendixs2.xlsx"
        )

    map_df = pd.read_csv(
        map_file,
        sep=r"\s+",
        header=None,
        names=[
            "Chromosome",
            "marker",
            "genetic_distance",
            "position",
        ],
        dtype={
            "marker": str
        }
    )

    if map_df[
        "marker"
    ].duplicated().any():
        raise RuntimeError(
            "Trout Map marker duplicate"
        )

    markers = (
        map_df["marker"]
        .astype(str)
        .tolist()
    )

    print(
        "Map SNP:",
        len(markers)
    )

    if len(markers) != 26068:
        raise RuntimeError(
            f"预期26068 SNP，实际"
            f"{len(markers)}"
        )

    ann = read_trout_annotation(
        ann_files[0]
    )

    missing = [
        x
        for x in markers
        if x not in ann
    ]

    print(
        "annotation matched:",
        len(markers) - len(missing),
        "/",
        len(markers)
    )

    if missing:
        print(
            "missing前20:",
            missing[:20]
        )

        raise RuntimeError(
            "Trout annotation不是100%匹配"
        )

    allele_a = np.asarray([
        ann[x][0]
        for x in markers
    ])

    allele_b = np.asarray([
        ann[x][1]
        for x in markers
    ])

    bad = [
        (
            markers[i],
            allele_a[i],
            allele_b[i]
        )
        for i in range(len(markers))
        if (
            allele_a[i]
            not in VALID_BASES
            or allele_b[i]
            not in VALID_BASES
            or allele_a[i]
            == allele_b[i]
        )
    ]

    if bad:
        print("bad allele:", bad[:20])
        raise RuntimeError(
            "Trout allele异常"
        )

    geno_ids, geno_all = (
        read_trout_genotype(
            geno_file,
            len(markers),
        )
    )

    print(
        "genotype:",
        geno_all.shape
    )

    print(
        "genotype codes:",
        dict(
            zip(
                *np.unique(
                    geno_all,
                    return_counts=True
                )
            )
        )
    )

    if len(geno_ids) != 2047:
        raise RuntimeError(
            f"预期2047 genotyped fish，"
            f"实际{len(geno_ids)}"
        )

    raw_pheno = pd.read_csv(
        pheno_file,
        sep=r"\s+"
    )

    required = [
        "Id",
        "SRS_Mortality",
        "End_Weight",
    ]

    for c in required:
        if c not in raw_pheno.columns:
            raise RuntimeError(
                f"Phenotype.txt 缺少 {c}"
            )

    raw_pheno["Id"] = (
        raw_pheno["Id"]
        .astype(str)
    )

    raw_pheno = (
        raw_pheno
        .drop_duplicates("Id")
        .set_index("Id")
    )

    geno_set = set(geno_ids)

    missing_pheno = [
        x
        for x in geno_ids
        if x not in raw_pheno.index
    ]

    if missing_pheno:
        raise RuntimeError(
            f"{len(missing_pheno)} genotype "
            "samples没有 phenotype"
        )

    # genotype 顺序即最终统一样本顺序
    pheno = (
        raw_pheno
        .loc[
            geno_ids,
            [
                "SRS_Mortality",
                "End_Weight",
            ]
        ]
        .reset_index()
        .rename(
            columns={
                "Id": "sample_id"
            }
        )
    )

    pheno[
        "SRS_Mortality"
    ] = pd.to_numeric(
        pheno[
            "SRS_Mortality"
        ],
        errors="coerce"
    )

    pheno[
        "End_Weight"
    ] = pd.to_numeric(
        pheno[
            "End_Weight"
        ],
        errors="coerce"
    )

    complete = (
        pheno[
            [
                "SRS_Mortality",
                "End_Weight"
            ]
        ]
        .notna()
        .all(axis=1)
    )

    if not complete.all():
        raise RuntimeError(
            "2047个基因型样本中存在"
            "双性状缺失"
        )

    print(
        "phenotype samples:",
        len(pheno)
    )

    print(
        "SRS_Mortality:"
    )
    print(
        pheno[
            "SRS_Mortality"
        ].value_counts()
    )

    print(
        "End_Weight mean:",
        pheno[
            "End_Weight"
        ].mean()
    )

    # 保存统一 phenotype
    pheno.to_csv(
        base / "phongraph.tsv",
        sep="\t",
        index=False,
    )

    train_idx, test_idx = (
        joint_outer_split(
            pheno,
            "SRS_Mortality",
            "End_Weight",
        )
    )

    write_splits_and_cv(
        base,
        pheno,
        train_idx,
        test_idx,
    )

    channels = make_channels(
        allele_a,
        allele_b
    )

    pos_feat = make_position_features(
        map_df["Chromosome"],
        map_df["position"],
        numeric_chr=True,
    )

    meta = map_df.copy()

    meta["AlleleA"] = (
        allele_a
    )

    meta["AlleleB"] = (
        allele_b
    )

    meta.to_csv(
        base / "snp_metadata.tsv",
        sep="\t",
        index=False,
    )

    phenotype = (
        pheno[
            [
                "SRS_Mortality",
                "End_Weight",
            ]
        ]
        .to_numpy(dtype=np.float32)
    )

    out_dir = (
        base / "processed"
    )

    train_h5 = (
        out_dir
        / "trout_barria2019_allele10_trainval.h5"
    )

    test_h5 = (
        out_dir
        / "trout_barria2019_allele10_test.h5"
    )

    orientation = (
        "BLUPF90: 0=AlleleA/AlleleA, "
        "1=AlleleA/AlleleB, "
        "2=AlleleB/AlleleB, "
        "5=missing"
    )

    write_h5_stream(
        train_h5,
        geno_all,
        geno_ids,
        phenotype,
        [
            "SRS_Mortality",
            "End_Weight"
        ],
        train_idx,
        channels,
        pos_feat,
        markers,
        allele_a,
        allele_b,
        orientation,
    )

    write_h5_stream(
        test_h5,
        geno_all,
        geno_ids,
        phenotype,
        [
            "SRS_Mortality",
            "End_Weight"
        ],
        test_idx,
        channels,
        pos_feat,
        markers,
        allele_a,
        allele_b,
        orientation,
    )

    final_h5_check(
        [
            train_h5,
            test_h5
        ]
    )

    print(
        "\nPASS: TROUT SOURCE ALLELE10"
    )


# ============================================================
# CARP
# ============================================================

def read_whitespace_or_csv(path):
    attempts = [
        dict(
            sep=r"\s+",
            engine="python"
        ),
        dict(
            sep=","
        ),
        dict(
            sep="\t"
        ),
    ]

    for kw in attempts:
        try:
            df = pd.read_csv(
                path,
                **kw
            )

            if len(df.columns) > 1:
                return df
        except Exception:
            pass

    raise RuntimeError(
        f"无法读取表格: {path}"
    )


def find_carp_phenotype(base):
    """
    自动在 S2 / S3 中寻找：
      Id
      survival
      SL
    """
    candidates = [
        base / "File_S3.txt",
        base / "File_S2.txt",
    ]

    aliases_id = {
        "id",
        "sample_id",
        "animal_id",
    }

    aliases_surv = {
        "survival",
        "surv",
    }

    aliases_sl = {
        "sl",
        "standard_length",
        "standardlength",
        "length",
    }

    reports = []

    for path in candidates:
        if not path.exists():
            continue

        df = read_whitespace_or_csv(
            path
        )

        cols = {
            str(c).strip().lower():
            c
            for c in df.columns
        }

        reports.append(
            (
                path.name,
                list(df.columns)
            )
        )

        def find_alias(names):
            for x in names:
                if x in cols:
                    return cols[x]
            return None

        c_id = find_alias(
            aliases_id
        )

        c_surv = find_alias(
            aliases_surv
        )

        c_sl = find_alias(
            aliases_sl
        )

        if (
            c_id is not None
            and c_surv is not None
            and c_sl is not None
        ):
            out = df[
                [
                    c_id,
                    c_surv,
                    c_sl
                ]
            ].copy()

            out.columns = [
                "sample_id",
                "survival",
                "SL",
            ]

            out[
                "sample_id"
            ] = out[
                "sample_id"
            ].astype(str)

            out[
                "survival"
            ] = pd.to_numeric(
                out["survival"],
                errors="coerce"
            )

            out[
                "SL"
            ] = pd.to_numeric(
                out["SL"],
                errors="coerce"
            )

            print(
                "Carp phenotype source:",
                path
            )

            return out

    print(
        "\nS2/S3 headers:"
    )

    for name, cols in reports:
        print(name, cols)

    raise RuntimeError(
        "没有在 File_S2/File_S3 中"
        "自动找到 Id + survival + SL"
    )


def read_carp_s4(
    path,
    expected_snps=15615,
):
    with path.open() as f:
        header = (
            f.readline()
            .strip()
            .split()
        )

        if len(header) != (
            expected_snps + 1
        ):
            raise RuntimeError(
                f"S4 header字段数="
                f"{len(header)}, "
                f"预期{expected_snps + 1}"
            )

        if header[0].lower() != "id":
            raise RuntimeError(
                "S4第一列不是 Id"
            )

        markers = [
            str(x)
            for x in header[1:]
        ]

        ids = []
        rows = []

        for line_no, line in enumerate(
            f,
            2
        ):
            p = (
                line.strip()
                .split()
            )

            if not p:
                continue

            if len(p) != len(header):
                raise RuntimeError(
                    f"S4第{line_no}行 "
                    f"字段数={len(p)}, "
                    f"预期={len(header)}"
                )

            sid = p[0]

            vals = np.fromiter(
                (
                    3
                    if x.upper()
                    in {
                        "NA",
                        "NAN",
                        "."
                    }
                    else int(x)
                    for x in p[1:]
                ),
                dtype=np.uint8,
                count=expected_snps,
            )

            if np.any(
                ~np.isin(
                    vals,
                    [0, 1, 2, 3]
                )
            ):
                raise RuntimeError(
                    f"S4非法 genotype: {sid}"
                )

            ids.append(sid)
            rows.append(vals)

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "S4 sample ID重复"
        )

    geno = np.stack(
        rows,
        axis=0
    )

    return ids, markers, geno


def process_carp():
    print(
        "\n"
        + "=" * 76
        + "\nCARP / Palaiokostas2019\n"
        + "=" * 76
    )

    base = CARP_DIR

    s1_path = (
        base / "File_S1.txt"
    )

    s4_path = (
        base / "File_S4.txt"
    )

    if not s1_path.exists():
        raise FileNotFoundError(
            s1_path
        )

    if not s4_path.exists():
        raise FileNotFoundError(
            s4_path
        )

    s1 = pd.read_csv(
        s1_path,
        sep=r"\s+",
        dtype={
            "Id": str,
            "Chromosome": str,
        }
    )

    required = [
        "Id",
        "Chromosome",
        "Ref_genome_position",
        "Allele1",
        "Allele2",
        "MAF",
        "Sequence",
    ]

    for c in required:
        if c not in s1.columns:
            raise RuntimeError(
                f"S1缺少字段: {c}\n"
                f"实际: {list(s1.columns)}"
            )

    s1[
        "Id"
    ] = s1[
        "Id"
    ].astype(str)

    if s1["Id"].duplicated().any():
        raise RuntimeError(
            "S1 SNP ID重复"
        )

    print(
        "S1 SNP:",
        len(s1)
    )

    if len(s1) != 15615:
        raise RuntimeError(
            f"预期15615 SNP，"
            f"实际{len(s1)}"
        )

    all_ids, markers, geno_all = (
        read_carp_s4(
            s4_path,
            15615
        )
    )

    print(
        "S4 genotype:",
        geno_all.shape
    )

    print(
        "S4 samples:",
        len(all_ids)
    )

    s1_ids = set(
        s1["Id"]
    )

    s4_ids = set(
        markers
    )

    missing_s1 = (
        s4_ids - s1_ids
    )

    extra_s1 = (
        s1_ids - s4_ids
    )

    print(
        "S1↔S4 marker match:",
        len(s4_ids & s1_ids),
        "/",
        len(markers)
    )

    if missing_s1 or extra_s1:
        print(
            "S4 not in S1:",
            list(missing_s1)[:20]
        )

        print(
            "S1 not in S4:",
            list(extra_s1)[:20]
        )

        raise RuntimeError(
            "Carp S1/S4 marker不是100%匹配"
        )

    # 严格按 S4 marker 顺序排列 metadata
    meta = (
        s1
        .set_index("Id")
        .loc[markers]
        .reset_index()
    )

    allele1 = (
        meta["Allele1"]
        .astype(str)
        .str.upper()
        .to_numpy()
    )

    allele2 = (
        meta["Allele2"]
        .astype(str)
        .str.upper()
        .to_numpy()
    )

    # --------------------------------------------------------
    # 用 S1 MAF 对 S4 0/1/2 做完整性 + 方向诊断
    # --------------------------------------------------------

    called = (
        geno_all != 3
    )

    n_called = (
        called.sum(axis=0)
    )

    dosage_sum = np.where(
        called,
        geno_all,
        0
    ).sum(
        axis=0,
        dtype=np.float64
    )

    p_code2 = (
        dosage_sum
        /
        (
            2.0
            * n_called
        )
    )

    maf_calc = np.minimum(
        p_code2,
        1.0 - p_code2
    )

    maf_s1 = pd.to_numeric(
        meta["MAF"],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64
    )

    maf_error = np.abs(
        maf_calc - maf_s1
    )

    print(
        "\nCarp MAF validation"
    )

    print(
        "median |calculated-S1|:",
        float(
            np.nanmedian(maf_error)
        )
    )

    print(
        "95% quantile error:",
        float(
            np.nanquantile(
                maf_error,
                0.95
            )
        )
    )

    print(
        "within 0.02:",
        float(
            np.nanmean(
                maf_error <= 0.02
            )
        )
    )

    if (
        np.nanmedian(maf_error)
        > 0.01
        or
        np.nanmean(
            maf_error <= 0.02
        ) < 0.90
    ):
        raise RuntimeError(
            "S4 genotype频率与S1 MAF"
            "不吻合，停止生成10D"
        )

    # 额外验证 code2 是否对应 S1 Allele2 的 minor dosage。
    direct = np.abs(
        p_code2 - maf_s1
    )

    direct_rate = np.nanmean(
        direct <= 0.02
    )

    print(
        "P(code2)≈S1 MAF 比例:",
        float(direct_rate)
    )

    if direct_rate < 0.90:
        raise RuntimeError(
            "无法高置信确认 "
            "0=A1/A1, 1=A1/A2, 2=A2/A2。"
            "先把这段 MAF 输出发给我，"
            "不要生成错误10D。"
        )

    print(
        "PASS: S4 0/1/2方向得到 "
        "S1 MAF支持"
    )

    channels = make_channels(
        allele1,
        allele2
    )

    pheno = find_carp_phenotype(
        base
    )

    pheno = (
        pheno
        .drop_duplicates(
            "sample_id"
        )
        .set_index(
            "sample_id"
        )
    )

    # 只取有 genotype + 完整双性状的 offspring
    valid_ids = [
        sid
        for sid in all_ids
        if (
            sid in pheno.index
            and
            pd.notna(
                pheno.loc[
                    sid,
                    "survival"
                ]
            )
            and
            pd.notna(
                pheno.loc[
                    sid,
                    "SL"
                ]
            )
        )
    ]

    print(
        "S4 ∩ complete phenotype:",
        len(valid_ids)
    )

    if len(valid_ids) != 1259:
        raise RuntimeError(
            f"预期1259 offspring，"
            f"实际{len(valid_ids)}"
        )

    row_lookup = {
        sid: i
        for i, sid
        in enumerate(all_ids)
    }

    valid_rows = np.asarray(
        [
            row_lookup[x]
            for x in valid_ids
        ],
        dtype=np.int64
    )

    # 这里开始完全丢掉 parents，
    # H5只使用1259 offspring
    geno = geno_all[
        valid_rows,
        :
    ]

    pheno_valid = (
        pheno
        .loc[
            valid_ids,
            [
                "survival",
                "SL"
            ]
        ]
        .reset_index()
    )

    pheno_valid.to_csv(
        base / "phongraph.tsv",
        sep="\t",
        index=False,
    )

    print(
        "survival:"
    )

    print(
        pheno_valid[
            "survival"
        ].value_counts()
    )

    print(
        "SL mean:",
        pheno_valid[
            "SL"
        ].mean()
    )

    train_idx, test_idx = (
        joint_outer_split(
            pheno_valid,
            "survival",
            "SL",
        )
    )

    write_splits_and_cv(
        base,
        pheno_valid,
        train_idx,
        test_idx,
    )

    pos_feat = make_position_features(
        meta["Chromosome"],
        meta["Ref_genome_position"],
        numeric_chr=False,
    )

    meta.to_csv(
        base / "snp_metadata.tsv",
        sep="\t",
        index=False,
    )

    phenotype = (
        pheno_valid[
            [
                "survival",
                "SL"
            ]
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    out_dir = (
        base / "processed"
    )

    train_h5 = (
        out_dir
        / "carp_palaiokostas2019_allele10_trainval.h5"
    )

    test_h5 = (
        out_dir
        / "carp_palaiokostas2019_allele10_test.h5"
    )

    orientation = (
        "S1/S4: "
        "0=Allele1/Allele1, "
        "1=Allele1/Allele2, "
        "2=Allele2/Allele2, "
        "NA=missing; "
        "validated against S1 MAF"
    )

    write_h5_stream(
        train_h5,
        geno,
        valid_ids,
        phenotype,
        [
            "survival",
            "SL"
        ],
        train_idx,
        channels,
        pos_feat,
        markers,
        allele1,
        allele2,
        orientation,
    )

    write_h5_stream(
        test_h5,
        geno,
        valid_ids,
        phenotype,
        [
            "survival",
            "SL"
        ],
        test_idx,
        channels,
        pos_feat,
        markers,
        allele1,
        allele2,
        orientation,
    )

    final_h5_check(
        [
            train_h5,
            test_h5
        ]
    )

    print(
        "\nPASS: CARP SOURCE ALLELE10"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    process_trout()

    gc.collect()

    process_carp()

    print(
        "\n"
        + "=" * 76
    )

    print(
        "ALL PASS"
    )

    print(
        "虹鳟 + 鲤鱼源数据 "
        "10D allele H5 已重新生成"
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":
    main()
