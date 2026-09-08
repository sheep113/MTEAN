#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import shutil

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parents[2]

TEST_RATIO = 0.15
SEED = 1


def backup(path):
    if path.exists():
        dst = path.with_name(
            path.name
            + ".bak_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        shutil.copy2(path, dst)
        print(f"备份: {path} -> {dst}")


def write_ids(path, ids):
    path.write_text(
        "\n".join(map(str, ids)) + "\n",
        encoding="utf-8"
    )


def split_dataset(
    dataset_name,
    pheno_file,
    survival_col,
    continuous_col,
    expected_valid
):
    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    df = pd.read_csv(
        pheno_file,
        sep="\t",
        dtype={"sample_id": str}
    )

    df["sample_id"] = df["sample_id"].astype(str).str.strip()

    if df["sample_id"].duplicated().any():
        raise RuntimeError(
            f"{dataset_name}: sample_id 存在重复"
        )

    # --------------------------------------------------
    # 只使用两个性状都完整的样本
    # --------------------------------------------------
    valid = df.dropna(
        subset=[
            survival_col,
            continuous_col
        ]
    ).copy()

    print("原始样本数:", len(df))
    print("双性状完整样本:", len(valid))
    print("排除不完整样本:", len(df) - len(valid))

    if len(valid) != expected_valid:
        raise RuntimeError(
            f"{dataset_name}: "
            f"预期完整样本 {expected_valid}，"
            f"实际 {len(valid)}"
        )

    # survival 必须为 0/1
    surv_values = set(
        valid[survival_col]
        .astype(int)
        .unique()
        .tolist()
    )

    if not surv_values.issubset({0, 1}):
        raise RuntimeError(
            f"{dataset_name}: "
            f"{survival_col} 出现非0/1值: {surv_values}"
        )

    # --------------------------------------------------
    # 联合分层
    #
    # survival:
    #   直接使用 0 / 1
    #
    # 连续性状:
    #   按中位数二分
    # --------------------------------------------------
    cont_bin = pd.qcut(
        valid[continuous_col],
        q=2,
        labels=False,
        duplicates="drop"
    )

    if cont_bin.nunique() != 2:
        raise RuntimeError(
            f"{dataset_name}: "
            f"{continuous_col} 无法正常二分"
        )

    valid["strata"] = (
        valid[survival_col]
        .astype(int)
        .astype(str)
        + "_"
        + cont_bin.astype(str)
    )

    print("\n完整样本 survival 分布:")
    print(
        valid[survival_col]
        .value_counts()
        .sort_index()
    )

    print("\n联合分层:")
    print(
        valid["strata"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------
    # 与青鱼相同：
    # StratifiedShuffleSplit
    # test_size = 0.15
    # random_state = 1
    # --------------------------------------------------
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_RATIO,
        random_state=SEED
    )

    trainval_idx, test_idx = next(
        sss.split(
            valid,
            valid["strata"]
        )
    )

    trainval = valid.iloc[
        trainval_idx
    ].copy()

    test = valid.iloc[
        test_idx
    ].copy()

    trainval_ids = trainval[
        "sample_id"
    ].tolist()

    test_ids = test[
        "sample_id"
    ].tolist()

    # --------------------------------------------------
    # 严格验证
    # --------------------------------------------------
    tv_set = set(trainval_ids)
    test_set = set(test_ids)
    valid_set = set(valid["sample_id"])

    if tv_set & test_set:
        raise RuntimeError(
            f"{dataset_name}: trainval/test 有重叠"
        )

    if (tv_set | test_set) != valid_set:
        raise RuntimeError(
            f"{dataset_name}: trainval/test union 不完整"
        )

    if len(trainval_ids) != len(tv_set):
        raise RuntimeError(
            f"{dataset_name}: trainval 内有重复 ID"
        )

    if len(test_ids) != len(test_set):
        raise RuntimeError(
            f"{dataset_name}: test 内有重复 ID"
        )

    out_dir = pheno_file.parent

    trainval_file = (
        out_dir / "trainval_samples.txt"
    )

    test_file = (
        out_dir / "test_samples.txt"
    )

    backup(trainval_file)
    backup(test_file)

    write_ids(
        trainval_file,
        trainval_ids
    )

    write_ids(
        test_file,
        test_ids
    )

    # --------------------------------------------------
    # 打印划分后的分布
    # --------------------------------------------------
    print("\n划分结果:")
    print("trainval:", len(trainval))
    print("test:", len(test))
    print(
        "实际 test ratio:",
        round(len(test) / len(valid), 6)
    )

    print("\n全部样本 strata:")
    print(
        valid["strata"]
        .value_counts()
        .sort_index()
    )

    print("\ntrainval strata:")
    print(
        trainval["strata"]
        .value_counts()
        .sort_index()
    )

    print("\ntest strata:")
    print(
        test["strata"]
        .value_counts()
        .sort_index()
    )

    print("\ntest 各层抽取比例:")
    ratio = (
        test["strata"]
        .value_counts()
        .sort_index()
        /
        valid["strata"]
        .value_counts()
        .sort_index()
    )

    print(ratio.round(4))

    print("\n输出:")
    print(trainval_file)
    print(test_file)

    return {
        "dataset": dataset_name,
        "total": len(df),
        "valid": len(valid),
        "trainval": len(trainval),
        "test": len(test)
    }


def main():

    trout = split_dataset(
        dataset_name="Trout1935",
        pheno_file=ROOT / "data/Trout2047/phongraph.tsv",
        survival_col="Binary_survival",
        continuous_col="Final_weight",
        expected_valid=1844
    )

    carp = split_dataset(
        dataset_name="Carp1259",
        pheno_file=ROOT / "data/Carp1259/phongraph.tsv",
        survival_col="survival",
        continuous_col="SL",
        expected_valid=1259
    )

    print("\n" + "=" * 70)
    print("PASS: 两个数据集固定独立测试集生成完成")
    print("=" * 70)

    print(
        f"Trout1935: "
        f"valid={trout['valid']}, "
        f"trainval={trout['trainval']}, "
        f"test={trout['test']}"
    )

    print(
        f"Carp1259: "
        f"valid={carp['valid']}, "
        f"trainval={carp['trainval']}, "
        f"test={carp['test']}"
    )

    print("\n参数:")
    print("test_size = 0.15")
    print("seed      = 1")
    print("split     = survival × continuous-trait median bin")


if __name__ == "__main__":
    main()
