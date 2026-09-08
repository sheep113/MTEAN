#!/usr/bin/env python3

import csv
from pathlib import Path
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "Trout1935": 1567,
    "Carp1259": 1070,
}

SEEDS = [1, 2, 3]
N_FOLDS = 5


def load_ids(path):
    ids = [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            f"{path} 中存在重复 sample_id"
        )

    return ids


def generate(dataset, expected_n):

    data_dir = ROOT / "data" / dataset

    trainval_file = (
        data_dir / "trainval_samples.txt"
    )

    test_file = (
        data_dir / "test_samples.txt"
    )

    trainval_ids = load_ids(trainval_file)
    test_ids = load_ids(test_file)

    if len(trainval_ids) != expected_n:
        raise RuntimeError(
            f"{dataset}: trainval 样本数错误，"
            f"预期 {expected_n}，实际 {len(trainval_ids)}"
        )

    if set(trainval_ids) & set(test_ids):
        raise RuntimeError(
            f"{dataset}: trainval/test 有重叠"
        )

    print("\n" + "=" * 60)
    print(dataset)
    print("=" * 60)

    print("trainval:", len(trainval_ids))
    print("independent test:", len(test_ids))

    for seed in SEEDS:

        kf = KFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=seed
        )

        rows = []

        print(f"\nseed = {seed}")

        for fold, (train_idx, val_idx) in enumerate(
            kf.split(trainval_ids)
        ):

            train_ids = [
                trainval_ids[i]
                for i in train_idx
            ]

            val_ids = [
                trainval_ids[i]
                for i in val_idx
            ]

            tr = set(train_ids)
            va = set(val_ids)

            if tr & va:
                raise RuntimeError(
                    f"{dataset} seed={seed} "
                    f"fold={fold}: train/val overlap"
                )

            if (
                tr | va
            ) != set(trainval_ids):
                raise RuntimeError(
                    f"{dataset} seed={seed} "
                    f"fold={fold}: union 错误"
                )

            if (
                (tr | va)
                &
                set(test_ids)
            ):
                raise RuntimeError(
                    f"{dataset}: independent test "
                    "泄漏到 CV"
                )

            print(
                f"fold {fold}: "
                f"train={len(train_ids)}, "
                f"val={len(val_ids)}"
            )

            rows.extend(
                [
                    [fold, sid, "train"]
                    for sid in train_ids
                ]
            )

            rows.extend(
                [
                    [fold, sid, "val"]
                    for sid in val_ids
                ]
            )

        out = (
            data_dir /
            f"cv_splits_{seed}.csv"
        )

        with open(
            out,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "fold",
                    "sample_id",
                    "split"
                ]
            )

            writer.writerows(rows)

        print("已生成:", out)

    print(
        f"\nPASS: {dataset} "
        "普通 5-fold CV 完成"
    )


def main():

    for dataset, expected_n in DATASETS.items():
        generate(
            dataset,
            expected_n
        )

    print("\n" + "=" * 60)
    print("PASS: 两个数据集 CV 全部完成")
    print("=" * 60)

    print("CV method : KFold")
    print("n_splits  : 5")
    print("shuffle   : True")
    print("seeds     : 1, 2, 3")
    print(
        "test set  : fixed stratified independent test"
    )


if __name__ == "__main__":
    main()
