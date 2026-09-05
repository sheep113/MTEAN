#!/usr/bin/env python3
"""
快速生成 trainval 样本的 5-fold CV 划分。

用法：
    python training/data/generate_cv_splits.py 1
    python training/data/generate_cv_splits.py 1 2 3

输入：
    data/blackcarp499/trainval_samples.txt

输出：
    data/blackcarp499/cv_splits_1.csv
    data/blackcarp499/cv_splits_2.csv
    ...
"""

import csv
import sys
from pathlib import Path

from sklearn.model_selection import KFold


DATA_DIR = Path("data/blackcarp499")
TRAINVAL_FILE = DATA_DIR / "trainval_samples.txt"
N_FOLDS = 5


def load_samples():
    if not TRAINVAL_FILE.exists():
        raise FileNotFoundError(
            f"未找到: {TRAINVAL_FILE}"
        )

    with open(TRAINVAL_FILE) as f:
        samples = [
            line.strip()
            for line in f
            if line.strip()
        ]

    # 检查重复 ID
    if len(samples) != len(set(samples)):
        raise RuntimeError(
            "trainval_samples.txt 中存在重复样本 ID"
        )

    return samples


def generate_one_seed(samples, seed):
    kf = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=seed
    )

    out_path = (
        DATA_DIR /
        f"cv_splits_{seed}.csv"
    )

    rows = []

    print(f"\n========== seed {seed} ==========")

    for fold, (train_idx, val_idx) in enumerate(
        kf.split(samples)
    ):
        train_ids = [
            samples[i]
            for i in train_idx
        ]

        val_ids = [
            samples[i]
            for i in val_idx
        ]

        # 安全检查
        overlap = (
            set(train_ids)
            &
            set(val_ids)
        )

        if overlap:
            raise RuntimeError(
                f"seed={seed}, fold={fold} "
                f"train/val 出现重叠"
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

    with open(
        out_path,
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

    print(
        f"已生成: {out_path}"
    )


def main():
    if len(sys.argv) < 2:
        print(
            "用法:\n"
            "  python training/data/"
            "generate_cv_splits.py 1\n"
            "或:\n"
            "  python training/data/"
            "generate_cv_splits.py 1 2 3"
        )
        sys.exit(1)

    seeds = [
        int(x)
        for x in sys.argv[1:]
    ]

    samples = load_samples()

    print(
        f"trainval 样本数: "
        f"{len(samples)}"
    )

    for seed in seeds:
        generate_one_seed(
            samples,
            seed
        )

    print(
        "\n全部 CV 划分生成完成。"
    )


if __name__ == "__main__":
    main()
