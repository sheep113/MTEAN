#!/usr/bin/env python3
"""
生成指定种子的 5 折交叉验证划分文件，基于训练验证池样本。
读取 trainval_samples.txt（或 train_samples.txt），输出 cv_splits_{seed}.csv。
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

OUT_DIR = "data/blackcarp"

def main():
    if len(sys.argv) < 2:
        print("用法: python generate_cv_splits.py <seed>")
        sys.exit(1)
    seed = int(sys.argv[1])

    # 读取训练验证池样本（已包含全部训练评估样本）
    trainval_file = os.path.join(OUT_DIR, "trainval_samples.txt")
    if not os.path.exists(trainval_file):
        raise FileNotFoundError(f"未找到 {trainval_file}")

    with open(trainval_file) as f:
        trainval_samples = [line.strip() for line in f if line.strip()]

    n = len(trainval_samples)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for fold_idx, (train_loc, val_loc) in enumerate(kf.split(indices)):
        fold_train = [trainval_samples[i] for i in train_loc]
        fold_val   = [trainval_samples[i] for i in val_loc]
        for sid in fold_train:
            rows.append((fold_idx, sid, 'train'))
        for sid in fold_val:
            rows.append((fold_idx, sid, 'val'))

    df = pd.DataFrame(rows, columns=['fold', 'sample_id', 'split'])
    out_path = os.path.join(OUT_DIR, f"cv_splits_{seed}.csv")
    df.to_csv(out_path, index=False)
    print(f"已生成: {out_path}")

if __name__ == "__main__":
    main()