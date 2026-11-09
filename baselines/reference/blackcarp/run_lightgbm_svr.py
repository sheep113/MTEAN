# Modified for METAN portable release 2026-11-09.
#!/usr/bin/env python3
"""
Black carp independent-test evaluation for LightGBM and SVR.

Design:
    424 trainval
        -> 3 seeds × 5-fold
        -> each fold trains only on its train split
        -> predict the same frozen 75-fish independent test

Outputs:
    1. 15 individual models
    2. per-seed 5-fold ensemble
    3. 15-model ensemble
    4. Pearson / MSE / RMSE / MAE / R2
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
# StandardScaler disabled
from sklearn.svm import SVR

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================
SEEDS = [1, 2, 3]
FOLDS = [0, 1, 2, 3, 4]
TRAITS = ["BW", "LE"]
MODELS = ["LightGBM", "SVR"]

GENO_CSV = Path(
    "data/blackcarp499/geno_fixed_for_python.csv"
)

PHENO_FILE = Path(
    "data/blackcarp499/phongraph_new.tsv"
)

TRAINVAL_SAMPLES = Path(
    "data/blackcarp499/trainval_samples.txt"
)

TEST_SAMPLES = Path(
    "data/blackcarp499/test_samples.txt"
)

OUT_DIR = Path(
    "evaluation/blackcarp/ml_models/independent_test_results"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# METRICS
# ============================================================
def calc_metrics(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=np.float64
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64
    )

    if (
        np.std(y_true) > 0
        and
        np.std(y_pred) > 0
    ):
        pearson = np.corrcoef(
            y_true,
            y_pred
        )[0, 1]
    else:
        pearson = np.nan

    error = (
        y_pred
        -
        y_true
    )

    mse = np.mean(
        error ** 2
    )

    rmse = np.sqrt(
        mse
    )

    mae = np.mean(
        np.abs(error)
    )

    ss_res = np.sum(
        error ** 2
    )

    ss_tot = np.sum(
        (
            y_true
            -
            y_true.mean()
        ) ** 2
    )

    r2 = (
        1.0
        -
        ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    return {
        "Pearson": pearson,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


# ============================================================
# HELPERS
# ============================================================
def read_id_file(path):

    with open(path) as f:

        return [
            line.strip()
            for line in f
            if line.strip()
        ]


def impute_train_median(
    X_train,
    X_test
):
    """
    Missing values are imputed using TRAIN-fold medians only.
    """

    medians = np.nanmedian(
        X_train,
        axis=0
    )

    medians = np.where(
        np.isnan(medians),
        0.0,
        medians
    )

    X_train = np.where(
        np.isnan(X_train),
        medians,
        X_train
    )

    X_test = np.where(
        np.isnan(X_test),
        medians,
        X_test
    )

    return X_train, X_test


# ============================================================
# MAIN
# ============================================================
def main():

    print("=" * 80)
    print(
        "BLACK CARP | LIGHTGBM + SVR | INDEPENDENT TEST"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Genotype
    # --------------------------------------------------------
    print(
        "读取 genotype CSV..."
    )

    geno_df = pd.read_csv(
        GENO_CSV,
        index_col=0
    )

    geno_df.index = (
        geno_df.index.astype(str)
    )

    geno_df.columns = (
        geno_df.columns.astype(str)
    )

    all_sample_ids = (
        geno_df.index.tolist()
    )

    all_snp_ids = set(
        geno_df.columns.tolist()
    )

    print(
        f"样本数     : {len(all_sample_ids)}"
    )

    print(
        f"SNP总数    : {geno_df.shape[1]}"
    )

    print(
        "基因型缺失率: "
        f"{geno_df.isna().mean().mean():.6f}"
    )

    # --------------------------------------------------------
    # Phenotype
    # --------------------------------------------------------
    pheno = pd.read_csv(
        PHENO_FILE,
        sep="\t"
    )

    pheno["sample_id"] = (
        pheno["sample_id"].astype(str)
    )

    pheno = pheno.set_index(
        "sample_id"
    )

    # --------------------------------------------------------
    # Fixed outer split
    # --------------------------------------------------------
    trainval_ids = read_id_file(
        TRAINVAL_SAMPLES
    )

    test_ids = read_id_file(
        TEST_SAMPLES
    )

    trainval_ids = [
        str(x)
        for x in trainval_ids
    ]

    test_ids = [
        str(x)
        for x in test_ids
    ]

    assert len(trainval_ids) == 424, (
        f"trainval 应为424，实际 {len(trainval_ids)}"
    )

    assert len(test_ids) == 75, (
        f"test 应为75，实际 {len(test_ids)}"
    )

    missing_test = [
        x
        for x in test_ids
        if x not in geno_df.index
    ]

    if missing_test:
        raise RuntimeError(
            f"{len(missing_test)} 个 test ID "
            "不在 genotype CSV 中"
        )

    print()
    print(
        "✅ trainval = 424"
    )
    print(
        "✅ independent test = 75"
    )

    # --------------------------------------------------------
    # Use ALL SNPs (same SNP set as METAN)
    # --------------------------------------------------------

    all_snps = geno_df.columns.tolist()

    snp_lists = {
        "BW": all_snps,
        "LE": all_snps,
    }

    print()
    print("=" * 80)
    print("FIXED SNP SETS")
    print("=" * 80)

    print(
        f"BW: 使用全部SNP={len(all_snps)}"
    )

    print(
        f"LE: 使用全部SNP={len(all_snps)}"
    )

    # --------------------------------------------------------
    # TEST TRUE VALUES
    # --------------------------------------------------------
    test_y = {
        trait:
            pheno.loc[
                test_ids,
                trait
            ].to_numpy(
                dtype=np.float64
            )
        for trait in TRAITS
    }

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------
    individual_rows = []

    # key:
    # (model, trait, seed, fold)


    # ========================================================
    # 3 seeds × 5 folds
    # ========================================================
    for seed in SEEDS:

        cv_file = Path(
            f"data/blackcarp499/"
            f"cv_splits_{seed}.csv"
        )

        cv = pd.read_csv(
            cv_file
        )

        cv["sample_id"] = (
            cv["sample_id"].astype(str)
        )

        print()
        print("#" * 80)
        print(
            f"SEED {seed}"
        )
        print("#" * 80)

        for fold in FOLDS:

            train_ids = cv[
                (cv["fold"] == fold)
                &
                (cv["split"] == "train")
            ]["sample_id"].tolist()

            val_ids = cv[
                (cv["fold"] == fold)
                &
                (cv["split"] == "val")
            ]["sample_id"].tolist()

            if (
                len(train_ids)
                +
                len(val_ids)
                != 424
            ):
                raise RuntimeError(
                    f"Seed {seed} Fold {fold}: "
                    f"train={len(train_ids)}, "
                    f"val={len(val_ids)}"
                )

            print()
            print(
                f"Seed {seed} Fold {fold}: "
                f"train={len(train_ids)}, "
                f"val={len(val_ids)}"
            )

            for trait in TRAITS:

                snps = snp_lists[
                    trait
                ]

                X_train = (
                    geno_df.loc[
                        train_ids,
                        snps
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                X_test = (
                    geno_df.loc[
                        test_ids,
                        snps
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                y_train = (
                    pheno.loc[
                        train_ids,
                        trait
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                y_test = test_y[
                    trait
                ]

                # --------------------------------------------
                # Fold-train-only imputation
                # --------------------------------------------
                X_train_i, X_test_i = (
                    impute_train_median(
                        X_train,
                        X_test
                    )
                )

                # ============================================
                # LIGHTGBM
                # ============================================
                lgb = LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=15,
                    max_depth=5,
                    random_state=seed,
                    n_jobs=-1,
                    verbose=-1,
                )

                lgb.fit(
                    X_train_i,
                    y_train
                )

                pred_lgb = lgb.predict(
                    X_test_i
                )


                m = calc_metrics(
                    y_test,
                    pred_lgb
                )

                individual_rows.append({
                    "model":
                        "LightGBM",

                    "trait":
                        trait,

                    "seed":
                        seed,

                    "fold":
                        fold,

                    "train_n":
                        len(train_ids),

                    "val_n":
                        len(val_ids),

                    **m,
                })

                # ============================================
                # SVR
                # ============================================
                # ================================
                # SVR raw SNP input (no scaling)
                # ================================
                X_train_s = X_train_i

                X_test_s = X_test_i

                svr = SVR(
                    kernel="rbf",
                    C=1.0,
                    epsilon=0.1,
                )

                svr.fit(
                    X_train_s,
                    y_train
                )

                pred_svr = svr.predict(
                    X_test_s
                )


                m_svr = calc_metrics(
                    y_test,
                    pred_svr
                )

                individual_rows.append({
                    "model":
                        "SVR",

                    "trait":
                        trait,

                    "seed":
                        seed,

                    "fold":
                        fold,

                    "train_n":
                        len(train_ids),

                    "val_n":
                        len(val_ids),

                    **m_svr,
                })

                print(
                    f"  {trait}: "
                    f"LightGBM={m['Pearson']:.4f} | "
                    f"SVR={m_svr['Pearson']:.4f}"
                )


    # ========================================================
    # ONLY FOLD RESULTS + SEED MEAN + ALL SEED MEAN
    # NO ENSEMBLE
    # ========================================================


    individual_df = pd.DataFrame(
        individual_rows
    )


    for model_name in MODELS:


        model_df = individual_df[
            individual_df["model"] == model_name
        ]


        output_rows = []


        # -------------------------
        # fold level
        # -------------------------

        for seed in SEEDS:


            seed_df = model_df[
                model_df["seed"] == seed
            ]


            for fold in FOLDS:


                fold_df = seed_df[
                    seed_df["fold"] == fold
                ]


                bw = fold_df[
                    fold_df["trait"]=="BW"
                ].iloc[0]


                le = fold_df[
                    fold_df["trait"]=="LE"
                ].iloc[0]


                output_rows.append({

                    "group":"fold",
                    "seed":seed,
                    "fold":fold,

                    "BW_Pearson":
                        bw["Pearson"],
                    "BW_MSE":
                        bw["MSE"],
                    "BW_R2":
                        bw["R2"],

                    "LE_Pearson":
                        le["Pearson"],
                    "LE_MSE":
                        le["MSE"],
                    "LE_R2":
                        le["R2"]
                })



            # -------------------------
            # seed mean
            # -------------------------

            bw_seed = seed_df[
                seed_df["trait"]=="BW"
            ][
                ["Pearson","MSE","R2"]
            ].mean()


            le_seed = seed_df[
                seed_df["trait"]=="LE"
            ][
                ["Pearson","MSE","R2"]
            ].mean()



            output_rows.append({

                "group":"seed_mean",
                "seed":seed,
                "fold":"mean",

                "BW_Pearson":
                    bw_seed["Pearson"],
                "BW_MSE":
                    bw_seed["MSE"],
                "BW_R2":
                    bw_seed["R2"],

                "LE_Pearson":
                    le_seed["Pearson"],
                "LE_MSE":
                    le_seed["MSE"],
                "LE_R2":
                    le_seed["R2"]
            })



        # -------------------------
        # all seed mean
        # -------------------------

        temp = pd.DataFrame(output_rows)

        seed_mean = temp[
            temp["group"]=="seed_mean"
        ]


        output_rows.append({

            "group":"all_seed_mean",
            "seed":"all",
            "fold":"mean",

            "BW_Pearson":
                seed_mean["BW_Pearson"].mean(),
            "BW_MSE":
                seed_mean["BW_MSE"].mean(),
            "BW_R2":
                seed_mean["BW_R2"].mean(),

            "LE_Pearson":
                seed_mean["LE_Pearson"].mean(),
            "LE_MSE":
                seed_mean["LE_MSE"].mean(),
            "LE_R2":
                seed_mean["LE_R2"].mean()

        })


        out = pd.DataFrame(
            output_rows
        )


        out_file = OUT_DIR / (
            model_name.lower()
            +
            "_test_performance_summary_seed1_3.csv"
        )


        out.to_csv(
            out_file,
            index=False
        )


        print()
        print("="*80)
        print(model_name)
        print("="*80)

        print(out.tail(1))

        print(
            "保存:",
            out_file
        )



if __name__ == "__main__":
    main()
