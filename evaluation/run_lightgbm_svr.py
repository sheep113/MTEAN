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
from sklearn.preprocessing import StandardScaler
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

FIXED_CANDIDATES = {
    "BW": Path(
        "data/blackcarp499/fixed_candidates_BW.txt"
    ),
    "LE": Path(
        "data/blackcarp499/fixed_candidates_LE.txt"
    ),
}

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
    # Candidate SNPs
    # --------------------------------------------------------
    snp_lists = {}

    print()
    print("=" * 80)
    print("FIXED SNP SETS")
    print("=" * 80)

    for trait in TRAITS:

        candidate_ids = read_id_file(
            FIXED_CANDIDATES[trait]
        )

        valid = [
            str(s)
            for s in candidate_ids
            if str(s) in all_snp_ids
        ]

        if not valid:
            raise RuntimeError(
                f"{trait}: 没有匹配到任何候选SNP"
            )

        snp_lists[
            trait
        ] = valid

        print(
            f"{trait}: "
            f"候选={len(candidate_ids)}, "
            f"实际匹配={len(valid)}"
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
    predictions = {}

    prediction_long_rows = []

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

                predictions[
                    (
                        "LightGBM",
                        trait,
                        seed,
                        fold
                    )
                ] = pred_lgb

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
                scaler = StandardScaler()

                X_train_s = (
                    scaler.fit_transform(
                        X_train_i
                    )
                )

                X_test_s = (
                    scaler.transform(
                        X_test_i
                    )
                )

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

                predictions[
                    (
                        "SVR",
                        trait,
                        seed,
                        fold
                    )
                ] = pred_svr

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

                # --------------------------------------------
                # save every prediction
                # --------------------------------------------
                for i, sid in enumerate(
                    test_ids
                ):

                    prediction_long_rows.append({
                        "sample_id":
                            sid,

                        "seed":
                            seed,

                        "fold":
                            fold,

                        "trait":
                            trait,

                        "label":
                            y_test[i],

                        "LightGBM_pred":
                            pred_lgb[i],

                        "SVR_pred":
                            pred_svr[i],
                    })

                print(
                    f"  {trait}: "
                    f"LightGBM={m['Pearson']:.4f} | "
                    f"SVR={m_svr['Pearson']:.4f}"
                )

    # ========================================================
    # SAVE INDIVIDUAL RESULTS
    # ========================================================
    individual_df = pd.DataFrame(
        individual_rows
    )

    individual_df.to_csv(
        OUT_DIR
        / "lightgbm_svr_independent_test_15models.csv",
        index=False
    )

    pd.DataFrame(
        prediction_long_rows
    ).to_csv(
        OUT_DIR
        / "lightgbm_svr_independent_test_all_predictions.csv",
        index=False
    )

    # ========================================================
    # 15 MODEL RAW MEAN ± SD
    # ========================================================
    print()
    print("=" * 80)
    print(
        "15 INDIVIDUAL MODELS | MEAN ± SD"
    )
    print("=" * 80)

    raw_summary_rows = []

    for model_name in MODELS:

        for trait in TRAITS:

            sub = individual_df[
                (individual_df["model"] == model_name)
                &
                (individual_df["trait"] == trait)
            ]

            row = {
                "model":
                    model_name,

                "trait":
                    trait,
            }

            for metric in [
                "Pearson",
                "MSE",
                "RMSE",
                "MAE",
                "R2",
            ]:

                row[
                    f"{metric}_mean"
                ] = sub[
                    metric
                ].mean()

                row[
                    f"{metric}_sd"
                ] = sub[
                    metric
                ].std(ddof=1)

            raw_summary_rows.append(
                row
            )

            print(
                f"{model_name:9s} {trait}: "
                f"Pearson="
                f"{row['Pearson_mean']:.4f} ± "
                f"{row['Pearson_sd']:.4f}"
            )

    pd.DataFrame(
        raw_summary_rows
    ).to_csv(
        OUT_DIR
        / "lightgbm_svr_independent_test_15models_summary.csv",
        index=False
    )

    # ========================================================
    # PER-SEED 5-FOLD ENSEMBLE
    # ========================================================
    print()
    print("=" * 80)
    print(
        "PER-SEED 5-FOLD ENSEMBLE"
    )
    print("=" * 80)

    seed_ensemble_rows = []
    seed_pred_rows = []

    for model_name in MODELS:

        for seed in SEEDS:

            trait_scores = []

            for trait in TRAITS:

                fold_preds = np.stack(
                    [
                        predictions[
                            (
                                model_name,
                                trait,
                                seed,
                                fold
                            )
                        ]
                        for fold in FOLDS
                    ],
                    axis=0
                )

                ensemble_pred = np.mean(
                    fold_preds,
                    axis=0
                )

                m = calc_metrics(
                    test_y[trait],
                    ensemble_pred
                )

                seed_ensemble_rows.append({
                    "model":
                        model_name,

                    "seed":
                        seed,

                    "trait":
                        trait,

                    **m,
                })

                trait_scores.append(
                    m["Pearson"]
                )

                for i, sid in enumerate(
                    test_ids
                ):

                    seed_pred_rows.append({
                        "sample_id":
                            sid,

                        "model":
                            model_name,

                        "seed":
                            seed,

                        "trait":
                            trait,

                        "label":
                            test_y[trait][i],

                        "prediction":
                            ensemble_pred[i],
                    })

            print(
                f"{model_name:9s} Seed {seed}: "
                f"BW={seed_ensemble_rows[-2]['Pearson']:.4f} | "
                f"LE={seed_ensemble_rows[-1]['Pearson']:.4f} | "
                f"Mean={np.mean(trait_scores):.4f}"
            )

    seed_df = pd.DataFrame(
        seed_ensemble_rows
    )

    seed_df.to_csv(
        OUT_DIR
        / "lightgbm_svr_seed_5fold_ensemble_metrics.csv",
        index=False
    )

    pd.DataFrame(
        seed_pred_rows
    ).to_csv(
        OUT_DIR
        / "lightgbm_svr_seed_5fold_ensemble_predictions.csv",
        index=False
    )

    # ========================================================
    # 3 SEEDS MEAN ± SD
    # ========================================================
    print()
    print("=" * 80)
    print(
        "3 SEEDS | 5-FOLD ENSEMBLE MEAN ± SD"
    )
    print("=" * 80)

    seed_summary_rows = []

    for model_name in MODELS:

        for trait in TRAITS:

            sub = seed_df[
                (seed_df["model"] == model_name)
                &
                (seed_df["trait"] == trait)
            ]

            row = {
                "model":
                    model_name,

                "trait":
                    trait,
            }

            for metric in [
                "Pearson",
                "MSE",
                "RMSE",
                "MAE",
                "R2",
            ]:

                row[
                    f"{metric}_mean"
                ] = sub[
                    metric
                ].mean()

                row[
                    f"{metric}_sd"
                ] = sub[
                    metric
                ].std(ddof=1)

            seed_summary_rows.append(
                row
            )

            print(
                f"{model_name:9s} {trait}: "
                f"Pearson="
                f"{row['Pearson_mean']:.4f} ± "
                f"{row['Pearson_sd']:.4f}"
            )

    seed_summary_df = pd.DataFrame(
        seed_summary_rows
    )

    seed_summary_df.to_csv(
        OUT_DIR
        / "lightgbm_svr_seed_5fold_ensemble_summary.csv",
        index=False
    )

    # ========================================================
    # FINAL 15-MODEL ENSEMBLE
    # ========================================================
    print()
    print("=" * 80)
    print(
        "15-MODEL ENSEMBLE | INDEPENDENT TEST"
    )
    print("=" * 80)

    final_rows = []
    final_prediction_rows = []

    for model_name in MODELS:

        trait_pearsons = []

        for trait in TRAITS:

            all_preds = np.stack(
                [
                    predictions[
                        (
                            model_name,
                            trait,
                            seed,
                            fold
                        )
                    ]
                    for seed in SEEDS
                    for fold in FOLDS
                ],
                axis=0
            )

            final_pred = np.mean(
                all_preds,
                axis=0
            )

            m = calc_metrics(
                test_y[trait],
                final_pred
            )

            final_rows.append({
                "model":
                    model_name,

                "trait":
                    trait,

                **m,
            })

            trait_pearsons.append(
                m["Pearson"]
            )

            for i, sid in enumerate(
                test_ids
            ):

                final_prediction_rows.append({
                    "sample_id":
                        sid,

                    "model":
                        model_name,

                    "trait":
                        trait,

                    "label":
                        test_y[trait][i],

                    "prediction":
                        final_pred[i],
                })

            print(
                f"{model_name:9s} {trait}: "
                f"Pearson={m['Pearson']:.4f} | "
                f"RMSE={m['RMSE']:.4f} | "
                f"MAE={m['MAE']:.4f} | "
                f"R2={m['R2']:.4f}"
            )

        print(
            f"{model_name:9s} Mean Pearson = "
            f"{np.mean(trait_pearsons):.4f}"
        )

    final_df = pd.DataFrame(
        final_rows
    )

    final_df.to_csv(
        OUT_DIR
        / "lightgbm_svr_15model_ensemble_metrics.csv",
        index=False
    )

    pd.DataFrame(
        final_prediction_rows
    ).to_csv(
        OUT_DIR
        / "lightgbm_svr_15model_ensemble_predictions.csv",
        index=False
    )

    print()
    print("=" * 80)
    print("RESULT FILES")
    print("=" * 80)

    for p in sorted(
        OUT_DIR.iterdir()
    ):
        print(p)

    print()
    print(
        "✅ LIGHTGBM + SVR INDEPENDENT TEST PASS"
    )


if __name__ == "__main__":
    main()
