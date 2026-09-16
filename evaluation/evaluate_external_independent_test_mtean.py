#!/usr/bin/env python3

import sys
import re
import json
from pathlib import Path
from contextlib import nullcontext

import h5py
import numpy as np
import pandas as pd
import torch

from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

# ============================================================
# 基础设置
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.models.DNAWhisper import DNAWhisper


torch.backends.cudnn.enabled = False

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


SEEDS = [1]
FOLDS = [0, 1, 2, 3, 4]
BATCH_SIZE = 8


DATASETS = {
    "Carp1259": {
        "data_dir": ROOT / "data/Carp1259",

        "train_h5":
            ROOT
            / "data/Carp1259/processed/"
              "carp_palaiokostas2019_allele10_trainval.h5",

        "test_h5":
            ROOT
            / "data/Carp1259/processed/"
              "carp_palaiokostas2019_allele10_test.h5",

        "checkpoint_root":
            ROOT / "output/carp/mtean_head_v1_cv",

        "checkpoint_prefix": "carp_head_v1",

        "status_dir":
            ROOT / "output/run_status/carp",

        "output_dir":
            ROOT / "evaluation/carp/independent_test_results",

        "traits": [
            "survival",
            "SL",
        ],

        "binary_traits": {
            "survival",
        },

        "expected_trainval": 1070,
        "expected_test": 189,
        "expected_snps": 15615,
    },


    "Trout2047": {
        "data_dir": ROOT / "data/Trout2047",

        "train_h5":
            ROOT
            / "data/Trout2047/processed/"
              "trout_barria2019_allele10_trainval.h5",

        "test_h5":
            ROOT
            / "data/Trout2047/processed/"
              "trout_barria2019_allele10_test.h5",

        "checkpoint_root":
            ROOT / "output/trout/mtean_cv",

        "checkpoint_prefix": "trout",

        "status_dir":
            ROOT / "output/run_status/trout",

        "output_dir":
            ROOT / "evaluation/trout/independent_test_results",

        "traits": [
            "SRS_Mortality",
            "End_Weight",
        ],

        "binary_traits": {
            "SRS_Mortality",
        },

        "expected_trainval": 1739,
        "expected_test": 308,
        "expected_snps": 26068,
    },
}


# ============================================================
# 工具函数
# ============================================================

def decode_strings(values):
    return [
        x.decode("utf-8")
        if isinstance(x, bytes)
        else str(x)
        for x in values
    ]


def check_done_files(cfg):

    missing = []

    for seed in SEEDS:
        for fold in FOLDS:

            p = (
                cfg["status_dir"]
                / f"seed_{seed}_fold_{fold}.done"
            )

            if not p.exists():
                missing.append(
                    f"Seed{seed} Fold{fold}"
                )

    if missing:
        raise RuntimeError(
            "存在未完成 fold:\n"
            + "\n".join(missing)
        )

    print("✅ 15/15 .done PASS")


def parse_val_score(path):

    m = re.search(
        r"val_pearson_corr_epoch[=:_-]([-+]?\d*\.?\d+)",
        path.name,
    )

    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    return None


def get_best_checkpoint(cfg, seed, fold):

    fold_dir = (
        cfg["checkpoint_root"]
        / f'{cfg["checkpoint_prefix"]}_{seed}'
        / f"fold_{fold}"
    )

    if not fold_dir.exists():
        raise FileNotFoundError(
            f"checkpoint目录不存在: {fold_dir}"
        )

    candidates = []

    for p in fold_dir.glob("*.ckpt"):

        name = p.name

        # 不用 last
        if name.startswith("last"):
            continue

        # 不用 Lightning 重复生成的 -v1 / -v2
        if re.search(
            r"-v\d+\.ckpt$",
            name
        ):
            continue

        candidates.append(p)

    if not candidates:
        raise RuntimeError(
            f"没有找到正式 best checkpoint: {fold_dir}"
        )

    scored = []

    for p in candidates:

        score = parse_val_score(p)

        scored.append(
            (
                -np.inf if score is None else score,
                p.stat().st_mtime,
                p,
            )
        )

    scored.sort(
        key=lambda x: (
            x[0],
            x[1],
        ),
        reverse=True,
    )

    return scored[0][2]


def load_h5_metadata(path):

    with h5py.File(
        path,
        "r",
    ) as f:

        geno_shape = (
            f[
                "features/genotype_features"
            ].shape
        )

        phenotype = (
            f["phenotypes"][:]
            .astype(np.float32)
        )

        sample_ids = decode_strings(
            f["sample_ids"][:]
        )

        phenotype_names = decode_strings(
            f["phenotype_names"][:]
        )

    return (
        geno_shape,
        phenotype,
        sample_ids,
        phenotype_names,
    )


def calculate_metrics(
    y_true,
    y_pred,
    binary=False,
):

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if (
        np.std(y_true) == 0
        or
        np.std(y_pred) == 0
    ):
        pearson = np.nan
    else:
        pearson = pearsonr(
            y_true,
            y_pred,
        )[0]

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    out = {
        "Pearson": float(pearson),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "R2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }

    if binary:

        unique = np.unique(
            y_true
        )

        if len(unique) == 2:
            try:
                out["AUC"] = float(
                    roc_auc_score(
                        y_true,
                        y_pred,
                    )
                )
            except Exception:
                out["AUC"] = np.nan
        else:
            out["AUC"] = np.nan

    return out


# ============================================================
# 模型推理
# ============================================================

def predict_test(
    checkpoint,
    test_h5,
    pheno_mean,
    pheno_std,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "    loading:",
        checkpoint.name
    )

    model = DNAWhisper.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
    )

    model.eval()
    model.requires_grad_(False)
    model.to(device)

    predictions = []

    with h5py.File(
        test_h5,
        "r",
    ) as f:

        geno = f[
            "features/genotype_features"
        ]

        n_samples = geno.shape[1]

        for start in range(
            0,
            n_samples,
            BATCH_SIZE,
        ):

            end = min(
                start + BATCH_SIZE,
                n_samples,
            )

            # H5:
            # [SNP, sample, 10]
            #
            # model:
            # [sample, SNP, 10]

            x_np = np.asarray(
                geno[
                    :,
                    start:end,
                    :
                ],
                dtype=np.float32,
            )

            x_np = np.transpose(
                x_np,
                (1, 0, 2),
            )

            x = torch.from_numpy(
                x_np
            ).to(device)

            if device.type == "cuda":
                amp_ctx = torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                )
            else:
                amp_ctx = nullcontext()

            with torch.inference_mode():
                with amp_ctx:

                    output = model(x)

                    pred_norm = output[
                        "final_pred"
                    ]

            predictions.append(
                pred_norm
                .float()
                .cpu()
                .numpy()
            )

    pred_norm = np.concatenate(
        predictions,
        axis=0,
    )

    # fold-specific phenotype反标准化
    pred_raw = (
        pred_norm
        * pheno_std[None, :]
        + pheno_mean[None, :]
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred_raw


# ============================================================
# 单数据集评价
# ============================================================

def evaluate_dataset(
    dataset_name,
    cfg,
):

    print()
    print("=" * 80)
    print(dataset_name)
    print("=" * 80)

    # --------------------------------------------------------
    # 1. 完整性检查
    # --------------------------------------------------------

    check_done_files(cfg)

    (
        train_shape,
        y_trainval,
        trainval_ids,
        train_traits,
    ) = load_h5_metadata(
        cfg["train_h5"]
    )

    (
        test_shape,
        y_test,
        test_ids,
        test_traits,
    ) = load_h5_metadata(
        cfg["test_h5"]
    )

    print(
        "train genotype:",
        train_shape
    )

    print(
        "test genotype :",
        test_shape
    )

    print(
        "traits        :",
        train_traits
    )

    if train_traits != cfg["traits"]:
        raise RuntimeError(
            f"train phenotype names错误: {train_traits}"
        )

    if test_traits != cfg["traits"]:
        raise RuntimeError(
            f"test phenotype names错误: {test_traits}"
        )

    if train_shape[0] != cfg["expected_snps"]:
        raise RuntimeError(
            "train SNP数量错误"
        )

    if test_shape[0] != cfg["expected_snps"]:
        raise RuntimeError(
            "test SNP数量错误"
        )

    if train_shape[1] != cfg["expected_trainval"]:
        raise RuntimeError(
            f"trainval样本数错误: {train_shape[1]}"
        )

    if test_shape[1] != cfg["expected_test"]:
        raise RuntimeError(
            f"test样本数错误: {test_shape[1]}"
        )

    print("✅ H5 CHECK PASS")

    # --------------------------------------------------------
    # 2. trainval sample ID lookup
    # --------------------------------------------------------

    id_to_index = {
        sid: i
        for i, sid
        in enumerate(trainval_ids)
    }

    if len(id_to_index) != len(
        trainval_ids
    ):
        raise RuntimeError(
            "trainval sample_id存在重复"
        )

    # --------------------------------------------------------
    # 3. 跑15个正式best checkpoint
    # --------------------------------------------------------

    cfg["output_dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []
    individual_metric_rows = []
    all_prediction_rows = []

    # 保存全部预测：
    # key = (seed, fold)
    model_predictions = {}

    for seed in SEEDS:

        cv_path = (
            cfg["data_dir"]
            / f"cv_splits_{seed}.csv"
        )

        cv = pd.read_csv(
            cv_path,
            dtype={
                "sample_id": str
            },
        )

        for fold in FOLDS:

            print()
            print(
                f">>> {dataset_name} "
                f"Seed {seed} Fold {fold}"
            )

            q = cv[
                cv["fold"] == fold
            ]

            train_ids = (
                q[
                    q["split"] == "train"
                ][
                    "sample_id"
                ]
                .astype(str)
                .tolist()
            )

            val_ids = (
                q[
                    q["split"] == "val"
                ][
                    "sample_id"
                ]
                .astype(str)
                .tolist()
            )

            if set(
                train_ids
            ) & set(
                val_ids
            ):
                raise RuntimeError(
                    f"Seed{seed} Fold{fold}: "
                    "train/val overlap"
                )

            if (
                set(train_ids)
                | set(val_ids)
            ) != set(
                trainval_ids
            ):
                raise RuntimeError(
                    f"Seed{seed} Fold{fold}: "
                    "CV IDs与trainval H5不一致"
                )

            train_idx = np.asarray(
                [
                    id_to_index[x]
                    for x in train_ids
                ],
                dtype=int,
            )

            # ------------------------------------------------
            # 与训练DataModule的 standard normalization一致
            # ddof=0
            # ------------------------------------------------

            fold_y = y_trainval[
                train_idx,
                :
            ]

            pheno_mean = np.mean(
                fold_y,
                axis=0,
            )

            pheno_std = np.std(
                fold_y,
                axis=0,
                ddof=0,
            )

            if np.any(
                pheno_std <= 0
            ):
                raise RuntimeError(
                    f"Seed{seed} Fold{fold}: "
                    "phenotype std异常"
                )

            ckpt = get_best_checkpoint(
                cfg,
                seed,
                fold,
            )

            print(
                "    train:",
                len(train_idx),
                "val:",
                len(val_ids),
            )

            print(
                "    checkpoint:",
                ckpt
            )

            pred = predict_test(
                ckpt,
                cfg["test_h5"],
                pheno_mean,
                pheno_std,
            )

            if pred.shape != y_test.shape:
                raise RuntimeError(
                    f"prediction shape错误: "
                    f"{pred.shape} != {y_test.shape}"
                )

            model_predictions[
                (seed, fold)
            ] = pred

            manifest_rows.append({
                "dataset":
                    dataset_name,

                "seed":
                    seed,

                "fold":
                    fold,

                "checkpoint":
                    str(ckpt),

                "train_n":
                    len(train_idx),

                "val_n":
                    len(val_ids),

                "phenotype_mean":
                    json.dumps(
                        pheno_mean.tolist()
                    ),

                "phenotype_std":
                    json.dumps(
                        pheno_std.tolist()
                    ),
            })

            for trait_idx, trait in enumerate(
                cfg["traits"]
            ):

                metrics = calculate_metrics(
                    y_test[:, trait_idx],
                    pred[:, trait_idx],
                    binary=(
                        trait
                        in cfg["binary_traits"]
                    ),
                )

                row = {
                    "dataset":
                        dataset_name,

                    "seed":
                        seed,

                    "fold":
                        fold,

                    "trait":
                        trait,
                }

                row.update(
                    metrics
                )

                individual_metric_rows.append(
                    row
                )

            # 保存逐样本预测
            for i, sid in enumerate(
                test_ids
            ):

                row = {
                    "dataset":
                        dataset_name,

                    "sample_id":
                        sid,

                    "seed":
                        seed,

                    "fold":
                        fold,
                }

                for j, trait in enumerate(
                    cfg["traits"]
                ):
                    row[
                        f"{trait}_true"
                    ] = float(
                        y_test[i, j]
                    )

                    row[
                        f"{trait}_pred"
                    ] = float(
                        pred[i, j]
                    )

                all_prediction_rows.append(
                    row
                )


    # --------------------------------------------------------
    # 4. 每个seed的5-fold ensemble
    # --------------------------------------------------------

    seed_metric_rows = []
    seed_prediction_rows = []

    seed_ensemble_predictions = {}

    for seed in SEEDS:

        preds = np.stack(
            [
                model_predictions[
                    (seed, fold)
                ]
                for fold in FOLDS
            ],
            axis=0,
        )

        ensemble = np.mean(
            preds,
            axis=0,
        )

        seed_ensemble_predictions[
            seed
        ] = ensemble

        for trait_idx, trait in enumerate(
            cfg["traits"]
        ):

            metrics = calculate_metrics(
                y_test[:, trait_idx],
                ensemble[:, trait_idx],
                binary=(
                    trait
                    in cfg["binary_traits"]
                ),
            )

            row = {
                "dataset":
                    dataset_name,

                "seed":
                    seed,

                "trait":
                    trait,
            }

            row.update(
                metrics
            )

            seed_metric_rows.append(
                row
            )

        for i, sid in enumerate(
            test_ids
        ):

            row = {
                "dataset":
                    dataset_name,

                "sample_id":
                    sid,

                "seed":
                    seed,
            }

            for j, trait in enumerate(
                cfg["traits"]
            ):

                row[
                    f"{trait}_true"
                ] = float(
                    y_test[i, j]
                )

                row[
                    f"{trait}_pred"
                ] = float(
                    ensemble[i, j]
                )

            seed_prediction_rows.append(
                row
            )


    # --------------------------------------------------------
    # 5. 15-model ensemble
    # --------------------------------------------------------

    all_15 = np.stack(
        [
            model_predictions[
                (seed, fold)
            ]
            for seed in SEEDS
            for fold in FOLDS
        ],
        axis=0,
    )

    ensemble_15 = np.mean(
        all_15,
        axis=0,
    )

    ensemble15_metric_rows = []

    for trait_idx, trait in enumerate(
        cfg["traits"]
    ):

        metrics = calculate_metrics(
            y_test[:, trait_idx],
            ensemble_15[:, trait_idx],
            binary=(
                trait
                in cfg["binary_traits"]
            ),
        )

        row = {
            "dataset":
                dataset_name,

            "trait":
                trait,
        }

        row.update(
            metrics
        )

        ensemble15_metric_rows.append(
            row
        )


    ensemble15_prediction_rows = []

    for i, sid in enumerate(
        test_ids
    ):

        row = {
            "dataset":
                dataset_name,

            "sample_id":
                sid,
        }

        for j, trait in enumerate(
            cfg["traits"]
        ):

            row[
                f"{trait}_true"
            ] = float(
                y_test[i, j]
            )

            row[
                f"{trait}_pred"
            ] = float(
                ensemble_15[i, j]
            )

        ensemble15_prediction_rows.append(
            row
        )


    # --------------------------------------------------------
    # 6. 保存
    # --------------------------------------------------------

    out = cfg["output_dir"]

    pd.DataFrame(
        manifest_rows
    ).to_csv(
        out / "checkpoint_manifest.csv",
        index=False,
    )

    pd.DataFrame(
        individual_metric_rows
    ).to_csv(
        out / "MTEAN_independent_test_15_best_models.csv",
        index=False,
    )

    pd.DataFrame(
        all_prediction_rows
    ).to_csv(
        out / "MTEAN_independent_test_all_model_predictions.csv",
        index=False,
    )

    seed_metrics_df = pd.DataFrame(
        seed_metric_rows
    )

    seed_metrics_df.to_csv(
        out / "MTEAN_seed_5fold_ensemble_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        seed_prediction_rows
    ).to_csv(
        out / "MTEAN_seed_5fold_ensemble_predictions.csv",
        index=False,
    )

    ensemble15_df = pd.DataFrame(
        ensemble15_metric_rows
    )

    ensemble15_df.to_csv(
        out / "MTEAN_15model_ensemble_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        ensemble15_prediction_rows
    ).to_csv(
        out / "MTEAN_15model_ensemble_predictions.csv",
        index=False,
    )


    # --------------------------------------------------------
    # 7. 打印最终摘要
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        f"{dataset_name} | "
        "3 SEEDS | 5-FOLD ENSEMBLE MEAN ± SD"
    )
    print("=" * 80)

    summary = {}

    for trait in cfg["traits"]:

        x = seed_metrics_df[
            seed_metrics_df[
                "trait"
            ] == trait
        ]

        p_mean = x[
            "Pearson"
        ].mean()

        p_sd = x[
            "Pearson"
        ].std(
            ddof=1
        )

        print(
            f"{trait:<18} "
            f"Pearson="
            f"{p_mean:.4f} ± {p_sd:.4f}"
        )

        summary[
            f"{trait}_seed_ensemble_pearson_mean"
        ] = float(
            p_mean
        )

        summary[
            f"{trait}_seed_ensemble_pearson_sd"
        ] = float(
            p_sd
        )


    print()
    print("=" * 80)
    print(
        f"{dataset_name} | "
        "15-MODEL ENSEMBLE | INDEPENDENT TEST"
    )
    print("=" * 80)

    pearsons = []

    for _, row in ensemble15_df.iterrows():

        trait = row["trait"]

        pearsons.append(
            row["Pearson"]
        )

        line = (
            f"{trait:<18} "
            f"Pearson={row['Pearson']:.4f} | "
            f"RMSE={row['RMSE']:.4f} | "
            f"MAE={row['MAE']:.4f} | "
            f"R2={row['R2']:.4f}"
        )

        if "AUC" in row.index and pd.notna(
            row["AUC"]
        ):
            line += (
                f" | AUC={row['AUC']:.4f}"
            )

        print(line)

    mean_pearson = float(
        np.mean(
            pearsons
        )
    )

    print(
        f"Mean Pearson = "
        f"{mean_pearson:.4f}"
    )

    summary[
        "15model_mean_pearson"
    ] = mean_pearson

    with open(
        out / "summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        "✅ saved:",
        out
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "DEVICE:",
        "CUDA"
        if torch.cuda.is_available()
        else "CPU"
    )

    for dataset_name, cfg in DATASETS.items():

        evaluate_dataset(
            dataset_name,
            cfg,
        )

    print()
    print("=" * 80)
    print("✅ CARP + TROUT INDEPENDENT TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
