# Modified for METAN portable release 2026-11-09.
# ============================================================
# GBLUP independent test evaluation
# Aligned with METAN / SVR output format
# Output table format:
# Extension: MAE/RMSE + one prediction CSV per trait/seed/fold.
# Original model fitting, genotype processing and splits are unchanged.
# ============================================================

rm(list = ls())

library(asreml)
library(ASRgenomics)
library(data.table)

# ----------------------------
# 配置
# ----------------------------
ROOT <- "${METAN_ROOT}"

GENO_CSV <- file.path(ROOT, "data/blackcarp499/geno_fixed_for_python.csv")
TRAINVAL_H5 <- file.path(ROOT, "data/blackcarp499/processed/blackcarp_allele10_trainval.h5")
TEST_H5     <- file.path(ROOT, "data/blackcarp499/processed/blackcarp_allele10_test.h5")

CV_DIR <- file.path(ROOT, "data/blackcarp499")

RUN_ID <- paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_", Sys.getpid())
OUT_DIR <- Sys.getenv("GBLUP_OUT_DIR", unset = file.path(
  ROOT, "evaluation/blackcarp/ml_models/independent_test_results",
  paste0("gblup_MAE_predictions_", RUN_ID)
))
# A fresh directory is mandatory: never overwrite historical results.
if (dir.exists(OUT_DIR) || file.exists(OUT_DIR)) stop("Output already exists: ", OUT_DIR)
if (!dir.create(OUT_DIR, recursive = TRUE)) stop("Cannot create output: ", OUT_DIR)
PRED_DIR <- file.path(OUT_DIR, "predictions")
METRIC_DIR <- file.path(OUT_DIR, "fold_metrics")
dir.create(PRED_DIR)
dir.create(METRIC_DIR)
cat("\nNEW OUTPUT DIRECTORY:", OUT_DIR, "\n")
writeLines(capture.output(sessionInfo()), file.path(OUT_DIR, "sessionInfo.txt"))

SEEDS <- 1:3
FOLDS <- 0:4

OUT_CSV <- file.path(OUT_DIR, "gblup_test_performance_summary_seed1_3.csv")

# ----------------------------
# 工具函数
# ----------------------------
calc_metrics <- function(obs, pred) {
  ok <- complete.cases(obs, pred)
  obs <- obs[ok]
  pred <- pred[ok]

  if (length(obs) == 0) {
    return(list(Pearson = NA_real_, MSE = NA_real_, RMSE = NA_real_,
                MAE = NA_real_, R2 = NA_real_))
  }

  pearson <- suppressWarnings(cor(obs, pred))
  mse <- mean((obs - pred)^2)
  sse <- sum((obs - pred)^2)
  sst <- sum((obs - mean(obs))^2)
  r2 <- ifelse(sst == 0, NA_real_, 1 - sse / sst)

  list(Pearson = pearson, MSE = mse, RMSE = sqrt(mse),
       MAE = mean(abs(obs - pred)), R2 = r2)
}

# Save each finished trait immediately; do not lose BW if LE later fails.
# Predictions are explicitly reordered to the original independent-Test IDs.
save_trait_result <- function(res, trait, seed, fold, test_ids) {
  v <- res$valid
  if (anyDuplicated(v$ID) || !setequal(as.character(v$ID), test_ids)) {
    stop("Incomplete/duplicate Test IDs: ", trait, " seed=", seed, " fold=", fold)
  }
  v <- v[match(test_ids, as.character(v$ID)), , drop = FALSE]
  stopifnot(nrow(v) == 75L, identical(as.character(v$ID), test_ids))
  if (!all(is.finite(v$Observed)) || !all(is.finite(v$Prediction))) {
    stop("Non-finite Test truth/prediction: ", trait, " seed=", seed, " fold=", fold)
  }
  pred <- data.frame(
    model = "GBLUP", trait = trait, seed = seed, fold = fold,
    split = "independent_test", sample_id = as.character(v$ID),
    Observed = v$Observed, Prediction = v$Prediction, GEBV = v$GEBV,
    residual = v$Observed - v$Prediction,
    absolute_error = abs(v$Observed - v$Prediction),
    squared_error = (v$Observed - v$Prediction)^2,
    stringsAsFactors = FALSE
  )
  stem <- paste0(trait, "_seed", seed, "_fold", fold)
  pred_path <- file.path(PRED_DIR, paste0(stem, "_predictions.csv"))
  metric_path <- file.path(METRIC_DIR, paste0(stem, "_metrics.csv"))
  stopifnot(!file.exists(pred_path), !file.exists(metric_path))
  fwrite(pred, pred_path)
  # Recompute from saved values to verify the prediction export.
  saved <- fread(pred_path, colClasses = list(character = "sample_id"))
  stopifnot(identical(saved$sample_id, test_ids), nrow(saved) == 75L)
  checks <- calc_metrics(saved$Observed, saved$Prediction)
  for (m in names(checks)) {
    if (!isTRUE(all.equal(checks[[m]], res$metrics[[m]], tolerance = 1e-9))) {
      stop("Saved-prediction metric mismatch: ", stem, " ", m)
    }
  }
  mr <- data.frame(model = "GBLUP", trait = trait, seed = seed, fold = fold,
                   test_n = nrow(saved), as.list(checks),
                   prediction_file = pred_path, stringsAsFactors = FALSE)
  fwrite(mr, metric_path)
  cat("SAVED:", stem, "| n=", nrow(saved), "| MAE=", checks$MAE, "\n")
  invisible(mr)
}

fit_one_trait <- function(dd_combined, pheno_all, test_ids, trait, ginv) {
  fixed_formula <- as.formula(paste(trait, "~ 1"))

  init <- asreml(
    fixed = fixed_formula,
    random = ~ vm(ID, ginv),
    dense = ~ vm(ID, ginv),
    residual = ~ idv(units),
    start.values = TRUE,
    data = dd_combined
  )

  vc <- init$vparameters.table

  mod <- asreml(
    fixed = fixed_formula,
    random = ~ vm(ID, ginv),
    dense = ~ vm(ID, ginv),
    residual = ~ idv(units),
    G.param = vc,
    R.param = vc,
    data = dd_combined
  )

  # 固定效应截距
  fx <- coef(mod)$fixed
  if (is.matrix(fx) || is.data.frame(fx)) {
    mu <- as.numeric(fx[1, 1])
  } else {
    mu <- as.numeric(fx[1])
  }

  # 随机效应 GEBV
  re <- coef(mod)$random
  if (is.vector(re)) {
    re <- matrix(re, ncol = 1, dimnames = list(names(re), "effect"))
  }
  re <- as.data.frame(re)
  re$ID <- sub("^vm\\(ID, ginv\\)_", "", rownames(re))
  colnames(re)[1] <- "GEBV"

  # 测试集真实值
  obs_df <- pheno_all[, c("ID", trait)]
  names(obs_df)[2] <- "Observed"

  valid <- merge(re, obs_df, by = "ID")
  valid <- valid[valid$ID %in% test_ids, , drop = FALSE]
  valid$Prediction <- mu + valid$GEBV

  metrics <- calc_metrics(valid$Observed, valid$Prediction)

  list(
    model = mod,
    valid = valid,
    metrics = metrics
  )
}


# ============================================================
# 读取当前 METAN/SVR 使用的 28804 SNP 基因型
# 保留 0/1/2/NA，不进行任何人为填补
# ============================================================

cat("\n===== LOADING GENOTYPE =====\n")

geno_df <- fread(
  GENO_CSV,
  data.table = FALSE,
  check.names = FALSE
)

stopifnot(names(geno_df)[1] == "sample_id")

geno_ids <- as.character(geno_df[[1]])

geno <- as.matrix(
  geno_df[, -1, drop = FALSE]
)

storage.mode(geno) <- "numeric"
rownames(geno) <- geno_ids

rm(geno_df)
gc()

cat("geno dim =", dim(geno), "\n")
cat("missing =", sum(is.na(geno)), "\n")

stopifnot(nrow(geno) == 499)
stopifnot(ncol(geno) == 28804)
stopifnot(!anyDuplicated(rownames(geno)))
stopifnot(!anyDuplicated(colnames(geno)))

bad <- !is.na(geno) &
       !(geno == 0 | geno == 1 | geno == 2)

stopifnot(!any(bad))

# 核对 SNP union
bw_snps <- as.character(fread(
  file.path(CV_DIR, "fixed_candidates_BW.txt"),
  header = FALSE
)[[1]])

le_snps <- as.character(fread(
  file.path(CV_DIR, "fixed_candidates_LE.txt"),
  header = FALSE
)[[1]])

union_snps <- unique(c(bw_snps, le_snps))

stopifnot(length(union_snps) == 28804)
stopifnot(setequal(colnames(geno), union_snps))

cat("✅ SNP union verified: 28804\n")
cat("✅ genotype coding: 0/1/2/NA\n")
cat("✅ no manual imputation\n")


# ============================================================
# 读取 METAN H5 导出的统一表型 CSV
# 避免 rhdf5 表型矩阵转置问题
# ============================================================

cat("\n===== LOADING METAN PHENOTYPES =====\n")

PHE_CSV <- file.path(
  CV_DIR,
  "metan_exact_phenotypes_499.csv"
)

phe <- fread(
  PHE_CSV,
  data.table = FALSE
)

stopifnot(
  all(c("ID", "BW", "LE", "split") %in% names(phe))
)

phe$ID <- as.character(phe$ID)
phe$split <- trimws(as.character(phe$split))

stopifnot(nrow(phe) == 499)
stopifnot(!anyDuplicated(phe$ID))
stopifnot(setequal(phe$ID, rownames(geno)))

stopifnot(sum(phe$split == "trainval") == 424)
stopifnot(sum(phe$split == "test") == 75)

stopifnot(!anyNA(phe$BW))
stopifnot(!anyNA(phe$LE))

common_ids <- rownames(geno)

pheno_all <- phe[
  match(common_ids, phe$ID),
  c("ID", "BW", "LE"),
  drop = FALSE
]

test_ids <- phe$ID[phe$split == "test"]

stopifnot(
  identical(pheno_all$ID, common_ids)
)

stopifnot(length(test_ids) == 75)

cat("total samples =", nrow(pheno_all), "\n")
cat("trainval =", sum(phe$split == "trainval"), "\n")
cat("independent test =", length(test_ids), "\n")
cat("phenotypes = BW, LE\n")

cat("\n✅ GENOTYPE / PHENOTYPE ALIGNMENT PASSED\n\n")

# ----------------------------
# 主循环
# ----------------------------
all_rows <- list()
row_idx <- 1

metric_cols <- c(
  "BW_Pearson", "BW_MSE", "BW_RMSE", "BW_MAE", "BW_R2",
  "LE_Pearson", "LE_MSE", "LE_RMSE", "LE_MAE", "LE_R2"
)

for (seed in SEEDS) {
  cat("\n############################################################\n")
  cat("SEED =", seed, "\n")
  cat("############################################################\n")

  cv_file <- file.path(CV_DIR, paste0("cv_splits_", seed, ".csv"))
  if (!file.exists(cv_file)) {
    stop(paste("CV split file not found:", cv_file))
  }

  cv <- fread(cv_file)
  if (!all(c("sample_id", "fold", "split") %in% names(cv))) {
    stop(paste("CV file missing required columns:", cv_file))
  }

  seed_fold_rows <- list()

  for (kf in FOLDS) {
    cat("\n----------------------------------------\n")
    cat("seed =", seed, "fold =", kf, "\n")
    cat("----------------------------------------\n")

    train_ids <- cv$sample_id[cv$fold == kf & cv$split == "train"]
    train_ids <- train_ids[train_ids %in% common_ids]

    if (length(train_ids) == 0) {
      stop(paste("No training IDs for seed", seed, "fold", kf))
    }

    all_ids <- c(train_ids, test_ids)
    all_ids <- unique(all_ids)

    cat("train =", length(train_ids), "| independent test =", length(test_ids), "\n")

    geno_fold <- geno[all_ids, , drop = FALSE]
    storage.mode(geno_fold) <- "integer"

    cat("G matrix input:", nrow(geno_fold), "samples x", ncol(geno_fold), "SNPs\n")

    # G matrix（不填补缺失）
    Gobj <- G.matrix(
      M = geno_fold,
      method = "VanRaden",
      na.string = NA
    )

    G <- Gobj$G
    diag(G) <- diag(G) + 0.01

    cat("G dim =", nrow(G), "x", ncol(G), "\n")
    cat("G NA =", sum(is.na(G)), "\n")

    ginv <- G.inverse(G, sparseform = TRUE)$Ginv.sparse

    # ---------------- BW ----------------
    dd_bw_train <- pheno_all[match(train_ids, pheno_all$ID), c("ID", "BW")]
    dd_bw_test  <- pheno_all[match(test_ids, pheno_all$ID), c("ID", "BW")]
    dd_bw_test$BW <- NA
    dd_bw_combined <- rbind(dd_bw_train, dd_bw_test)
    dd_bw_combined$ID <- factor(dd_bw_combined$ID)

    bw_res <- fit_one_trait(
      dd_combined = dd_bw_combined,
      pheno_all = pheno_all,
      test_ids = test_ids,
      trait = "BW",
      ginv = ginv
    )
    save_trait_result(bw_res, "BW", seed, kf, test_ids)

    # ---------------- LE ----------------
    dd_le_train <- pheno_all[match(train_ids, pheno_all$ID), c("ID", "LE")]
    dd_le_test  <- pheno_all[match(test_ids, pheno_all$ID), c("ID", "LE")]
    dd_le_test$LE <- NA
    dd_le_combined <- rbind(dd_le_train, dd_le_test)
    dd_le_combined$ID <- factor(dd_le_combined$ID)

    le_res <- fit_one_trait(
      dd_combined = dd_le_combined,
      pheno_all = pheno_all,
      test_ids = test_ids,
      trait = "LE",
      ginv = ginv
    )
    save_trait_result(le_res, "LE", seed, kf, test_ids)

    # 汇总当前 fold
    fold_row <- data.frame(
      group = "fold",
      seed = seed,
      fold = kf,
      BW_Pearson = bw_res$metrics$Pearson,
      BW_MSE = bw_res$metrics$MSE,
      BW_RMSE = bw_res$metrics$RMSE,
      BW_MAE = bw_res$metrics$MAE,
      BW_R2 = bw_res$metrics$R2,
      LE_Pearson = le_res$metrics$Pearson,
      LE_MSE = le_res$metrics$MSE,
      LE_RMSE = le_res$metrics$RMSE,
      LE_MAE = le_res$metrics$MAE,
      LE_R2 = le_res$metrics$R2,
      stringsAsFactors = FALSE
    )

    print(fold_row)

    seed_fold_rows[[length(seed_fold_rows) + 1]] <- fold_row
    all_rows[[row_idx]] <- fold_row
    row_idx <- row_idx + 1
  }

  # 当前 seed 平均
  seed_fold_df <- rbindlist(seed_fold_rows, fill = TRUE)

  seed_mean_row <- data.frame(
    group = "seed_mean",
    seed = seed,
    fold = "mean",
    as.list(colMeans(seed_fold_df[, metric_cols, with = FALSE], na.rm = TRUE)),
    stringsAsFactors = FALSE
  )

  print(seed_mean_row)

  all_rows[[row_idx]] <- seed_mean_row
  row_idx <- row_idx + 1
}

# 所有 seed 总平均（只对 fold 行求平均）
all_df <- rbindlist(all_rows, fill = TRUE)
fold_only <- all_df[group == "fold"]

all_seed_mean_row <- data.frame(
  group = "all_seed_mean",
  seed = "all",
  fold = "mean",
  as.list(colMeans(fold_only[, metric_cols, with = FALSE], na.rm = TRUE)),
  stringsAsFactors = FALSE
)

all_df <- rbindlist(list(all_df, all_seed_mean_row), fill = TRUE)

# ----------------------------
# 保存结果
# ----------------------------
fwrite(all_df, OUT_CSV)

# Tidy per-trait metrics plus five-fold/15-model Mean and sample SD (ddof=1).
mf <- list.files(METRIC_DIR, pattern = "_metrics\\.csv$", full.names = TRUE)
stopifnot(length(mf) == 30L)
long <- rbindlist(lapply(mf, fread), use.names = TRUE)
stopifnot(nrow(long) == 30L, !anyDuplicated(long[, .(trait, seed, fold)]))
setorder(long, trait, seed, fold)
fwrite(long, file.path(OUT_DIR, "gblup_test_fold_metrics_long.csv"))
metric_names <- c("Pearson", "MSE", "RMSE", "MAE", "R2")
summary_rows <- list()
for (tt in c("BW", "LE")) {
  for (ss in c("1", "2", "3", "all")) {
    z <- long[trait == tt]
    if (ss != "all") z <- z[seed == as.integer(ss)]
    stopifnot(nrow(z) == if (ss == "all") 15L else 5L)
    for (mm in metric_names) {
      if (!all(is.finite(z[[mm]]))) stop("Non-finite metric: ", tt, " ", ss, " ", mm)
      summary_rows[[length(summary_rows) + 1L]] <- data.frame(
        trait = tt, seed = ss, metric = mm, n_models = nrow(z),
        mean = mean(z[[mm]]), SD = sd(z[[mm]]), stringsAsFactors = FALSE)
    }
  }
}
fwrite(rbindlist(summary_rows), file.path(OUT_DIR, "gblup_test_seed_overall_mean_sd.csv"))
writeLines(c("SUCCEEDED", "30 trait/model evaluations; 30 prediction files; 2250 prediction rows.",
             "Same 75 independent-Test fish evaluated by 15 models, for BW and LE."),
           file.path(OUT_DIR, "SUCCESS.txt"))

cat("\n============================================================\n")
cat("FINAL SUMMARY\n")
cat("============================================================\n")
print(all_df[group %in% c("seed_mean", "all_seed_mean")])

cat("\nCSV saved:\n", OUT_CSV, "\n")
cat("✅ done\n")
