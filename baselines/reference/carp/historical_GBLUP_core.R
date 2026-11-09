# Modified for METAN portable release 2026-11-09.
# GBLUP500: historical ASReml core, explicit manual --run-fits only.
calc_metrics <- function(obs, pred) {
  ok <- complete.cases(obs, pred)
  obs <- obs[ok]
  pred <- pred[ok]

  if (length(obs) == 0) {
    return(list(Pearson = NA_real_, MSE = NA_real_, R2 = NA_real_))
  }

  pearson <- suppressWarnings(cor(obs, pred))
  mse <- mean((obs - pred)^2)
  sse <- sum((obs - pred)^2)
  sst <- sum((obs - mean(obs))^2)
  r2 <- ifelse(sst == 0, NA_real_, 1 - sse / sst)

  list(Pearson = pearson, MSE = mse, R2 = r2)
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

