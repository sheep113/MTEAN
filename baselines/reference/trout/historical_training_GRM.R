# Modified for METAN portable release 2026-11-09.
train_statistics_grm <- function(train, validation) {
  assert_ok(is.matrix(train) && is.matrix(validation) && ncol(train) == ncol(validation), "Dosage matrix dimensions differ")
  assert_ok(all(is.na(train) | train %in% 0:2) && all(is.na(validation) | validation %in% 0:2), "Expected original0/1/2 dosage/NA")
  mean_train <- colMeans(train, na.rm = TRUE)
  assert_ok(all(is.finite(mean_train)), "Training fold contains all-missing SNP: frequency cannot be estimated; stop without filtering or assuming a value")
  p <- mean_train / 2
  fill <- function(x) {
    miss <- which(is.na(x), arr.ind = TRUE)
    if (nrow(miss)) x[miss] <- mean_train[miss[, 2L]]
    sweep(x, 2L, mean_train, "-")
  }
  Zt <- fill(train); Zv <- fill(validation); denominator <- 2 * sum(p * (1 - p))
  assert_ok(is.finite(denominator) && denominator > 0, "VanRaden denominator is nonpositive")
  # ASRgenomics1.1.5 G.matrix defaults round the identical VanRaden formula to8digits.
  Z <- rbind(Zt, Zv); G <- round(tcrossprod(Z) / denominator, digits = 8L)
  diag(G) <- diag(G) + 0.01
  rownames(G) <- colnames(G) <- c(rownames(train), rownames(validation))
  nt <- nrow(train); nv <- nrow(validation)
  list(G = G, G_train = G[seq_len(nt), seq_len(nt), drop = FALSE],
       G_validation_train = G[nt + seq_len(nv), seq_len(nt), drop = FALSE],
       training_frequency = p, training_mean = mean_train, denominator = denominator)
}
