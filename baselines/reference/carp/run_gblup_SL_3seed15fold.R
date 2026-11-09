# Modified for METAN portable release 2026-11-09.
# Versioned adapter: historical VanRaden +0.01 and ASReml REML; training-only statistics.
# Historical calc_metrics() and fit_one_trait() are sourced verbatim from the adjacent core file.
assert_ok <- function(x, msg) if (!isTRUE(x)) stop(msg, call. = FALSE)
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
synthetic_check <- function() {
  # Matrix calculation only: never invokes fit_one_trait()/asreml(), never reads fish data.
  train <- matrix(c(0,1,2,0,1,2, 2,2,1,0,0,1, 0,0,1,1,2,2,
                    1,2,1,0,2,0, 2,1,0,2,0,1, 0,2,0,1,1,2), nrow = 6)
  train[2, 1] <- NA_real_; rownames(train) <- paste0("syn_train", 1:6); colnames(train) <- paste0("snp",1:6)
  validation <- rbind(c(0,1,2,0,1,2),c(2,0,1,2,1,0));rownames(validation) <- c("syn_val1","syn_val2");colnames(validation) <- colnames(train)
  one <- train_statistics_grm(train, validation)
  validation_changed <- validation;validation_changed[,] <- 2 - validation
  two <- train_statistics_grm(train, validation_changed)
  assert_ok(identical(one$training_frequency, two$training_frequency) && identical(one$G_train,two$G_train), "Validation must not influence fitted statistics/training GRM")
  assert_ok(requireNamespace("ASRgenomics", quietly = TRUE), "ASRgenomics missing")
  original <- ASRgenomics::G.matrix(M = train, method = "VanRaden", na.string = NA, message = FALSE)$G
  diag(original) <- diag(original) + .01
  err <- max(abs(one$G_train - original))
  assert_ok(is.finite(err) && err <= 1e-8, paste("Historical VanRaden formula mismatch",err))
  cat("SYNTHETIC_GRM_PASS: historical training G maximum absolute error",err,
      "; validation statistics isolation PASS; ASReml fits=0; real data reads=0\n")
}
extended_SL_metrics <- function(obs, pred) {
  base <- calc_metrics(obs,pred)
  c(unlist(base), RMSE=sqrt(base$MSE), MAE=mean(abs(obs-pred)),
    prediction_mean=mean(pred),true_mean=mean(obs),prediction_SD=sd(pred),true_SD=sd(obs),
    prediction_SD_ratio=sd(pred)/sd(obs))
}
summary_rows <- function(d, by_seed) {
  rows <- list(); groups <- if (by_seed) 1:3 else NA_integer_
  for (seed in groups) for (metric in c("Pearson","MSE","RMSE","MAE","R2","prediction_mean","true_mean","prediction_SD","true_SD","prediction_SD_ratio")) {
    x <- if (by_seed) d[d$seed==seed,metric] else d[[metric]]; good <- x[is.finite(x)]
    rows[[length(rows)+1L]] <- data.frame(model="GBLUP",trait="SL",seed=seed,metric=metric,
       n_total=length(x),n_valid=length(good),n_nan=length(x)-length(good),
       mean=if(length(good))mean(good)else NA_real_,SD=if(length(good)>1L)sd(good)else NA_real_,ddof=1L)
  }
  do.call(rbind,rows)
}
main <- function() {
  args <- commandArgs(trailingOnly=TRUE)
  script_arg <- grep("^--file=",commandArgs(),value=TRUE)
  assert_ok(length(script_arg)==1L,"Run with Rscript")
  here <- dirname(normalizePath(sub("^--file=","",script_arg)))
  source(file.path(here,"historical_GBLUP_core.R"),local=.GlobalEnv)
  if (identical(args,"--help")) {cat("Rscript run_gblup_SL_3seed15fold.R --config config.json --output UNIQUE_DIRECTORY --run-fits\nDefault never fits; --synthetic-check uses synthetic matrices only.\n");return(invisible(NULL))}
  if (identical(args,"--synthetic-check")) {synthetic_check();return(invisible(NULL))}
  assert_ok(length(args)==5L && args[1L]=="--config" && args[3L]=="--output" && args[5L] %in% c("--run-fits","--check-inputs"), "Explicit --config PATH --output UNIQUE_DIRECTORY --run-fits required")
  assert_ok(requireNamespace("jsonlite",quietly=TRUE),"jsonlite missing")
  cfg <- jsonlite::fromJSON(args[2L],simplifyVector=FALSE)
  assert_ok(identical(cfg$trait,"SL") && identical(cfg$preprocessing_scope,"training_fold_only"),"Unconfirmed trait or preprocessing scope")
  assert_ok(identical(as.integer(cfg$total_n),500L) && identical(as.integer(cfg$train_n),400L) && identical(as.integer(cfg$validation_n),100L),"Expected all500 fixed400/100 protocol")
  for (p in names(cfg$input_SHA256)) {
    assert_ok(file.exists(p),paste("Missing input",p))
    got <- system2("sha256sum",c("--",shQuote(p)),stdout=TRUE)
    assert_ok(length(got)==1L && identical(strsplit(got,"[[:space:]]+")[[1L]][1L],cfg$input_SHA256[[p]]),paste("Input changed",p))
  }
  for (p in names(cfg$package_versions)) assert_ok(requireNamespace(p,quietly=TRUE) &&
    packageVersion(p)==package_version(cfg$package_versions[[p]]),paste("Historical package missing/version differs",p))
  assert_ok(getRversion()==package_version(cfg$R_version),"Historical R version differs")
  library(data.table);setDTthreads(1L)
  gd <- fread(cfg$genotype_csv,header=TRUE,sep=",",data.table=FALSE,check.names=FALSE,na.strings=c("NA","NaN"),colClasses=list(character="sample_id"))
  geno <- as.matrix(gd[-1L]);storage.mode(geno)<-"numeric";rownames(geno)<-gd$sample_id
  phe <- fread(cfg$phenotype_csv,data.table=FALSE,colClasses=list(character="sample_id"))
  ids <- readLines(cfg$development_IDs,warn=FALSE)
  alleles <- fread(cfg$snp_alleles,data.table=FALSE,colClasses="character")
  assert_ok(identical(dim(geno),c(500L,15615L)) && identical(rownames(geno),ids) && !anyDuplicated(ids) &&
    identical(phe$sample_id,ids) && identical(names(phe),c("sample_id","SL")) && identical(colnames(geno),alleles$snp_id),"500/SL/SNP/sample order mismatch")
  assert_ok(all(is.finite(phe$SL)) && all(is.na(geno)|geno %in% 0:2),"Bad labels or dosage")
  cvs <- lapply(cfg$CV_files,function(p)fread(p,data.table=FALSE,colClasses=list(character="sample_id")))
  for (seed in 1:3) {
    cv<-cvs[[as.character(seed)]];coverage<-character()
    assert_ok(identical(sort(unique(cv$fold)),0:4),"Missing fixed folds")
    for (fold in 0:4) {
      tr<-cv$sample_id[cv$fold==fold & cv$split=="train"];va<-cv$sample_id[cv$fold==fold & cv$split=="val"]
      assert_ok(length(tr)==400L && length(va)==100L && !anyDuplicated(tr) && !anyDuplicated(va) && !length(intersect(tr,va)) && setequal(c(tr,va),ids),"Fixed400/100CV mismatch")
      coverage<-c(coverage,va)
    }
    assert_ok(length(coverage)==500L && !anyDuplicated(coverage) && setequal(coverage,ids),"CV validation coverage mismatch")
  }
  if (identical(args[5L],"--check-inputs")) {cat("REAL_INPUT_CHECK_PASS: 500 unique IDs; 15615 ordered SNPs; SL labels aligned; all15 fixed400/100 folds; fits=0; predictions=0\n");return(invisible(NULL))}
  # Libraries loaded only after validation and explicit manual --run-fits.
  library(asreml);library(ASRgenomics)
  out<-args[4L];assert_ok(!file.exists(out),"Refusing to overwrite existing GBLUP output")
  dir.create(out,recursive=TRUE);out<-normalizePath(out);for(d in c("tasks","predictions","tmp"))dir.create(file.path(out,d))
  manifest<-list(status="RUNNING",model="GBLUP",trait="SL",expected=15L,completed=0L,
    seeds=as.list(1:3),folds=as.list(0:4),train_n=400L,validation_n=100L,holdout_access=FALSE,
    original_Test_access=FALSE,GRM="historical VanRaden rounded8digits +diag0.01; ASRgenomics G.inverse sparse",
    statistic_scope="training400 only; validation response masked throughout ASReml; raw SL",
    variance_estimation="historical ASReml REML start.values then G.param/R.param",ensemble=FALSE,OOF=FALSE,ddof=1L)
  save_manifest<-function()jsonlite::write_json(manifest,file.path(out,"run_manifest.json"),auto_unbox=TRUE,pretty=TRUE,na="null")
  save_manifest();on.exit({if(!identical(manifest$status,"SUCCEEDED")){manifest$status<-"FAILED_OR_INTERRUPTED";save_manifest()}},add=TRUE)
  oldwd<-getwd();setwd(file.path(out,"tmp"));on.exit(setwd(oldwd),add=TRUE)
  capture.output(sessionInfo(),file=file.path(out,"sessionInfo.txt"));file.copy(args[2L],file.path(out,"run_config.json"))
  rows<-list()
  for (seed in 1:3) for (fold in 0:4) {
    manifest$current<-list(seed=seed,fold=fold);save_manifest();cat("GBLUP Seed",seed,"Fold",fold,"Completed",manifest$completed,"/15\n");flush.console()
    cv<-cvs[[as.character(seed)]];tr<-cv$sample_id[cv$fold==fold & cv$split=="train"];va<-cv$sample_id[cv$fold==fold & cv$split=="val"]
    task<-file.path(out,"tasks",paste0("seed",seed,"_fold",fold));dir.create(task)
    relationship<-train_statistics_grm(geno[tr,,drop=FALSE],geno[va,,drop=FALSE]);G<-relationship$G
    assert_ok(all(is.finite(G)),"Non-finite GRM")
    # Exactly the successful historical sparse inverse and ASReml fitting core.
    ginv<-G.inverse(G,sparseform=TRUE)$Ginv.sparse
    dd<-data.frame(ID=factor(c(tr,va)),SL=c(phe$SL[match(tr,phe$sample_id)],rep(NA_real_,100L)))
    assert_ok(all(is.na(dd$SL[match(va,as.character(dd$ID))])),"Validation labels not masked")
    truth_after_fit<-function(){data.frame(ID=va,SL=phe$SL[match(va,phe$sample_id)])}
    t0<-proc.time()[[3L]];fit<-fit_one_trait(dd,truth_after_fit(),va,"SL",ginv)
    assert_ok(is.logical(fit$model$converge) && length(fit$model$converge)==1L && isTRUE(fit$model$converge),"ASReml convergence unconfirmed/failed; stop without retry")
    valid<-fit$valid[match(va,as.character(fit$valid$ID)),,drop=FALSE]
    saveRDS(list(validation=valid,expected_IDs=va,expected_SL=phe$SL[match(va,phe$sample_id)],model_converged=fit$model$converge),file.path(task,"validation_before_acceptance.rds"))
    assert_ok(nrow(valid)==100L && identical(as.character(valid$ID),va) && all(is.finite(valid$Prediction)) && identical(as.numeric(valid$Observed),as.numeric(phe$SL[match(va,phe$sample_id)])),"Validation prediction/ID mismatch")
    mm<-extended_SL_metrics(valid$Observed,valid$Prediction)
    pr<-data.frame(Seed=seed,Fold=fold,sample_id=va,true_SL=valid$Observed,pred_SL=valid$Prediction,model="GBLUP",trait="SL")
    fwrite(pr,file.path(out,"predictions",paste0("seed",seed,"_fold",fold,".csv")),na="NaN")
    row<-data.frame(model="GBLUP",trait="SL",seed=seed,fold=fold,train_n=400L,validation_n=100L,as.list(mm),fit_seconds=proc.time()[[3L]]-t0,check.names=FALSE)
    fwrite(row,file.path(task,"metrics.csv"),na="NaN");saveRDS(list(training_IDs=tr,validation_IDs=va,training_mean=relationship$training_mean,training_frequency=relationship$training_frequency,denominator=relationship$denominator,SNP_IDs=colnames(geno)),file.path(task,"preprocessing.rds"))
    fwrite(as.data.frame(summary(fit$model)$varcomp),file.path(task,"variance_components.csv"));rows[[length(rows)+1L]]<-row
    manifest$completed<-length(rows);save_manifest();rm(fit,relationship,G,ginv);gc()
  }
  details<-do.call(rbind,rows);assert_ok(nrow(details)==15L,"Incomplete CV; no full summary")
  fwrite(details,file.path(out,"fold_metrics.csv"),na="NaN");fwrite(summary_rows(details,TRUE),file.path(out,"seed_summary.csv"),na="NaN");fwrite(summary_rows(details,FALSE),file.path(out,"overall_15fold_summary.csv"),na="NaN")
  writeLines(c("# 鲤鱼SL单性状GBLUP固定十五折", "15/15完成；400训练/100验证，仅SL。VanRaden统计量只从训练折估计，ASReml REML沿用历史算法；未使用独立留出评价或原Test。", "结果为逐折指标Mean±SD(ddof=1)，不是独立生物学重复，无OOF或集成。"),file.path(out,"中文GBLUP结果报告.md"))
  manifest$status<-"SUCCEEDED";manifest$current<-NULL;save_manifest();cat("GBLUP15/15 SUCCEEDED:",out,"\n")
}
main()
