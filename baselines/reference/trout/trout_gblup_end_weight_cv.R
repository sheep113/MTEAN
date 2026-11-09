# Modified for METAN portable release 2026-11-09.
# Isolated adapter. historical_GBLUP_core.R and historical GRM are copied unchanged.
assert_ok <- function(x,msg) if(!isTRUE(x)) stop(msg,call.=FALSE)
gb_hash <- function(path) {
 assert_ok(file.exists(path),paste('Missing file',path))
 z<-system2('sha256sum',c('--',shQuote(path)),stdout=TRUE)
 assert_ok(length(z)==1L && is.null(attr(z,'status')),paste('SHA256 failed',path))
 strsplit(z,'[[:space:]]+')[[1L]][1L]
}
gb_json <- function(x,path) jsonlite::write_json(x,path,auto_unbox=TRUE,pretty=TRUE,na='null',null='null',digits=17)
gb_dependencies <- function(cfg,fit=FALSE) {
 assert_ok(getRversion()==package_version(cfg$R_version),'R version differs from historical protocol')
 versions<-cfg$package_versions
 for(p in names(versions)) {
  if(!fit && p %in% c('asreml','ASRgenomics','AGHmatrix'))next
  assert_ok(requireNamespace(p,quietly=TRUE) && packageVersion(p)==package_version(versions[[p]]),paste('Missing/historical version differs:',p))
 }
}
gb_validate <- function(cfg) {
 assert_ok(identical(cfg$model,'GBLUP') && identical(cfg$trait,'End_Weight') && identical(cfg$species,'Trout500'),'Only Trout500 End_Weight supported')
 assert_ok(identical(cfg$preprocessing_scope,'training_fold_only'),'Unconfirmed preprocessing scope')
 syn<-identical(cfg$scope,'SYNTHETIC_TEST')
 if(!syn) assert_ok(identical(as.integer(c(cfg$total_n,cfg$train_n,cfg$validation_n,cfg$n_snps)),c(500L,400L,100L,26068L)),'Only verified500 /400/100 /26068 real protocol; old425 rejected')
 assert_ok(identical(unlist(cfg$seeds,use.names=FALSE),c(1L,2L,3L)) && identical(unlist(cfg$folds,use.names=FALSE),0:4),'Seed/Fold definition differs')
 assert_ok(length(cfg$input_SHA256)>0,'Missing frozen input hashes')
 cvpaths<-unlist(cfg$CV_files,use.names=FALSE)
 assert_ok(length(cvpaths)==3L && !anyDuplicated(normalizePath(cvpaths,mustWork=TRUE)) && !anyDuplicated(vapply(cvpaths,gb_hash,character(1))),'Each Seed requires its own distinct verified CV; duplicated Seed1 mapping rejected')
 for(p in names(cfg$input_SHA256))assert_ok(identical(gb_hash(p),cfg$input_SHA256[[p]]),paste('Input changed',p))
 assert_ok(requireNamespace('data.table',quietly=TRUE),'data.table missing');data.table::setDTthreads(1L)
 ids<-readLines(cfg$sample_ids,warn=FALSE);assert_ok(length(ids)==cfg$total_n && !anyNA(ids) && all(nzchar(ids)) && !anyDuplicated(ids),'Sample ID count/uniqueness')
 if(syn)assert_ok(all(grepl('^SYN_',ids)),'Synthetic mode refuses fish sample IDs')
 gd<-data.table::fread(cfg$genotype_csv,header=TRUE,sep=',',data.table=FALSE,check.names=FALSE,na.strings=c('NA','NaN'),colClasses=list(character=cfg$id_column))
 assert_ok(identical(names(gd)[1L],cfg$id_column),'Genotype ID column mismatch')
 gid<-gd[[cfg$id_column]];geno<-as.matrix(gd[-1L]);storage.mode(geno)<-'numeric';rownames(geno)<-gid
 assert_ok(!anyDuplicated(gid) && identical(gid,ids) && identical(as.integer(dim(geno)),as.integer(c(cfg$total_n,cfg$n_snps))) && !anyDuplicated(colnames(geno)),'Genotype ID/order/SNP dimensions differ')
 assert_ok(all(is.na(geno)|geno %in% 0:2),'Dosage must be0/1/2/NA; raw5 is invalid, not an allele')
 ph<-data.table::fread(cfg$phenotype_csv,header=TRUE,sep=',',data.table=FALSE,colClasses=list(character=cfg$id_column))
 assert_ok(identical(names(ph),c(cfg$id_column,cfg$trait)) && identical(ph[[cfg$id_column]],ids) && !anyDuplicated(ph[[cfg$id_column]]),'Phenotype target/ID/order mismatch')
 assert_ok(all(is.finite(ph[[cfg$trait]])),'Missing/nonfinite target labels')
 snps<-data.table::fread(cfg$snp_order_csv,header=TRUE,sep=',',data.table=FALSE,colClasses='character')
 assert_ok(identical(colnames(geno),snps$snp_id),'SNP order differs')
 cvs<-list()
 for(seed in 1:3) {
  path<-cfg$CV_files[[as.character(seed)]];assert_ok(!is.null(path),'Missing specified CV file')
  cv<-data.table::fread(path,header=TRUE,data.table=FALSE,colClasses=list(character='sample_id',integer='fold'))
  assert_ok(identical(names(cv),c('fold','sample_id','split')) && identical(sort(unique(cv$fold)),0:4) && setequal(unique(cv$split),c('train','val')) && !anyNA(cv),'Malformed fixed CV')
  coverage<-character()
  for(fold in 0:4) {
   tr<-cv$sample_id[cv$fold==fold & cv$split=='train'];va<-cv$sample_id[cv$fold==fold & cv$split=='val']
   assert_ok(length(tr)==cfg$train_n && length(va)==cfg$validation_n && !anyDuplicated(tr) && !anyDuplicated(va) && !length(intersect(tr,va)) && setequal(c(tr,va),ids),'Fixed training/validation IDs invalid; no fallback')
   coverage<-c(coverage,va)
  }
  assert_ok(length(coverage)==cfg$total_n && !anyDuplicated(coverage) && setequal(coverage,ids),'Validation coverage invalid')
  cvs[[as.character(seed)]]<-cv
 }
 list(geno=geno,y=as.numeric(ph[[cfg$trait]]),ids=ids,cvs=cvs,synthetic=syn)
}
gb_metrics <- function(obs,pred) {
 assert_ok(length(obs)==length(pred) && all(is.finite(obs)) && all(is.finite(pred)),'Nonfinite prediction/label')
 base<-calc_metrics(obs,pred)
 c(unlist(base),RMSE=sqrt(base$MSE),MAE=mean(abs(obs-pred)),prediction_mean=mean(pred),true_mean=mean(obs),prediction_SD=sd(pred),true_SD=sd(obs),prediction_SD_ratio=if(sd(obs)>0)sd(pred)/sd(obs)else NA_real_)
}
gb_fit <- function(train,validation,train_y,truth_after_fit,trait='End_Weight') {
 assert_ok(length(train_y)==nrow(train) && all(is.finite(train_y)),'Training phenotype invalid')
 assert_ok(!anyDuplicated(rownames(train)) && !anyDuplicated(rownames(validation)) && !length(intersect(rownames(train),rownames(validation))),'Overlap/duplicate fit IDs')
 build_start<-proc.time()[[3L]];relationship<-train_statistics_grm(train,validation)
 # Qualified function is the same historical G.inverse(sparseform=TRUE), defaults unchanged.
 ginv<-ASRgenomics::G.inverse(relationship$G,sparseform=TRUE)$Ginv.sparse
 build_seconds<-proc.time()[[3L]]-build_start;tr<-rownames(train);va<-rownames(validation)
 dd<-data.frame(ID=factor(c(tr,va)));dd[[trait]]<-c(train_y,rep(NA_real_,length(va)))
 assert_ok(all(is.na(dd[[trait]][match(va,as.character(dd$ID))])),'Validation response must be masked')
 before<-proc.time()[[3L]]
 # R lazy promise: truth_after_fit is first evaluated inside the unchanged core AFTER REML and coefficient extraction.
 fit<-fit_one_trait(dd,truth_after_fit,va,trait,ginv)
 total_seconds<-proc.time()[[3L]]-before
 assert_ok(isTRUE(fit$model$converge),'ASReml REML did not converge; no automatic retry or changed parameters')
 valid<-fit$valid[match(va,as.character(fit$valid$ID)),,drop=FALSE]
 assert_ok(nrow(valid)==length(va) && !anyDuplicated(valid$ID) && identical(as.character(valid$ID),va) && all(is.finite(valid$Prediction)),'Prediction/ID mismatch')
 list(fit=fit,valid=valid,relationship=relationship,GRM_seconds=build_seconds,fit_and_coefficient_prediction_seconds=total_seconds)
}
gb_summary <- function(details) {
 rows<-list()
 for(seed in c(1:3,0L)) {
  d<-if(seed==0L)details else details[details$Seed==seed,,drop=FALSE]
  if(!nrow(d))next
  for(metric in c('Pearson','MSE','RMSE','MAE','R2','prediction_mean','true_mean','prediction_SD','true_SD','prediction_SD_ratio')) {
   x<-d[[metric]];ok<-is.finite(x);rows[[length(rows)+1L]]<-data.frame(Model='GBLUP',Trait='End_Weight',Seed=seed,Metric=metric,Mean=if(any(ok))mean(x[ok])else NA_real_,SD=if(sum(ok)>1L)sd(x[ok])else NA_real_,ddof=1L,n_total=length(x),n_valid=sum(ok),n_nan=sum(!ok))
  }
 }
 do.call(rbind,rows)
}
run_gblup_end_weight <- function(config,output=NULL,seed=NULL,fold=NULL,run_fits=FALSE) {
 assert_ok(requireNamespace('jsonlite',quietly=TRUE),'jsonlite missing')
 config<-normalizePath(config,mustWork=TRUE);cfg<-jsonlite::fromJSON(config,simplifyVector=FALSE)
 here<-dirname(config);source(file.path(here,'historical_GBLUP_core.R'),local=.GlobalEnv);source(file.path(here,'historical_training_GRM.R'),local=.GlobalEnv)
 gb_dependencies(cfg,fit=run_fits);data<-gb_validate(cfg)
 assert_ok(is.null(seed)==is.null(fold),'Specify both Seed and Fold, or neither')
 assert_ok(is.null(seed) || (length(seed)==1L && seed %in% 1:3 && length(fold)==1L && fold %in% 0:4),'Invalid Seed/Fold')
 if(!run_fits) {cat('INPUT_CHECK_PASS:',cfg$total_n,'samples;',cfg$n_snps,'SNPs;15 fixed folds; fits=0,predictions=0\n');return(invisible(list(status='INPUTS_VERIFIED_NOT_RUN',folds=15L)))}
 assert_ok(!is.null(output) && !file.exists(output),'Unique new output path required; refusing overwrite')
 library(asreml);library(ASRgenomics)
 dir.create(output,recursive=TRUE);out<-normalizePath(output,mustWork=TRUE)
 for(n in c('tasks','predictions','tmp'))dir.create(file.path(out,n))
 jobs<-if(is.null(seed))expand.grid(Fold=0:4,Seed=1:3) else data.frame(Fold=as.integer(fold),Seed=as.integer(seed))
 state<-list(status='RUNNING',scope=cfg$scope,model='GBLUP',trait='End_Weight',expected=nrow(jobs),completed=0L,current=NULL,config=config,config_SHA256=gb_hash(config),CV_files=cfg$CV_files,started=format(Sys.time(),'%FT%T%z'),real_fits=0L,synthetic_fits=0L)
 save_state<-function()gb_json(state,file.path(out,'run_status.json'))
 save_state();oldwd<-getwd();setwd(file.path(out,'tmp'))
 logcon<-file(file.path(out,'run.log'),'wt');sink(logcon,split=TRUE)
 on.exit({if(state$status=='RUNNING'){state$status<-'FAILED_OR_INTERRUPTED';save_state()};sink();close(logcon);setwd(oldwd)},add=TRUE)
 file.copy(config,file.path(out,'run_config.json'),overwrite=FALSE);capture.output(sessionInfo(),file=file.path(out,'sessionInfo.txt'))
 rows<-list()
 tryCatch({
  for(j in seq_len(nrow(jobs))) {
   s<-jobs$Seed[j];f<-jobs$Fold[j];state$current<-list(seed=s,fold=f);save_state();cat('GBLUP Seed',s,'Fold',f,'Completed',state$completed,'/',state$expected,'\n');flush.console()
   cv<-data$cvs[[as.character(s)]];tr<-cv$sample_id[cv$fold==f & cv$split=='train'];va<-cv$sample_id[cv$fold==f & cv$split=='val']
   task<-file.path(out,'tasks',paste0('seed',s,'_fold',f));dir.create(task)
   writeLines(tr,file.path(task,'training_IDs.txt'));writeLines(va,file.path(task,'validation_IDs.txt'))
   fit<-gb_fit(data$geno[tr,,drop=FALSE],data$geno[va,,drop=FALSE],data$y[match(tr,data$ids)],data.frame(ID=va,End_Weight=data$y[match(va,data$ids)]))
   valid<-fit$valid;assert_ok(identical(as.numeric(valid$Observed),as.numeric(data$y[match(va,data$ids)])),'Truth mismatch')
   mm<-gb_metrics(valid$Observed,valid$Prediction)
   pp<-data.frame(Sample_ID=va,True_End_Weight=as.numeric(valid$Observed),Pred_End_Weight=as.numeric(valid$Prediction),Seed=s,Fold=f,Model='GBLUP')
   path<-file.path(out,'predictions',paste0('seed',s,'_fold',f,'.csv'));data.table::fwrite(pp,path,na='NaN')
   reread<-data.table::fread(path,data.table=FALSE,colClasses=list(character='Sample_ID'))
   assert_ok(identical(reread$Sample_ID,va) && nrow(reread)==cfg$validation_n,'Saved prediction ID mismatch')
   back<-gb_metrics(reread$True_End_Weight,reread$Pred_End_Weight)
   assert_ok(isTRUE(all.equal(mm,back,tolerance=1e-11,check.attributes=TRUE)),'Saved metrics independently recomputed differ')
   trc<-fit$fit$model$trace;trace_rows<-if(!is.null(trc))nrow(as.matrix(trc)) else NA_integer_
   diag<-list(converged=fit$fit$model$converge,trace_rows=trace_rows,trace=trc,model_iteration_field=fit$fit$model$iter,GRM_method=cfg$GRM,variance_estimation=cfg$variance_estimation,model_parameters=cfg$algorithm_parameters,GRM_seconds=fit$GRM_seconds,fit_and_prediction_seconds=fit$fit_and_coefficient_prediction_seconds,prediction_time_scope='Historical core combines REML, coefficient extraction and mu+GEBV; combined duration reported, separate prediction duration unavailable without changing core',training_IDs_SHA256=gb_hash(file.path(task,'training_IDs.txt')),validation_IDs_SHA256=gb_hash(file.path(task,'validation_IDs.txt')),CV_file=cfg$CV_files[[as.character(s)]],CV_SHA256=gb_hash(cfg$CV_files[[as.character(s)]]),prediction_SHA256=gb_hash(path))
   gb_json(diag,file.path(task,'diagnostics.json'));saveRDS(trc,file.path(task,'iteration_trace.rds'))
   vc<-as.data.frame(summary(fit$fit$model)$varcomp);vc$component<-rownames(vc);data.table::fwrite(vc,file.path(task,'variance_components.csv'))
   saveRDS(list(training_mean=fit$relationship$training_mean,training_frequency=fit$relationship$training_frequency,denominator=fit$relationship$denominator,SNP_order=colnames(data$geno),training_IDs=tr,validation_IDs=va),file.path(task,'preprocessing.rds'))
   row<-data.frame(Model='GBLUP',Trait='End_Weight',Seed=s,Fold=f,train_n=length(tr),validation_n=length(va),as.list(mm),GRM_seconds=fit$GRM_seconds,fit_and_prediction_seconds=fit$fit_and_coefficient_prediction_seconds,converged=TRUE,trace_rows=trace_rows,check.names=FALSE)
   row$Pearson_NA_reason<-if(is.finite(mm[['Pearson']]))'' else 'constant prediction or target; Pearson undefined'
   data.table::fwrite(row,file.path(task,'metrics.csv'),na='NaN');rows[[length(rows)+1L]]<-row;state$completed<-length(rows);if(data$synthetic)state$synthetic_fits<-state$completed else state$real_fits<-state$completed;save_state();rm(fit);gc()
  }
  details<-do.call(rbind,rows);assert_ok(nrow(details)==nrow(jobs),'Incomplete requested jobs')
  data.table::fwrite(details,file.path(out,'fold_metrics.csv'),na='NaN')
  summaries<-gb_summary(details);data.table::fwrite(summaries,file.path(out,'requested_tasks_summary.csv'),na='NaN')
  if(nrow(details)==15L){data.table::fwrite(summaries[summaries$Seed>0,,drop=FALSE],file.path(out,'seed_summary.csv'),na='NaN');data.table::fwrite(summaries[summaries$Seed==0,,drop=FALSE],file.path(out,'overall_15fold_summary.csv'),na='NaN')}
  state$status<-'SUCCEEDED';state$current<-NULL;state$finished<-format(Sys.time(),'%FT%T%z');save_state();gb_json(state,file.path(out,'final_status.json'));cat('REQUESTED_GBLUP_TASKS_SUCCEEDED',state$completed,'\n')
 },error=function(e){state$status<-'FAILED';state$error<-conditionMessage(e);state$finished<-format(Sys.time(),'%FT%T%z');save_state();gb_json(state,file.path(out,'final_status.json'));stop(e)})
 invisible(state)
}
gb_main <- function(args=commandArgs(trailingOnly=TRUE)) {
 if(!length(args) || '--help' %in% args){cat('Rscript trout_gblup_end_weight_cv.R --config CONFIG [--check-inputs | --output UNIQUE --run-fits [--seed 1 --fold 0]]\nNo fits by default. In R Console: source(script); run_gblup_end_weight(config, output, seed=1, fold=0, run_fits=TRUE).\n');return(invisible(NULL))}
 allowed<-c('--config','--output','--seed','--fold','--run-fits','--check-inputs');a<-list();i<-1L
 while(i<=length(args)){key<-args[i];assert_ok(key %in% allowed && is.null(a[[key]]),paste('Unknown/duplicate argument',key));if(key %in% c('--run-fits','--check-inputs')){a[[key]]<-TRUE;i<-i+1L}else{assert_ok(i<length(args),'Missing argument value');a[[key]]<-args[i+1L];i<-i+2L}}
 assert_ok(!is.null(a[['--config']]),'--config required');assert_ok(!(isTRUE(a[['--run-fits']]) && isTRUE(a[['--check-inputs']])),'Choose check or fit')
 run_gblup_end_weight(a[['--config']],a[['--output']],if(!is.null(a[['--seed']]))as.integer(a[['--seed']]) else NULL,if(!is.null(a[['--fold']]))as.integer(a[['--fold']])else NULL,isTRUE(a[['--run-fits']]))
}
if(sys.nframe()==0L)gb_main()
