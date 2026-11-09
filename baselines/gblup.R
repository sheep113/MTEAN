# Modified METAN portable wrapper. ASReml model and VanRaden conventions retained.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1L)
library(jsonlite)
library(data.table)
library(asreml)
library(ASRgenomics)
cfg <- fromJSON(args[1])
folder <- cfg$directory
read_geno <- function(name) {
  d <- fread(file.path(folder,name),data.table=FALSE,check.names=FALSE,colClasses=list(character='sample_id'))
  x <- as.matrix(d[-1]); storage.mode(x)<-'numeric';rownames(x)<-d$sample_id
  x
}
train <- read_geno('train_dosage.csv')
evaluation <- read_geno('eval_dosage.csv')
stopifnot(identical(colnames(train),colnames(evaluation)),!anyDuplicated(c(rownames(train),rownames(evaluation))))
ids <- c(rownames(train),rownames(evaluation))
settings <- cfg$settings
if (settings$centering=='joint_genotypes') {
  x <- rbind(train,evaluation);storage.mode(x)<-'integer'
  G <- G.matrix(M=x,method='VanRaden',na.string=NA)$G
} else if (settings$centering=='training') {
  mean_train <- colMeans(train,na.rm=TRUE)
  stopifnot(all(is.finite(mean_train)))
  fill <- function(x) {
    idx <- which(is.na(x),arr.ind=TRUE)
    if(nrow(idx))x[idx]<-mean_train[idx[,2]]
    sweep(x,2,mean_train,'-')
  }
  p <- mean_train/2;denom<-2*sum(p*(1-p));stopifnot(denom>0)
  G <- round(tcrossprod(rbind(fill(train),fill(evaluation)))/denom,digits=settings$round_digits)
} else stop('Unknown GRM centering convention')
diag(G)<-diag(G)+settings$diagonal_offset
rownames(G)<-colnames(G)<-ids
ginv<-G.inverse(G,sparseform=TRUE)$Ginv.sparse
pheno<-fread(file.path(folder,'train_phenotype.csv'),data.table=FALSE,colClasses=list(character='sample_id'))
stopifnot(identical(pheno$sample_id,rownames(train)),all(is.finite(pheno$phenotype)))
dd<-data.frame(ID=factor(ids),phenotype=c(pheno$phenotype,rep(NA_real_,nrow(evaluation))))
init<-asreml(fixed=phenotype~1,random=~vm(ID,ginv),dense=~vm(ID,ginv),residual=~idv(units),start.values=TRUE,data=dd)
vc<-init$vparameters.table
fit<-asreml(fixed=phenotype~1,random=~vm(ID,ginv),dense=~vm(ID,ginv),residual=~idv(units),G.param=vc,R.param=vc,data=dd)
if(!isTRUE(fit$converge))stop('ASReml fit did not converge; this task is incomplete')
fixed<-coef(fit)$fixed
mu<-if(is.matrix(fixed)||is.data.frame(fixed))as.numeric(fixed[1,1])else as.numeric(fixed[1])
random<-coef(fit)$random
if(is.vector(random))random<-matrix(random,ncol=1,dimnames=list(names(random),'effect'))
random<-as.data.frame(random)
sid<-sub('^vm\\(ID, ginv\\)_','',rownames(random))
stopifnot(!anyDuplicated(sid),all(rownames(evaluation)%in%sid))
prediction<-mu+as.numeric(random[match(rownames(evaluation),sid),1])
stopifnot(all(is.finite(prediction)))
fwrite(data.frame(sample_id=rownames(evaluation),prediction=prediction),file.path(folder,'fitted_predictions.csv'))
saveRDS(fit,file.path(folder,'model.rds'))
