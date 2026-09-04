# =======================================================
# 1. 设置工作目录和文件名 (已适配到你的数据)
# =======================================================
setwd("/mnt/HTT_3T/github orign code/Whisper_of_DNA_pl")
#plink_prefix <- "data/runs/maize1404_analysis.final_geno"  # PLINK文件前缀
#pheno_file   <- "data/runs/maize1404_analysis.matched_pheno.tsv"  # 表型文件（TSV格式）
#trait_name   <- "KWPE"              # 你要分析的性状列名 (如 DTA, PH, EH 等)
#output_dir   <- "results/gwas"     # 输出目录 (图和结果都放这里)
#use_pca      <- FALSE               # 是否使用PCA作为协变量 (TRUE/FALSE)

plink_prefix <- "data/maize1404_genotype/cubic_1404_qc_ldpruned_traintag"  # PLINK文件前缀_old 
pheno_file   <- "data/maize1404_genotype/maize1404_analysis.matched_pheno.tsv"  # 表型文件（TSV格式）
trait_name   <- "DTT"              # "DTA", "DTS", "DTT"   
output_dir   <- "logs0/DNAWhisperfinetune/run0_blockembedding_f3p_30k"     # 输出目录 (图和结果都放这里)
use_pca      <- TRUE                # 是否使用PCA作为协变量 (TRUE/FALSE)
# =======================================================


library(rMVP)

# =======================================================
# 2. 转换 PLINK 数据 (只需跑一次)
# =======================================================
# 创建输出目录
dir.create(output_dir, showWarnings=FALSE, recursive=TRUE)

# rMVP 会把 PLINK 转成高效格式，生成 .geno.desc 等文件
# 改变工作目录到输出目录，避免路径问题
old_wd <- getwd()
setwd(output_dir)

mvp_prefix <- paste0("mvp.", basename(plink_prefix))
MVP.Data(fileBed=file.path(old_wd, plink_prefix),
         out=mvp_prefix,
         priority="speed")

# 返回原工作目录
setwd(old_wd)

# =======================================================
# 3. 读取并清洗表型数据 (关键步骤)
# =======================================================
# 读取表型 (TSV格式，处理 NaN, ., -9 等常见缺失标记)
raw_pheno <- read.table(pheno_file, header=TRUE, sep="\t", 
                        na.strings=c("NaN", "NA", ".", "-9", ""), 
                        stringsAsFactors=FALSE)

# 读取刚刚转换好的基因型"点名册" (.fam 信息的替代品)
# map文件第二列通常是 SNP ID，但我们需要样本 ID。
# rMVP 转换后的 .geno.ind 文件存储了样本顺序
ind_file <- file.path(output_dir, paste0(mvp_prefix, ".geno.ind"))
if(!file.exists(ind_file)) stop("Error: 基因型转换失败，找不到 .ind 文件:", ind_file)
geno_ind <- read.table(ind_file, header=FALSE)
target_IDs <- as.character(geno_ind$V1) # 基因型文件里的样本顺序

# === 自动对齐 (Match) ===
# 创建符合 rMVP 要求的表型表：第一列必须是 Taxa，顺序必须和基因型完全一致
mvp_pheno <- data.frame(Taxa = target_IDs)

# 如果表型文件里有这个样，就填入数值；如果没有，就自动填 NA
if(trait_name %in% colnames(raw_pheno)){
  # 假设表型文件第一列是样本名，如果不是，请把 raw_pheno[,1] 改成 raw_pheno$SampleID
  mvp_pheno[[trait_name]] <- raw_pheno[[trait_name]][match(target_IDs, raw_pheno[,1])]
} else {
  stop(paste("错误: 在表型文件中找不到列名", trait_name))
}

print(paste("对齐完成。样本总数:", nrow(mvp_pheno), 
            "有效表型数:", sum(!is.na(mvp_pheno[[trait_name]]))))

# =======================================================
# 4. 运行 GWAS (混合线性模型 MLM)
# =======================================================
genotype <- attach.big.matrix(file.path(output_dir, paste0(mvp_prefix, ".geno.desc")))
map <- read.table(file.path(output_dir, paste0(mvp_prefix, ".geno.map")), header = TRUE)

imMVP <- MVP(
    phe=mvp_pheno,
    geno=genotype,
    map=map,
    method=c("MLM"),                 # 混合线性模型 (MLM) 作为 Baseline
    nPC.MLM=ifelse(use_pca, 3, 0),    # 0=不用PCA, 3=使用前3个PCs
    file.output=TRUE,                # 保留MVP默认输出（所有报告图表）
    outpath=output_dir,              # 输出目录
    threshold=0.05,                  # Suggestive threshold
    ncpus=4                          # 使用4个CPU核心加速
)

# =======================================================
# 额外生成发表级别高质量曼哈顿图（适合拼图排版）
# =======================================================
# 1. 矩形曼哈顿图 (7x6英寸 - 适合两张并排拼图)
MVP.Report(imMVP,
           plot.type="m",
           LOG10=TRUE,
           threshold=1.0e-5,
           outpath=output_dir,
           
           # 发表级别尺寸和质量
           height=5, width=10,              # 7x6英寸（两张并排=14英寸总宽）
           dpi=400,                        # 600 DPI
           file.type="jpg",
           
           # 文字清晰度
           cex=1.0,                        # 点大小
           cex.axis=1.4,                   # 轴刻度字号（发表级别）
           cex.lab=1.5,                    # 轴标签字号（发表级别）
           
           # 保持默认配色和SNP密度显示
           chr.den.col=c("darkgreen", "yellow", "red"),  # SNP密度颜色
           
           memo="publication"              # 文件名标记，不覆盖默认
)

# 2. 环状曼哈顿图 (8x8英寸 - 正方形)
MVP.Report(imMVP,
           plot.type="c",
           LOG10=TRUE,
           threshold=1.0e-5,
           outpath=output_dir,
           
           # 发表级别尺寸和质量
           height=5, width=5,              # 8x8英寸正方形
           dpi=400,
           file.type="jpg",
           
           # 环状图参数
           r=0.4,                          # 圆半径
           cir.chr.h=1.5,                  # 染色体高度
           cir.legend=TRUE,                # 显示图例
           cir.legend.cex=1.2,             # 图例字号
           
           # 保持默认配色和SNP密度显示
           chr.den.col=c("darkgreen", "yellow", "red"),  # SNP密度颜色
           
           memo="publication"
)
