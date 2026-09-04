#!/bin/bash

# --- 请在此处修改您的输入文件路径和前缀 ---
# 例如: FULL_INPUT_PREFIX="/home/user/project/dataset_europe"
# 或: FULL_INPUT_PREFIX="/mnt/data/my_gwas_data/cohort1_genotypes"
FULL_INPUT_PREFIX="/media/marxin/softs/workspace/Whisper_of_DNA_pl/data/Grapedata/gwas"
# -------------------------------------------------

# --- 检查输入参数是否已配置 ---
if [ "${FULL_INPUT_PREFIX}" == "YOUR_FULL_INPUT_PLINK_FILE_PREFIX_HERE" ] || [ -z "${FULL_INPUT_PREFIX}" ]; then
    echo "错误: 请在脚本中修改 'FULL_INPUT_PREFIX' 变量来指定您的输入PLINK文件前缀 (包含完整路径)。"
    echo "例如: FULL_INPUT_PREFIX=\"/path/to/your/input_file_prefix\""
    exit 1
fi

# --- 输入和输出路径配置 ---
INPUT_DIR=$(dirname "${FULL_INPUT_PREFIX}")
INPUT_BASENAME=$(basename "${FULL_INPUT_PREFIX}")

OUTPUT_PREFIX_STEP1="${INPUT_DIR}/${INPUT_BASENAME}_filtered_step1"
OUTPUT_PREFIX_LD="${INPUT_DIR}/${INPUT_BASENAME}_ld_pruned_list"
# 修改步骤3的输出名，使其成为中间文件
OUTPUT_PREFIX_STEP3_LD_PRUNED="${INPUT_DIR}/${INPUT_BASENAME}_snps_ld_pruned"
# 最终抽样后的输出文件名
OUTPUT_PREFIX_FINAL_THINNED="${INPUT_DIR}/${INPUT_BASENAME}_final_snps_100k"

# --- 筛选步骤 ---
echo "开始SNP筛选..."
echo "输入文件前缀: ${FULL_INPUT_PREFIX}"
echo "输出文件将保存在目录: ${INPUT_DIR}"
echo "最终输出文件名前缀将是: ${INPUT_BASENAME}_final_snps"
echo "-----------------------------------------------------"

echo "步骤 1: 移除INDEL并应用SNP质量控制 (第一轮强QC)..."
plink --bfile "${FULL_INPUT_PREFIX}" \
    --allow-extra-chr \
    --snps-only just-acgt \
    --maf 0.3 --geno 0.0005 --hwe 1e-20 \
    --make-bed \
    --out "${OUTPUT_PREFIX_STEP1}" 

if [ $? -ne 0 ]; then
    echo "错误：步骤1 PLINK筛选失败。请检查PLINK日志文件 ${OUTPUT_PREFIX_STEP1}.log"
    exit 1
fi
echo "步骤 1 完成. 中间文件: ${OUTPUT_PREFIX_STEP1}.bed/bim/fam"
echo "-----------------------------------------------------"

echo "步骤 2: 生成LD剪枝列表 (基于第一轮QC结果)..."
plink --bfile "${OUTPUT_PREFIX_STEP1}" \
    --allow-extra-chr \
    --indep-pairwise 10000 100 0.05 \
    --out "${OUTPUT_PREFIX_LD}"

if [ $? -ne 0 ]; then
    echo "错误：步骤2 LD剪枝列表生成失败。请检查PLINK日志文件 ${OUTPUT_PREFIX_LD}.log"
    exit 1
fi
echo "步骤 2 完成. LD剪枝列表: ${OUTPUT_PREFIX_LD}.prune.in 和 ${OUTPUT_PREFIX_LD}.prune.out"
echo "-----------------------------------------------------"

echo "步骤 3: 根据LD剪枝列表提取SNP..."
plink --bfile "${OUTPUT_PREFIX_STEP1}" \
    --allow-extra-chr \
    --extract "${OUTPUT_PREFIX_LD}.prune.in" \
    --make-bed \
    --out "${OUTPUT_PREFIX_STEP3_LD_PRUNED}" # 输出到新的中间文件

if [ $? -ne 0 ]; then
    echo "错误：步骤3 根据LD剪枝列表提取SNP失败。请检查PLINK日志文件 ${OUTPUT_PREFIX_STEP3_LD_PRUNED}.log"
    exit 1
fi
echo "步骤 3 完成. LD剪枝后文件: ${OUTPUT_PREFIX_STEP3_LD_PRUNED}.bed/bim/fam"
echo "-----------------------------------------------------"

echo "步骤 4: 随机抽样到约10万个SNP..."
plink --bfile "${OUTPUT_PREFIX_STEP3_LD_PRUNED}" \
    --allow-extra-chr \
    --thin-count 32768 \
    --make-bed \
    --out "${OUTPUT_PREFIX_FINAL_THINNED}"

if [ $? -ne 0 ]; then
    echo "错误：步骤4 随机抽样失败。请检查PLINK日志文件 ${OUTPUT_PREFIX_FINAL_THINNED}.log"
    exit 1
fi

echo "-----------------------------------------------------"
echo "SNP筛选完成!"
echo "最终输出文件 (约10万SNP): ${OUTPUT_PREFIX_FINAL_THINNED}.bed/bim/fam"
echo "保留的SNP列表在 ${OUTPUT_PREFIX_FINAL_THINNED}.bim"
echo "详细日志请查看 ${OUTPUT_PREFIX_STEP1}.log, ${OUTPUT_PREFIX_LD}.log 和 ${OUTPUT_PREFIX_FINAL_THINNED}.log"
echo "-----------------------------------------------------"

# --- 参数说明 (用于严格筛选) ---
# --snps-only just-acgt: 仅保留等位基因为A, C, G, T的SNP，有效移除插入缺失(INDEL)。
# --maf 0.1: 最小等位基因频率阈值。移除次要等位基因频率低于10%的SNP。
#            (常用值为0.01或0.05，0.1更为严格)
# --geno 0.01: SNP缺失率阈值。移除缺失基因型比例高于1%的SNP。
#             (常用值为0.05或0.1，0.01更为严格)
# --hwe 1e-10: Hardy-Weinberg平衡检验p值阈值。移除p值小于1e-10的SNP。
#              (常用值为1e-6，1e-10更为严格。注意：此筛选通常在对照组中应用，
#               或在确信群体结构不是主要混杂因素时用于一般QC)
# --indep-pairwise 50 5 0.2: 进行LD剪枝。
#   - 50: 计算LD的窗口大小 (单位：SNP数量)。
#   - 5: 窗口移动的步长 (单位：SNP数量)。
#   - 0.2: r^2阈值。高于此阈值的SNP对将被视为处于LD状态，其中一个会被移除。
#          (常用值为0.2-0.5，0.2相对严格)
# --make-bed: 输出PLINK二进制文件 (bed, bim, fam)。
# --out: 指定输出文件名前缀。
# --extract: 提取指定的SNP列表中的SNP。