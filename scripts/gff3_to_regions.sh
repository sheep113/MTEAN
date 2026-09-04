#!/bin/bash

# 检查参数个数
if [ "$#" -ne 2 ]; then
    echo "用法: $0 <输入GFF3文件> <输出BED文件>"
    echo "  例如: $0 input.gff3 output.bed"
    exit 1
fi

input_file=$1
output_file=$2

# 检查输入文件是否存在
if [ ! -f "$input_file" ]; then
    echo "错误：输入文件 $input_file 不存在"
    exit 1
fi

# 创建临时文件
temp_bed=$(mktemp)
temp_seq_regions=$(mktemp)

# 添加错误处理
if [ -z "$temp_bed" ] || [ -z "$temp_seq_regions" ]; then
    echo "错误：无法创建临时文件"
    exit 1
fi

echo "正在处理GFF3文件: $input_file"

# 首先提取染色体sequence-region信息
echo "提取染色体信息..."
grep "^##sequence-region" "$input_file" | awk '{
    # 提取染色体名称和范围
    chr = $2
    start = $3 - 1  # 转换为0-based
    end = $4
    print chr"\t"start"\t"end"\t""chromosome:"chr"\t0\t."
}' > "$temp_seq_regions"

# 如果没有找到sequence-region行，尝试从特征行推断
if [ ! -s "$temp_seq_regions" ]; then
    echo "未找到##sequence-region信息，尝试从特征行推断..."
    awk 'BEGIN{FS="\t"; OFS="\t"}
    $1 !~ /^#/ && $1 != "" {
        chr=$1
        end=$5
        if(!(chr in max_pos) || end > max_pos[chr]) {
            max_pos[chr] = end
        }
        if(!(chr in min_pos) || $4 < min_pos[chr]) {
            min_pos[chr] = $4
        }
    }
    END {
        for(chr in max_pos) {
            # 输出推断的染色体区域
            print chr, 0, max_pos[chr], "chromosome:"chr, 0, "."
        }
    }' "$input_file" > "$temp_seq_regions"
fi

# 提取所有编码区域
# 这里包含常见的编码区类型：gene, CDS, exon, mRNA, transcript
echo "提取所有编码区域..."
awk 'BEGIN{FS="\t"; OFS="\t"} 
$1 !~ /^#/ && ($3=="gene") {
    # 提取ID和类型信息
    feature_type = $3
    feature_id = ""
    parent_id = ""
    gene_name = ""
    
    # 解析属性字段
    split($9, attrs, ";")
    for(i in attrs) {
        if(attrs[i] ~ /^ID=/) {
            feature_id = substr(attrs[i], 4)
        }
        if(attrs[i] ~ /^Parent=/) {
            parent_id = substr(attrs[i], 8)
        }
        if(attrs[i] ~ /^Name=/) {
            gene_name = substr(attrs[i], 6)
        }
    }
    
    # 构建标识符
    identifier = feature_type
    if(feature_id != "") {
        identifier = identifier":"feature_id
    } else if(gene_name != "") {
        identifier = identifier":"gene_name
    } else if(parent_id != "") {
        identifier = identifier":parent="parent_id
    } else {
        identifier = identifier":"$1"_"$4"_"$5
    }
    
    # 输出标准BED格式：染色体、起始位置(0-based)、结束位置、ID、分数、链方向
    print $1, $4-1, $5, identifier, "0", $7
}' "$input_file" | sort -k1,1 -k2,2n > "$temp_bed"

# 检查临时文件是否为空
if [ ! -s "$temp_bed" ]; then
    echo "警告：没有找到任何编码区域特征"
    exit 1
fi

# 创建临时文件存储合并后的区域
temp_merged=$(mktemp)

# 合并连续区域
echo "合并连续或重叠区域..."
awk 'BEGIN{FS="\t"; OFS="\t"}
{
    if (NR == 1) {
        # 初始化第一个区域
        current_chr=$1;
        current_start=$2;
        current_end=$3;
        current_name=$4;
        current_score=$5;
        current_strand=$6;
        next;
    }
    
    # 如果染色体和链方向相同，且区域连续或重叠
    if ($1 == current_chr && $6 == current_strand && $2 <= current_end+1) {
        # 扩展当前区域
        if ($3 > current_end) current_end = $3;
        current_name = current_name "," $4;
    } else {
        # 输出当前区域并开始新区域
        print current_chr, current_start, current_end, current_name, current_score, current_strand;
        current_chr=$1;
        current_start=$2;
        current_end=$3;
        current_name=$4;
        current_score=$5;
        current_strand=$6;
    }
}
END {
    # 输出最后一个区域
    if (NR > 0) print current_chr, current_start, current_end, current_name, current_score, current_strand;
}' "$temp_bed" > "$temp_merged"

# 保存合并前的特征数量
before_merge=$(wc -l < "$temp_bed")

# 合并染色体信息和合并后的编码区域
echo "合并染色体信息和编码区域..."
cat "$temp_seq_regions" "$temp_merged" > "$output_file"

# 清理临时文件
rm -f "$temp_bed" "$temp_seq_regions" "$temp_merged"

# 输出统计信息
total_chroms=$(grep -c "chromosome:" "$output_file")
total_features=$(($(wc -l < "$output_file") - $total_chroms))

echo "已成功创建BED文件: $output_file"
echo "染色体数量: $total_chroms"
echo "合并前编码区域数量: $before_merge"
echo "合并后编码区域数量: $total_features"
echo "合并减少的区域数量: $(($before_merge - $total_features))"
echo "文件格式: 染色体 起始位置(0-based) 结束位置 特征ID 分数 链方向"

# 显示文件前几行作为示例
echo -e "\n染色体信息示例:"
grep "chromosome:" "$output_file" | head -n 3

echo -e "\n合并后的编码区域示例:"
grep -v "chromosome:" "$output_file" | head -n 5

echo -e "\n注意：此BED文件已格式化为PLINK兼容格式，第一部分包含染色体sequence-region信息，其后是合并后的编码区域。"
