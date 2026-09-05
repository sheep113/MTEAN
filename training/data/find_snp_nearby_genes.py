#!/usr/bin/env python3
"""
查找top50 SNPs附近的已命名基因 - V2版本
支持从GFF3文件提取功能基因命名（如ZCN8）

使用方法:
    python find_snp_nearby_genes_v2.py
    
    或自定义参数:
    python find_snp_nearby_genes_v2.py --snp-csv top50.csv --bim-file data.bim --gene-bed genes.bed --gff-file genes.gff3 --window 3000000 --output results.tsv
    
输出:
    - top50_snps_nearby_genes.tsv: 只包含有功能命名的基因 (ZCN8等)
    - top50_snps_nearby_genes_all.tsv: 包含所有基因 (完整列表)
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
import time
import re

# ==================== 默认参数设置 ====================
DEFAULT_SNP_CSV = "../logs0/DNAWhisperfinetune/run0_blockembedding_f3p_30k/manhattan_enrichment_clean_v5/manhattan_plots_clean/top50_snps_combined_v2_DTS.csv"
DEFAULT_BIM_FILE = "../data/maize1404_genotype/cubic_1404_v4_converted.bim"
DEFAULT_GFF_FILE = "../data/maize1404_genotype/Zea_mays.AGPv4.38.gff3"  # V4版本，Zm00001d格式，包含坐标和功能命名
DEFAULT_WINDOW_SIZE = 3000000  # 3Mb
DEFAULT_OUTPUT = "../output/DTS_top50_snps_nearby_genes.tsv"
DEFAULT_OUTPUT_ALL = "../output/DTS_top50_snps_nearby_genes_all.tsv"

def load_snp_list(snp_csv_path):
    """
    加载top50 SNPs CSV文件
    返回: DataFrame with columns [SNP_ID, neg_log10_p]
    注意：只保留SNP_ID和p值，CSV中的V2位置坐标被忽略（因为不正确）
    """
    print(f"正在加载SNP列表: {snp_csv_path}")
    df = pd.read_csv(snp_csv_path)
    
    # 只保留SNP_ID和neg_log10_p列
    if 'neg_log10_p' not in df.columns:
        print("警告: CSV文件中没有neg_log10_p列")
        df['neg_log10_p'] = None
    
    df = df[['SNP_ID', 'neg_log10_p']].copy()
    
    print(f"成功加载 {len(df)} 个SNPs")
    return df

def load_bim_file(bim_path):
    """
    加载BIM文件（PLINK格式）- 优化版
    BIM格式: chromosome, snp_id, genetic_distance, position, allele1, allele2
    返回: DataFrame with snp_id as index for O(1) lookup
    """
    print(f"正在加载BIM文件: {bim_path}")
    start = time.time()
    df = pd.read_csv(
        bim_path,
        sep='\t',
        header=None,
        names=['chromosome', 'snp_id', 'genetic_distance', 'position', 'allele1', 'allele2']
    )
    # 设置snp_id为索引，实现O(1)查询
    df.set_index('snp_id', inplace=True)
    print(f"成功加载 {len(df):,} 个SNPs (耗时: {time.time()-start:.2f}s)")
    return df

def load_gene_data_from_gff3(gff_file):
    """
    直接从GFF3文件中加载所有基因数据（坐标、功能命名）
    
    返回:
        dict: {chromosome: DataFrame}, 每条染色体上的基因及其功能命名
    """
    if not Path(gff_file).exists():
        print(f"错误: GFF3文件不存在: {gff_file}")
        return {}
    
    print(f"正在加载GFF3文件（包含基因坐标和功能命名）: {gff_file}")
    start_time = time.time()
    
    genes_data = []
    # 匹配模式
    id_pattern = re.compile(r'ID=gene:([^;]+)')
    name_pattern = re.compile(r'Name=([^;]+)')
    
    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            
            # 只处理gene类型的行
            if parts[2] != 'gene':
                continue
            
            chromosome = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attributes = parts[8]
            
            # 提取ID和Name
            id_match = id_pattern.search(attributes)
            name_match = name_pattern.search(attributes)
            
            if id_match:
                gene_id = id_match.group(1)
                functional_name = name_match.group(1) if name_match else None
                
                # 只保存非Zm开头的功能命名
                if functional_name and functional_name.startswith('Zm'):
                    functional_name = None
                
                genes_data.append({
                    'chromosome': chromosome,
                    'start': start,
                    'end': end,
                    'gene_id': gene_id,
                    'functional_name': functional_name,
                    'strand': strand
                })
    
    # 转换为DataFrame并按染色体分组
    df = pd.DataFrame(genes_data)
    gene_by_chr = {chrom: group.reset_index(drop=True) 
                   for chrom, group in df.groupby('chromosome')}
    
    elapsed = time.time() - start_time
    print(f"成功加载 {len(df):,} 个基因")
    print(f"  - 有功能命名的基因: {df['functional_name'].notna().sum()}")
    print(f"  - 覆盖 {len(gene_by_chr)} 条染色体 (耗时: {elapsed:.1f}秒)")
    
    return gene_by_chr

def extract_gene_names(name_field):
    """
    从BED文件的name字段提取基因名称
    例如: "gene:gene:Zm00001eb000010,gene:transcript:Zm00001eb000010_T001"
    提取: ['Zm00001eb000010']
    """
    genes = []
    parts = name_field.split(',')
    for part in parts:
        if 'gene:Zm' in part:
            # 提取基因ID
            gene_id = part.split(':')[-1]
            # 只保留基因ID，不包含transcript
            if '_T0' not in gene_id:
                genes.append(gene_id)
                break  # 只取第一个基因ID
    return genes

def find_nearby_genes(chromosome, position, gene_by_chr, window_size):
    """
    查找指定位置附近的基因
    
    参数:
        chromosome: 染色体号
        position: 位置
        gene_by_chr: 按染色体索引的基因字典
        window_size: 窗口大小（bp）
    
    返回: List of dicts containing gene info
    """
    # 从预索引字典中获取该染色体的基因
    if chromosome not in gene_by_chr:
        return []
    
    chr_genes = gene_by_chr[chromosome]
    
    # 使用向量化操作计算距离
    window_start = position - window_size
    window_end = position + window_size
    
    # 筛选可能在范围内的基因（粗筛选）
    candidates = chr_genes[
        (chr_genes['end'] >= window_start) & 
        (chr_genes['start'] <= window_end)
    ]
    
    if candidates.empty:
        return []
    
    nearby_genes = []
    for idx, gene in candidates.iterrows():
        # 精确计算SNP到基因的距离
        if position < gene['start']:
            distance = gene['start'] - position
        elif position > gene['end']:
            distance = position - gene['end']
        else:
            distance = 0  # SNP在基因内部
        
        gene_info = {
            'gene_id': gene['gene_id'],
            'functional_name': gene['functional_name'],
            'chromosome': gene['chromosome'],
            'start': gene['start'],
            'end': gene['end'],
            'strand': gene['strand'],
            'distance': distance
        }
        
        nearby_genes.append(gene_info)
    
    return nearby_genes

def format_gene_display(gene_id, functional_name):
    """
    格式化基因显示名称
    
    如果有功能命名: ZCN8 (Zm00001eb000010)
    如果没有: Zm00001eb000010
    """
    if functional_name:
        return f"{functional_name} ({gene_id})"
    else:
        return gene_id

def main():
    parser = argparse.ArgumentParser(description='查找top50 SNPs附近的已命名基因')
    parser.add_argument('--snp-csv', default=DEFAULT_SNP_CSV, help='Top50 SNPs CSV文件路径')
    parser.add_argument('--bim-file', default=DEFAULT_BIM_FILE, help='BIM文件路径（V4坐标）')
    parser.add_argument('--gff-file', default=DEFAULT_GFF_FILE, help='GFF3文件路径（V4版本，用于基因坐标和功能命名）')
    parser.add_argument('--window', type=int, default=DEFAULT_WINDOW_SIZE, help='搜索窗口大小（bp）')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='输出文件路径（只包含有功能命名的基因）')
    parser.add_argument('--output-all', default=DEFAULT_OUTPUT_ALL, help='输出文件路径（包含所有基因）')
    
    args = parser.parse_args()
    
    # 记录总运行时间
    total_start = time.time()
    
    # 1. 加载数据
    snp_list = load_snp_list(args.snp_csv)
    bim_data = load_bim_file(args.bim_file)
    gene_data = load_gene_data_from_gff3(args.gff_file)
    
    # 2. 处理每个SNP
    print(f"\n开始分析 {len(snp_list)} 个SNPs附近的基因...")
    print(f"搜索窗口: {args.window:,} bp (±{args.window/1e6:.1f} Mb)")
    
    results_all = []  # 所有基因
    results_named = []  # 只包含有功能命名的基因
    missing_snps = []
    analysis_start = time.time()
    
    for idx, row in snp_list.iterrows():
        snp_id = row['SNP_ID']
        neg_log10_p = row['neg_log10_p']
        
        # 在BIM文件中查找SNP的V4位置
        if snp_id not in bim_data.index:
            missing_snps.append(snp_id)
            continue
        
        snp_info = bim_data.loc[snp_id]
        chromosome = str(snp_info['chromosome'])
        position = snp_info['position']
        
        # 查找附近的基因
        nearby_genes = find_nearby_genes(chromosome, position, gene_data, args.window)
        
        # 保存结果
        for gene in nearby_genes:
            gene_display = format_gene_display(gene['gene_id'], gene['functional_name'])
            
            result = {
                'SNP_ID': snp_id,
                'SNP_Chromosome': chromosome,
                'SNP_Position': position,
                'neg_log10_p': neg_log10_p,
                'Gene': gene_display,
                'Gene_ID': gene['gene_id'],
                'Functional_Name': gene['functional_name'] if gene['functional_name'] else '-',
                'Gene_Chromosome': gene['chromosome'],
                'Gene_Start': gene['start'],
                'Gene_End': gene['end'],
                'Gene_Strand': gene['strand'],
                'Distance_to_SNP': gene['distance']
            }
            
            # 所有基因
            results_all.append(result)
            
            # 只保存有功能命名的基因
            if gene['functional_name']:
                results_named.append(result)
    
    analysis_elapsed = time.time() - analysis_start
    
    # 3. 输出结果
    if missing_snps:
        print(f"\n警告: {len(missing_snps)} 个SNPs在BIM文件中未找到V4坐标:")
        for snp in missing_snps[:10]:  # 只显示前10个
            print(f"  - {snp}")
        if len(missing_snps) > 10:
            print(f"  ... 还有 {len(missing_snps)-10} 个")
    
    # 保存完整结果（所有基因）
    if results_all:
        df_all = pd.DataFrame(results_all)
        df_all.to_csv(args.output_all, sep='\t', index=False)
        print(f"\n✓ 完整结果已保存: {args.output_all}")
        print(f"  - 共 {len(df_all)} 个基因")
        print(f"  - 平均每个SNP: {len(df_all)/len(snp_list):.1f} 个基因")
    
    # 保存有功能命名的基因
    if results_named:
        df_named = pd.DataFrame(results_named)
        df_named.to_csv(args.output, sep='\t', index=False)
        print(f"\n✓ 功能命名基因已保存: {args.output}")
        print(f"  - 共 {len(df_named)} 个有功能命名的基因")
        print(f"  - 占比: {len(df_named)/len(df_all)*100:.1f}%")
    else:
        print(f"\n警告: 未找到有功能命名的基因！")
        print(f"  可能的原因:")
        print(f"  1. GFF3文件中没有Name属性")
        print(f"  2. 或所有Name属性都是Zm开头的ID")
    
    # 统计信息
    total_elapsed = time.time() - total_start
    print(f"\n" + "="*60)
    print(f"性能统计:")
    print(f"  - 总耗时: {total_elapsed:.2f}秒")
    print(f"  - 分析耗时: {analysis_elapsed:.2f}秒")
    print(f"  - 处理速度: {len(snp_list)/analysis_elapsed:.1f} SNPs/秒")
    print("="*60)
    
    # 2. 处理每个SNP
    print(f"\n开始分析 {len(snp_list)} 个SNPs附近的基因...")
    print(f"搜索窗口: {args.window:,} bp (±{args.window/1e6:.1f} Mb)")
    
    results_all = []  # 所有基因
    results_named = []  # 只包含有功能命名的基因
    missing_snps = []
    analysis_start = time.time()
    
    for idx, row in snp_list.iterrows():
        snp_id = row['SNP_ID']
        neg_log10_p = row['neg_log10_p']
        
        # 在BIM文件中查找SNP的V4位置
        if snp_id not in bim_data.index:
            missing_snps.append(snp_id)
            continue
        
        snp_info = bim_data.loc[snp_id]
        chromosome = snp_info['chromosome']
        position = snp_info['position']
        
        # 查找附近的基因
        nearby_genes = find_nearby_genes(chromosome, position, gene_data, args.window)
        
        # 保存结果
        for gene in nearby_genes:
            gene_display = format_gene_display(gene['gene_id'], gene['functional_name'])
            
            result = {
                'SNP_ID': snp_id,
                'SNP_Chromosome': chromosome,
                'SNP_Position': position,
                'neg_log10_p': neg_log10_p,
                'Gene': gene_display,
                'Gene_ID': gene['gene_id'],
                'Functional_Name': gene['functional_name'] if gene['functional_name'] else '-',
                'Gene_Chromosome': gene['chromosome'],
                'Gene_Start': gene['start'],
                'Gene_End': gene['end'],
                'Gene_Strand': gene['strand'],
                'Distance_to_SNP': gene['distance']
            }
            
            # 所有基因
            results_all.append(result)
            
            # 只保存有功能命名的基因
            if gene['functional_name']:
                results_named.append(result)
    
    analysis_elapsed = time.time() - analysis_start
    
    # 3. 输出结果
    if missing_snps:
        print(f"\n警告: {len(missing_snps)} 个SNPs在BIM文件中未找到V4坐标:")
        for snp in missing_snps[:10]:  # 只显示前10个
            print(f"  - {snp}")
        if len(missing_snps) > 10:
            print(f"  ... 还有 {len(missing_snps)-10} 个")
    
    # 保存完整结果（所有基因）
    if results_all:
        df_all = pd.DataFrame(results_all)
        df_all.to_csv(args.output_all, sep='\t', index=False)
        print(f"\n✓ 完整结果已保存: {args.output_all}")
        print(f"  - 共 {len(df_all)} 个基因")
        print(f"  - 平均每个SNP: {len(df_all)/len(snp_list):.1f} 个基因")
    
    # 保存有功能命名的基因
    if results_named:
        df_named = pd.DataFrame(results_named)
        df_named.to_csv(args.output, sep='\t', index=False)
        print(f"\n✓ 功能命名基因已保存: {args.output}")
        print(f"  - 共 {len(df_named)} 个有功能命名的基因")
        print(f"  - 占比: {len(df_named)/len(df_all)*100:.1f}%")
    else:
        print(f"\n警告: 未找到有功能命名的基因！")
        print(f"  可能的原因:")
        print(f"  1. GFF3文件不存在或格式不正确")
        print(f"  2. GFF3文件中没有Name属性")
        print(f"  3. 基因ID在GFF3和BED文件中不匹配")
    
    # 统计信息
    total_elapsed = time.time() - total_start
    print(f"\n" + "="*60)
    print(f"性能统计:")
    print(f"  - 总耗时: {total_elapsed:.2f}秒")
    print(f"  - 分析耗时: {analysis_elapsed:.2f}秒")
    print(f"  - 处理速度: {len(snp_list)/analysis_elapsed:.1f} SNPs/秒")
    print("="*60)

if __name__ == '__main__':
    main()
