#!/usr/bin/env python3
"""
Convert SNP format from:
chr1A	.	0	11772842	A	T
to:
1	chr1.s_11730	0	11730	T	C
"""

import re
import sys

def convert_snp_format(input_file, output_file):
    """
    Convert SNP format according to specified rules
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 6:
                continue
            
            # 解析输入格式
            chr_col = parts[0]  # chr1A
            pos = parts[3]      # 11772842
            ref_allele = parts[4]  # A
            alt_allele = parts[5]  # T
            
            # 提取chromosome编号 (chr和A之间的字符)
            match = re.match(r'chr(\d+)[A-Z]', chr_col)
            if match:
                chr_num = match.group(1)
            else:
                # 如果不匹配预期格式，跳过或使用默认值
                chr_num = "1"
            
            # 去除A后接.s_位置信息
            chr_base = re.sub(r'[A-Z]$', '', chr_col)  # 去除末尾的A
            new_chr_id = f"{chr_base}.s_{pos}"
            
            # 构建输出格式
            output_line = f"{chr_num}\t{new_chr_id}\t0\t{pos}\t{ref_allele}\t{alt_allele}"
            outfile.write(output_line + '\n')

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_format.py <input_file> <output_file>")
        print("Example: python convert_format.py input.txt output.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        convert_snp_format(input_file, output_file)
        print(f"转换完成！输出文件: {output_file}")
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
    except Exception as e:
        print(f"转换过程中出现错误: {e}")

if __name__ == "__main__":
    main()