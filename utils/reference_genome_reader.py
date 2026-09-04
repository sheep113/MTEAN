import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from . import plink_reader as pr

class ReferenceGenomeReader:
    def __init__(self, bed_file_path):
        """
        初始化参考基因组读取器
        
        参数:
            bed_file_path: BED文件路径
        """
        self.bed_file_path = bed_file_path
        self.chromosomes = {}  # 存储染色体编号及其长度
        self.genes = defaultdict(list)  # 存储染色体及其对应基因列表
        self.intervals = defaultdict(list)  # 存储染色体及其对应区间划分
        self.interval_stats = {}  # 存储区间统计信息
        self.read_bed_file()
        self.construct_intervals()
    
    def read_bed_file(self):
        """读取BED文件，获取染色体长度和基因位置信息"""
        with open(self.bed_file_path, 'r') as file:
            for line in file:
                fields = line.strip().split('\t')
                chrom = fields[0]  # 染色体编号
                start = int(fields[1])  # 起始位置
                end = int(fields[2])  # 结束位置
                
                # 如果是染色体长度信息行（第4列包含"chromosome"）
                if len(fields) > 3 and "chromosome" in fields[3]:
                    self.chromosomes[chrom] = end
                # 如果是基因信息行（第4列包含"gene"）
                elif len(fields) > 3 and "gene" in fields[3]:
                    self.genes[chrom].append((start, end))
    
    def construct_intervals(self):
        """
        构建非基因区域-基因区域交替的区间划分
        
        返回：
            intervals: 字典，键为染色体编号，值为区间列表，每个区间为(start, end, interval_type)
        """
        for chrom, length in self.chromosomes.items():
            # 排序基因区间（按起始位置）
            if chrom in self.genes:
                genes = sorted(self.genes[chrom], key=lambda x: x[0])
                
                # 处理第一个非基因区间（如果存在）
                if genes[0][0] > 0:
                    self.intervals[chrom].append((0, genes[0][0]-1, 'non-gene'))
                
                # 添加第一个基因区间
                self.intervals[chrom].append((genes[0][0], genes[0][1], 'gene'))
                
                # 处理中间的非基因和基因区间
                for i in range(1, len(genes)):
                    prev_end = genes[i-1][1]
                    curr_start = genes[i][0]
                    
                    # 添加非基因区间
                    if curr_start - prev_end > 1:
                        self.intervals[chrom].append((prev_end + 1, curr_start - 1, 'non-gene'))
                    
                    # 添加基因区间
                    self.intervals[chrom].append((genes[i][0], genes[i][1], 'gene'))
                
                # 处理最后一个非基因区间（如果存在）
                last_gene_end = genes[-1][1]
                if last_gene_end < length - 1:
                    self.intervals[chrom].append((last_gene_end + 1, length, 'non-gene'))
            else:
                # 如果该染色体没有基因，则整个染色体视为非基因区间
                self.intervals[chrom].append((0, length, 'non-gene'))
        
        return self.intervals
        
    def build_interval_index(self):
        """
        构建区间索引，使用区间树优化查询效率
        
        返回:
            interval_index: 字典，键为染色体号，值为区间索引
        """
        try:
            from intervaltree import IntervalTree
            
            self.interval_index = {}
            for chrom, intervals in self.intervals.items():
                tree = IntervalTree()
                for start, end, interval_type in intervals:
                    # 存储原始区间元组作为数据
                    tree[start:end+1] = (start, end, interval_type)
                self.interval_index[chrom] = tree
            
            # 标记索引已构建
            self.index_built = True
            return self.interval_index
            
        except ImportError:
            # 如果没有intervaltree库，则使用原始方法
            self.index_built = False
            return None
    
    def get_interval_for_position(self, chrom, position):
        """
        获取指定位置所在的区间（优化版）
        
        参数:
            chrom: 染色体编号
            position: 位置
            
        返回:
            interval: 区间信息 (start, end, type)
        """
        # 标准化染色体编号
        std_chrom = chrom
        
        # 1. 尝试可能的染色体格式
        possible_formats = []
        
        # 根据输入类型生成可能的格式
        if isinstance(chrom, (int, np.integer)):
            possible_formats.extend([str(chrom), f"chr{chrom}", chrom])
        elif isinstance(chrom, float):
            possible_formats.extend([str(int(chrom)), f"chr{int(chrom)}", int(chrom)])
        elif isinstance(chrom, str):
            # 如果是字符串格式
            if chrom.startswith('chr'):
                possible_formats.extend([chrom, chrom[3:]])
            else:
                possible_formats.extend([chrom, f"chr{chrom}"])
            # 如果是数字字符串，还可能是整数
            if chrom.isdigit():
                possible_formats.extend([int(chrom)])
        
        # 2. 检查各种格式是否存在于intervals字典中
        found = False
        for fmt in possible_formats:
            if fmt in self.intervals:
                std_chrom = fmt
                found = True
                break
        
        # 3. 如果找不到匹配的染色体编号，返回None
        if not found:
            return None   
                     
        # 二分查找
        intervals = self.intervals[std_chrom]
        low, high = 0, len(intervals) - 1
        
        while low <= high:
            mid = (low + high) // 2
            start, end, _ = intervals[mid]
            
            if start <= position <= end:
                return intervals[mid]
            elif position < start:
                high = mid - 1
            else:
                low = mid + 1
        
        return None
        
    def count_snps_in_intervals(self, chrom, positions):
        """
        计算指定染色体上每个区间内的SNP数量
        
        参数:
            chrom: 染色体编号
            positions: 该染色体上的SNP位置数组
            
        返回:
            counts: 字典，键为(chrom, start, end)，值为该区间内的SNP数量
        """
        if chrom not in self.intervals:
            return {}
        
        counts = {}
        
        # 对每个区间进行计数
        for interval in self.intervals[chrom]:
            start, end, _ = interval
            # 使用NumPy的向量化操作进行计数
            count = np.sum((positions >= start) & (positions <= end))
            if count > 0:
                counts[(chrom, start, end)] = count
        
        return counts

    def process_snp_batch(self, batch_indices, chrom_array, pos_array, interval_snp_counts):
        """
        处理一批SNP数据
        
        参数:
            batch_indices: 要处理的SNP索引列表
            chrom_array: 染色体数组
            pos_array: 位置数组
            interval_snp_counts: 区间SNP计数字典
            
        返回:
            batch_results: 处理结果列表
        """
        batch_results = []
        for idx in batch_indices:
            chrom = chrom_array[idx]
            pos = pos_array[idx]
            
            interval = self.get_interval_for_position(chrom, pos)
            if interval:
                start, end, interval_type = interval
                
                # 区间中心
                interval_center = (start + end) / 2
                
                # 区间长度
                interval_length = end - start + 1
                
                # SNP密度
                # 处理不同类型的染色体号，确保键匹配
                chrom_key = str(chrom)  # 首先尝试字符串格式
                key = (chrom_key, start, end)
                snp_count = interval_snp_counts.get(key, 0)                
                # 如果找不到，尝试使用带"chr"前缀的键
                if snp_count == 0 and not chrom_key.startswith('chr'):
                    key = (f"chr{chrom_key}", start, end)
                    snp_count = interval_snp_counts.get(key, 0)                
                # 如果还找不到，尝试不带"chr"前缀的键
                if snp_count == 0 and chrom_key.startswith('chr'):
                    key = (chrom_key[3:], start, end)
                    snp_count = interval_snp_counts.get(key, 0)                
                snp_density = snp_count / interval_length if interval_length > 0 else 0
                
                # 是否基因区域
                is_gene = 1 if interval_type == 'gene' else 0
                
                batch_results.append((idx, [interval_center, interval_length, snp_density, is_gene]))
        return batch_results

    def process_snps(self, bim_data):
        """
        处理SNP信息，构造6维向量（优化版）
        
        参数:
            bim_data: SNP信息数据框 (DataFrame)
            
        返回:
            snp_vectors: NumPy数组，每一行是一个SNP的6维向量
        """
        import numpy as np
        from multiprocessing import Pool, cpu_count
        
        # 确保区间索引已构建
        if not hasattr(self, 'index_built'):
            self.build_interval_index()
            
        # 提取必要的列，转换为NumPy数组以提高操作效率
        chrom_array = np.array(bim_data.chr)
        pos_array = np.array(bim_data.position, dtype=np.int32)
        
        # 创建用于存储结果的数组
        num_snps = len(bim_data)
        result_array = np.zeros((num_snps, 6), dtype=np.float64)
        result_array[:, 0] = chrom_array  # 染色体列
        result_array[:, 1] = pos_array    # 位置列
                
        # 使用字典预先计算每个染色体中存在的SNP位置
        snp_by_chrom = {}
        unique_chroms = np.unique(chrom_array)
        for chrom in unique_chroms:
            # 将染色体号转换为字符串以匹配self.intervals的键类型
            chrom_str = str(chrom)
            if chrom_str in self.intervals:
                # 获取当前染色体的所有SNP位置
                mask = (chrom_array == chrom)
                snp_by_chrom[chrom_str] = pos_array[mask]
            # 如果需要支持"chr"前缀格式
            elif f"chr{chrom}" in self.intervals:
                mask = (chrom_array == chrom)
                snp_by_chrom[f"chr{chrom}"] = pos_array[mask]
        
        # 初始化区间SNP计数字典
        interval_snp_counts = {}
        
        # 将数据准备为适合并行处理的格式
        chrom_position_pairs = []
        for chrom in unique_chroms:
            chrom_str = str(chrom)
            if chrom_str in snp_by_chrom:
                chrom_position_pairs.append((chrom_str, snp_by_chrom[chrom_str]))
            elif f"chr{chrom}" in snp_by_chrom:
                chrom_position_pairs.append((f"chr{chrom}", snp_by_chrom[f"chr{chrom}"]))
                
        # 并行处理每个染色体的区间计数
        if len(chrom_position_pairs) > 0:
            with Pool(processes=min(cpu_count(), len(chrom_position_pairs))) as pool:
                count_results = pool.starmap(self.count_snps_in_intervals, chrom_position_pairs)
            
            # 合并计数结果
            for result in count_results:
                interval_snp_counts.update(result)
                
        # 处理SNP批次
        batch_size = 50000  # 每批处理50000个SNP
        all_indices = list(range(num_snps))
        batches = [all_indices[i:i+batch_size] for i in range(0, num_snps, batch_size)]
        
        # 为避免多进程中的pickling问题，使用单进程处理批次
        all_results = [self.process_snp_batch(batch, chrom_array, pos_array, interval_snp_counts) for batch in batches]
        
        # 将结果填入结果数组
        for batch_results in all_results:
            for idx, values in batch_results:
                result_array[idx, 2:] = values
        
        return result_array

    def get_intervals_statistics(self):
        """
        获取并返回intervals的统计信息
        
        返回:
            stats: 包含intervals统计信息的字典
        """
        if not self.interval_stats:
            stats = {}
            total_intervals = 0
            gene_intervals = 0
            non_gene_intervals = 0
            interval_lengths = []
            gene_interval_lengths = []
            non_gene_interval_lengths = []
            
            # 统计每个染色体的intervals
            chrom_stats = {}
            for chrom, intervals in self.intervals.items():
                chrom_gene_intervals = sum(1 for _, _, type in intervals if type == 'gene')
                chrom_non_gene_intervals = len(intervals) - chrom_gene_intervals
                
                chrom_stats[chrom] = {
                    'total': len(intervals),
                    'gene': chrom_gene_intervals,
                    'non-gene': chrom_non_gene_intervals
                }
                
                total_intervals += len(intervals)
                gene_intervals += chrom_gene_intervals
                non_gene_intervals += chrom_non_gene_intervals
                
                # 计算区间长度
                for start, end, type in intervals:
                    length = end - start + 1
                    interval_lengths.append(length)
                    if type == 'gene':
                        gene_interval_lengths.append(length)
                    else:
                        non_gene_interval_lengths.append(length)
            
            # 全局统计
            stats['total_intervals'] = total_intervals
            stats['gene_intervals'] = gene_intervals
            stats['non_gene_intervals'] = non_gene_intervals
            stats['by_chromosome'] = chrom_stats
            
            # 区间长度统计
            if interval_lengths:
                stats['interval_lengths'] = {
                    'min': min(interval_lengths),
                    'max': max(interval_lengths),
                    'mean': sum(interval_lengths) / len(interval_lengths),
                    'median': sorted(interval_lengths)[len(interval_lengths) // 2]
                }
            
            if gene_interval_lengths:
                stats['gene_interval_lengths'] = {
                    'min': min(gene_interval_lengths),
                    'max': max(gene_interval_lengths),
                    'mean': sum(gene_interval_lengths) / len(gene_interval_lengths),
                    'median': sorted(gene_interval_lengths)[len(gene_interval_lengths) // 2]
                }
            
            if non_gene_interval_lengths:
                stats['non_gene_interval_lengths'] = {
                    'min': min(non_gene_interval_lengths),
                    'max': max(non_gene_interval_lengths),
                    'mean': sum(non_gene_interval_lengths) / len(non_gene_interval_lengths),
                    'median': sorted(non_gene_interval_lengths)[len(non_gene_interval_lengths) // 2]
                }
            
            self.interval_stats = stats
            
        return self.interval_stats
    
    def print_intervals_statistics(self):
        """
        打印intervals的统计信息
        """
        stats = self.get_intervals_statistics()
        print(f"区间统计信息:")
        print(f"  总区间数: {stats['total_intervals']}")
        print(f"  基因区间数: {stats['gene_intervals']}")
        print(f"  非基因区间数: {stats['non_gene_intervals']}")
        
        if 'interval_lengths' in stats:
            length_stats = stats['interval_lengths']
            print(f"  区间长度: 最小={length_stats['min']}, 最大={length_stats['max']}, "
                 f"平均={length_stats['mean']:.2f}, 中位数={length_stats['median']}")
        
        if 'gene_interval_lengths' in stats:
            length_stats = stats['gene_interval_lengths']
            print(f"  基因区间长度: 最小={length_stats['min']}, 最大={length_stats['max']}, "
                 f"平均={length_stats['mean']:.2f}, 中位数={length_stats['median']}")
        
        if 'non_gene_interval_lengths' in stats:
            length_stats = stats['non_gene_interval_lengths']
            print(f"  非基因区间长度: 最小={length_stats['min']}, 最大={length_stats['max']}, "
                 f"平均={length_stats['mean']:.2f}, 中位数={length_stats['median']}")
        
        # 打印每个染色体的统计信息
        print("\n按染色体统计:")
        for chrom, chrom_stat in stats['by_chromosome'].items():
            print(f"  染色体 {chrom}: 总区间={chrom_stat['total']}, "
                 f"基因区间={chrom_stat['gene']}, 非基因区间={chrom_stat['non-gene']}")

def read_bed(bed_file_path):
    """
    读取BED文件的辅助函数
    
    参数:
        bed_file_path: BED文件路径
        
    返回:
        reader: ReferenceGenomeReader对象
    """
    reader = ReferenceGenomeReader(bed_file_path)
    return reader

def get_snp_vectors(bed_file_path, bim_data):
    """
    获取SNP向量的辅助函数
    
    参数:
        bed_file_path: BED文件路径
        bim_data: SNP信息数据框
        
    返回:
        snp_vectors: SNP向量列表
    """
    reader = read_bed(bed_file_path)
    return reader.process_snps(bim_data)

class GenomePartitioner:
    """
    全基因组SNP划分器，用于实现分组注意力的SNP划分策略
    """
    def __init__(self, reference_reader, snp_info):
        """
        初始化全基因组SNP划分器
        
        参数:
            reference_reader: ReferenceGenomeReader对象
            snp_info: SNP信息数据框 (DataFrame)
        """
        self.reference_reader = reference_reader
        self.snp_info = snp_info
        self.snps = []
        self.partitions = []
        self.snp_to_partition = {}  # 存储SNP到分区的映射
        self._load_snps()

    def _load_snps(self):
        """加载所有SNP信息"""    
        # 将SNP按照染色体和位置排序
        self.snps = [(row.chr, int(row.position), row) for row in self.snp_info.itertuples()]
        self.snps.sort(key=lambda x: (x[0], x[1]))  # 按染色体号和位置排序

    def partition_genome(self, partition_size):
        """
        对全基因组SNP进行划分
        
        参数:
            partition_size: 每个分区的SNP数量
            
        返回:
            partitions: 划分结果，每个元素为(chrom, start_pos, end_pos, snp_count)
        """
        self.partitions = []
        self.snp_to_partition = {}  # 重置SNP到分区的映射
        
        # 步骤1: 根据SNP数量均分全基因组
        total_snps = len(self.snps)
        if total_snps == 0:
            return self.partitions
            
        num_partitions = (total_snps + partition_size - 1) // partition_size  # 向上取整
        
        # 初步划分
        for i in range(num_partitions):
            start_idx = i * partition_size
            end_idx = min((i + 1) * partition_size - 1, total_snps - 1)
            
            if start_idx >= total_snps:
                break
                
            start_chrom, start_pos = self.snps[start_idx][0], self.snps[start_idx][1]
            end_chrom, end_pos = self.snps[end_idx][0], self.snps[end_idx][1]
            
            self.partitions.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_chrom': start_chrom,
                'start_pos': start_pos,
                'end_chrom': end_chrom,
                'end_pos': end_pos,
                'snp_count': end_idx - start_idx + 1
            })
        
        # 步骤2: 检测分区边界是否处于基因区间内，若是则调整边界
        self._adjust_partition_boundaries()
        
        # 步骤3: 合并小分区并切分大分区
        self._adjust_partition_sizes(partition_size)
        
        # 步骤4: 处理跨染色体的分区，确保每个分区都不跨染色体
        self._split_cross_chromosome_partitions()
        
        # 步骤5: 构建SNP到分区的映射，确保每个SNP都被分配到一个唯一的分区
        self._map_snps_to_partitions()
        
        # 转换成简化的输出格式
        result_partitions = []
        for p in self.partitions:
            result_partitions.append((
                p['start_chrom'], 
                p['start_pos'], 
                p['end_pos'], 
                p['snp_count']
            ))
        
        return result_partitions
    
    def _adjust_partition_boundaries(self):
        """调整分区边界，使其不穿过基因区间"""
        if not self.partitions:
            return
            
        for i in range(1, len(self.partitions)):
            prev_partition = self.partitions[i-1]
            curr_partition = self.partitions[i]
            
            # 如果分区边界在不同染色体上，无需调整
            if prev_partition['end_chrom'] != curr_partition['start_chrom']:
                continue
                
            chrom = curr_partition['start_chrom']
            boundary_pos = curr_partition['start_pos']
            
            # 检查边界位置是否位于基因区间内
            interval = self.reference_reader.get_interval_for_position(chrom, boundary_pos)
            if interval and interval[2] == 'gene':
                # 找到基因区间的结束位置
                gene_start, gene_end, _ = interval
                
                # 将基因区间划入前一个分区
                new_boundary_pos = gene_end + 1
                
                # 寻找新边界所在的SNP索引
                new_boundary_idx = curr_partition['start_idx']
                while (new_boundary_idx < len(self.snps) and 
                       self.snps[new_boundary_idx][0] == chrom and 
                       self.snps[new_boundary_idx][1] <= gene_end):
                    new_boundary_idx += 1
                
                # 更新分区边界
                if new_boundary_idx < len(self.snps):
                    # 更新当前分区的起始位置
                    curr_partition['start_idx'] = new_boundary_idx
                    curr_partition['start_pos'] = self.snps[new_boundary_idx][1]
                    curr_partition['snp_count'] = curr_partition['end_idx'] - curr_partition['start_idx'] + 1
                    
                    # 更新前一个分区的结束位置
                    prev_partition['end_idx'] = new_boundary_idx - 1
                    prev_partition['end_pos'] = self.snps[new_boundary_idx - 1][1]
                    prev_partition['snp_count'] = prev_partition['end_idx'] - prev_partition['start_idx'] + 1
    
    def _adjust_partition_sizes(self, target_size):
        """
        调整分区大小：合并过小的分区(小于目标大小1/5)，切分过大的分区
        
        参数:
            target_size: 目标分区大小
        """
        if not self.partitions:
            return
        
        # 计算最小分区阈值为目标大小的1/5
        min_threshold = target_size / 5
        
        # 首先合并小分区
        i = 0
        while i < len(self.partitions):
            if self.partitions[i]['snp_count'] < min_threshold:
                merged = False
                # 尝试合并到前一个或后一个分区
                if i > 0:
                    # 优先合并到前一个分区
                    prev_partition = self.partitions[i-1]
                    small_partition = self.partitions[i]
                    
                    prev_partition['end_idx'] = small_partition['end_idx']
                    prev_partition['end_chrom'] = small_partition['end_chrom']
                    prev_partition['end_pos'] = small_partition['end_pos']
                    prev_partition['snp_count'] = prev_partition['end_idx'] - prev_partition['start_idx'] + 1
                    
                    # 删除当前小分区
                    self.partitions.pop(i)
                    merged = True
                elif i < len(self.partitions) - 1:
                    # 合并到后一个分区
                    next_partition = self.partitions[i+1]
                    small_partition = self.partitions[i]
                    
                    next_partition['start_idx'] = small_partition['start_idx']
                    next_partition['start_chrom'] = small_partition['start_chrom']
                    next_partition['start_pos'] = small_partition['start_pos']
                    next_partition['snp_count'] = next_partition['end_idx'] - next_partition['start_idx'] + 1
                    
                    # 删除当前小分区
                    self.partitions.pop(i)
                    merged = True
                
                if not merged:
                    # 这是唯一的分区，无法合并
                    i += 1
            else:
                i += 1
                
        # 然后切分大分区
        i = 0
        while i < len(self.partitions):
            partition = self.partitions[i]
            # 如果分区大小超过目标大小的两倍，考虑切分
            if partition['snp_count'] > target_size * 2:
                # 找到合适的切分点，避免切分基因区域
                start_idx = partition['start_idx']
                end_idx = partition['end_idx']
                chrom = partition['start_chrom']
                
                # 尝试在中间位置切分
                mid_idx = start_idx + partition['snp_count'] // 2
                ideal_mid_idx = mid_idx
                
                # 确保切分点不在基因区域内
                while mid_idx < end_idx:
                    mid_pos = self.snps[mid_idx][1]
                    interval = self.reference_reader.get_interval_for_position(chrom, mid_pos)
                    if interval and interval[2] == 'non-gene':
                        break
                    mid_idx += 1
                
                # 如果向后找不到合适的切分点，向前尝试
                if mid_idx == end_idx:
                    mid_idx = ideal_mid_idx
                    while mid_idx > start_idx:
                        mid_pos = self.snps[mid_idx][1]
                        interval = self.reference_reader.get_interval_for_position(chrom, mid_pos)
                        if interval and interval[2] == 'non-gene':
                            break
                        mid_idx -= 1
                
                # 如果找到了合适的切分点
                if start_idx < mid_idx < end_idx:
                    mid_pos = self.snps[mid_idx][1]
                    
                    # 创建新的分区
                    new_partition = {
                        'start_idx': mid_idx,
                        'end_idx': end_idx,
                        'start_chrom': chrom,
                        'start_pos': mid_pos,
                        'end_chrom': partition['end_chrom'],
                        'end_pos': partition['end_pos'],
                        'snp_count': end_idx - mid_idx + 1
                    }
                    
                    # 更新原分区
                    partition['end_idx'] = mid_idx - 1
                    partition['end_pos'] = self.snps[mid_idx - 1][1]
                    partition['snp_count'] = mid_idx - start_idx
                    
                    # 添加新分区
                    self.partitions.insert(i + 1, new_partition)
                    
                    # 检查是否需要进一步切分
                    if partition['snp_count'] > target_size * 2:
                        continue  # 不增加i，再次检查当前分区
                
            i += 1
        
        # 确保所有SNP都被分配（填补可能的空隙）
        self._ensure_complete_coverage()
                
    def _ensure_complete_coverage(self):
        """确保所有SNP都被分配到分区，填补可能的空隙"""
        if not self.partitions or len(self.snps) == 0:
            return
            
        # 按染色体和起始位置排序分区
        self.partitions.sort(key=lambda p: (p['start_chrom'], p['start_pos']))
        
        # 检查分区之间是否有空隙
        for i in range(1, len(self.partitions)):
            prev_partition = self.partitions[i-1]
            curr_partition = self.partitions[i]
            
            # 如果在同一条染色体上且存在空隙
            if (prev_partition['end_chrom'] == curr_partition['start_chrom'] and 
                prev_partition['end_idx'] + 1 < curr_partition['start_idx']):
                # 将空隙SNP分配给前一个分区
                prev_partition['end_idx'] = curr_partition['start_idx'] - 1
                prev_partition['end_pos'] = self.snps[curr_partition['start_idx'] - 1][1]
                prev_partition['snp_count'] = prev_partition['end_idx'] - prev_partition['start_idx'] + 1
        
        # 检查第一个分区之前的SNP
        first_partition = self.partitions[0]
        if first_partition['start_idx'] > 0:
            # 创建新分区包含前面的SNP
            new_partition = {
                'start_idx': 0,
                'end_idx': first_partition['start_idx'] - 1,
                'start_chrom': self.snps[0][0],
                'start_pos': self.snps[0][1],
                'end_chrom': self.snps[first_partition['start_idx'] - 1][0],
                'end_pos': self.snps[first_partition['start_idx'] - 1][1],
                'snp_count': first_partition['start_idx']
            }
            self.partitions.insert(0, new_partition)
        
        # 检查最后一个分区之后的SNP
        last_partition = self.partitions[-1]
        if last_partition['end_idx'] < len(self.snps) - 1:
            # 创建新分区包含后面的SNP
            new_partition = {
                'start_idx': last_partition['end_idx'] + 1,
                'end_idx': len(self.snps) - 1,
                'start_chrom': self.snps[last_partition['end_idx'] + 1][0],
                'start_pos': self.snps[last_partition['end_idx'] + 1][1],
                'end_chrom': self.snps[-1][0],
                'end_pos': self.snps[-1][1],
                'snp_count': len(self.snps) - 1 - last_partition['end_idx']
            }
            self.partitions.append(new_partition)
    
    def _map_snps_to_partitions(self):
        """构建SNP到分区的映射"""
        self.snp_to_partition = {}
        for i, partition in enumerate(self.partitions):
            for snp_idx in range(partition['start_idx'], partition['end_idx'] + 1):
                snp_key = (self.snps[snp_idx][0], self.snps[snp_idx][1])  # (chrom, pos)
                self.snp_to_partition[snp_key] = i
    
    def get_partition_for_snp(self, chrom, pos):
        """
        查找包含指定SNP的分区
        
        参数:
            chrom: 染色体号
            pos: 位置
            
        返回:
            partition_index: 分区索引，如果未找到则返回-1
        """
        snp_key = (chrom, pos)
        return self.snp_to_partition.get(snp_key, -1)
    
    def get_partition_indices(self):
        """
        获取所有SNP的分区索引
        
        返回:
            indices: 列表，包含每个SNP所属的分区索引
        """
        indices = []
        for snp in self.snps:
            chrom, pos = snp[0], snp[1]
            partition_idx = self.get_partition_for_snp(chrom, pos)
            indices.append(partition_idx)
        return indices
    
    def get_partition_structure(self):
        """
        获取分区结构
        
        返回:
            structure: 列表，每个元素为(partition_index, chrom, start_pos)
        """
        structure = []
        for i, partition in enumerate(self.partitions):
            structure.append((i, partition['start_chrom'], partition['start_pos']))
        return structure
    
    def _split_cross_chromosome_partitions(self):
        """处理跨染色体的分区，将其切分为不跨染色体的多个分区"""
        if not self.partitions:
            return
            
        i = 0
        while i < len(self.partitions):
            partition = self.partitions[i]
            
            # 检查是否是跨染色体分区
            if partition['start_chrom'] != partition['end_chrom']:
                # 分区跨越染色体，需要分割
                start_chrom = partition['start_chrom']
                end_chrom = partition['end_chrom']
                
                # 获取当前染色体的长度
                chrom_length = self.reference_reader.chromosomes.get(start_chrom, 0)
                
                # 在SNP列表中找到染色体边界
                split_idx = partition['start_idx']
                while (split_idx <= partition['end_idx'] and 
                      self.snps[split_idx][0] == start_chrom):
                    split_idx += 1
                
                if split_idx <= partition['end_idx']:  # 找到了边界点
                    # 创建第一个分区：从当前起始位置到染色体末尾
                    first_partition = {
                        'start_idx': partition['start_idx'],
                        'end_idx': split_idx - 1,
                        'start_chrom': start_chrom,
                        'start_pos': partition['start_pos'],
                        'end_chrom': start_chrom,
                        'end_pos': chrom_length,  # 使用染色体的末尾位置
                        'snp_count': split_idx - partition['start_idx']
                    }
                    
                    # 创建第二个分区：从下一染色体开始到当前结束位置
                    second_partition = {
                        'start_idx': split_idx,
                        'end_idx': partition['end_idx'],
                        'start_chrom': self.snps[split_idx][0],
                        'start_pos': 0,  # 下一染色体的起始位置
                        'end_chrom': partition['end_chrom'],
                        'end_pos': partition['end_pos'],
                        'snp_count': partition['end_idx'] - split_idx + 1
                    }
                    
                    # 用这两个新分区替换当前分区
                    self.partitions[i] = first_partition
                    self.partitions.insert(i + 1, second_partition)
                    
                    # 不增加i，因为新插入的分区也可能跨染色体，需要再次检查
                    continue
            
            i += 1
        
        # 确保所有分区都不跨染色体（递归检查可能产生的新跨染色体分区）
        has_cross_chrom = any(p['start_chrom'] != p['end_chrom'] for p in self.partitions)
        if has_cross_chrom:
            self._split_cross_chromosome_partitions()

# 在主函数中添加测试代码
def main():
    """主函数，用于测试"""
    if len(sys.argv) < 3:
        print("Usage: python reference_genome_reader.py <bed_file_path> <bim_file_path>")
        return
        
    bed_file = sys.argv[1]
    bim_file = sys.argv[2]
    
    # 创建PlinkReader实例
    plink_reader = pr.PlinkReader()
    # 获取BIM文件前缀（去除.bim扩展名）
    bim_prefix = Path(bim_file).with_suffix('')
    # 加载SNP信息
    bim_data = plink_reader.load_snp_info(bim_prefix)
    
    # 测试ReferenceGenomeReader
    reader = read_bed(bed_file)
    snp_vectors = reader.process_snps(bim_data)
    
    # 打印前10个SNP的向量
    for i, vector in enumerate(snp_vectors[:10]):
        print(f"SNP {i+1}: {vector}")
    print(f"Total SNPs: {len(snp_vectors)}")
    
    # 测试GenomePartitioner
    partitioner = GenomePartitioner(reader, bim_data)
    partitions = partitioner.partition_genome(partition_size=1000)
    
    print("\nGenome Partitioning Results:")
    print(f"Total partitions: {len(partitions)}")
    for i, (chrom, start, end, count) in enumerate(partitions[:5]):
        print(f"Partition {i+1}: Chromosome {chrom}, Start: {start}, End: {end}, SNP Count: {count}")

if __name__ == "__main__":
    main()
