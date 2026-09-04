from pathlib import Path
import logging
import sys
import numpy as np
import pandas as pd
import multiprocessing
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import time
import concurrent.futures
from functools import partial
import os

# 添加h5py导入
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logging.warning("未检测到h5py库，HDF5功能将不可用。请使用'pip install h5py'安装")

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from utils.data_loader import DataLoader, timer
from scripts.snp_qc import SNPQualityControl
from scripts.calc_mic import GWASAnalyzer
from config.config import Config, ConfigValidationError

@dataclass
class PreprocessResult:
    """预处理结果简化类，用于返回处理状态"""
    success: bool
    message: str
    stats: Optional[Dict[str, Any]] = None

# 添加在类外部，用于多进程并行处理
def normalize_batch_genotypes(batch_data):
    """
    对基因型数据批次进行归一化处理
    
    Args:
        batch_data: 包含批次索引和基因型数据的元组
    
    Returns:
        tuple: (批次索引, 归一化后的基因型数据)
    """
    batch_idx, batch_genotypes = batch_data
    
    # 归一化处理逻辑 - 对每个SNP位点的基因型编码进行处理
    normalized = np.zeros_like(batch_genotypes, dtype=np.float32)
    
    for i in range(batch_genotypes.shape[0]):
        snp_values = batch_genotypes[i]
        # 统计每个编码值的频率
        unique_vals, counts = np.unique(snp_values, return_counts=True)
        total = np.sum(counts)
        
        # 为每个基因型计算得分 - 根据其在样本中的频率
        for val_idx, val in enumerate(unique_vals):
            # 修复：确保 total 大于 0，避免除零错误
            if total > 0:
                frequency = counts[val_idx] / total
                # 使用频率的负对数作为稀有变异的权重
                if frequency > 0:
                    weight = -np.log(frequency)
                else:
                    weight = 0
                # 应用权重到对应的基因型值
                normalized[i, snp_values == val] = weight
    
    return batch_idx, normalized

class PreprocessUtils:
    """预处理工具类 - 从PreprocessPipeline中提取的辅助方法"""

    def __init__(self, logger=None, n_jobs=15, data_loader=None, config_path=None):
        """
        初始化预处理工具
        
        Args:
            logger: 日志记录器
            n_jobs: 并行处理的作业数
            data_loader: 数据加载器实例
            config_path: 配置文件路径
        """
        self.logger = logger or self._setup_logger()
        self.n_jobs = n_jobs
        self.data_loader = data_loader
        self.config_path = config_path
        self.start_time = time.time()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(self.__class__.__name__)
        # 检查是否已有处理器
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def run_quality_control(self) -> Optional[Path]:
        """运行质量控制"""
        if not self.data_loader.config.preprocessing.snp_filtering.geno['enable']:
            self.logger.info("跳过质量控制步骤")
            return None

        self.logger.info("开始SNP质量控制...")
        qc = SNPQualityControl(self.config_path, logger=self.logger, n_jobs=self.n_jobs)
        return qc.run_qc()

    def run_mic_analysis(self, input_prefix: Path) -> Optional[Path]:
        """运行MIC分析"""
        if not self.data_loader.config.preprocessing.mic_analysis.enable:
            self.logger.info("跳过MIC分析步骤")
            return None

        self.logger.info("开始MIC分析...")
        analyzer = GWASAnalyzer(self.config_path, logger=self.logger, n_jobs=self.n_jobs)
        return analyzer.analyze()

    def calcul_transformer_params(self, transformer_config) -> int:
        """
        计算Transformer模型的参数量（公共方法）
        
        Args:
            transformer_config: Transformer配置对象
                
        Returns:
            int: 估计的参数量
        """
        return self._estimate_transformer_params(transformer_config)

    def _extract_block_features(self, snp_info: pd.DataFrame, processed_snps: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        废弃
        提取基因块特征 - 将SNP位置信息和基因型编码分开存储
        位置特征向量只需要存储一次（所有样本共享），基因型编码按照样本存储
        废弃
        
        Args:
            snp_info: SNP信息数据框
            processed_snps: 处理后的SNP数据 (已编码为0-9的整数)
            
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: 分开存储的特征和特征统计
        """
        self.logger.info("提取SNP特征与统计信息（优化存储版）...")
        
        try:
            # 1. 首先获取SNPs的6维位置特征向量 (所有样本共享)
            from utils.reference_genome_reader import ReferenceGenomeReader
            
            with timer("生成SNP位置特征向量", self.logger):
                reference_reader = ReferenceGenomeReader(self.data_loader.reference_genome_path)
                # 输出intervals统计信息
                self.logger.info("区间统计信息:")
                stats = reference_reader.get_intervals_statistics()
                self.logger.info(f"  总区间数: {stats['total_intervals']}")
                self.logger.info(f"  基因区间数: {stats['gene_intervals']}")
                self.logger.info(f"  非基因区间数: {stats['non_gene_intervals']}")
                
                # 使用已加载的snp_info获取位置向量
                position_vectors = reference_reader.process_snps(snp_info)
                self.logger.info(f"成功生成位置特征向量，包含{len(position_vectors)}个SNP")
                
                # 将位置向量转换为numpy数组
                position_array = np.array(position_vectors, dtype=np.float32)
                self.logger.info(f"位置特征向量形状: {position_array.shape}")
            
            # 计算数据维度
            n_snps, n_samples = processed_snps.shape
            position_dim = position_array.shape[1]  # 6维位置特征
            
            # 2. 分别归一化位置特征向量和基因型编码
            self.logger.info(f"开始归一化位置特征向量，形状: [{n_snps}, {position_dim}]")
            
            # 导入归一化函数
            from utils.snp_utils import normalize_feature
            
            # 归一化位置特征向量 - 并行处理各维度
            normalized_position = np.zeros_like(position_array)
            
            # 获取最大工作进程数
            import os
            max_workers = min(self.n_jobs, os.cpu_count() or 4)
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 为每个维度创建一个归一化任务
                future_to_dim = {
                    executor.submit(normalize_feature, position_array[:, dim]): dim
                    for dim in range(position_dim)
                }
                
                # 收集归一化结果
                for future in concurrent.futures.as_completed(future_to_dim):
                    dim = future_to_dim[future]
                    normalized_position[:, dim] = future.result()
            
            self.logger.info("位置特征向量归一化完成")
            
            # 3. 归一化基因型编码 - 按批次处理
            self.logger.info(f"开始归一化基因型编码，形状: [{n_snps}, {n_samples}]")
            
            # 优化：根据可用内存动态调整批处理大小
            import psutil
            available_mem = psutil.virtual_memory().available
            mem_per_batch = n_snps * 4  # 每个样本批次的估计内存使用量（字节）
            optimal_batch_size = max(100, min(500, int(available_mem * 0.2 / mem_per_batch)))
            
            batch_size = max(50, optimal_batch_size)  # 确保批次足够大以减少IO操作
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            self.logger.info(f"并行处理配置：{max_workers}个工作线程，{n_batches}个批次，每批次{batch_size}个样本")
            
            # 初始化归一化后的基因型编码数组
            normalized_genotypes = np.zeros_like(processed_snps, dtype=np.float32)
            
            # 使用并行处理归一化基因型数据
            if self.n_jobs > 1 and n_snps >= 1000:  # 只在数据量较大时使用并行
                # 分批处理
                batches = [(i, processed_snps[i:i+batch_size]) 
                        for i in range(0, n_snps, batch_size)]
                
                # 使用外部定义的函数进行并行处理
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    futures = [executor.submit(normalize_batch_genotypes, batch) for batch in batches]
                    
                    normalized_genotypes = np.zeros((n_snps, n_samples), dtype=np.float32)
                    for future in concurrent.futures.as_completed(futures):
                        batch_idx, batch_normalized = future.result()
                        end_idx = min(batch_idx + batch_normalized.shape[0], n_snps)
                        normalized_genotypes[batch_idx:end_idx] = batch_normalized
            else:
                # 串行处理
                normalized_genotypes = np.zeros((n_snps, n_samples), dtype=np.float32)
                for i in range(n_snps):
                    snp_values = processed_snps[i]
                    unique_vals, counts = np.unique(snp_values, return_counts=True)
                    total = np.sum(counts)
                    
                    for val_idx, val in enumerate(unique_vals):
                        frequency = counts[val_idx] / total
                        if frequency > 0:
                            weight = -np.log(frequency)
                        else:
                            weight = 0
                        normalized_genotypes[i, snp_values == val] = weight
            
            # 4. 计算位置特征统计信息
            position_stats = self._calculate_position_stats(normalized_position)
            
            # 5. 采样计算基因型编码统计信息
            genotype_stats = self._calculate_genotype_stats(normalized_genotypes, n_snps, n_samples)
            
            # 6. 构建返回结果
            feature_data = {
                "position_features": normalized_position,  # 位置特征向量 [n_snps, 6]
                "genotype_features": normalized_genotypes,  # 基因型编码 [n_snps, n_samples]
                "position_dim": position_dim,
                "n_snps": n_snps,
                "n_samples": n_samples
            }
            
            feature_stats = {
                "feature_dim": position_dim + 1,  # 总特征维度 = 位置特征维度 + 基因型编码维度
                "position_dim": position_dim,     # 位置特征维度
                "genotype_dim": 1,               # 基因型编码维度
                "n_snps": n_snps,                # SNP数量
                "n_samples": n_samples,           # 样本数量
                "position_stats": position_stats,  # 位置特征统计
                "genotype_stats": genotype_stats   # 基因型编码统计
            }
            
            self.logger.info("特征提取和统计完成")
            
            return feature_data, feature_stats
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _calculate_position_stats(self, normalized_position: np.ndarray) -> Dict[str, Any]:
        """
        计算位置特征向量的统计信息

        Args:
            normalized_position: 归一化后的位置特征向量

        Returns:
            Dict[str, Any]: 位置特征统计信息
        """
        self.logger.info("计算位置特征向量统计信息")

        n_snps, position_dim = normalized_position.shape

        # 计算每个维度的统计值
        stats = {
            "dims": position_dim,
            "value_ranges": []
        }

        for dim in range(position_dim):
            dim_values = normalized_position[:, dim]
            stats["value_ranges"].append({
                "min": float(np.min(dim_values)),
                "max": float(np.max(dim_values)),
                "mean": float(np.mean(dim_values)),
                "std": float(np.std(dim_values))
            })

        # 额外分析维度间的相关性
        try:
            # 只对少量SNP计算相关性以节省计算资源
            sample_size = min(5000, n_snps)
            if sample_size < n_snps:
                sample_indices = np.random.choice(n_snps, sample_size, replace=False)
                sample_data = normalized_position[sample_indices]
            else:
                sample_data = normalized_position

            # 计算每个维度的标准差，找出有效的维度（标准差大于阈值）
            std_values = np.std(sample_data, axis=0)
            eps = 1e-8  # 定义一个小的阈值
            valid_dims = np.where(std_values > eps)[0]

            # 只有在有足够的有效维度时才计算相关系数
            if len(valid_dims) >= 2:
                # 只使用有效维度的数据计算相关系数
                valid_data = sample_data[:, valid_dims]
                corr_matrix = np.corrcoef(valid_data.T)

                # 提取相关系数（排除对角线）
                corr_values = []
                for i in range(len(valid_dims)):
                    for j in range(i+1, len(valid_dims)):
                        corr_values.append(float(corr_matrix[i, j]))

                # 记录维度间的相关性统计
                if corr_values:
                    stats["correlations"] = {
                        "min": float(min(corr_values)),
                        "max": float(max(corr_values)),
                        "mean": float(np.mean(corr_values)),
                        "abs_mean": float(np.mean(np.abs(corr_values))),
                        "valid_dims_count": len(valid_dims),
                        "invalid_dims_count": position_dim - len(valid_dims)
                    }
                
                    # 如果有无效维度，记录它们的索引
                    if len(valid_dims) < position_dim:
                        stats["correlations"]["invalid_dims"] = [int(i) for i in range(position_dim) if i not in valid_dims]
            else:
                self.logger.warning(f"维度相关性分析跳过：至少需要2个有效维度，但只找到{len(valid_dims)}个")
                stats["correlations"] = {
                    "warning": "无法计算相关性，有效维度不足",
                    "valid_dims_count": len(valid_dims),
                    "invalid_dims_count": position_dim - len(valid_dims)
                }
        except Exception as e:
            self.logger.warning(f"计算位置特征相关性时出错: {str(e)}")
    
        return stats

    def _calculate_genotype_stats(self, normalized_genotypes: np.ndarray, n_snps: int, n_samples: int) -> Dict[str, Any]:
        """
        计算基因型编码的统计信息（使用抽样计算）
        
        Args:
            normalized_genotypes: 归一化后的基因型编码
            n_snps: SNP数量
            n_samples: 样本数量
            
        Returns:
            Dict[str, Any]: 基因型编码统计信息
        """
        self.logger.info("计算基因型编码统计信息（抽样）")
        
        # 对大数据集抽样进行统计
        sample_size = min(500, n_snps * n_samples // 10)  # 减少采样数量
        sample_snp_indices = np.random.randint(0, n_snps, sample_size)
        sample_sample_indices = np.random.randint(0, n_samples, sample_size)
        
        # 批量读取采样数据点
        sample_values = []
        batch_size = 50  # 每次读取50个采样点
        
        for i in range(0, sample_size, batch_size):
            batch_end = min(i + batch_size, sample_size)
            batch_snps = sample_snp_indices[i:batch_end]
            batch_samples = sample_sample_indices[i:batch_end]
            
            for j in range(len(batch_snps)):
                s, idx = batch_snps[j], batch_samples[j]
                sample_values.append(normalized_genotypes[s, idx])
        
        # 计算统计值
        sample_array = np.array(sample_values)
        
        stats = {
            "min": float(np.min(sample_array)),
            "max": float(np.max(sample_array)),
            "mean": float(np.mean(sample_array)),
            "std": float(np.std(sample_array)),
            "sample_size": sample_size
        }
        
        # 添加非零值统计
        non_zero = sample_array[sample_array > 0]
        if len(non_zero) > 0:
            stats["non_zero"] = {
                "count": len(non_zero),
                "ratio": len(non_zero) / len(sample_array),
                "min": float(np.min(non_zero)),
                "max": float(np.max(non_zero)),
                "mean": float(np.mean(non_zero)),
                "std": float(np.std(non_zero))
            }
        
        return stats

    def _build_partition_index(self, snp_info: pd.DataFrame, feature_dim: int) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
        """
        废弃
        构建分区索引，用于批处理和模型训练
        使用GenomePartitioner进行SNP分组
        废弃
        
        Args:
            snp_info: SNP信息数据框
            feature_dim: 特征维度
            
        Returns:
            Tuple[np.ndarray, Dict[str, Dict[str, Any]]]: 分区索引数组和染色体位置信息
        """
        self.logger.info("构建SNP分区索引...")
        
        try:
            # 使用GenomePartitioner进行SNP分组
            from utils.reference_genome_reader import ReferenceGenomeReader, GenomePartitioner
            
            # 初始化ReferenceGenomeReader和GenomePartitioner
            reference_reader = ReferenceGenomeReader(self.data_loader.reference_genome_path)
            # 直接传递已加载的snp_info
            partitioner = GenomePartitioner(reference_reader, snp_info)
            
            # 获取分区配置 - 使用data_loader获取配置
            partition_config = self.data_loader.get_partition_config()
            default_max_partition_size = partition_config['max_size']  # 从配置获取最大分区大小
            
            # 计算SNP Transformer和Gene Transformer的参数量
            snp_transformer = self.data_loader.config.model.snp_transformer  # 第一个transformer块
            gene_transformer = self.data_loader.config.model.gene_transformer  # 第二个transformer块
            
            # 估算参数量
            snp_params = self._estimate_transformer_params(snp_transformer)
            gene_params = self._estimate_transformer_params(gene_transformer)
            
            # 计算分区大小
            n_snps = len(snp_info)
            
            # 确保不会除以0
            if snp_params <= 0:
                snp_params = 1
                
            # 计算比例
            ratio = gene_params / snp_params
            
            # 按公式计算分区大小: sqrt(SNP数量 / (Gene Transformer参数 / SNP Transformer参数))
            if ratio > 0 and n_snps > 0:
                max_partition_size = int(np.sqrt(n_snps / ratio))
                # 确保分区大小至少为1
                max_partition_size = max(1, max_partition_size)
            else:
                # 使用默认值作为后备
                max_partition_size = default_max_partition_size
            
            self.logger.info(f"计算的最大分区大小: {max_partition_size}")
            self.logger.info(f"SNP数量: {n_snps}, SNP Transformer参数量: {snp_params}, Gene Transformer参数量: {gene_params}, 参数比: {ratio:.2f}")
            
            # 进行SNP分组
            with timer("SNP分组", self.logger):
                partitions_info = partitioner.partition_genome(max_partition_size)
                self.logger.info(f"SNP分组完成，生成了{len(partitions_info)}个分区")
            
            # 构建分区索引数组
            partitions = []
            chrom_positions = {}
            
            # 处理每个分区
            for i, (chrom, start_pos, end_pos, snp_count) in enumerate(partitions_info):
                # 将分区信息转换为索引范围
                start_idx = sum([p[3] for p in partitions_info[:i]])  # 累加前面分区的SNP数量
                end_idx = start_idx + snp_count
                partitions.append([start_idx, end_idx])
                
                # 记录染色体位置信息
                chrom_key = str(chrom)
                if (chrom_key) not in chrom_positions:
                    chrom_positions[chrom_key] = {
                        'start': start_idx,
                        'end': end_idx,
                        'size': snp_count,
                        'feature_dim': feature_dim,
                        'partitions': []
                    }
                else:
                    # 更新染色体结束位置
                    chrom_positions[chrom_key]['end'] = end_idx
                    chrom_positions[chrom_key]['size'] += snp_count
                
                # 添加分区信息到染色体记录中
                partition_info = {
                    'local_start': 0,  # 相对于当前染色体的开始位置
                    'local_end': snp_count,
                    'global_start': start_idx,
                    'global_end': end_idx,
                    'size': snp_count,
                    'positions': {
                        'start': int(start_pos),
                        'end': int(end_pos)
                    }
                }
                chrom_positions[chrom_key]['partitions'].append(partition_info)
            
            # 转换为numpy数组
            partitions = np.array(partitions)
            
            # 获取SNP到分区的映射
            partition_indices = partitioner.get_partition_indices()
            
            # 打印分区统计信息
            self._print_partition_stats(partitions)
            
            return partitions, chrom_positions
        except Exception as e:
            self.logger.error(f"构建分区索引失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
                    
    def _analyze_block_features(self, features: np.ndarray, snp_info: pd.DataFrame) -> Dict[str, Any]:
        """
        分析基因块特征统计
        
        Args:
            features: 特征数组
            snp_info: SNP信息
            
        Returns:
            Dict[str, Any]: 特征统计信息
        """
        self.logger.info("分析基因块特征...")
        
        # 按染色体分组分析特征
        chrom_groups = snp_info.groupby('CHR')
        stats = {'by_chrom': {}}
        
        for chrom, group in chrom_groups:
            indices = group.index.tolist()
            chrom_features = features[indices]
            
            # 计算每个染色体的统计信息
            chrom_stats = {
                'count': len(indices),
                'mean': np.mean(chrom_features, axis=0).tolist(),
                'std': np.std(chrom_features, axis=0).tolist(),
                'min': np.min(chrom_features, axis=0).tolist(),
                'max': np.max(chrom_features, axis=0).tolist(),
            }
            
            stats['by_chrom'][str(chrom)] = chrom_stats
            
            # 分析特征分布
            self._analyze_feature_distribution(
                np.mean(chrom_features, axis=1),
                f"染色体{chrom}特征均值"
            )
        
        # 全局统计信息
        stats['global'] = {
            'count': features.shape[0],
            'mean': np.mean(features, axis=0).tolist(),
            'std': np.std(features, axis=0).tolist(),
            'min': np.min(features, axis=0).tolist(),
            'max': np.max(features, axis=0).tolist(),
        }
        
        return stats

    def _print_partition_stats(self, partitions: np.ndarray) -> None:
        """
        打印分区统计信息
        
        Args:
            partitions: 分区索引数组
        """
        lengths = partitions[:, 1] - partitions[:, 0]
        self.logger.info(f"分区统计:")
        self.logger.info(f"  总分区数: {len(partitions)}")
        self.logger.info(f"  最小长度: {lengths.min()}")
        self.logger.info(f"  最大长度: {lengths.max()}")
        self.logger.info(f"  平均长度: {lengths.mean():.2f}")
        self.logger.info(f"  中位数长度: {np.median(lengths):.2f}")

    def _analyze_feature_distribution(self, values: np.array, name: str) -> None:
        """
        分析特征分布并提供变换建议
        
        Args:
            values: 特征值数组
            name: 特征名称
        """
        # 排除零值和无效值以更好地分析分布
        non_zero = values[values > 0]
        non_zero_ratio = len(non_zero) / len(values) if len(values) > 0 else 0
        
        # 基本统计信息
        self.logger.info(f"{name}特征分析:")
        self.logger.info(f"  总值数量: {len(values)}")
        self.logger.info(f"  非零值数量: {len(non_zero)} ({non_zero_ratio:.2%})")
        
        if len(non_zero) > 0:
            # 计算分位数
            percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
            perc_values = np.percentile(non_zero, percentiles)
            perc_info = ", ".join([f"{p}%: {v:.6g}" for p, v in zip(percentiles, perc_values)])
            self.logger.info(f"  非零值分位数: {perc_info}")
            
            # 计算偏度指标
            mean_val = np.mean(non_zero)
            median_val = np.median(non_zero)
            std_val = np.std(non_zero)
            
            if std_val > 0:
                skew_estimate = 3*(mean_val - median_val) / std_val
                self.logger.info(f"  均值: {mean_val:.6g}, 中位数: {median_val:.6g}, 标准差: {std_val:.6g}")
                self.logger.info(f"  偏度估计: {skew_estimate:.4f}")
                
                # 提供变换建议
                if abs(skew_estimate) > 2:
                    if skew_estimate > 0:  # 右偏(正偏)
                        self.logger.info(f"  分布右偏，建议使用对数变换")
                    else:  # 左偏(负偏)
                        self.logger.info(f"  分布左偏，建议使用平方根变换")

    def _estimate_transformer_params(self, transformer_config) -> int:
        """
        估算 Transformer 模型的参数量
        
        Args:
            transformer_config: Transformer 配置对象
            
        Returns:
            int: 估计的参数量，默认为1
        """
        if not transformer_config:
            return 1
        
        try:
            # 获取基本模型参数
            # 尝试获取嵌入维度 - 可能在不同位置定义
            if hasattr(transformer_config, 'encoder') and hasattr(transformer_config.encoder, 'dim'):
                embed_dim = transformer_config.encoder.dim
            elif hasattr(transformer_config, 'encoder') and hasattr(transformer_config.encoder, 'attention') and \
                 hasattr(transformer_config.encoder.attention, 'num_heads') and \
                 hasattr(transformer_config.encoder.attention, 'd_attention'):
                # 修复：确保transformer_config.encoder存在且有attention属性
                # 如果维度未直接指定，尝试从头数和注意力维度计算
                embed_dim = transformer_config.encoder.attention.num_heads * transformer_config.encoder.attention.d_attention
            else:
                # 使用默认值
                embed_dim = 768
                
            # 获取层数和前馈层维度
            num_layers = transformer_config.encoder.num_layers
            ff_dim = transformer_config.encoder.ff_dim
            num_heads = transformer_config.encoder.attention.num_heads
            
            # 每个头的维度
            # 修复：确保embed_dim和num_heads大于0，避免除零错误
            head_dim = embed_dim // max(1, num_heads)
            
            # 计算参数量
            # 多头注意力层: 4 * head_dim * embed_dim (Q, K, V, Output 投影)
            attention_params = 4 * embed_dim * embed_dim  # 简化计算
            
            # 前馈网络: 2 * embed_dim * ff_dim
            ff_params = 2 * embed_dim * ff_dim
            
            # 层归一化: 2 * embed_dim
            ln_params = 4 * embed_dim
            
            # 总参数量
            total_params = num_layers * (attention_params + ff_params + ln_params)
            
            return max(1, total_params)
        except Exception as e:
            self.logger.warning(f"估算模型参数量时出错: {str(e)}，使用默认值")
            return 1

    def _save_features_to_disk(self, output_path: Path, feature_data: Dict[str, Any], feature_stats: Dict[str, Any]) -> None:
        """
        将特征数据保存到磁盘，采用节省空间的格式
        位置特征向量和基因型编码分别保存
        
        Args:
            output_path: 输出路径
            feature_data: 特征数据
            feature_stats: 特征统计信息
        """
        # 创建输出目录
        output_dir = output_path.parent
        os.makedirs(output_dir, exist_ok=True)
        
        # 分别保存位置特征和基因型编码
        position_path = output_path.with_name(f"{output_path.stem}_position").with_suffix(".npy")
        genotype_path = output_path.with_name(f"{output_path.stem}_genotype").with_suffix(".npy")
        stats_path = output_path.with_name(f"{output_path.stem}_stats").with_suffix(".json")
        
        self.logger.info(f"保存位置特征向量到: {position_path}")
        np.save(position_path, feature_data["position_features"])
        
        self.logger.info(f"保存基因型编码到: {genotype_path}")
        np.save(genotype_path, feature_data["genotype_features"])
        
        self.logger.info(f"保存特征统计信息到: {stats_path}")
        
        # 更新统计信息中的文件路径
        feature_stats["files"] = {
            "position_features": str(position_path),
            "genotype_features": str(genotype_path)
        }
        
        # 将统计信息保存为JSON格式
        import json
        with open(stats_path, 'w') as f:
            json.dump(feature_stats, f, indent=2)
        
        self.logger.info("所有数据保存完成")

    def _save_to_hdf5(self, output_path: Path, feature_data: Dict[str, Any], feature_stats: Dict[str, Any], 
                     partitions: np.ndarray, chromosomes: Dict[str, Dict[str, Any]]) -> None:
        """
        将特征数据和元数据保存为HDF5格式
        位置特征向量和基因型编码分别保存为不同的数据集
        
        Args:
            output_path: 输出路径
            feature_data: 特征数据
            feature_stats: 特征统计信息
            partitions: 分区索引数组
            chromosomes: 染色体位置信息
        """
        if not HAS_H5PY:
            self.logger.warning("未安装h5py库，将使用NumPy格式保存数据")
            self._save_features_to_disk(output_path, feature_data, feature_stats)
            return
        
        output_path = output_path.with_suffix('.h5')
        self.logger.info(f"保存数据到HDF5文件: {output_path}")
        
        try:
            import json
            with h5py.File(output_path, 'w') as f:
                # 创建主要组
                features_group = f.create_group('features')
                meta_group = f.create_group('meta')
                
                # 保存位置特征（所有样本共享）
                self.logger.info("保存位置特征向量...")
                features_group.create_dataset(
                    'position_features', 
                    data=feature_data["position_features"],
                    chunks=(min(1000, feature_data["n_snps"]), feature_data["position_dim"]),
                    compression="gzip", 
                    compression_opts=4
                )
                
                # 保存基因型编码
                self.logger.info("保存基因型编码...")
                # 使用块大小优化来提高访问效率
                features_group.create_dataset(
                    'genotype_features', 
                    data=feature_data["genotype_features"],
                    chunks=(min(1000, feature_data["n_snps"]), min(100, feature_data["n_samples"])),
                    compression="gzip", 
                    compression_opts=4
                )
                
                # 保存分区信息
                self.logger.info("保存分区索引...")
                meta_group.create_dataset('partitions', data=partitions)
                
                # 保存染色体信息（转换为JSON字符串）
                self.logger.info("保存染色体位置信息...")
                chromosomes_json = json.dumps(chromosomes)
                meta_group.create_dataset('chromosomes', data=np.string_(chromosomes_json))
                
                # 保存特征统计信息
                self.logger.info("保存特征统计信息...")
                stats_json = json.dumps(feature_stats)
                meta_group.create_dataset('stats', data=np.string_(stats_json))
                
                # 保存基本元数据
                meta_group.attrs['n_snps'] = feature_data["n_snps"]
                meta_group.attrs['n_samples'] = feature_data["n_samples"]
                meta_group.attrs['position_dim'] = feature_data["position_dim"]
                meta_group.attrs['feature_dim'] = feature_stats["feature_dim"]
                meta_group.attrs['creation_time'] = str(time.time())
                
            self.logger.info(f"HDF5文件保存完成: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存HDF5文件失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # 回退到NumPy保存方式
            self.logger.info("尝试使用NumPy格式保存数据...")
            self._save_features_to_disk(output_path, feature_data, feature_stats)

    def _save_numpy_fallback(self, output_path: Path, feature_data: Dict[str, Any], feature_stats: Dict[str, Any],
                            partitions: np.ndarray = None, chromosomes: Dict[str, Dict[str, Any]] = None) -> None:
        """
        当HDF5不可用时，使用NumPy格式分别保存位置特征和基因型编码（优化版）
        
        Args:
            output_path: 输出路径
            feature_data: 要保存的特征数据字典
            feature_stats: 特征统计信息
            partitions: 分区索引数组
            chromosomes: 染色体位置信息
        """
        # 直接调用优化后的保存方法
        self._save_features_to_disk(output_path, feature_data, feature_stats)
        
        # 额外保存分区和染色体信息（如果提供）
        if partitions is not None:
            partition_path = output_path.with_name(f"{output_path.stem}_partitions").with_suffix(".npy")
            self.logger.info(f"保存分区索引到: {partition_path}")
            np.save(partition_path, partitions)
        
        if chromosomes is not None:
            chrom_path = output_path.with_name(f"{output_path.stem}_chromosomes").with_suffix(".json")
            self.logger.info(f"保存染色体信息到: {chrom_path}")
            import json
            with open(chrom_path, 'w') as f:
                json.dump(chromosomes, f, indent=2)

    def _save_additional_data(self, output_path: Path, split_results: Dict[str, Any], phenotype_array: np.ndarray = None) -> None:
        """
        保存额外数据（索引、分区信息等）
        
        Args:
            output_path: 基础输出路径
            split_results: 包含索引和其他元数据的字典
            phenotype_array: 表型数据数组
        """
        # 保存分区信息
        if 'partitions' in split_results:
            partition_path = output_path.with_name(f"{output_path.stem}_partitions").with_suffix('.npy')
            np.save(partition_path, split_results['partitions'])
            self.logger.info(f"分区信息已保存至: {partition_path}")
        
        # 保存染色体位置信息
        if 'chromosomes' in split_results:
            import json
            chrom_path = output_path.with_name(f"{output_path.stem}_chromosomes").with_suffix('.json')
            with open(chrom_path, 'w') as f:
                json.dump(split_results['chromosomes'], f, indent=2)
            self.logger.info(f"染色体位置信息已保存至: {chrom_path}")
        
        # 保存划分索引
        if 'train_valid_indices' in split_results:
            indices_path = output_path.with_name(f"{output_path.stem}_indices").with_suffix('.npz')
            np.savez(
                indices_path,
                train_valid_indices=np.array(split_results['train_valid_indices']),
                test_indices=np.array(split_results['test_indices'])
            )
            self.logger.info(f"划分索引已保存至: {indices_path}")
            
            # 保存交叉验证折索引
            cv_path = output_path.with_name(f"{output_path.stem}_cv_folds").with_suffix('.npz')
            cv_data = {}
            for fold_idx, (train_idx, val_idx) in enumerate(split_results['cv_folds']):
                cv_data[f'fold_{fold_idx}_train'] = np.array(train_idx)
                cv_data[f'fold_{fold_idx}_val'] = np.array(val_idx)
            
            np.savez(cv_path, **cv_data)
            self.logger.info(f"交叉验证折索引已保存至: {cv_path}")
        
        # 保存表型数据（如果有）
        if phenotype_array is not None:
            pheno_path = output_path.with_name(f"{output_path.stem}_phenotypes").with_suffix('.npy')
            np.save(pheno_path, phenotype_array)
            self.logger.info(f"表型数据已保存至: {pheno_path}")
