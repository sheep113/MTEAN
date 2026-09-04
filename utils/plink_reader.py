from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from contextlib import contextmanager
import time
import sys

# 确保可以导入snp_utils
sys.path.append(str(Path(__file__).parent.parent.parent))
import utils.snp_utils as snp_utils

class DataValidationError(Exception):
    """数据验证错误"""
    pass

@contextmanager
def timer(description: str, logger: logging.Logger):
    """计时器上下文管理器"""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"{description}: {elapsed:.2f} seconds")

class PlinkReader:
    """PLINK文件读取器，处理PLINK格式文件的加载和处理"""
    
    def __init__(self, config, logger: Optional[logging.Logger] = None, n_jobs: int = 1):
        """
        初始化PLINK读取器
        
        Args:
            config: 配置对象
            logger: 可选的日志记录器
            n_jobs: 并行处理的作业数
        """
        self.config = config
        self.logger = logger or logging.getLogger("PlinkReader")
        self.n_jobs = n_jobs
        self.sample_ids = None  # 用于存储从FAM文件加载的样本ID
    
    def load_sample_ids(self, plink_prefix: Path) -> List[str]:
        """
        从FAM文件读取样本ID
        
        Args:
            plink_prefix: PLINK文件前缀
            
        Returns:
            样本ID列表
        """
        try:
            fam_path = f"{plink_prefix}.fam"
            fam_df = pd.read_csv(
                fam_path, 
                sep='\s+', 
                header=None,
                usecols=[0, 1],  # 只读取家族ID和个体ID
                names=['FID', 'IID']
            )
            
            # 使用FID和IID组合作为样本标识符，与PLINK的标准一致
            sample_ids = [f"{row.FID}_{row.IID}" for _, row in fam_df.iterrows()]
            self.sample_ids = sample_ids
            
            self.logger.info(f"从FAM文件加载了{len(sample_ids)}个样本ID")
            return sample_ids
            
        except Exception as e:
            self.logger.error(f"样本ID加载失败: {e}")
            raise
    
    def load_phenotype(self, pheno_path: Path, plink_prefix: Optional[Path] = None) -> pd.DataFrame:
        """
        读取表型数据，并确保与基因型样本匹配
        
        Args:
            pheno_path: 表型文件路径
            plink_prefix: 可选的PLINK文件前缀，用于验证样本匹配
            
        Returns:
            表型数据DataFrame（已按基因型样本顺序排列）
        """
        try:
            pheno_df = pd.read_csv(
                pheno_path,
                index_col=0,
                float_precision='high',
                header=0,
                sep="\t"
            )
            
            self._validate_phenotype_data(pheno_df)
            self._process_phenotype_data(pheno_df)
            
            # 如果提供了PLINK前缀，检查并确保表型数据与基因型样本匹配
            if plink_prefix is not None:
                pheno_df = self._match_phenotype_with_genotype(pheno_df, plink_prefix)
            
            return pheno_df
            
        except Exception as e:
            self.logger.error(f"表型数据加载失败: {e}")
            raise
    
    def _validate_phenotype_data(self, df: pd.DataFrame) -> None:
        """验证表型数据"""
        if df.empty:
            raise DataValidationError("表型数据为空")
            
        if df.isnull().all().any():
            raise DataValidationError("存在完全为空的表型列")
            
    def _process_phenotype_data(self, df: pd.DataFrame) -> None:
        """处理表型数据"""
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        self.logger.info(f"读取到的表型数: {len(df.columns)}")
        self.logger.info(f"表型名称: {', '.join(df.columns)}")
        self.logger.info(f"样本数: {len(df)}")
    
    def _match_phenotype_with_genotype(self, pheno_df: pd.DataFrame, plink_prefix: Path) -> pd.DataFrame:
        """
        确保表型数据与基因型样本匹配
        
        Args:
            pheno_df: 表型数据DataFrame
            plink_prefix: PLINK文件前缀
            
        Returns:
            重排后的表型数据DataFrame（按基因型样本顺序，缺失样本填充NA）
        """
        # 如果样本ID尚未加载，则加载
        if self.sample_ids is None:
            self.load_sample_ids(plink_prefix)
        
        # 获取表型数据的样本ID
        pheno_sample_ids = pheno_df.index.tolist()
        
        # 创建一个新的DataFrame，顺序与基因型样本一致，并初始化为NA
        final_pheno_df = pd.DataFrame(index=self.sample_ids, columns=pheno_df.columns)
        final_pheno_df[:] = "NA"  # 所有值初始化为NA
        
        # 初始化匹配信息
        match_info = {
            "total_genotype_samples": len(self.sample_ids),
            "total_phenotype_samples": len(pheno_sample_ids),
            "matching_method": "direct",
            "matching_samples": 0,
            "missing_samples": 0
        }
        
        # 首先尝试直接匹配
        common_samples = set(self.sample_ids) & set(pheno_sample_ids)
        
        # 如果直接匹配的结果不佳（匹配率低于10%），尝试智能匹配
        if len(common_samples) < 0.1 * len(self.sample_ids) and pheno_sample_ids:
            self.logger.warning("直接匹配样本ID效果不佳，尝试智能匹配...")
            
            # 检查样本ID命名模式
            is_pheno_subset = any(p_id in g_id for p_id in pheno_sample_ids[:5] for g_id in self.sample_ids[:5])
            is_geno_subset = any(g_id in p_id for g_id in self.sample_ids[:5] for p_id in pheno_sample_ids[:5])
            
            if is_pheno_subset and not is_geno_subset:
                # 表型ID可能是基因型ID的一部分
                self.logger.info("检测到表型ID可能是基因型ID的子集")
                id_map = {}
                for g_id in self.sample_ids:
                    for p_id in pheno_sample_ids:
                        if p_id in g_id:
                            id_map[g_id] = p_id
                            break
                
                if id_map:
                    match_info["matching_method"] = "pheno_subset"
                    match_info["matching_samples"] = len(id_map)
                    match_info["missing_samples"] = len(self.sample_ids) - len(id_map)
                    
                    self.logger.info(f"通过子集匹配到{len(id_map)}/{len(self.sample_ids)}个样本")
                    
                    # 按基因型样本顺序填充表型值
                    for g_id in self.sample_ids:
                        if g_id in id_map:
                            p_id = id_map[g_id]
                            final_pheno_df.loc[g_id] = pheno_df.loc[p_id]
            
            elif is_geno_subset and not is_pheno_subset:
                # 基因型ID可能是表型ID的一部分
                self.logger.info("检测到基因型ID可能是表型ID的子集")
                id_map = {}
                for p_id in pheno_sample_ids:
                    for g_id in self.sample_ids:
                        if g_id in p_id:
                            id_map[g_id] = p_id
                            break
                
                if id_map:
                    match_info["matching_method"] = "geno_subset"
                    match_info["matching_samples"] = len(id_map)
                    match_info["missing_samples"] = len(self.sample_ids) - len(id_map)
                    
                    self.logger.info(f"通过子集匹配到{len(id_map)}/{len(self.sample_ids)}个样本")
                    
                    # 按基因型样本顺序填充表型值
                    for g_id in self.sample_ids:
                        if g_id in id_map:
                            p_id = id_map[g_id]
                            final_pheno_df.loc[g_id] = pheno_df.loc[p_id]
            
            else:
                # 尝试通过提取可能的个体ID部分进行匹配
                self.logger.info("尝试从样本ID中提取并匹配可能的个体ID部分")
                
                # 从FAM文件直接读取FID和IID
                fam_path = f"{plink_prefix}.fam"
                fam_df = pd.read_csv(
                    fam_path, 
                    sep='\s+', 
                    header=None,
                    usecols=[0, 1],
                    names=['FID', 'IID']
                )
                
                # 创建IID到完整ID的映射
                iid_to_full_id = {row.IID: f"{row.FID}_{row.IID}" for _, row in fam_df.iterrows()}
                
                # 检查表型样本ID是否与IID匹配
                common_iids = set(iid_to_full_id.keys()) & set(pheno_sample_ids)
                
                if common_iids:
                    match_info["matching_method"] = "iid_only"
                    match_info["matching_samples"] = len(common_iids)
                    match_info["missing_samples"] = len(self.sample_ids) - len(common_iids)
                    
                    self.logger.info(f"通过个体ID匹配到{len(common_iids)}/{len(self.sample_ids)}个样本")
                    
                    # 按基因型样本顺序填充表型值
                    for iid in common_iids:
                        full_id = iid_to_full_id[iid]
                        if full_id in self.sample_ids:
                            final_pheno_df.loc[full_id] = pheno_df.loc[iid]
        
        else:
            # 使用直接匹配结果
            match_info["matching_method"] = "direct"
            match_info["matching_samples"] = len(common_samples)
            match_info["missing_samples"] = len(self.sample_ids) - len(common_samples)
            
            self.logger.info(f"直接匹配到{len(common_samples)}/{len(self.sample_ids)}个样本")
            
            # 按基因型样本顺序填充表型值
            for sample_id in self.sample_ids:
                if sample_id in common_samples:
                    final_pheno_df.loc[sample_id] = pheno_df.loc[sample_id]
        
        # 报告匹配情况
        self.logger.info(f"表型-基因型样本匹配情况:")
        self.logger.info(f"  - 总基因型样本数: {match_info['total_genotype_samples']}")
        self.logger.info(f"  - 总表型样本数: {match_info['total_phenotype_samples']}")
        self.logger.info(f"  - 匹配方法: {match_info['matching_method']}")
        self.logger.info(f"  - 匹配样本数: {match_info['matching_samples']}")
        self.logger.info(f"  - 缺失样本数: {match_info['missing_samples']}")
        
        # 如果没有匹配的样本，发出警告
        if match_info["matching_samples"] == 0:
            self.logger.warning("没有匹配的样本！所有表型值将设置为NA")
        
        return final_pheno_df

    def load_snp_info(self, plink_prefix: Path) -> pd.DataFrame:
        """
        读取SNP信息数据
        
        Args:
            plink_prefix: PLINK文件前缀
            
        Returns:
            SNP信息DataFrame
        """
        try:
            # 包含等位基因信息以便进行详细的基因型编码
            snp_info = pd.read_csv(
                f"{plink_prefix}.bim",
                sep='\s+',
                header=None,
                names=['chr', 'snp_id', 'genetic_dist', 'position', 'allele1', 'allele2'],
                dtype={
                    'chr': np.int32,
                    'snp_id': str,
                    'genetic_dist': np.float32,
                    'position': np.int32,
                    'allele1': str,
                    'allele2': str
                },
                engine='c'  # 使用C引擎加速
            )
            
            self.logger.info(f"SNP信息加载完成，形状: {snp_info.shape}")
            self.logger.info(f"染色体分布: {snp_info['chr'].value_counts().sort_index().to_dict()}")
            return snp_info
            
        except Exception as e:
            self.logger.error(f"SNP信息加载失败: {e}")
            raise
    
    def load_bed_data(self, plink_prefix: Path, SNPs_batch_size: int = 10000, 
                      Samples_batch_idx: int = 0, Samples_batch_size: Optional[int] = None,
                      n_jobs: Optional[int] = None,
                      missing_value_config: Optional[Dict[str, Any]] = None) -> tuple[np.ndarray, List[str]]:
        """
        读取和处理BED文件数据，支持样本批次处理

        Args:
            plink_prefix: PLINK文件前缀
            SNPs_batch_size: 每次处理的SNP数量（并行时的块大小），默认为10000
            Samples_batch_idx: 样本批次的起始索引，默认为0
            Samples_batch_size: 每批处理的样本数量，None表示处理所有剩余样本
            n_jobs: 并行处理的作业数，如果为None则使用初始化时设置的值
            missing_value_config: 缺失值处理配置字典 (例如 {'enable': True, 'method': 'mode'})

        Returns:
            tuple: (处理后的SNP数据数组 (独热编码格式), 对应的样本ID列表)
        """
        try:
            # 如果n_jobs未指定，使用类初始化时设置的值
            n_jobs_actual = n_jobs if n_jobs is not None else self.n_jobs
            self.logger.info(f"使用{n_jobs_actual}个作业进行并行处理")

            bed_path = f"{plink_prefix}.bed"
            with open(bed_path, 'rb') as f:
                magic = f.read(3)
                if magic != b'\x6c\x1b\x01':
                    raise DataValidationError("Invalid BED file format")

            # 调用更新后的 _process_bed_file，传递 missing_value_config
            return self._process_bed_file(
                bed_path,
                SNPs_batch_size,
                Samples_batch_idx,
                Samples_batch_size,
                n_jobs_actual,
                missing_value_config
            )

        except Exception as e:
            self.logger.error(f"BED数据加载失败: {str(e)}")
            raise

    def _process_bed_file(self, bed_path: str, SNPs_batch_size: int = 10000, 
                          Samples_batch_idx: int = 0, Samples_batch_size: Optional[int] = None, 
                          n_jobs: int = 1,
                          missing_value_config: Optional[Dict[str, Any]] = None) -> tuple[np.ndarray, List[str]]:
        """
        处理BED文件，支持样本批次处理，并进行独热编码

        Args:
            bed_path: BED文件路径
            SNPs_batch_size: 每次并行处理的SNP块大小，默认为10000
            Samples_batch_idx: 样本批次的起始索引，默认为0
            Samples_batch_size: 每批处理的样本数量，None表示处理所有剩余样本
            n_jobs: 并行处理的作业数
            missing_value_config: 缺失值处理配置字典

        Returns:
            tuple: (处理后的SNP数据数组 (独热编码格式), 对应的样本ID列表)
        """
        try:
            # 获取SNP和样本数量信息
            if not Path(bed_path).exists():
                raise FileNotFoundError(f"BED文件不存在: {bed_path}")
            
            # 从对应的BIM和FAM文件获取SNP和样本数量
            bim_path = bed_path.replace('.bed', '.bim')
            fam_path = bed_path.replace('.bed', '.fam')
            
            # 获取SNP数量和样本数量
            n_snps = sum(1 for _ in open(bim_path, 'r'))
            n_samples_total = sum(1 for _ in open(fam_path, 'r'))
            
            # 计算样本批次范围
            if Samples_batch_size is None:
                Samples_batch_size = n_samples_total - Samples_batch_idx
            
            # 确保批次索引和大小在有效范围内
            if Samples_batch_idx < 0 or Samples_batch_idx >= n_samples_total:
                raise ValueError(f"样本批次索引 {Samples_batch_idx} 超出范围 [0, {n_samples_total-1}]")
            
            if Samples_batch_idx + Samples_batch_size > n_samples_total:
                Samples_batch_size = n_samples_total - Samples_batch_idx
                self.logger.warning(f"样本批次大小超出范围，调整为 {Samples_batch_size}")
            
            batch_end_idx = Samples_batch_idx + Samples_batch_size
            
            # 加载对应批次的样本ID
            if self.sample_ids is None:
                self.load_sample_ids(Path(bed_path).with_suffix(''))
            
            batch_sample_ids = self.sample_ids[Samples_batch_idx:batch_end_idx]
            
            self.logger.info(f"处理样本批次 [{Samples_batch_idx}-{batch_end_idx-1}]，共{Samples_batch_size}个样本")
            self.logger.info(f"BED文件: {bed_path}, SNP数量: {n_snps}, 总样本数量: {n_samples_total}")
            
            # 获取SNP信息，用于编码
            snp_info = self.load_snp_info(Path(bed_path).with_suffix(''))
            allele1_array = snp_info['allele1'].values
            allele2_array = snp_info['allele2'].values
            
            with timer(f"处理{n_snps}个SNP的批次样本数据", self.logger):
                # 读取BED文件
                with open(bed_path, 'rb') as f:
                    # 检查文件头3字节
                    magic_bytes = f.read(3)
                    if magic_bytes != b'\x6c\x1b\x01':
                        raise DataValidationError(f"无效的BED文件格式: {bed_path}")
                    
                    # 计算每个SNP需要的字节数 (向上取整到4)
                    bytes_per_snp = (n_samples_total + 3) // 4
                    
                    # 创建用于存储所有SNP字节数据的数组
                    bed_data = np.zeros((n_snps, bytes_per_snp), dtype=np.uint8)
                    
                    # 读取所有SNP的字节数据
                    for i in range(n_snps):
                        # 定位到当前SNP的起始位置
                        f.seek(3 + i * bytes_per_snp)
                        # 读取当前SNP的所有样本数据
                        snp_bytes = f.read(bytes_per_snp)
                        bed_data[i, :len(snp_bytes)] = np.frombuffer(snp_bytes, dtype=np.uint8)
                
                # 解码SNP数据 - 利用decode_snps内部的并行处理
                with timer(f"解码SNP数据", self.logger):
                    decoded_data = snp_utils.decode_snps(
                        bed_data, 
                        n_snps, 
                        n_samples_total,
                        n_jobs=n_jobs,
                        chunk_size=SNPs_batch_size
                    )
                    
                    # 提取当前批次样本的数据
                    batch_decoded_data = decoded_data[:, Samples_batch_idx:batch_end_idx]
                
                # 编码SNP数据 (独热编码) - 利用batch_encode_genotypes内部的并行处理
                with timer(f"独热编码SNP数据", self.logger):
                    encoded_data = snp_utils.batch_encode_genotypes(
                        batch_decoded_data,
                        allele1_array,
                        allele2_array,
                        n_jobs=n_jobs,
                        chunk_size=SNPs_batch_size
                    )
                
                # 填充缺失值 (独热编码) - 利用fill_missing_values内部的并行处理
                with timer(f"填充缺失值 (独热编码)", self.logger):
                    filled_data = snp_utils.fill_missing_values(
                        encoded_data, 
                        config=missing_value_config,
                        n_jobs=n_jobs
                    )
                
                self.logger.info(f"批次样本SNP数据处理完成 (独热编码)，形状: {filled_data.shape}")
                return filled_data, batch_sample_ids
                
        except Exception as e:
            self.logger.error(f"BED文件处理失败: {e}", exc_info=True)
            raise
