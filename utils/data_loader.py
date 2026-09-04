from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple, List
from contextlib import contextmanager
import sys
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import gc
from tqdm import tqdm

# 检查h5py库是否可用
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logging.warning("未检测到h5py库，HDF5功能将不可用。请使用'pip install h5py'安装")

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config, ConfigValidationError, MICAnalysisConfig

# 导入拆分后的组件模块
from utils.config_manager import ConfigManager
from utils.hdf5_handler import HDF5Handler
from utils.plink_reader import PlinkReader, DataValidationError
from utils.reference_genome_reader import ReferenceGenomeReader
from utils.hdf5_dataset import HDF5Dataset

class DataLoaderError(Exception):
    """数据加载错误基类"""
    pass

class FileNotFoundError(DataLoaderError):
    """文件未找到错误"""
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

class ParallelProcessor:
    """并行处理工具类"""
    
    def __init__(self, n_jobs: int = 1, logger: Optional[logging.Logger] = None):
        """初始化并行处理器"""
        self.n_jobs = n_jobs
        self.logger = logger or logging.getLogger("ParallelProcessor")
    
    def process_map(self, func, items, desc: str = None):
        """并行映射处理"""
        if self.n_jobs <= 1:
            # 单进程模式
            results = []
            for item in tqdm(items, desc=desc):
                results.append(func(item))
            return results
        
        # 多进程模式
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            if desc:
                return list(tqdm(executor.map(func, items), total=len(items), desc=desc))
            else:
                return list(executor.map(func, items))
    
    def process_parallel_tasks(self, tasks, desc: str = None):
        """并行执行多个任务"""
        if self.n_jobs <= 1:
            # 单进程模式
            results = []
            for task in tqdm(tasks, desc=desc):
                results.append(task())
            return results
        
        # 多进程模式
        results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                results.append(future.result())
        return results

class DataLoader:
    """增强的数据加载器，支持配置文件和并行处理，基于组件化设计"""
    
    def __init__(self, config_path: Union[str, Path], logger: Optional[logging.Logger] = None, n_jobs: Optional[int] = None):
        """
        初始化数据加载器
        
        Args:
            config_path: 配置文件路径
            logger: 可选的日志记录器
            n_jobs: 并行处理的作业数，如果为None则使用配置文件中的值
        """
        # 如果提供了外部logger，则使用它，否则创建自己的logger
        self.logger = logger if logger is not None else self._setup_logger()
        
        # 检查HDF5支持
        if not HAS_H5PY:
            self.logger.warning("未检测到h5py库，HDF5功能将不可用。请使用'pip install h5py'安装")
        
        # 初始化配置管理器
        self.config_manager = ConfigManager(config_path, self.logger)
        self.config = self.config_manager.config
            
        # 设置并行处理参数 - 使用配置值或传入值
        config_n_jobs = self.config.preprocessing.file_processing.n_jobs
        self.n_jobs = n_jobs if n_jobs is not None else config_n_jobs
        self.n_jobs = min(max(1, self.n_jobs), multiprocessing.cpu_count())
        self.logger.info(f"数据加载器使用{self.n_jobs}个并行处理核心")
        
        # 确保输出目录存在
        self._ensure_output_directory()
        self._validate_paths()
        self._setup_cache()
        self._check_config_version()
        
        # 初始化组件
        self.plink_reader = PlinkReader(self.config, self.logger, n_jobs=self.n_jobs)
        
        # 初始化 ReferenceGenomeReader，并处理文件不存在的情况
        self.reference_reader: Optional[ReferenceGenomeReader] = None
        try:
            ref_genome_path_str = self.config_manager.get_reference_genome_path()
            if ref_genome_path_str and str(ref_genome_path_str).strip(): # 确保路径非空
                ref_genome_path_obj = Path(ref_genome_path_str)
                if ref_genome_path_obj.exists() and ref_genome_path_obj.is_file():
                    self.reference_reader = ReferenceGenomeReader(str(ref_genome_path_obj))
                else:
                    self.logger.warning(
                        f"参考基因组文件 '{ref_genome_path_str}' 不存在或不是一个有效文件。"
                        f" ReferenceGenomeReader 将不会被有效初始化。"
                    )
            else:
                self.logger.warning(
                    "参考基因组文件路径未在配置中提供或为空。"
                    " ReferenceGenomeReader 将不会被有效初始化。"
                )
        except FileNotFoundError: # 捕获 ReferenceGenomeReader 初始化时因文件不存在引发的错误
            self.logger.warning(
                f"尝试初始化 ReferenceGenomeReader 时，文件 '{ref_genome_path_str}' 未找到。"
                f" ReferenceGenomeReader 将不会被有效初始化。"
            )
        except Exception as e: # 捕获其他可能的初始化错误
            self.logger.error(
                f"初始化 ReferenceGenomeReader 时发生意外错误: {e}", exc_info=True
            )
            # self.reference_reader 保持为 None

        self.hdf5_handler = HDF5Handler(self.config, self.logger)
        
        # 初始化并行处理器
        self.parallel = ParallelProcessor(n_jobs=self.n_jobs, logger=self.logger)
        
    def _ensure_output_directory(self) -> None:
        """确保输出目录存在，如果不存在则创建"""
        output_dir = Path(self.config.preprocessing.file_processing.output_directory)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"输出目录已确认: {output_dir}")
        except Exception as e:
            self.logger.error(f"创建输出目录失败: {e}")
            raise
            
    @property
    def mic_config(self) -> MICAnalysisConfig:
        """获取MIC分析配置"""
        return self.config.preprocessing.mic_analysis
    
    @property
    def phenotype_path(self) -> Path:
        """获取表型文件路径，使用ConfigManager"""
        return self.config_manager.get_phenotype_path()
    
    @property
    def plink_prefix(self) -> Path:
        """获取PLINK文件前缀，使用ConfigManager"""
        return self.config_manager.get_plink_prefix()
        
    @property
    def reference_genome_path(self) -> Path:
        """获取参考基因组文件路径，使用ConfigManager"""
        return self.config_manager.get_reference_genome_path()
    
    @property
    def output_path(self) -> Path:
        """获取输出文件路径，使用ConfigManager"""
        return self.config_manager.get_output_path()
    
    @property
    def embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.config.model.embedding.dim
    
    @property
    def transformer_configs(self) -> Dict[str, Any]:
        """获取转换器配置"""
        return {
            'snp': self.config.model.snp_transformer,
            'gene': self.config.model.gene_transformer  # 从 dna 改为 gene
        }
        
    def get_attention_params(self, transformer_type: str = 'snp') -> Dict[str, Any]:
        """获取指定转换器的注意力参数"""
        return self.config_manager.get_attention_params(transformer_type)
        
    def get_pooling_config(self, transformer_type: str = 'snp') -> Dict[str, Any]:
        """获取指定转换器的池化配置"""
        return self.config_manager.get_pooling_config(transformer_type)
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('DataLoader')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def _setup_cache(self) -> None:
        """初始化缓存"""
        self._cache: Dict[str, Any] = {}
        
    def _validate_paths(self) -> None:
        """验证输入路径 - 使用委托给ConfigManager"""
        self.config_manager.validate_paths()
            
    def _check_config_version(self):
        """检查配置版本兼容性"""
        self.config_manager.check_config_version()
            
    def load_phenotype(self) -> pd.DataFrame:
        """读取表型数据，使用缓存优化"""
        if 'phenotype' in self._cache:
            return self._cache['phenotype']
            
        with timer("Reading phenotype data", self.logger):
            try:
                pheno_df = self.plink_reader.load_phenotype(self.phenotype_path)
                self._cache['phenotype'] = pheno_df
                return pheno_df
                
            except Exception as e:
                self.logger.error(f"表型数据加载失败: {e}")
                raise DataLoaderError(f"表型数据加载失败: {e}")
                
    def load_snp_info(self) -> pd.DataFrame:
        """读取SNP信息数据，使用缓存优化"""
        if 'snp_info' in self._cache:
            return self._cache['snp_info']
            
        with timer("Reading SNP info", self.logger):
            try:
                snp_info = self.plink_reader.load_snp_info(self.plink_prefix)
                self._cache['snp_info'] = snp_info
                return snp_info
                
            except Exception as e:
                self.logger.error(f"SNP信息加载失败: {e}")
                raise DataLoaderError(f"SNP信息加载失败: {e}")

    def load_bed_data_all(self, chunk_size: Optional[int] = None, parallel: bool = True, n_jobs: Optional[int] = None) -> np.ndarray:
        """
        读取和处理BED文件数据
        
        Args:
            chunk_size: 块大小，如果为None则使用配置中的值
            parallel: 是否启用并行处理
            n_jobs: 并行处理的作业数，如果为None则使用实例化时设置的值
            
        Returns:
            处理后的SNP数据
        """
        # 使用配置中的chunk_size，如果未提供则默认使用
        SNPs_batch_size = self.config.preprocessing.file_processing.SNPs_batch_size
        Samples_batch_size = self.config.preprocessing.file_processing.Samples_batch_size
        n_jobs = self.config.preprocessing.file_processing.n_jobs
        
        # 如果n_jobs未指定，使用实例化时设置的值
        n_jobs_actual = n_jobs if n_jobs is not None else self.n_jobs
        
        with timer("Processing SNPs", self.logger):
            try:
                return self.plink_reader.load_bed_data(
                    self.plink_prefix, 
                    SNPs_batch_size, 
                    Samples_batch_idx=0,
                    Samples_batch_size=None, 
                    n_jobs=n_jobs_actual
                )
                
            except Exception as e:
                self.logger.error(f"BED数据加载失败: {e}")
                raise DataLoaderError(f"BED数据加载失败: {e}")
    
    def load_bed_data(self, output_file: Optional[Path] = None, samples_per_batch: Optional[int] = None) -> Path:
        """
        分批次加载BED文件数据并保存为HDF5格式，以减少内存占用。
        现在支持独热编码和缺失值填充。

        Args:
            output_file: 输出HDF5文件路径，默认使用配置中的路径
            samples_per_batch: 每批处理的样本数量，默认使用配置中的值

        Returns:
            保存的HDF5文件路径
        """
        from utils.preprocess_utils import PreprocessUtils

        # 使用配置中的批处理参数
        config_batch_size = self.config.preprocessing.file_processing.Samples_batch_size
        samples_per_batch = samples_per_batch or config_batch_size

        # 如果样本批次大小未指定，使用默认值
        if not samples_per_batch or samples_per_batch <= 0:
            samples_per_batch = 1000
            self.logger.info(f"未指定样本批次大小，使用默认值: {samples_per_batch}")

        # 确定输出文件路径
        if output_file is None:
            output_dir = Path(self.config.preprocessing.file_processing.output_directory)
            # 使用配置中的 output_file 作为基础文件名
            base_filename = self.config.preprocessing.file_processing.output_file
            # 确保文件名有 .h5 后缀
            if not base_filename.endswith(".h5"):
                base_filename += ".h5"
            output_file = output_dir / base_filename
            output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"开始分批处理BED数据，输出文件: {output_file}")

        # 检查是否已存在输出文件，如存在则可能进行追加模式
        file_exists = output_file.exists()
        if file_exists:
            user_input = input(f"输出文件已存在: {output_file}，是否覆盖? (y/n): ").strip().lower()
            if user_input == 'y':
                try:
                    output_file.unlink()
                except Exception as e:
                    self.logger.error(f"无法删除现有文件: {e}")
                    raise
            else:
                output_file = output_file.with_name(f"{output_file.stem}_new{output_file.suffix}")
                self.logger.info(f"将使用新文件名: {output_file}")

        # 1. 首先加载所有样本ID，确保与表型数据一致的顺序
        with timer("加载样本ID", self.logger):
            if not hasattr(self.plink_reader, 'sample_ids') or self.plink_reader.sample_ids is None:
                sample_ids = self.plink_reader.load_sample_ids(self.plink_prefix)
            else:
                sample_ids = self.plink_reader.sample_ids
            total_samples = len(sample_ids)
            self.logger.info(f"总样本数: {total_samples}")

        # 2. 读取表型数据，并匹配
        with timer("加载并匹配表型数据", self.logger):
            # 读取原始表型数据
            # 使用plink_reader的方法加载并匹配表型数据，确保与样本ID顺序一致
            phenotype_df = self.plink_reader.load_phenotype(self.phenotype_path, self.plink_prefix)

        # 3. 获取SNP信息
        with timer("加载SNP信息", self.logger):
            snp_info = self.load_snp_info()
            total_snps = len(snp_info)
            self.logger.info(f"总SNP数: {total_snps}")

        # 4. 计算批次数量
        n_batches = (total_samples + samples_per_batch - 1) // samples_per_batch
        self.logger.info(f"将分 {n_batches} 批处理数据，每批 {samples_per_batch} 个样本")

        # 创建预处理工具实例用于统计分析
        preprocess_utils = PreprocessUtils(logger=self.logger, n_jobs=self.n_jobs)

        # 获取缺失值处理配置
        missing_handling_dc = self.config.preprocessing.missing_value_handling
        missing_value_config = {
            'enable': missing_handling_dc.enable,
            'method': missing_handling_dc.method
        }
        self.logger.info(f"缺失值处理配置: {missing_value_config}")

        # 检查参考基因组文件是否存在，决定位置特征生成策略
        ref_genome_available = False
        position_dim = 6  # 默认位置特征维度
        
        # 获取参考基因组路径
        ref_genome_path = self.reference_genome_path
        if ref_genome_path and Path(ref_genome_path).exists():
            try:
                # 尝试初始化ReferenceGenomeReader以验证文件有效性
                test_reader = ReferenceGenomeReader(str(ref_genome_path))
                ref_genome_available = True
                self.logger.info(f"参考基因组文件可用: {ref_genome_path}")
            except Exception as e:
                self.logger.warning(f"参考基因组文件无效: {e}")
                ref_genome_available = False
        else:
            self.logger.warning(f"参考基因组文件不存在或路径为空: {ref_genome_path}")
            ref_genome_available = False
    
        if not ref_genome_available:
            self.logger.info("将使用简化的位置特征（无基因区域信息）")

        # 创建一个空的HDF5文件并设置必要的组
        if HAS_H5PY:
            with h5py.File(output_file, 'w') as f:
                # 创建特征组和元数据组
                f.create_group('features')
                meta_group = f.create_group('meta')
                meta_group.attrs['total_snps'] = total_snps
                meta_group.attrs['total_samples'] = total_samples
                meta_group.attrs['creation_time'] = str(time.time())
                meta_group.attrs['encoding'] = 'one-hot-10d' # 记录编码方式
                meta_group.attrs['ref_genome_available'] = ref_genome_available  # 记录参考基因组可用性

                # 处理并保存表型数据，确保类型兼容
                self.logger.info("处理表型数据，确保类型兼容性并保存NA值信息")

                # 转换表型数据为浮点数，NaN值会被保留
                pheno_float = phenotype_df.astype(float)

                # 保存表型数据
                f.create_dataset(
                    'phenotypes',
                    data=pheno_float.values,
                    compression="gzip",
                    compression_opts=4
                )

                # 创建并保存NA值掩码（True表示是NA值）
                na_mask = phenotype_df.isna().values
                if np.any(na_mask):
                    self.logger.info(f"表型数据中存在NA值，创建掩码数组")
                    f.create_dataset(
                        'phenotypes_na_mask',
                        data=na_mask,
                        compression="gzip",
                        compression_opts=4
                    )
                    # 保存每列的NA值比例，便于后续分析
                    na_ratio = na_mask.mean(axis=0)
                    na_stats = f.create_group('phenotype_na_stats')
                    na_stats.create_dataset('na_ratio_by_column', data=na_ratio)

                    # 记录哪些列含有NA值
                    cols_with_na = np.where(na_mask.any(axis=0))[0]
                    na_stats.create_dataset('columns_with_na', data=cols_with_na)

                    # 记录日志中显示NA信息
                    for i, col in enumerate(phenotype_df.columns):
                        if i in cols_with_na:
                            na_count = na_mask[:, i].sum()
                            self.logger.info(f"列 '{col}' 包含 {na_count} 个NA值 ({na_count/len(phenotype_df):.2%})")

                # 保存表型名称
                pheno_names = phenotype_df.columns.tolist()
                dt = h5py.special_dtype(vlen=str)
                pheno_names_dset = f.create_dataset(
                    'phenotype_names',
                    shape=(len(pheno_names),),
                    dtype=dt
                )
                for i, name in enumerate(pheno_names):
                    pheno_names_dset[i] = name

                # 保存样本ID索引 - 这是很重要的，后续数据集划分需要使用
                dt = h5py.special_dtype(vlen=str)
                sample_ids_dset = f.create_dataset(
                    'sample_ids',
                    shape=(total_samples,),
                    dtype=dt
                )
                for i, sid in enumerate(sample_ids):
                    sample_ids_dset[i] = sid
        else:
            self.logger.warning("未安装h5py库，无法创建HDF5文件")
            raise ImportError("需要安装h5py库以支持HDF5格式")

        # 5. 循环处理每个批次
        for batch_idx in range(n_batches):
            batch_start = batch_idx * samples_per_batch
            batch_end = min(batch_start + samples_per_batch, total_samples)
            batch_size = batch_end - batch_start

            self.logger.info(f"处理批次 {batch_idx+1}/{n_batches}: 样本 {batch_start}-{batch_end-1}")

            # 使用上下文管理器监控资源
            with self._monitor_resources(f"批次 {batch_idx+1} 处理"):
                # 5.1 读取、解码、编码并填充当前批次的SNP数据
                try:
                    with timer(f"读取并处理批次 {batch_idx+1} SNP数据", self.logger):
                        # 调用 plink_reader.load_bed_data，它现在返回独热编码且填充后的数据
                        snp_data, batch_sample_ids = self.plink_reader.load_bed_data(
                            self.plink_prefix,
                            SNPs_batch_size=self.config.preprocessing.file_processing.SNPs_batch_size,
                            Samples_batch_idx=batch_start,
                            Samples_batch_size=batch_size,
                            n_jobs=self.n_jobs,
                            missing_value_config=missing_value_config # 传递缺失值配置
                        )
                        # snp_data 的形状现在是 [n_snps, batch_size, 10]

                    # 验证当前批次的样本ID与预期一致
                    expected_ids = sample_ids[batch_start:batch_end]
                    if not all(a == b for a, b in zip(batch_sample_ids, expected_ids)):
                        self.logger.warning(f"批次{batch_idx+1}的样本ID与预期顺序不一致")

                    # 5.2 保存当前批次数据到HDF5文件
                    with timer(f"保存批次 {batch_idx+1} 数据", self.logger):
                        # 追加到HDF5文件
                        with h5py.File(output_file, 'a') as f:
                            features_group = f['features']

                            # 检查并创建数据集
                            if 'genotype_features' not in features_group:
                                # 首个批次，创建数据集
                                genotype_dset = features_group.create_dataset(
                                    'genotype_features',
                                    shape=(total_snps, batch_size, 10), # 更新形状
                                    maxshape=(total_snps, total_samples, 10),  # 更新maxshape
                                    chunks=(min(1000, total_snps), min(100, batch_size), 10), # 更新chunks
                                    dtype=np.float32, # 保持float32
                                    compression="gzip",
                                    compression_opts=4
                                )
                                # 首次写入数据
                                genotype_dset[:, :batch_size, :] = snp_data # 更新写入逻辑
                            else:
                                # 非首个批次，扩展并追加数据
                                genotype_dset = features_group['genotype_features']
                                current_size = genotype_dset.shape[1]
                                if current_size + batch_size > total_samples:
                                    self.logger.warning(f"当前批次加上已处理的样本数量({current_size + batch_size})超过总样本数({total_samples})，将调整")
                                    batch_size = total_samples - current_size
                                    batch_end = batch_start + batch_size  # 更新batch_end以匹配新的batch_size
                                    snp_data = snp_data[:, :batch_size, :] # 调整snp_data大小

                                if batch_size > 0:
                                    genotype_dset.resize(current_size + batch_size, axis=1) # axis=1 是样本维度
                                    genotype_dset[:, current_size:current_size + batch_size, :] = snp_data # 更新写入逻辑

                            # 第一批次时保存位置特征（所有批次共享同一份位置特征）
                            if batch_idx == 0 and 'position_features' not in features_group:
                                # 生成位置特征向量
                                self.logger.info("生成位置特征向量")
                                
                                if ref_genome_available:
                                    # 使用参考基因组生成完整的位置特征
                                    try:
                                        reference_reader = ReferenceGenomeReader(str(ref_genome_path))
                                        position_vectors = reference_reader.process_snps(snp_info)
                                        position_array = np.array(position_vectors, dtype=np.float32)
                                        self.logger.info("使用参考基因组生成了完整的位置特征")
                                    except Exception as e:
                                        self.logger.error(f"使用参考基因组生成位置特征失败: {e}")
                                        ref_genome_available = False
                                
                                if not ref_genome_available:
                                    # 生成简化的位置特征（无基因区域信息）
                                    self.logger.info("生成简化的位置特征（无基因区域信息）")
                                    position_array = self._generate_simplified_position_features(snp_info)

                                # 保存位置特征
                                features_group.create_dataset(
                                    'position_features',
                                    data=position_array,
                                    chunks=(min(1000, total_snps), position_dim),
                                    compression="gzip",
                                    compression_opts=4
                                )

                                # 添加位置特征的统计分析
                                with timer("对SNP位置特征进行统计分析", self.logger):
                                    # 打印基本信息统计
                                    self.logger.info(f"总样本数: {total_samples}")
                                    self.logger.info(f"总SNP数: {total_snps}")
                                    self.logger.info(f"位置特征维度: {position_dim}")

                                    # 打印位置特征基本信息
                                    self.logger.info(f"位置特征维度分布情况:")
                                    for dim in range(position_dim):
                                        dim_values = position_array[:, dim]
                                        non_zero = np.count_nonzero(dim_values)

                                        self.logger.info(f"  维度 #{dim}:")
                                        self.logger.info(f"    非零值数量: {non_zero} ({non_zero/total_snps:.2%})")
                                        self.logger.info(f"    值范围: [{np.min(dim_values):.4f}, {np.max(dim_values):.4f}]")
                                        self.logger.info(f"    均值: {np.mean(dim_values):.4f}, 标准差: {np.std(dim_values):.4f}")

                    # 5.3 如果是最后一个批次，对部分样本进行统计分析 (注意：此部分可能需要调整以适应独热编码)
                    if batch_idx == n_batches - 1:
                        with timer("对最后批次样本进行统计分析 (独热编码)", self.logger):
                            import random
                            num_samples_to_analyze = min(5, len(batch_sample_ids))
                            sample_indices = random.sample(range(len(batch_sample_ids)), num_samples_to_analyze)

                            self.logger.info(f"对最后批次随机选取的{num_samples_to_analyze}个样本进行统计分析 (基于独热编码)")

                            for idx in sample_indices:
                                sample_id = batch_sample_ids[idx]
                                # sample_snps 的形状是 [n_snps, 10]
                                sample_snps_one_hot = snp_data[:, idx, :]

                                # 将独热编码转回标量值 (0-9) 以进行简单统计
                                # 注意：全零向量（原始缺失值）会被argmax错误地映射为0
                                # 更准确的统计需要考虑原始缺失值信息，或直接分析独热向量分布
                                sample_snps_scalar = np.argmax(sample_snps_one_hot, axis=1)
                                missing_mask = np.all(sample_snps_one_hot == 0, axis=1)
                                num_missing = np.sum(missing_mask)

                                self.logger.info(f"样本 {sample_id} 的SNP统计信息 (基于标量转换):")
                                if num_missing > 0:
                                    self.logger.info(f"  原始缺失值数量 (填充前): {num_missing} ({num_missing/total_snps:.2%})")

                                if total_snps > num_missing:
                                    valid_scalars = sample_snps_scalar[~missing_mask]
                                    self.logger.info(f"  标量值范围 (0-9): [{np.min(valid_scalars)}, {np.max(valid_scalars)}]")
                                    self.logger.info(f"  标量值均值: {np.mean(valid_scalars):.4f}")
                                    self.logger.info(f"  标量值标准差: {np.std(valid_scalars):.4f}")
                                    # 可以添加更复杂的独热向量分布统计
                                else:
                                    self.logger.info("  所有SNP均为原始缺失值。")

                    # 5.4 清理内存
                    del snp_data, batch_sample_ids
                    gc.collect()

                except Exception as e:
                    self.logger.error(f"处理批次 {batch_idx+1} 时出错: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    raise

        # 6. 处理完成后，保存最终的统计信息
        with timer("保存最终统计信息", self.logger):
            with h5py.File(output_file, 'a') as f:
                meta_group = f['meta']
                meta_group.attrs['completed'] = True
                meta_group.attrs['completion_time'] = str(time.time())

                # 添加一些有用的统计信息
                stats_group = meta_group.create_group('stats')
                stats_group.attrs['total_snps'] = total_snps
                stats_group.attrs['total_samples'] = total_samples
                stats_group.attrs['phenotype_columns'] = len(phenotype_df.columns)

                # 添加数据集划分需要的信息
                # 根据配置计算默认的训练/验证/测试集划分
                split_config = self.config.preprocessing.data_split

                # 检查是否启用数据划分
                if split_config.enable:
                    self.logger.info("根据配置执行数据划分...")
                    # 生成随机索引进行划分
                    rng = np.random.RandomState(split_config.random_seed)
                    indices = np.arange(total_samples)
                    rng.shuffle(indices)

                    # 首先分离出测试集
                    test_size = int(total_samples * split_config.test_ratio)
                    train_valid_indices = indices[:-test_size]
                    test_indices = indices[-test_size:]

                    # 保存划分索引
                    f.create_dataset('train_valid_indices', data=train_valid_indices)
                    f.create_dataset('test_indices', data=test_indices)

                    # 如果启用交叉验证，也预先计算交叉验证折
                    if split_config.cross_validation.enable:
                        self.logger.info("根据配置计算交叉验证折...")
                        n_splits = split_config.cross_validation.n_splits
                        cv_group = f.create_group('cv_folds')

                        if split_config.cross_validation.shuffle:
                            rng = np.random.RandomState(split_config.cross_validation.cv_random_seed)
                            rng.shuffle(train_valid_indices)

                        # 计算每折的大小
                        fold_size = len(train_valid_indices) // n_splits

                        # 创建交叉验证折
                        for i in range(n_splits):
                            start = i * fold_size
                            end = start + fold_size if i < n_splits - 1 else len(train_valid_indices)
                            valid_idx = train_valid_indices[start:end]
                            train_idx = np.concatenate([
                                train_valid_indices[:start],
                                train_valid_indices[end:]
                            ])

                            # 保存当前折
                            fold_group = cv_group.create_group(f'fold_{i}')
                            fold_group.create_dataset('train', data=train_idx)
                            fold_group.create_dataset('valid', data=valid_idx)
                    else:
                        self.logger.info("交叉验证未启用，跳过计算交叉验证折。")
                else:
                    self.logger.info("数据划分未启用 (data_split.enable is false)，跳过划分步骤。")

        self.logger.info(f"所有批次处理完成，数据已保存到: {output_file}")
        return output_file

    def normalize_snp_features(self, snp_data: np.ndarray, snp_info: pd.DataFrame, 
                              block_map: Optional[Dict[int, Dict[str, float]]] = None) -> np.ndarray:
        """编码并归一化SNP数据特征 - 委托给PlinkReader"""
        return self.plink_reader.normalize_snp_features(snp_data, snp_info, block_map)

    def generate_feature_statistics(self, processed_data: np.ndarray) -> Dict[str, Any]:
        """生成特征统计信息 - 委托给PlinkReader"""
        return self.plink_reader.generate_feature_statistics(processed_data)
        
    def split_dataset_indices(self, n_samples: int, phenotypes: np.ndarray) -> Dict[str, np.ndarray]:
        """
        根据配置文件划分数据集，只返回索引（不处理特征数据）
        
        Args:
            n_samples: 样本总数
            phenotypes: 表型数据数组
            
        Returns:
            包含不同数据集划分索引的字典
        """
        split_config = self.config.preprocessing.data_split
        
        # 检查样本数和表型数据样本数是否匹配
        if phenotypes.shape[0] != n_samples:
            msg = f"样本数({n_samples})与表型数据样本数({phenotypes.shape[0]})不匹配"
            self.logger.error(msg)
            raise DataValidationError(msg)
            
        # 生成随机索引
        rng = np.random.RandomState(split_config.random_seed)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        
        # 首先分离出测试集
        test_size = int(n_samples * split_config.test_ratio)
        train_valid_indices = indices[:-test_size]
        test_indices = indices[-test_size:]
        
        # 创建交叉验证折
        cv_folds = []
        if split_config.cross_validation.enable:
            n_splits = split_config.cross_validation.n_splits
            if split_config.cross_validation.shuffle:
                rng = np.random.RandomState(split_config.cross_validation.cv_random_seed)
                rng.shuffle(train_valid_indices)
                
            # 计算每折的大小
            fold_size = len(train_valid_indices) // n_splits
            
            # 创建交叉验证折
            for i in range(n_splits):
                start = i * fold_size
                end = start + fold_size if i < n_splits - 1 else len(train_valid_indices)
                valid_idx = train_valid_indices[start:end]
                train_idx = np.concatenate([
                    train_valid_indices[:start],
                    train_valid_indices[end:]
                ])
                cv_folds.append((train_idx, valid_idx))
                
        return {
            'train_valid_indices': train_valid_indices,
            'test_indices': test_indices,
            'cv_folds': cv_folds,
            'phenotypes': phenotypes
        }
        
    def split_dataset(self, data: np.ndarray, phenotypes: np.ndarray) -> Dict[str, np.ndarray]:
        """
        根据配置划分数据集
        
        Args:
            data: 特征数据
            phenotypes: 表型数据
            
        Returns:
            包含不同数据集划分和相应数据的字典
        """
        n_samples = data.shape[1]
        
        # 调用索引划分方法
        split_indices = self.split_dataset_indices(n_samples, phenotypes)
        
        # 合并特征数据和索引
        return {
            'processed_data': data,
            'train_valid_indices': split_indices['train_valid_indices'],
            'test_indices': split_indices['test_indices'], 
            'cv_folds': split_indices['cv_folds'],
            'phenotypes': phenotypes
        }
        
    def save_processed_data(self, split_results: Dict[str, Any]) -> None:
        """保存预处理后的数据，委托给HDF5Handler"""
        return self.hdf5_handler.save_processed_data(split_results)
    
    def save_hdf5_data(self, split_results: Dict[str, Any]) -> None:
        """保存预处理后的数据为HDF5格式，委托给HDF5Handler"""
        return self.hdf5_handler.save_hdf5_data(split_results)
    
    def load_hdf5_data(self, h5_file: Union[str, Path], 
                      dataset_name: str = 'data', 
                      indices: Optional[np.ndarray] = None) -> HDF5Dataset:
        """加载HDF5格式的数据，委托给HDF5Handler"""
        return self.hdf5_handler.load_hdf5_data(h5_file, dataset_name, indices)
        
    def create_data_loaders(self, h5_file: Union[str, Path], batch_size: int = 32, 
                         num_workers: int = 4, shuffle_train: bool = True):
        """从HDF5文件创建数据加载器，委托给HDF5Handler"""
        return self.hdf5_handler.create_data_loaders(
            h5_file, batch_size, num_workers, shuffle_train, self.config
        )
        
    def print_data_validation(self, processed_data: np.ndarray, phenotypes: np.ndarray, 
                             samples_to_show: int = 3, snps_to_show: int = 5) -> None:
        """打印数据验证信息 - 保留原有功能"""
        # 确定要展示的样本数和SNP数
        n_samples_to_show = min(samples_to_show, processed_data.shape[1])
        n_snps_to_show = min(snps_to_show, processed_data.shape[0])
        
        # 转置为 [n_samples, n_snps, features] 方便展示
        transposed_data = np.transpose(processed_data, (1, 0, 2))
        
        # 为每个样本打印信息
        for i in range(n_samples_to_show):
            self.logger.info(f"\n样本 #{i} 数据验证:")
            
            # 打印SNP向量化值
            self.logger.info(f"前 {n_snps_to_show} 个SNP的向量化值:")
            for j in range(n_snps_to_show):
                feature_values = transposed_data[i, j]
                feature_str = ", ".join([f"{val:.4f}" for val in feature_values])
                self.logger.info(f"  SNP #{j}: [{feature_str}]")
                
            # 打印表型值
            if i < phenotypes.shape[0]:
                if phenotypes.ndim == 1:
                    pheno_str = f"{phenotypes[i]:.4f}"
                else:
                    pheno_str = ", ".join([f"{val:.4f}" for val in phenotypes[i]])
                self.logger.info(f"表型值: {pheno_str}")
            else:
                self.logger.warning(f"样本 #{i} 没有对应的表型值")
                
    def print_feature_statistics(self, stats: Dict[str, Any]) -> None:
        """打印特征统计信息"""
        self.plink_reader.print_feature_statistics(stats)

    def load_reference_genome(self, snp_info: Optional[pd.DataFrame] = None) -> Dict[int, Dict[str, float]]:
        """读取并解析参考基因组BED文件，委托给ReferenceGenomeReader"""
        # 根据新的ReferenceGenomeReader API调整
        if snp_info is None:
            snp_info = self.load_snp_info()
        
        try:
            return self.reference_reader.process_snps(self.plink_prefix)
        except Exception as e:
            self.logger.error(f"参考基因组加载失败: {e}")
            raise DataLoaderError(f"参考基因组加载失败: {e}")

    def check_input_files(self) -> None:
        """检查输入文件是否存在，委托给ConfigManager"""
        self.config_manager.check_input_files()

    def get_partition_config(self) -> Dict[str, Any]:
        """获取分区配置，委托给ConfigManager"""
        return self.config_manager.get_partition_config()

    def _get_memory_usage(self):
        """获取当前内存使用情况"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # 转换为MB
        except ImportError:
            return 0  # 如果psutil不可用

    @contextmanager
    def _monitor_resources(self, task_name: str):
        """监控任务执行过程中的资源使用情况"""
        start_time = time.time()
        start_mem = self._get_memory_usage()
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            end_mem = self._get_memory_usage()
            mem_diff = end_mem - start_mem
            
            self.logger.info(f"{task_name} - 耗时: {elapsed:.2f}秒, "
                             f"内存使用: {end_mem:.1f}MB (变化: {mem_diff:+.1f}MB)")
    
    def _generate_simplified_position_features(self, snp_info: pd.DataFrame) -> np.ndarray:
        """
        当参考基因组不可用时，生成简化的位置特征
    
        Args:
            snp_info: SNP信息数据框
        
        Returns:
            简化的位置特征数组，形状为 [n_snps, 6]
        """
        n_snps = len(snp_info)
        position_features = np.zeros((n_snps, 6), dtype=np.float64)
    
        # 填充基本位置信息
        position_features[:, 0] = snp_info['chr'].astype(float)  # 染色体号
        position_features[:, 1] = snp_info['position'].astype(float)  # 位置
    
        # 其余维度设为默认值：
        # 维度2: 区间中心 - 使用位置本身
        position_features[:, 2] = snp_info['position'].astype(float)
    
        # 维度3: 区间长度 - 设为固定值（如1000bp）
        position_features[:, 3] = 1000.0
    
        # 维度4: SNP密度 - 设为固定值（如0.001）
        position_features[:, 4] = 0.001
    
        # 维度5: 是否基因区域 - 全部设为0.5（未知）
        position_features[:, 5] = 0.5
    
        self.logger.info("生成了简化的位置特征：染色体号、位置、区间中心(=位置)、固定区间长度、固定SNP密度、未知基因区域状态")
    
        return position_features
