from __future__ import annotations

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import time
import sys

# 检查h5py库是否可用
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logging.warning("未检测到h5py库，HDF5功能将不可用。请使用'pip install h5py'安装")

# 导入HDF5Dataset
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from utils.hdf5_dataset import HDF5Dataset, create_data_loaders
except ImportError:
    if HAS_H5PY:
        logging.warning("无法导入HDF5Dataset类，HDF5功能可能不可用")
    # 创建空类型避免引用错误
    class HDF5Dataset:
        pass

class HDF5Handler:
    """HDF5文件处理器，处理HDF5格式数据的保存和加载"""
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        """
        初始化HDF5处理器
        
        Args:
            config: 配置对象
            logger: 可选的日志记录器
        """
        self.config = config
        self.logger = logger or logging.getLogger("HDF5Handler")
        
        if not HAS_H5PY:
            self.logger.warning("无法使用HDF5功能，h5py库不可用")
    
    def save_processed_data(self, feature_data: Dict[str, Any], feature_stats: Dict[str, Any], 
                            partitions: Optional[np.ndarray] = None, 
                            chromosomes: Optional[Dict[str, Dict[str, Any]]] = None,
                            split_results: Optional[Dict[str, Any]] = None,
                            phenotypes: Optional[np.ndarray] = None) -> None:
        """
        保存预处理后的数据，默认调用HDF5保存方法
        
        Args:
            feature_data: 包含位置特征和基因型编码的字典
            feature_stats: 特征统计信息
            partitions: 分区索引数组
            chromosomes: 染色体位置信息
            split_results: 包含划分结果的字典
            phenotypes: 表型数据数组
        """
        return self.save_hdf5_data(feature_data, feature_stats, partitions, chromosomes, split_results, phenotypes)
    
    def save_hdf5_data(self, feature_data: Dict[str, Any], feature_stats: Dict[str, Any],
                      partitions: Optional[np.ndarray] = None,
                      chromosomes: Optional[Dict[str, Dict[str, Any]]] = None,
                      split_results: Optional[Dict[str, Any]] = None,
                      phenotypes: Optional[np.ndarray] = None) -> None:
        """
        保存预处理后的数据为HDF5格式
        位置特征向量和基因型编码分开存储
        
        Args:
            feature_data: 包含位置特征和基因型编码的字典
            feature_stats: 特征统计信息
            partitions: 分区索引数组
            chromosomes: 染色体位置信息
            split_results: 包含划分结果的字典
            phenotypes: 表型数据数组
        """
        if not HAS_H5PY:
            self.logger.error("无法保存HDF5格式，h5py库不可用")
            raise ImportError("保存HDF5文件需要h5py库。请使用'pip install h5py'安装")
            
        output_dir = Path(self.config.preprocessing.file_processing.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取位置特征和基因型编码
        position_features = feature_data.get("position_features")
        genotype_features = feature_data.get("genotype_features")
        
        if position_features is None or genotype_features is None:
            raise ValueError("位置特征或基因型编码数据缺失")
        
        # 获取数据维度信息
        position_dim = feature_data.get("position_dim", position_features.shape[1])
        n_snps = feature_data.get("n_snps", position_features.shape[0])
        n_samples = feature_data.get("n_samples", genotype_features.shape[1])
        
        # 创建HDF5文件并保存数据
        output_file = output_dir / "whisper_dna_dataset.h5"
        self.logger.info(f"保存数据为HDF5格式: {output_file}")
        
        try:
            with h5py.File(output_file, 'w') as f:
                # 创建特征组
                features_group = f.create_group('features')
                
                # 保存位置特征向量（所有样本共享）
                self.logger.info(f"保存位置特征向量，形状: {position_features.shape}")
                features_group.create_dataset(
                    'position_features', 
                    data=position_features,
                    chunks=(min(1000, n_snps), position_dim),
                    compression="gzip", 
                    compression_opts=4
                )
                
                # 保存基因型编码
                self.logger.info(f"保存基因型编码，形状: {genotype_features.shape}")
                features_group.create_dataset(
                    'genotype_features', 
                    data=genotype_features,
                    chunks=(min(1000, n_snps), min(100, n_samples)),
                    compression="gzip", 
                    compression_opts=4
                )
                
                # 保存表型数据
                if phenotypes is not None:
                    self.logger.info(f"保存表型数据，形状: {phenotypes.shape}")
                    f.create_dataset(
                        'phenotypes', 
                        data=phenotypes, 
                        chunks=True,
                        compression="gzip", 
                        compression_opts=4
                    )
                
                # 保存训练/验证/测试索引
                if split_results:
                    self.logger.info("保存数据集划分索引")
                    if 'train_valid_indices' in split_results:
                        f.create_dataset('train_valid_indices', data=split_results['train_valid_indices'])
                    if 'test_indices' in split_results:
                        f.create_dataset('test_indices', data=split_results['test_indices'])
                    
                    # 保存交叉验证折索引
                    cv_folds = split_results.get('cv_folds', [])
                    if cv_folds:
                        cv_group = f.create_group('cv_folds')
                        for i, (train_idx, valid_idx) in enumerate(cv_folds):
                            fold_group = cv_group.create_group(f'fold_{i}')
                            fold_group.create_dataset('train', data=train_idx)
                            fold_group.create_dataset('valid', data=valid_idx)
                
                # 保存分区信息
                if partitions is not None and len(partitions) > 0:
                    self.logger.info("保存分区索引")
                    f.create_dataset('partitions', data=partitions)
                
                # 保存染色体信息（转换为JSON字符串）
                if chromosomes:
                    self.logger.info("保存染色体位置信息")
                    import json
                    chromosomes_json = json.dumps(chromosomes)
                    f.create_dataset('chromosomes', data=np.string_(chromosomes_json))
                
                # 保存特征统计信息
                if feature_stats:
                    self.logger.info("保存特征统计信息")
                    import json
                    stats_json = json.dumps(feature_stats)
                    f.create_dataset('stats', data=np.string_(stats_json))
                
                # 保存元数据
                meta_group = f.create_group('meta')
                meta_group.attrs['n_snps'] = n_snps
                meta_group.attrs['n_samples'] = n_samples
                meta_group.attrs['position_dim'] = position_dim
                meta_group.attrs['feature_dim'] = feature_stats.get("feature_dim", position_dim + 1)
                meta_group.attrs['creation_time'] = str(time.time())
                
            self.logger.info(f"HDF5文件保存完成: {output_file}")
            
        except Exception as e:
            self.logger.error(f"保存HDF5文件失败: {str(e)}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            
            # 回退到NumPy保存方式
            self.logger.info("尝试使用NumPy格式分别保存数据...")
            self._save_numpy_fallback(output_dir / "whisper_dna_dataset", feature_data, feature_stats)
    
    def _save_numpy_fallback(self, output_path: Path, feature_data: Dict[str, Any], feature_stats: Dict[str, Any]) -> None:
        """
        当HDF5不可用时，使用NumPy格式分别保存位置特征和基因型编码
        
        Args:
            output_path: 输出路径
            feature_data: 要保存的特征数据字典
            feature_stats: 特征统计信息
        """
        # 保存位置特征向量
        position_path = output_path.with_name(f"{output_path.stem}_position").with_suffix(".npy")
        self.logger.info(f"保存位置特征向量到: {position_path}")
        np.save(position_path, feature_data["position_features"])
        
        # 保存基因型编码
        genotype_path = output_path.with_name(f"{output_path.stem}_genotype").with_suffix(".npy")
        self.logger.info(f"保存基因型编码到: {genotype_path}")
        np.save(genotype_path, feature_data["genotype_features"])
        
        # 保存特征统计信息
        stats_path = output_path.with_name(f"{output_path.stem}_stats").with_suffix(".json")
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
    
    def load_hdf5_data(self, h5_file: Union[str, Path], 
                      indices: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        加载HDF5格式的数据
        
        Args:
            h5_file: HDF5文件路径
            indices: 要使用的索引数组，如果为None则使用全部数据
            
        Returns:
            包含加载数据的字典
        """
        if not HAS_H5PY:
            self.logger.error("无法加载HDF5格式，h5py库不可用")
            raise ImportError("加载HDF5文件需要h5py库。请使用'pip install h5py'安装")
            
        h5_file = Path(h5_file)
        if not h5_file.exists():
            raise FileNotFoundError(f"HDF5文件不存在: {h5_file}")
            
        self.logger.info(f"加载HDF5数据: {h5_file}")
        
        result = {}
        
        with h5py.File(h5_file, 'r') as f:
            # 检查文件结构
            if 'features' not in f:
                self.logger.warning("旧格式HDF5文件，尝试兼容模式加载")
                # 兼容旧格式
                if 'data' in f:
                    self.logger.info("加载旧格式数据...")
                    data = f['data'][()]
                    # 提取位置特征和基因型编码
                    if data.ndim == 3:  # [n_samples, n_snps, features]
                        n_samples, n_snps, features_dim = data.shape
                        # 转置为 [n_snps, n_samples, features]
                        data = np.transpose(data, (1, 0, 2))
                        # 假设最后一维是基因型编码
                        position_features = data[:, 0, :-1]  # 取第一个样本的位置特征
                        genotype_features = data[:, :, -1]   # 所有样本的基因型编码
                        
                        result['position_features'] = position_features
                        result['genotype_features'] = genotype_features
                        result['position_dim'] = position_features.shape[1]
                        result['n_snps'] = n_snps
                        result['n_samples'] = n_samples
                    else:
                        self.logger.error(f"不支持的数据维度: {data.shape}")
                        raise ValueError(f"不支持的数据维度: {data.shape}")
                else:
                    raise KeyError("HDF5文件结构不匹配，缺少 'features' 或 'data' 组")
            else:
                # 新格式，分别加载位置特征和基因型编码
                features_group = f['features']
                
                # 加载位置特征向量
                if 'position_features' in features_group:
                    position_features = features_group['position_features'][()]
                    result['position_features'] = position_features
                    result['position_dim'] = position_features.shape[1]
                else:
                    raise KeyError("HDF5文件中缺少位置特征向量数据")
                
                # 加载基因型编码
                if 'genotype_features' in features_group:
                    genotype_features = features_group['genotype_features'][()]
                    # 如果提供了索引，只加载指定样本
                    if indices is not None:
                        genotype_features = genotype_features[:, indices]
                    result['genotype_features'] = genotype_features
                else:
                    raise KeyError("HDF5文件中缺少基因型编码数据")
                
                # 获取维度信息
                result['n_snps'] = position_features.shape[0]
                result['n_samples'] = genotype_features.shape[1]
            
            # 加载表型数据
            if 'phenotypes' in f:
                phenotypes = f['phenotypes'][()]
                # 如果提供了索引，只加载指定样本的表型
                if indices is not None:
                    phenotypes = phenotypes[indices]
                result['phenotypes'] = phenotypes
            
            # 加载元数据
            if 'meta' in f:
                meta_group = f['meta']
                meta_data = {}
                for key, value in meta_group.attrs.items():
                    meta_data[key] = value
                result['meta'] = meta_data
            
            # 加载特征统计信息
            if 'stats' in f:
                import json
                stats_json = f['stats'][()].decode('utf-8')
                result['feature_stats'] = json.loads(stats_json)
            
            # 加载分区信息
            if 'partitions' in f:
                result['partitions'] = f['partitions'][()]
            
            # 加载染色体信息
            if 'chromosomes' in f:
                import json
                chromosomes_json = f['chromosomes'][()].decode('utf-8')
                result['chromosomes'] = json.loads(chromosomes_json)
        
        self.logger.info(f"数据加载完成: {result['n_snps']} SNPs, {result['n_samples']} 样本, {result['position_dim']} 位置特征维度")
        
        return result
        
    def create_data_loaders(self, h5_file: Union[str, Path], batch_size: int = 32, 
                          num_workers: int = 4, shuffle_train: bool = True, 
                          distributed: bool = False, 
                          local_rank: int = -1, world_size: int = 1):
        """
        从HDF5文件创建训练、验证和测试数据加载器
        
        Args:
            h5_file: HDF5文件路径
            batch_size: 批处理大小
            num_workers: 数据加载的工作进程数
            shuffle_train: 是否打乱训练数据
            distributed: 是否使用分布式训练
            local_rank: 当前进程的本地排名
            world_size: 分布式训练的进程总数
            
        Returns:
            包含训练、验证和测试数据加载器的字典
        """
        return create_data_loaders(
            h5_file, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            pin_memory=True,
            shuffle_train=shuffle_train,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size
        )
