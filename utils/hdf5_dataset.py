"""
HDF5数据集工具 - 用于高效加载和处理预处理好的HDF5格式数据
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

class HDF5Dataset(Dataset):
    """HDF5格式数据集，支持高效的内存管理和数据访问"""
    
    def __init__(self, 
                h5_file: str, 
                dataset_name: str = 'features',
                position_name: str = 'position_features',
                genotype_name: str = 'genotype_features',
                phenotype_name: str = 'phenotypes',
                indices: Optional[np.ndarray] = None,
                transform = None,
                cache_position: bool = True):
        """
        初始化HDF5数据集
        
        Args:
            h5_file: HDF5文件路径
            dataset_name: 特征数据组名称
            position_name: 位置特征数据集名称（所有样本共享）
            genotype_name: 基因型编码数据集名称（样本特定）
            phenotype_name: 表型数据集名称
            indices: 数据索引，用于选择数据子集
            transform: 可选的数据转换函数
            cache_position: 是否缓存位置特征到内存（提高性能）
        """
        self.h5_file = h5_file
        self.dataset_name = dataset_name
        self.position_name = position_name
        self.genotype_name = genotype_name
        self.phenotype_name = phenotype_name
        self.transform = transform
        self.cache_position = cache_position
        self.position_features = None
        
        # 获取数据形状并验证
        with h5py.File(self.h5_file, 'r') as f:
            # 检查文件格式
            self.is_legacy_format = self.dataset_name in f
            
            if self.is_legacy_format:
                # 旧格式：所有特征存储在一个数据集中
                self.data_shape = f[dataset_name].shape
                self.feature_dim = self.data_shape[2] if len(self.data_shape) > 2 else 1
                self.has_phenotypes = phenotype_name in f
                
                if self.has_phenotypes:
                    self.phenotype_shape = f[phenotype_name].shape
            else:
                # 新格式：位置特征和基因型编码分开存储
                self.has_position_features = self.position_name in f
                self.has_genotype_features = self.genotype_name in f
                
                if not self.has_position_features or not self.has_genotype_features:
                    raise ValueError(f"在{h5_file}中找不到位置特征或基因型编码数据集")
                
                # 获取位置特征形状
                self.position_shape = f[self.position_name].shape
                # 获取基因型编码形状
                self.genotype_shape = f[self.genotype_name].shape
                
                # 计算总特征维度
                self.feature_dim = self.position_shape[1] + 1  # 位置特征维度 + 基因型编码维度(1)
                
                # 检查表型数据
                indices_file = Path(h5_file).with_name(f"{Path(h5_file).stem}_phenotypes").with_suffix('.npy')
                self.has_phenotypes = indices_file.exists()
                
                if self.has_phenotypes:
                    # 从单独文件加载表型形状信息
                    try:
                        phenotype_array = np.load(indices_file, mmap_mode='r')
                        self.phenotype_shape = phenotype_array.shape
                    except Exception as e:
                        logging.warning(f"无法加载表型文件{indices_file}: {e}")
                        self.has_phenotypes = False
                
                # 缓存位置特征（所有样本共享）
                if self.cache_position:
                    self.position_features = f[self.position_name][:]
                    logging.info(f"已缓存{self.position_shape[0]}个SNP的位置特征向量")
            
            # 检查indices的有效性
            if indices is not None:
                # 确保所有索引有效
                max_idx = self.genotype_shape[1] - 1 if not self.is_legacy_format else self.data_shape[0] - 1
                if np.any(indices > max_idx) or np.any(indices < 0):
                    raise ValueError(f"无效的索引，最大索引应为 {max_idx}")
                self.indices = indices
            else:
                if self.is_legacy_format:
                    self.indices = np.arange(self.data_shape[0])
                else:
                    self.indices = np.arange(self.genotype_shape[1])
                
            # 获取并存储元数据
            self.metadata = {}
            if 'meta' in f:
                meta_group = f['meta']
                # 读取属性
                for key, value in meta_group.attrs.items():
                    self.metadata[key] = value
                
                # 尝试读取统计信息
                if 'stats' in meta_group:
                    try:
                        import json
                        stats_data = meta_group['stats'][()]
                        if isinstance(stats_data, bytes):
                            self.metadata['stats'] = json.loads(stats_data.decode('utf-8'))
                    except Exception as e:
                        logging.warning(f"无法解析统计信息: {e}")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            (features, labels) 元组，如果没有标签则返回 (features, None)
        """
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
            
        # 获取实际索引
        actual_idx = self.indices[idx]
        
        if self.is_legacy_format:
            # 旧格式：直接加载完整特征
            with h5py.File(self.h5_file, 'r') as f:
                # 加载特征
                features = f[self.dataset_name][actual_idx]
                
                # 加载标签（如果有）
                labels = None
                if self.has_phenotypes:
                    labels = f[self.phenotype_name][actual_idx]
            
            # 转换为张量
            features = torch.from_numpy(features).float()
            if labels is not None:
                labels = torch.from_numpy(labels).float()
        else:
            # 新格式：组合位置特征和基因型编码
            # 加载位置特征（所有样本共享）
            if self.position_features is not None:
                # 使用缓存的位置特征
                position_features_all = self.position_features
            else:
                # 从文件加载位置特征
                with h5py.File(self.h5_file, 'r') as f:
                    position_features_all = f[self.position_name][:]
            
            # 加载该样本的基因型编码
            with h5py.File(self.h5_file, 'r') as f:
                genotype_features = f[self.genotype_name][:, actual_idx]
            
            # 将位置特征和基因型编码组合为完整特征
            # 初始化完整特征数组
            n_snps = position_features_all.shape[0]
            combined_features = np.zeros((n_snps, self.feature_dim), dtype=np.float32)
            
            # 填充位置特征（前6列）
            combined_features[:, :position_features_all.shape[1]] = position_features_all
            
            # 填充基因型编码（最后一列）
            combined_features[:, -1] = genotype_features
            
            # 转换为张量
            features = torch.from_numpy(combined_features).float()
            
            # 加载标签（如果有）
            labels = None
            if self.has_phenotypes:
                # 从单独文件加载表型
                phenotype_file = Path(self.h5_file).with_name(f"{Path(self.h5_file).stem}_phenotypes").with_suffix('.npy')
                try:
                    phenotype_array = np.load(phenotype_file)
                    labels = torch.from_numpy(phenotype_array[actual_idx]).float()
                except Exception as e:
                    logging.warning(f"加载表型失败: {e}")
        
        # 应用转换（如果有）
        if self.transform:
            features = self.transform(features)
            
        return features, labels if labels is not None else torch.zeros(1)
    
    @staticmethod
    def get_metadata(h5_file):
        """获取HDF5文件的元数据"""
        metadata = {}
        try:
            with h5py.File(h5_file, 'r') as f:
                # 检查文件格式
                is_legacy_format = 'data' in f
                
                if is_legacy_format:
                    # 旧格式
                    if 'data' in f:
                        metadata['data_shape'] = f['data'].shape
                else:
                    # 新格式
                    if 'features' in f and 'position_features' in f['features']:
                        metadata['position_shape'] = f['features']['position_features'].shape
                    
                    if 'features' in f and 'genotype_features' in f['features']:
                        metadata['genotype_shape'] = f['features']['genotype_features'].shape
                
                # 获取存储的元数据
                if 'meta' in f:
                    meta_group = f['meta']
                    for key, value in meta_group.attrs.items():
                        metadata[key] = value
                    
                    # 尝试读取统计信息
                    if 'stats' in meta_group:
                        try:
                            import json
                            stats_data = meta_group['stats'][()]
                            if isinstance(stats_data, bytes):
                                metadata['stats'] = json.loads(stats_data.decode('utf-8'))
                        except Exception as e:
                            logging.warning(f"无法解析统计信息: {e}")
                            
                # 尝试读取染色体信息
                if 'meta' in f and 'chromosomes' in f['meta']:
                    try:
                        import json
                        chrom_data = f['meta']['chromosomes'][()]
                        if isinstance(chrom_data, bytes):
                            metadata['chromosomes'] = json.loads(chrom_data.decode('utf-8'))
                    except Exception as e:
                        logging.warning(f"无法解析染色体信息: {e}")
        except Exception as e:
            logging.error(f"读取HDF5元数据失败: {e}")
            
        return metadata

def create_data_loaders(h5_file: str, 
                       batch_size: int = 32, 
                       num_workers: int = 4, 
                       pin_memory: bool = True,
                       shuffle_train: bool = True,
                       distributed: bool = False,
                       local_rank: int = -1,
                       world_size: int = 1,
                       cache_position: bool = True):
    """
    创建训练、验证和测试数据加载器
    
    Args:
        h5_file: HDF5文件路径
        batch_size: 批处理大小
        num_workers: 数据加载的工作进程数
        pin_memory: 是否将张量固定在内存中
        shuffle_train: 是否打乱训练数据
        distributed: 是否使用分布式训练
        local_rank: 当前进程的本地排名
        world_size: 分布式训练的进程总数
        cache_position: 是否缓存位置特征（提高性能）
        
    Returns:
        包含训练、验证和测试数据加载器的字典
    """
    h5_file = Path(h5_file)
    logger = logging.getLogger("HDF5Dataset")
    logger.info(f"从HDF5文件创建数据加载器: {h5_file}")
    
    # 检查文件是否存在
    if not h5_file.exists():
        raise FileNotFoundError(f"HDF5文件不存在: {h5_file}")
    
    # 检查是否需要从单独的索引文件加载
    indices_file = h5_file.with_name(f"{h5_file.stem}_indices").with_suffix('.npz')
    separate_indices = indices_file.exists()
    
    if separate_indices:
        logger.info(f"从单独文件加载分割索引: {indices_file}")
        try:
            indices_data = np.load(indices_file)
            train_valid_indices = indices_data['train_valid_indices']
            test_indices = indices_data['test_indices']
            logger.info(f"成功加载索引: 训练/验证={len(train_valid_indices)}, 测试={len(test_indices)}")
        except Exception as e:
            logger.error(f"加载索引文件失败: {e}")
            raise
    else:
        # 从主HDF5文件中读取分割索引
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'train_valid_indices' not in f or 'test_indices' not in f:
                    raise KeyError(f"HDF5文件中缺少必要的索引数据")
                    
                train_valid_indices = f['train_valid_indices'][()]
                test_indices = f['test_indices'][()]
        except Exception as e:
            logger.error(f"从HDF5文件加载索引失败: {e}")
            raise
    
    # 计算训练/验证集分割
    valid_ratio = 0.2  # 默认验证集比例
    train_size = int(len(train_valid_indices) * (1 - valid_ratio))
    
    train_indices = train_valid_indices[:train_size]
    valid_indices = train_valid_indices[train_size:]
    
    logger.info(f"分割索引: 训练={len(train_indices)}, 验证={len(valid_indices)}, 测试={len(test_indices)}")
    
    # 创建数据集
    try:
        # 检测文件格式
        is_legacy_format = False
        with h5py.File(h5_file, 'r') as f:
            is_legacy_format = 'data' in f
        
        # 根据格式创建适当的数据集
        if is_legacy_format:
            logger.info("检测到旧格式HDF5文件")
            train_dataset = HDF5Dataset(str(h5_file), indices=train_indices)
            valid_dataset = HDF5Dataset(str(h5_file), indices=valid_indices)
            test_dataset = HDF5Dataset(str(h5_file), indices=test_indices)
        else:
            logger.info("检测到新格式HDF5文件（位置特征和基因型编码分离存储）")
            # 对训练集缓存位置特征以提高性能
            train_dataset = HDF5Dataset(
                str(h5_file), 
                dataset_name='features',
                position_name='position_features',
                genotype_name='genotype_features',
                indices=train_indices,
                cache_position=cache_position
            )
            
            valid_dataset = HDF5Dataset(
                str(h5_file), 
                dataset_name='features',
                position_name='position_features',
                genotype_name='genotype_features',
                indices=valid_indices,
                cache_position=cache_position
            )
            
            test_dataset = HDF5Dataset(
                str(h5_file), 
                dataset_name='features',
                position_name='position_features',
                genotype_name='genotype_features',
                indices=test_indices,
                cache_position=cache_position
            )
    except Exception as e:
        logger.error(f"创建数据集失败: {e}")
        raise
    
    # 分布式训练设置
    train_sampler = None
    valid_sampler = None
    test_sampler = None
    
    if distributed and local_rank != -1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=shuffle_train
        )
        
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False
        )
        
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False
        )
        
        logger.info(f"创建分布式采样器: 进程排名={local_rank}, 总进程数={world_size}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None and shuffle_train),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size*2,  # 验证时可以用更大的批量
        shuffle=False,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size*2,  # 测试时可以用更大的批量
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    logger.info(f"数据加载器创建完成")
    logger.info(f"- 训练数据: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    logger.info(f"- 验证数据: {len(valid_dataset)} 样本, {len(valid_loader)} 批次")
    logger.info(f"- 测试数据: {len(test_dataset)} 样本, {len(test_loader)} 批次")
    
    return {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

def preload_and_cache_batches(data_loader, num_batches=10, device='cpu'):
    """
    预加载和缓存若干批次的数据，用于降低I/O等待时间
    
    Args:
        data_loader: 数据加载器
        num_batches: 要预加载的批次数
        device: 设备
        
    Returns:
        预加载的数据批次列表
    """
    cached_batches = []
    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break
        # 将数据移到指定设备
        features, targets = batch
        features = features.to(device)
        targets = targets.to(device)
        cached_batches.append((features, targets))
    return cached_batches