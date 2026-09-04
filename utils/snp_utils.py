# Description: SNP数据处理工具函数
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import sys
import time
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# 尝试导入numba以加速处理
try:
    import numba
    from numba import njit, prange, float32, int32, int64, optional
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # 创建空装饰器
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(*args):
        return range(*args)
    
    logging.warning("未检测到numba库，将使用未优化版本。推荐使用'pip install numba'安装numba以提高性能。")
# 尝试使用SKlearn的分位数变换
try:
    from sklearn.preprocessing import QuantileTransformer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# 创建一个全局的独热编码映射
ONE_HOT_MAP = np.eye(10, dtype=np.float32)
ZERO_VECTOR = np.zeros(10, dtype=np.float32)

@njit
def get_mode(arr: np.ndarray) -> float:
    """
    计算数组中的众数 - 优化版本
    
    Args:
        arr: 输入数组
        
    Returns:
        众数
    """
    if len(arr) == 0:
        return 0.0
    
    # 使用预分配字典大小和直接数组索引来优化性能
    unique_vals = np.unique(arr)
    if len(unique_vals) == 1:
        return unique_vals[0]
    
    counts = np.zeros(len(unique_vals), dtype=np.int32)
    
    # 计数每个唯一值
    for val in arr:
        for i, uval in enumerate(unique_vals):
            if val == uval:
                counts[i] += 1
                break
    
    # 找出最大计数的索引
    max_idx = 0
    for i in range(1, len(counts)):
        if counts[i] > counts[max_idx]:
            max_idx = i
            
    return unique_vals[max_idx]

def normalize_feature(data: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    归一化特征向量
    
    Args:
        data: 输入数据数组
        eps: 避免除零的小值
        
    Returns:
        np.ndarray: 归一化后的数据
    """
    if len(data) == 0:
        return data
        
    min_val = np.min(data)
    max_val = np.max(data)
    
    # 如果值域太小，直接返回零数组
    if max_val - min_val < eps:
        return np.zeros_like(data, dtype=np.float32)
    
    normalized = (data - min_val) / (max_val - min_val + eps)
    return normalized

@njit(parallel=True)
def log_transform(values: np.ndarray) -> np.ndarray:
    """
    对输入值应用对数变换 - 优化版本
    
    Args:
        values: 输入值数组
        
    Returns:
        变换后的数组
    """
    # 创建输出数组，避免多次分配内存
    result = np.zeros_like(values, dtype=np.float32)
    
    # 快速检查是否有正值
    has_positive = False
    min_positive = float('inf')
    
    # 单次循环找出最小正值
    for i in range(len(values)):
        if values[i] > 0:
            has_positive = True
            if values[i] < min_positive:
                min_positive = values[i]
    
    # 如果没有正值，直接返回零数组
    if not has_positive:
        return result
    
    # 计算偏移量
    offset = min_positive / 10.0
    
    # 并行计算对数变换
    for i in prange(len(values)):
        if values[i] > 0:
            result[i] = np.log(values[i] + offset)
    
    # 归一化到[0,1]
    return normalize_feature(result)

@njit(parallel=True)
def enhance_density_contrast(densities: np.ndarray, power: float = 0.5) -> np.ndarray:
    """
    使用幂变换增强密度对比度 - 优化版本
    
    Args:
        densities: 密度值数组
        power: 幂参数，默认为0.5（平方根变换）
        
    Returns:
        变换后的数组
    """
    # 预分配内存
    result = np.zeros_like(densities, dtype=np.float32)
    
    # 并行处理幂变换
    for i in prange(len(densities)):
        if densities[i] > 0:
            result[i] = densities[i] ** power
    
    # 归一化到[0,1]
    return normalize_feature(result)

# 非numba函数
def apply_adaptive_normalization(values: np.ndarray) -> np.ndarray:
    """
    应用自适应归一化，根据数据分布选择合适的变换
    
    Args:
        values: 输入值数组
        
    Returns:
        变换后的数组
    """
    # 处理空数组
    if len(values) == 0:
        return np.array([], dtype=np.float32)
    
    # 计算基本统计量
    non_zero = values[values > 0]
    if len(non_zero) == 0:
        return np.zeros_like(values, dtype=np.float32)
    
    mean_val = np.mean(non_zero)
    median_val = np.median(non_zero)
    max_val = np.max(non_zero)
    
    # 估计分布的偏斜度
    if mean_val > 2 * median_val:
        # 严重右偏，使用对数变换
        return log_transform(values)
    elif max_val > 5 * median_val:
        # 中度右偏，使用平方根变换
        return enhance_density_contrast(values, power=0.5)
    else:
        # 轻微偏斜或对称分布，使用线性归一化
        return normalize_feature(values)

def quantile_transform(values: np.ndarray) -> np.ndarray:
    """
    使用分位数变换处理严重偏斜的分布 - 优化版本
    
    Args:
        values: 输入值数组
        
    Returns:
        变换后的数组
    """
    if not HAS_SKLEARN:
        return log_transform(values)
    
    # 处理空数组
    if len(values) == 0:
        return np.array([], dtype=np.float32)
    
    # 使用布尔掩码代替索引操作，减少内存分配
    zeros_mask = values == 0
    zero_ratio = np.mean(zeros_mask)
    
    if (zero_ratio > 0.5):  # 零值占大多数
        non_zeros_mask = ~zeros_mask
        non_zeros_count = np.sum(non_zeros_mask)
        
        if (non_zeros_count > 0):
            # 创建结果数组（先分配内存）
            result = np.zeros_like(values, dtype=np.float32)
            
            # 变换非零值
            non_zeros = values[non_zeros_mask].reshape(-1, 1)
            transformer = QuantileTransformer(output_distribution='normal')
            transformed_non_zeros = transformer.fit_transform(non_zeros).flatten()
            
            # 归一化非零值
            min_val = np.min(transformed_non_zeros)
            max_val = np.max(transformed_non_zeros)
            if max_val > min_val:
                transformed_non_zeros = (transformed_non_zeros - min_val) / (max_val - min_val)
            
            # 更新结果数组中的非零值
            result[non_zeros_mask] = transformed_non_zeros
            
            return result
        else:
            return np.zeros_like(values, dtype=np.float32)
    else:
        # 直接变换所有值
        transformer = QuantileTransformer(output_distribution='normal')
        transformed = transformer.fit_transform(values.reshape(-1, 1)).flatten()
        
        # 归一化到[0,1]
        min_val = np.min(transformed)
        max_val = np.max(transformed)
        if max_val > min_val:
            transformed = (transformed - min_val) / (max_val - min_val)
            
        return transformed.astype(np.float32)

def encode_genotype(allele1: str, allele2: str, genotype: int) -> np.ndarray:
    """
    根据提供的等位基因和基因型代码对SNP进行独热编码

    编码规则:
    AA (0), AT (1), TA (1), AC (2), CA (2), AG (3), GA (3),
    TT (4), TC (5), CT (5), TG (6), GT (6),
    CC (7), CG (8), GC (8), GG (9)
    缺失值 (genotype=3 or invalid) -> 全零向量 [0, 0, ..., 0]

    Args:
        allele1: 第一个等位基因 (A,T,C或G)
        allele2: 第二个等位基因 (A,T,C或G)
        genotype: 原始基因型值 (0=纯合子第一等位基因, 1=杂合子, 2=纯合子第二等位基因, 3=缺失值)

    Returns:
        np.ndarray: 10维独热编码向量，缺失值返回全零向量
    """
    # 处理缺失基因型 (代码为3) 或无效代码
    if genotype == 3 or genotype < 0 or genotype > 2:
        return ZERO_VECTOR # 返回全零向量

    # 标准化等位基因 (转为大写)
    a1 = allele1.upper()
    a2 = allele2.upper()

    # 确保等位基因是有效碱基
    valid_bases = ('A', 'T', 'C', 'G')
    if a1 not in valid_bases or a2 not in valid_bases:
        logging.warning(f"发现无效碱基: {allele1}, {allele2}. SNP将编码为缺失值.")
        return ZERO_VECTOR # 返回全零向量

    # 根据基因型确定实际的等位基因组合
    if genotype == 0:  # 纯合子第一等位基因 (a1/a1)
        bases = a1 + a1
    elif genotype == 1:  # 杂合子 (a1/a2 或 a2/a1)
        # 排序等位基因对以保持一致
        bases = "".join(sorted((a1, a2))) # 排序确保 AT 和 TA 映射到同一个值
    else:  # genotype == 2, 纯合子第二等位基因 (a2/a2)
        bases = a2 + a2

    # 应用编码规则 - 修复：添加了所有可能的杂合子组合
    encoding_map = {
        "AA": 0, 
        "AT": 1, "TA": 1,  # AT 和 TA 都映射到 1
        "AC": 2, "CA": 2,  # AC 和 CA 都映射到 2
        "AG": 3, "GA": 3,  # AG 和 GA 都映射到 3
        "TT": 4, 
        "TC": 5, "CT": 5,  # TC 和 CT 都映射到 5
        "TG": 6, "GT": 6,  # TG 和 GT 都映射到 6
        "CC": 7, 
        "CG": 8, "GC": 8,  # CG 和 GC 都映射到 8
        "GG": 9
    }

    scalar_code = encoding_map.get(bases, -1)

    if scalar_code != -1:
        return ONE_HOT_MAP[scalar_code]
    else:
        # 不应该到达这里，除非有无效碱基组合（理论上已在前面检查过）
        logging.error(f"无法识别的碱基组合: {bases}. SNP将编码为缺失值.")
        return ZERO_VECTOR # 返回全零向量

@njit
def decode_snp_byte(byte_val: np.uint8, sample_count: int, sample_offset: int) -> np.ndarray:
    """
    解码表示4个样本的单个字节
    
    BED文件编码规则:
    00: 与 .bim 文件中第一个等位基因纯合 (0)
    01: 缺失的基因型 (3)
    10: 杂合 (1)
    11: 与 .bim 文件中第二个等位基因纯合 (2)
    
    Args:
        byte_val: 存储4个样本基因型的字节
        sample_count: 总样本数，用于边界检查
        sample_offset: 此字节的样本偏移量
        
    Returns:
        包含基因型编码的数组 (最多4个样本)
    """
    # 每个字节包含4个2位基因型代码
    genotypes = np.zeros(4, dtype=np.float32)
    
    # 提取每个2位编码并映射到基因型值
    for i in range(4):
        # 如果已经超出样本数，不再处理
        if sample_offset + i >= sample_count:
            break
            
        # 提取2位值 (00, 01, 10, 11)
        code = (byte_val >> (i * 2)) & 0x03
        
        # 映射到基因型:
        # 00: 与 .bim 文件中第一个等位基因纯合 (0)
        # 01: 缺失的基因型 (3)
        # 10: 杂合 (1)
        # 11: 与 .bim 文件中第二个等位基因纯合 (2)
        if code == 0:
            genotypes[i] = 0
        elif code == 1:
            genotypes[i] = 3
        elif code == 2:
            genotypes[i] = 1
        else:  # code == 3
            genotypes[i] = 2
            
    return genotypes

@njit
def decode_snp_row(byte_row: np.ndarray, n_samples: int) -> np.ndarray:
    """
    解码表示一个SNP的所有字节
    
    Args:
        byte_row: SNP的字节行
        n_samples: 样本数量
        
    Returns:
        包含所有样本该SNP基因型的数组
    """
    # 每个字节有4个样本，计算需要多少字节
    n_bytes = len(byte_row)
    genotypes = np.zeros(n_samples, dtype=np.float32)
    
    for b in range(n_bytes):
        # 计算当前字节对应的样本偏移量
        sample_offset = b * 4
        
        # 如果已经超出样本数，不再处理
        if sample_offset >= n_samples:
            break
            
        # 解码当前字节
        byte_genotypes = decode_snp_byte(byte_row[b], n_samples, sample_offset)
        
        # 将解码结果复制到结果数组
        for i in range(4):
            if sample_offset + i < n_samples:
                genotypes[sample_offset + i] = byte_genotypes[i]
                
    return genotypes

def _decode_chunk(chunk_idx, bed_data, n_snps, n_samples, chunk_size):
    """
    解码BED数据块，被decode_snps并行调用
    
    Args:
        chunk_idx: 块索引
        bed_data: BED字节数据
        n_snps: SNP总数
        n_samples: 样本数量
        chunk_size: 每个块的SNP数量
        
    Returns:
        tuple: 块索引和解码后的SNP数据块
    """
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, n_snps)
    
    chunk_decoded = np.zeros((end_idx - start_idx, n_samples), dtype=np.float32)
    
    for i, snp_idx in enumerate(range(start_idx, end_idx)):
        if snp_idx < len(bed_data):  # 确保不超出bed_data范围
            chunk_decoded[i] = decode_snp_row(bed_data[snp_idx], n_samples)
    
    return chunk_idx, chunk_decoded

def decode_snps(bed_data: np.ndarray, n_snps: int, n_samples: int, 
                n_jobs: int = 5, chunk_size: int = 10000) -> np.ndarray:
    """
    并行解码BED数据，支持多进程处理
    
    Args:
        bed_data: BED字节数据
        n_snps: SNP数量
        n_samples: 样本数量
        n_jobs: 并行作业数
        chunk_size: 每个并行任务处理的SNP数量
        
    Returns:
        解码后的基因型数组 (n_snps × n_samples)
    """
    # 创建结果数组
    result = np.zeros((n_snps, n_samples), dtype=np.float32)
    
    # 确保n_snps不超过实际数据行数
    n_snps = min(n_snps, bed_data.shape[0])
    
    # 计算分块数
    n_chunks = (n_snps + chunk_size - 1) // chunk_size
    
    # 并行处理
    n_jobs = min(n_jobs, n_chunks, multiprocessing.cpu_count())
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(
                _decode_chunk, i, bed_data, n_snps, n_samples, chunk_size
            ) for i in range(n_chunks)]
            for future in futures:
                chunk_idx, chunk_decoded = future.result()
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, n_snps)
                result[start_idx:end_idx] = chunk_decoded
    else:
        # 串行处理
        for chunk_idx in range(n_chunks):
            chunk_idx, chunk_decoded = _decode_chunk(
                chunk_idx, bed_data, n_snps, n_samples, chunk_size
            )
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_snps)
            result[start_idx:end_idx] = chunk_decoded
    
    return result

def batch_normalize_features(features: np.ndarray, n_jobs: int = 1) -> np.ndarray:
    """
    并行批量归一化特征数组
    
    Args:
        features: 输入特征数组，形状为 (n_samples, n_features)
        n_jobs: 并行作业数
        
    Returns:
        np.ndarray: 归一化后的特征数组
    """
    n_features = features.shape[1]
    normalized = np.zeros_like(features, dtype=np.float32)
    
    # 定义单特征归一化函数
    def _normalize_single_feature(i):
        return i, normalize_feature(features[:, i])
    
    # 并行处理每个特征
    n_jobs = min(n_jobs, n_features, multiprocessing.cpu_count())
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            for i, norm_feature in executor.map(_normalize_single_feature, range(n_features)):
                normalized[:, i] = norm_feature
    else:
        # 串行处理
        for i in range(n_features):
            normalized[:, i] = normalize_feature(features[:, i])
    
    return normalized

def _encode_chunk(chunk_idx, raw_snps, allele1, allele2, n_snps, n_samples, chunk_size):
    """
    编码基因型数据块（独热编码），被batch_encode_genotypes调用

    Args:
        chunk_idx: 块索引
        raw_snps: 原始SNP数据 (0,1,2,3)
        allele1: 第一个等位基因序列
        allele2: 第二个等位基因序列
        n_snps: SNP总数
        n_samples: 样本总数
        chunk_size: 每个块的SNP数量

    Returns:
        tuple: 块索引和编码后的SNP数据块 (独热编码, shape: [chunk_snps, n_samples, 10])
    """
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, n_snps)
    num_snps_in_chunk = end_idx - start_idx

    # 注意形状变化：添加了最后一个维度 10
    chunk_encoded = np.zeros((num_snps_in_chunk, n_samples, 10), dtype=np.float32)

    for i, snp_idx in enumerate(range(start_idx, end_idx)):
        # 获取当前SNP的等位基因
        a1, a2 = allele1[snp_idx], allele2[snp_idx]

        # 对当前SNP的每个样本进行编码
        for j in range(n_samples):
            # raw_snps[snp_idx, j] 包含 0, 1, 2, 或 3 (缺失)
            # encode_genotype 现在返回 10 维向量
            chunk_encoded[i, j, :] = encode_genotype(a1, a2, int(raw_snps[snp_idx, j]))

    return chunk_idx, chunk_encoded

def batch_encode_genotypes(raw_snps: np.ndarray, allele1: np.ndarray, allele2: np.ndarray,
                           n_jobs: int = 5, chunk_size: int = 10000) -> np.ndarray:
    """
    批量编码基因型数据（独热编码），支持并行处理

    Args:
        raw_snps: 原始SNP数据，形状为 (n_snps, n_samples)，值为 0,1,2,3
        allele1: 第一个等位基因序列
        allele2: 第二个等位基因序列
        n_jobs: 并行作业数
        chunk_size: 每个并行任务处理的SNP数量

    Returns:
        np.ndarray: 编码后的SNP数据，形状为 (n_snps, n_samples, 10)
    """
    n_snps, n_samples = raw_snps.shape
    # 注意形状变化：添加了最后一个维度 10
    encoded = np.zeros((n_snps, n_samples, 10), dtype=np.float32)

    # 计算分块数
    n_chunks = (n_snps + chunk_size - 1) // chunk_size

    # 并行处理
    n_jobs = min(n_jobs, n_chunks, multiprocessing.cpu_count())
    if n_jobs > 1:
        logging.info(f"使用 {n_jobs} 个进程并行编码SNP...")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(
                _encode_chunk, i, raw_snps, allele1, allele2, n_snps, n_samples, chunk_size
            ) for i in range(n_chunks)]
            for future in futures:
                chunk_idx, chunk_encoded = future.result()
                start_idx = chunk_idx * chunk_size
                end_idx = start_idx + chunk_encoded.shape[0] # 使用实际返回的块大小
                # 确保索引在范围内
                if start_idx < n_snps and end_idx <= n_snps:
                     encoded[start_idx:end_idx] = chunk_encoded
                else:
                     # 处理可能的边界情况或错误
                     logging.warning(f"Chunk index {chunk_idx} result shape mismatch or out of bounds.")
    else:
        logging.info("使用单进程串行编码SNP...")
        # 串行处理
        for chunk_idx in range(n_chunks):
            chunk_idx, chunk_encoded = _encode_chunk(
                chunk_idx, raw_snps, allele1, allele2, n_snps, n_samples, chunk_size
            )
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_encoded.shape[0] # 使用实际返回的块大小
            if start_idx < n_snps and end_idx <= n_snps:
                 encoded[start_idx:end_idx] = chunk_encoded
            else:
                 logging.warning(f"Chunk index {chunk_idx} result shape mismatch or out of bounds.")

    logging.info("SNP编码完成。")
    return encoded

def get_mode_scalar(arr: np.ndarray) -> Optional[int]:
    """计算标量数组的众数"""
    if len(arr) == 0:
        return None
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]

def fill_missing_values(data: np.ndarray, config: Optional[Dict[str, Any]] = None, n_jobs: int = 1) -> np.ndarray:
    """
    根据配置填充独热编码数组中的缺失值（全零向量）。

    Args:
        data: 输入独热编码数据数组，形状为 (n_features, n_samples, 10)
        config: 包含 missing_value_handling 配置的字典
                例如: {'enable': True, 'method': 'mode'}
        n_jobs: 并行作业数

    Returns:
        np.ndarray: 填充后的数组
    """
    if data.size == 0:
        return data.copy()

    # 检查配置
    # 从字典中安全地获取配置值
    missing_handling_config = config or {}
    handle_missing = missing_handling_config.get('enable', False)
    fill_method = missing_handling_config.get('method', 'mode') if handle_missing else None

    if not handle_missing:
        logging.info("缺失值填充已禁用，跳过填充步骤。")
        return data.copy() # 直接返回，缺失值保持为全零向量

    # 验证填充方法
    if fill_method not in ['mode', 'mean']:
        logging.warning(f"无效的填充方法: {fill_method}，将使用默认 'mode'。")
        fill_method = 'mode'

    logging.info(f"开始填充缺失值，方法: {fill_method}")

    n_features, n_samples, encoding_dim = data.shape
    if encoding_dim != 10:
         raise ValueError("输入数据似乎不是10维独热编码")

    filled_data = data.copy() # 创建副本进行填充

    # 定义单行填充函数 (线程安全)
    def _fill_row_thread(idx):
        row = filled_data[idx] # 直接修改副本
        # 找到缺失值的位置 (全零向量)
        missing_mask = np.all(row == 0, axis=1)

        if not np.any(missing_mask):
            return # 没有缺失值，无需处理

        # 获取非缺失值用于计算填充值
        valid_vectors = row[~missing_mask]
        if len(valid_vectors) == 0:
             logging.warning(f"特征 {idx} 所有样本值都缺失，无法填充。")
             return

        # 将有效的独热编码转回标量值 (0-9)
        valid_scalars = np.argmax(valid_vectors, axis=1)

        # 计算填充值 (标量)
        fill_scalar = None
        if fill_method == 'mean':
            fill_scalar = int(np.round(np.mean(valid_scalars)))
            fill_scalar = np.clip(fill_scalar, 0, 9)
        elif fill_method == 'mode':
            fill_scalar_opt = get_mode_scalar(valid_scalars)
            if fill_scalar_opt is None:
                 logging.warning(f"特征 {idx} 无法计算众数，跳过填充。")
                 return
            fill_scalar = fill_scalar_opt

        if fill_scalar is None:
            logging.warning(f"特征 {idx} 未能计算出填充值，跳过填充。")
            return

        # 获取填充值对应的独热向量
        fill_vector = ONE_HOT_MAP[fill_scalar]

        # 填充缺失位置
        row[missing_mask] = fill_vector

    # 使用线程池并行处理每一行（特征）
    n_jobs = min(n_jobs, n_features, multiprocessing.cpu_count())
    if n_jobs > 1:
        logging.info(f"使用 {n_jobs} 个线程并行填充缺失值...")
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            list(executor.map(_fill_row_thread, range(n_features)))
    else:
        logging.info("使用单线程串行填充缺失值...")
        for i in range(n_features):
            _fill_row_thread(i)

    logging.info("缺失值填充完成。")
    return filled_data
