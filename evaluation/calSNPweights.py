# -*- coding: utf-8 -*-
"""
SNP 权重提取与展开脚本（基于原项目 DNAWhisper 的核心权重计算）
功能：
  1. 读取训练时保存的池化权重 H5 文件（如 val_pooling_weights.h5）
  2. 将 Embedding 层与 GFI 块的注意力权重逐层展开，得到每个 SNP 的贡献权重
  3. 支持批量处理样本、归一化、调整最后一个 context block
  4. 输出每个样本-专家-SNP 的权重矩阵（.npy），用于后续重要性分析

用法示例：
  python3 evaluation/calSNPweights.py --h5_path <权重H5路径> --output_dir <输出目录> --batch_size 32 --normalize --experts 0,1
"""

import h5py
import numpy as np
import os
from typing import Optional, List, Tuple, Dict, Union

def expand_pooling_weights(
    h5_path: str,
    output_path: str = None,
    sample_indices=None,
    expert_indices=None,
    normalize: bool = True,
    adjust_last_block: bool = True  # 新增参数：是否调整最后一个context_block
):
    """
    展开HDF5保存的pooling权重，计算每个样本每个专家每个SNP对最终输出的权重。
    """
    print(f"正在读取权重文件: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        # 读取embedding权重
        if 'embedding' in f and 'pooling_weights' in f['embedding']:
            emb_weights = f['embedding/pooling_weights'][:]  # [B, N_emb, E, S_emb]
            print(f"Embedding权重形状: {emb_weights.shape}")
        else:
            raise KeyError("HDF5文件中未找到'embedding/pooling_weights'，请检查权重文件。")
        
        # 读取所有GFI block权重
        gfi_weights_list = []
        if 'gfi_blocks' in f:
            i = 0
            while f'gfi_blocks/block_{i}' in f and 'pooling_weights' in f[f'gfi_blocks/block_{i}']:
                weight = f[f'gfi_blocks/block_{i}/pooling_weights'][:]  # [B, N_gfi, E, S_gfi]
                gfi_weights_list.append(weight)
                print(f"GFI块{i}权重形状: {weight.shape}")
                i += 1
            print(f"共读取 {len(gfi_weights_list)} 个GFI块权重")
        else:
            raise KeyError("HDF5文件中未找到'gfi_blocks'，请检查权重文件。")

    B, N_emb, E, S_emb = emb_weights.shape
    num_gfi = len(gfi_weights_list)
    sample_indices = sample_indices if sample_indices is not None else range(B)
    expert_indices = expert_indices if expert_indices is not None else range(E)
    print(f"处理 {len(sample_indices)} 个样本, {len(expert_indices)} 个专家")

    # 如果需要调整最后一个块，预处理所有GFI块的权重
    if adjust_last_block:
        print("正在调整各GFI块最后一个context_block的权重...")
        adjusted_gfi_weights_list = []
        
        for gfi_idx, gfi_weights in enumerate(gfi_weights_list):
            adjusted_weights = gfi_weights.copy()  # [B, N_gfi, E, S_gfi]
            B_gfi, N_gfi, E_gfi, S_gfi = gfi_weights.shape
            
            # 统计每个样本最后一个context_block的非零权重数量
            last_block_stats = []
            for b in range(B_gfi):
                # 获取最后一个context_block的权重 [E, S_gfi]
                last_block = gfi_weights[b, -1, :, :]  # 最后一个context_block
                
                # 统计非零权重数量（对所有expert求平均）
                non_zero_counts = []
                for e in range(E_gfi):
                    non_zero_count = np.count_nonzero(last_block[e, :])
                    non_zero_counts.append(non_zero_count)
                
                avg_non_zero_count = np.mean(non_zero_counts)
                scaling_factor = avg_non_zero_count / S_gfi if S_gfi > 0 else 0
                last_block_stats.append((avg_non_zero_count, scaling_factor))
                
                # 应用缩放因子到最后一个context_block
                if scaling_factor > 0:
                    adjusted_weights[b, -1, :, :] = gfi_weights[b, -1, :, :] * scaling_factor
                
                if b == 0:  # 只打印第一个样本的统计信息
                    print(f"  GFI块{gfi_idx} 样本{b}: 非零权重数={avg_non_zero_count:.1f}/{S_gfi}, 缩放因子={scaling_factor:.3f}")
            
            adjusted_gfi_weights_list.append(adjusted_weights)
            
        # 替换原始权重列表
        gfi_weights_list = adjusted_gfi_weights_list

    all_snp_weights = []
    for b in sample_indices:
        snp_weights_per_expert = []
        for e in expert_indices:
            # 1. 最后一层GFI块，拼接所有token
            tokens = gfi_weights_list[-1][b, :, min(e, gfi_weights_list[-1].shape[2]-1), :].reshape(-1)  # [N_last * S_last]
            
            # 2. 依次向前展开
            if num_gfi == 1:
                if b == sample_indices[0] and e == expert_indices[0]:  # 只打印一次
                    print("注意：当前只有一个GFI块，不进行多层展开。")
            
            for gfi_idx in reversed(range(num_gfi - 1)):
                current_block = gfi_weights_list[gfi_idx][b, :, min(e, gfi_weights_list[gfi_idx].shape[2]-1), :]  # [N_gfi, S_gfi]
                N_gfi = current_block.shape[0]
                # 截取前N_gfi个token
                tokens = tokens[:N_gfi]
                # 矩阵乘法展开
                tokens = (tokens[:, None] * current_block).reshape(-1)
            
            # 3. 与embedding层对齐
            tokens = tokens[:N_emb]
            embedding_weights = emb_weights[b, :, e, :]  # [N_emb, S_emb]
            snp_weights = (tokens[:, None] * embedding_weights).reshape(-1)  # [N_emb * S_emb]
            
            # 归一化
            if normalize:
                s = np.sum(np.abs(snp_weights))
                if s > 0:
                    snp_weights = snp_weights / s
                    
            snp_weights_per_expert.append(snp_weights)
        all_snp_weights.append(np.stack(snp_weights_per_expert, axis=0))  # [E, S_SNP]
    
    all_snp_weights = np.stack(all_snp_weights, axis=0)  # [B, E, S_SNP]
    print(f"计算完成，最终权重形状: {all_snp_weights.shape}")
    
    if output_path:
        np.save(output_path, all_snp_weights)
        print(f"已保存展开权重到 {output_path}")
    
    return all_snp_weights

def analyze_snp_weights(
    weights: np.ndarray,
    output_dir: str = None,
    top_k: int = 100,
    expert_indices: List[int] = None
):
    """
    分析SNP权重并生成统计信息
    
    :param weights: 计算得到的SNP权重，形状 [B, E, S_SNP]
    :param output_dir: 可选，输出目录
    :param top_k: 输出前K个重要SNP
    :param expert_indices: 指定要分析的专家索引
    :return: Dict，包含统计信息
    """
    B, E, S = weights.shape
    expert_indices = expert_indices if expert_indices is not None else range(E)
    
    results = {}
    
    # 1. 计算每个SNP的平均权重（按专家）
    avg_weights_per_expert = np.mean(np.abs(weights), axis=0)  # [E, S]
    
    # 2. 计算总体平均权重
    avg_weights_all = np.mean(avg_weights_per_expert[expert_indices], axis=0)  # [S]
    
    # 3. 找出每个专家的top-k SNPs
    for e in expert_indices:
        expert_weights = avg_weights_per_expert[e]  # [S]
        top_indices = np.argsort(-expert_weights)[:top_k]  # 按权重降序排列
        top_weights = expert_weights[top_indices]
        
        results[f"expert_{e}_top_indices"] = top_indices
        results[f"expert_{e}_top_weights"] = top_weights
    
    # 4. 找出总体的top-k SNPs
    top_indices_all = np.argsort(-avg_weights_all)[:top_k]
    top_weights_all = avg_weights_all[top_indices_all]
    
    results["top_indices_all"] = top_indices_all
    results["top_weights_all"] = top_weights_all
    
    # 5. 保存结果到文件
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # 保存总体top-k
        with open(os.path.join(output_dir, "top_snps_all.txt"), "w") as f:
            f.write("SNP_index\tAverage_weight\n")
            for idx, weight in zip(top_indices_all, top_weights_all):
                f.write(f"{idx}\t{weight:.6f}\n")
        
        # 保存每个专家的top-k
        for e in expert_indices:
            with open(os.path.join(output_dir, f"top_snps_expert_{e}.txt"), "w") as f:
                f.write("SNP_index\tAverage_weight\n")
                top_indices = results[f"expert_{e}_top_indices"]
                top_weights = results[f"expert_{e}_top_weights"]
                for idx, weight in zip(top_indices, top_weights):
                    f.write(f"{idx}\t{weight:.6f}\n")
    
    return results

def batch_process_weights(
    h5_path: str,
    output_dir: str,
    batch_size: int = 10,
    normalize: bool = True,
    expert_indices: List[int] = None,
    adjust_last_block: bool = True  # 新增参数
):
    """
    批量处理权重文件
    
    :param h5_path: HDF5权重文件路径
    :param output_dir: 输出目录
    :param batch_size: 批量处理的样本数
    :param normalize: 是否对每个样本的权重进行归一化 
    :param expert_indices: 指定要处理的专家索引
    :param adjust_last_block: 是否调整最后一个context_block的权重
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 首先获取样本总数
    with h5py.File(h5_path, 'r') as f:
        if 'embedding' in f and 'pooling_weights' in f['embedding']:
            emb_weights = f['embedding/pooling_weights']
            total_samples = emb_weights.shape[0]
        else:
            # 兼容旧的文件结构
            emb_weights = f['embedding_pooling_weights']
            total_samples = emb_weights.shape[0]
        
    print(f"总样本数: {total_samples}")
    
    # 按批次处理
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        print(f"处理样本 {start_idx} 到 {end_idx-1}")
        
        sample_indices = list(range(start_idx, end_idx))
        batch_output_path = os.path.join(output_dir, f"snp_weights_batch_{start_idx}_{end_idx-1}.npy")
        
        # 计算该批次的权重
        weights = expand_pooling_weights(
            h5_path=h5_path,
            output_path=batch_output_path,
            sample_indices=sample_indices,
            expert_indices=expert_indices,
            normalize=normalize,
            adjust_last_block=adjust_last_block
        )
        
        # 分析该批次的权重（可选）
        batch_analysis_dir = os.path.join(output_dir, f"analysis_batch_{start_idx}_{end_idx-1}")
        analyze_snp_weights(weights, batch_analysis_dir, expert_indices=expert_indices)
    
    print("批量处理完成")

# 用法示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='计算SNP对最终输出的影响权重')
    parser.add_argument('--h5_path', type=str, required=True, help='HDF5权重文件路径')
    parser.add_argument('--output_dir', type=str, default='snp_weights_output', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=10, help='批量处理的样本数')
    parser.add_argument('--normalize', action='store_true', help='是否对每个样本的权重进行归一化')
    parser.add_argument('--no_adjust_last_block', action='store_true', help='不调整最后一个context_block的权重')  # 新增参数
    parser.add_argument('--experts', type=str, default=None, help='要处理的专家索引，用逗号分隔')
    parser.add_argument('--samples', type=str, default=None, help='要处理的样本索引，用逗号分隔')
    parser.add_argument('--analyze_only', action='store_true', help='仅分析已存在的权重文件，不计算新权重')
    parser.add_argument('--single_file', action='store_true', help='单独处理，不分批')
    
    args = parser.parse_args()
    
    adjust_last_block = not args.no_adjust_last_block  # 默认启用调整
    
    expert_indices = None
    if args.experts:
        expert_indices = [int(e) for e in args.experts.split(',')]
    
    sample_indices = None
    if args.samples:
        sample_indices = [int(s) for s in args.samples.split(',')]
    
    if args.analyze_only:
        # 分析已存在的权重文件
        weight_files = [f for f in os.listdir(args.output_dir) if f.startswith('snp_weights_') and f.endswith('.npy')]
        for file in weight_files:
            print(f"分析权重文件: {file}")
            weights = np.load(os.path.join(args.output_dir, file))
            analyze_dir = os.path.join(args.output_dir, f"analysis_{file[:-4]}")
            analyze_snp_weights(weights, analyze_dir, expert_indices=expert_indices)
    elif args.single_file:
        # 单独处理所有样本
        output_path = os.path.join(args.output_dir, "snp_weights_all.npy")
        weights = expand_pooling_weights(
            h5_path=args.h5_path,
            output_path=output_path,
            sample_indices=sample_indices,
            expert_indices=expert_indices,
            normalize=args.normalize,
            adjust_last_block=adjust_last_block
        )
        analyze_dir = os.path.join(args.output_dir, "analysis_all")
        analyze_snp_weights(weights, analyze_dir, expert_indices=expert_indices)
    else:
        # 批量处理
        batch_process_weights(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            normalize=args.normalize,
            expert_indices=expert_indices,
            adjust_last_block=adjust_last_block
        )