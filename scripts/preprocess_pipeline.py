from pathlib import Path
import logging
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing
from typing import Optional, Tuple, Dict, List, Any
import time

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
from utils.preprocess_utils import PreprocessUtils, PreprocessResult

# 在程序早期配置 root logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PreprocessPipeline:
    """预处理流水线类 - 大部分文件处理逻辑已移至DataLoader中"""
    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config"

    def __init__(self, config_path=None):
        """
        初始化预处理流水线
        
        Args:
            config_path: 配置文件目录路径，默认使用项目根目录下的config目录
        """
        # 设置日志记录器
        self.logger = logging.getLogger("PreprocessPipeline")
        # 检查是否已有处理器，避免重复添加
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.logger.propagate = False # 添加此行防止日志向上传播
        
        # 加载配置
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self.logger.info(f"使用配置目录: {self.config_path}")
        
        # 初始化DataLoader，并将当前logger传递给它
        self.data_loader = DataLoader(self.config_path, logger=self.logger)
        
        # 初始化预处理工具
        self.preprocess_utils = PreprocessUtils(logger=self.logger) # 传递logger
        
        # 设置结果
        self.result = PreprocessResult(success=False, message="")

    def run(self, enable_qc=False, enable_mic=False) -> PreprocessResult:
        """
        运行完整的预处理流水线
        
        Args:
            enable_qc: 是否启用质量控制，默认不启用
            enable_mic: 是否启用MIC分析，默认不启用
            
        Returns:
            PreprocessResult: 处理结果
        """
        start_time = time.time()
        
        try:
            # 检查HDF5支持
            if not HAS_H5PY:
                self.logger.warning("未检测到h5py库，数据将只能以NumPy格式保存。请使用'pip install h5py'获得HDF5支持")
                
            # 1. 质量控制
            if enable_qc:
                self.logger.info("开始质量控制...")
                try:
                    from scripts.snp_qc import SNPQualityControl
                    
                    # 创建SNP质量控制对象，传递配置路径和logger
                    snp_qc = SNPQualityControl(
                        config_path=str(self.config_path), 
                        logger=self.logger
                    )
                    
                    # 运行质量控制
                    output_prefix = snp_qc.run_qc()
                    
                    # 如果启用了LD pruning，执行LD pruning
                    if snp_qc.config.preprocessing.ld_pruning.enable:
                        final_prefix = snp_qc.run_ld_pruning(output_prefix)
                        self.logger.info(f"LD pruning完成，最终输出文件前缀: {final_prefix}")
                    else:
                        self.logger.info(f"质量控制完成，输出文件前缀: {output_prefix}")
                        
                    # 将质控后的文件路径保存到结果中
                    self.result.qc_output_prefix = final_prefix if snp_qc.config.preprocessing.ld_pruning.enable else output_prefix
                    
                except Exception as e:
                    self.logger.error(f"质量控制过程出错: {str(e)}")
                    if enable_qc:
                        raise  # 如果明确启用了QC，则失败时抛出异常
                
                self.logger.info("质量控制完成")

            # 2. MIC分析
            if enable_mic:
                self.logger.info("开始MIC分析...")
                try:
                    from scripts.calc_mic import GWASAnalyzer
                    
                    # 创建GWAS分析器对象
                    analyzer = GWASAnalyzer(config_path=str(self.config_path))
                    
                    # 运行MIC分析
                    mic_results = analyzer.analyze()
                    
                    # 将MIC分析结果保存到结果对象中
                    self.result.mic_results = mic_results
                    
                    self.logger.info(f"MIC分析完成，共分析 {len(mic_results.columns) - 4} 个表型")
                    
                except Exception as e:
                    self.logger.error(f"MIC分析过程出错: {str(e)}")
                    if enable_mic:
                        raise  # 如果明确启用了MIC分析，则失败时抛出异常
                
                self.logger.info("MIC分析完成")
            
            # 3. 调用data_loader的load_bed_data方法处理SNP数据
            self.logger.info("开始处理SNP数据...")
            processedfile_path = self.data_loader.load_bed_data()
            import gc
            gc.collect()  # 强制垃圾回收
            # 加载SNP信息（bim数据）
            snp_info = self.data_loader.load_snp_info()
            self.logger.info(f"SNP数据处理完成，共加载 {len(snp_info)} 个SNP")

            # 创建snp_data字典，包含所需信息
            snp_data = {'snp_info': snp_info}

            # 4. 定义一个函数处理分区
            def process_genome_partitioning(snp_info, bed_file_path, partition_size=1000):
                """
                使用GenomePartitioner处理基因组分区
                
                Args:
                    snp_info: SNP信息数据框
                    bed_file_path: BED文件路径
                    partition_size: 分区大小
                    
                Returns:
                    partition_info: 分区信息
                """
                from utils.reference_genome_reader import ReferenceGenomeReader, GenomePartitioner
                
                # 确保 bed_file_path 是一个有效的路径字符串
                if not isinstance(bed_file_path, (str, Path)) or not bed_file_path:
                    self.logger.warning(f"无效的参考基因组文件路径: {bed_file_path}。将跳过基因组分区。")
                    return self._create_simple_partitioning(snp_info, partition_size)

                # 检查文件是否存在
                bed_path = Path(bed_file_path)
                if not bed_path.exists():
                    self.logger.warning(f"参考基因组文件 {bed_file_path} 未找到。将使用简单分区策略。")
                    return self._create_simple_partitioning(snp_info, partition_size)

                try:
                    # 创建参考基因组读取器
                    self.logger.info(f"尝试读取参考基因组: {bed_file_path}")
                    reader = ReferenceGenomeReader(str(bed_file_path)) # 确保是字符串路径
                except Exception as e: # 捕获其他可能的初始化错误
                    self.logger.error(f"读取参考基因组 {bed_file_path} 时发生错误: {str(e)}。将使用简单分区策略。")
                    return self._create_simple_partitioning(snp_info, partition_size)
                
                # 创建基因组分区器
                self.logger.info("创建基因组分区器...")
                partitioner = GenomePartitioner(reader, snp_info)
                
                # 执行分区
                self.logger.info(f"执行基因组分区，目标分区大小: {partition_size}")
                partitions = partitioner.partition_genome(partition_size)
                
                # 获取分区索引
                partition_indices = partitioner.get_partition_indices()
                
                return {
                    'partitions': partitions,
                    'partition_indices': partition_indices,
                    'partition_structure': partitioner.get_partition_structure()
                }

            # 5. 调用上面定义的函数，实现分区处理
            self.logger.info("开始基因组分区...")
            bed_file_path = self.data_loader.reference_genome_path
            
            # 获取模型配置以确定分区大小
            from config.config import Config
            try:
                # 从配置文件加载模型配置
                model_config_path = str(self.config_path / "model_configblackcarp.json")
                preprocess_config_path = str(self.config_path / "preprocessingblackcarp_config.json")
                full_config = Config.from_json(preprocess_config_path, model_config_path)
                
                # 确定分区大小
                partition_size = 300  # 默认值
                
                if hasattr(full_config.preprocessing, 'model_input') and full_config.preprocessing.model_input:
                    partition_config = full_config.preprocessing.model_input.partition
                    
                    if partition_config.enable:
                        # 如果启用自定义分区，直接使用max_size
                        partition_size = partition_config.max_size
                        self.logger.info(f"使用配置设定的分区大小: {partition_size}")
                    else:
                        # 如果未启用自定义分区，根据模型参数量计算分区大小
                        self.logger.info("根据模型参数量动态计算分区大小...")
                        
                        # 导入参数量计算工具
                        from utils.preprocess_utils import PreprocessUtils
                        utils = PreprocessUtils()
                        
                        # 获取模型的Transformer块
                        blocks = full_config.model.GFI_FormerBLOCKS.blocks
                        if len(blocks) >= 2:
                            # 获取SNP transformer和Gene transformer
                            snp_transformer = blocks[0]
                            gene_transformer = blocks[1]
                            
                            # 计算参数量
                            snp_pa = utils.calcul_transformer_params(snp_transformer)                            
                            gene_pa = utils.calcul_transformer_params(gene_transformer)
                            
                            # 获取总SNP数量
                            n_snp = len(snp_data['snp_info'])
                            
                            # 计算partition_size = sqrt(n_snp/(gene_pa/snp_pa))
                            if snp_pa > 0:
                                param_ratio = gene_pa / snp_pa
                                partition_size = int(np.sqrt(n_snp / param_ratio))
                                
                                # 确保分区大小在合理范围内
                                if hasattr(partition_config, 'min_size') and partition_size < partition_config.min_size:
                                    partition_size = partition_config.min_size
                                
                                self.logger.info(f"动态计算的分区大小: {partition_size}")
                                self.logger.info(f"计算依据: SNP数量={n_snp}, SNP参数量={snp_pa}, 基因参数量={gene_pa}, 参数比={param_ratio:.2f}")
                        else:
                            self.logger.warning("模型块数量不足，无法计算参数比例，使用默认分区大小")
                    
            except Exception as e:
                self.logger.warning(f"配置处理出错，使用默认分区大小(300): {str(e)}")
                partition_size = 300
            
            # 执行分区
            partition_info = process_genome_partitioning(
                snp_data['snp_info'], 
                bed_file_path,
                partition_size
            )
            
            # 打印分区信息
            partitions = partition_info.get('partitions', []) # 使用 .get 获取，更安全
            num_partitions = len(partitions)
            self.logger.info(f"基因组分区处理完成，共生成 {num_partitions} 个分区") # 调整日志消息
            
            # 计算分区统计信息
            if partitions: # 检查 partitions 是否非空
                snp_counts = [count for _, _, _, count in partitions]
                if snp_counts: # 确保 snp_counts 实际有内容
                    min_snps = min(snp_counts)
                    max_snps = max(snp_counts)
                    avg_snps = sum(snp_counts) / len(snp_counts) if len(snp_counts) > 0 else 0
                    self.logger.info(f"分区统计: 最小分区 {min_snps} SNPs, 最大分区 {max_snps} SNPs, 平均 {avg_snps:.2f} SNPs")
                else:
                    self.logger.info("分区内无SNP计数信息可供统计。")
            elif bed_file_path: # 仅在尝试了分区（即 bed_file_path 有效）但未成功时记录
                self.logger.info("未生成有效分区（可能由于文件未找到或读取错误），跳过分区统计。")
            # 如果 bed_file_path 本身就无效，则跳过分区的信息已在前面记录 (如 "参考基因组BED文件路径未配置...")
            
            # 添加分区信息到结果
            self.result.partition_info = partition_info
            self.result.snp_data = snp_data
            self.result.success = True
                      
            # 计算总运行时间
            end_time = time.time()
            total_time = end_time - start_time
            self.logger.info(f"预处理流水线完成，总耗时: {total_time:.2f} 秒")
            
            return self.result
            
        except Exception as e:
            self.logger.error(f"预处理流水线出错: {str(e)}", exc_info=True)
            self.result.success = False
            self.result.message = str(e)
            return self.result


    def _create_simple_partitioning(self, snp_info: pd.DataFrame, partition_size: int = 1000) -> Dict[str, Any]:
        """
        当参考基因组不可用时，创建简单的基于位置的分区
        
        Args:
            snp_info: SNP信息数据框
            partition_size: 目标分区大小
            
        Returns:
            简单分区信息
        """
        self.logger.info(f"创建简单分区策略，目标分区大小: {partition_size}")
        
        # 按染色体和位置排序
        snp_info_sorted = snp_info.sort_values(['chr', 'position']).reset_index(drop=True)
        
        total_snps = len(snp_info_sorted)
        n_partitions = (total_snps + partition_size - 1) // partition_size
        
        partitions = []
        partition_indices = []
        
        for i in range(n_partitions):
            start_idx = i * partition_size
            end_idx = min((i + 1) * partition_size - 1, total_snps - 1)
            
            if start_idx >= total_snps:
                break
                
            start_chrom = snp_info_sorted.iloc[start_idx]['chr']
            start_pos = snp_info_sorted.iloc[start_idx]['position']
            end_chrom = snp_info_sorted.iloc[end_idx]['chr']
            end_pos = snp_info_sorted.iloc[end_idx]['position']
            snp_count = end_idx - start_idx + 1
            
            partitions.append((start_chrom, start_pos, end_pos, snp_count))
            
            # 为这个分区中的所有SNP分配分区索引
            for j in range(start_idx, end_idx + 1):
                partition_indices.append(i)
        
        # 创建分区结构
        partition_structure = []
        for i, (chrom, start_pos, _, _) in enumerate(partitions):
            partition_structure.append((i, chrom, start_pos))
        
        self.logger.info(f"创建了 {len(partitions)} 个简单分区")
        
        return {
            'partitions': partitions,
            'partition_indices': partition_indices,
            'partition_structure': partition_structure
        }

def main():
    """主程序入口"""
    import argparse
    parser = argparse.ArgumentParser(description="SNP预处理流水线")
    parser.add_argument("--config", type=str, help="配置文件目录路径")
    parser.add_argument("--enable-qc", action="store_true", help="启用质量控制，默认不启用")
    parser.add_argument("--enable-mic", action="store_true", help="启用MIC分析，默认不启用")
    
    args = parser.parse_args()
    
    try:
        pipeline = PreprocessPipeline(args.config)
        result = pipeline.run(
            enable_qc=args.enable_qc,
            enable_mic=args.enable_mic
        )
        if not result.success:
            print(f"错误: {result.message}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
