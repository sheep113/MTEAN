import numpy as np
import pandas as pd
from minepy import MINE
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, NamedTuple, Optional
import warnings
from tqdm import tqdm
import sys

# 导入配置类
sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config, ConfigValidationError
from utils.data_loader import DataLoader, timer


class MICResult(NamedTuple):
    """MIC计算结果"""
    mic: float
    samples: int

def process_chunk(chunk_data: Tuple[np.ndarray, np.ndarray, float, int, int]) -> List[Tuple[float, int]]:
    """处理数据块
    Args:
        chunk_data: (snp_chunk, pheno_values, alpha, c, min_samples)
    Returns:
        List of (mic_value, sample_count) tuples
    """
    snp_chunk, pheno_values, alpha, c, min_samples = chunk_data
    results = []
    
    if snp_chunk is None or len(snp_chunk) == 0:
        return results
    
    for snp in snp_chunk:
        # 检查SNP数据的有效性
        if snp is None or len(snp) == 0:
            results.append((np.nan, 0))
            continue
            
        # 计算单个SNP的MIC值
        mask = ~(np.isnan(snp) | np.isnan(pheno_values))
        x_clean = snp[mask]
        y_clean = pheno_values[mask]
        
        if len(x_clean) < min_samples:
            results.append((np.nan, len(x_clean)))
            continue
        
        # 检查值的唯一性
        if len(np.unique(x_clean)) <= 1 or len(np.unique(y_clean)) <= 1:
            results.append((0.0, len(x_clean)))
            continue
            
        try:
            mine = MINE(alpha=alpha, c=c)
            mine.compute_score(x_clean, y_clean)
            results.append((mine.mic(), len(x_clean)))
        except Exception as e:
            # 记录具体错误但继续处理
            print(f"计算MIC时出错: {e}")
            results.append((np.nan, len(x_clean)))
    
    return results

class GWASAnalyzer:
    """GWAS分析主类"""
    def __init__(self, config_path: str):
        """
        初始化分析器
        Args:
            config_path: 配置文件目录路径
        """
        # 先设置好日志记录器
        self.logger = self._setup_logger()
        try:
            # 将logger传递给DataLoader以避免重复日志
            self.data_loader = DataLoader(config_path, logger=self.logger)
            self.config = self.data_loader.config
            self.mic_config = self.config.preprocessing.mic_analysis
            
            if not self.mic_config.enable:
                raise ConfigValidationError("MIC分析未启用")
                
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            raise

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('GWASAnalyzer')
        # 检查是否已经有处理器，避免重复添加
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(handler)
        return logger

    def read_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """读取所需的所有数据"""
        pheno_df = self.data_loader.load_phenotype()
        snp_df = self.data_loader.load_snp_info()
        return pheno_df, snp_df

    def analyze(self) -> pd.DataFrame:
        """执行完整的分析流程"""
        try:
            pheno_df, snp_df = self.read_data()
            
            # 验证数据的有效性
            if pheno_df.empty or snp_df.empty:
                self.logger.error("表型数据或SNP数据为空")
                raise ValueError("无效的数据集")
                
            output_df = snp_df.copy()
            
            # 获取输出文件路径
            output_path = self.data_loader.output_path
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 检查是否存在已有结果文件
            #if output_path.exists():
               # self.logger.info("检测到已有结果文件，将继续未完成的计算")
               # output_df = pd.read_csv(output_path, sep='\t')
               # processed_phenos = {col.replace('_mic', '') for col in output_df.columns 
               #                  if col.endswith('_mic')}
               # pheno_df = pheno_df.drop(columns=processed_phenos, errors='ignore')
                #if len(pheno_df.columns) == 0:
                #    self.logger.info("所有表型都已计算完成")
                #    return output_df
            
            # 获取SNP数据
            snp_data = self.data_loader.load_bed_data_all()
            # 如果是元组，取第一个元素作为基因型矩阵
            if isinstance(snp_data, tuple):
                snp_data = snp_data[0]
            if snp_data.ndim == 3:
                snp_data = snp_data[:, :, 0] + snp_data[:, :, 1]
            
            with ProcessPoolExecutor(max_workers=self.mic_config.num_threads) as executor:
                for pheno_name in pheno_df.columns:
                    try:
                        self.logger.info(f"\n处理表型: {pheno_name}")
                        pheno_values = pheno_df[pheno_name].astype(np.float64).values
                        
                        chunks = np.array_split(
                            snp_data,
                            max(1, len(snp_data) // self.mic_config.chunk_size)
                        )
                        
                        futures = [
                            executor.submit(
                                process_chunk, 
                                (chunk, pheno_values, self.mic_config.alpha, 
                                 self.mic_config.c, self.mic_config.min_samples)
                            )
                            for chunk in chunks
                        ]
                        
                        chunk_results = []
                        for future in tqdm(futures, desc=f"计算 {pheno_name} 的MIC值"):
                            chunk_results.extend(future.result())
                        
                        # 处理结果并保存
                        self._process_and_save_results(
                            output_df, pheno_name, chunk_results, output_path
                        )
                        
                    except Exception as e:
                        self.logger.error(f"处理表型 {pheno_name} 时发生错误: {str(e)}")
                        continue
            
            return output_df
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

    def _process_and_save_results(
        self, output_df: pd.DataFrame, pheno_name: str, 
        chunk_results: List[Tuple[float, int]], output_path: Path   
    ) -> None:
        self.logger.info(f"结果文件已保存至: {output_path}")
        """处理和保存分析结果"""
        mic_values, sample_counts = map(np.array, zip(*chunk_results))
        output_df[f'{pheno_name}_mic'] = pd.Series(mic_values, dtype=np.float32)
        output_df[f'{pheno_name}_samples'] = pd.Series(sample_counts, dtype=np.int32)
        
        # 确保列的顺序正确
        cols = ['chr', 'snp_id', 'genetic_dist', 'position']
        for col in output_df.columns:
            if col not in cols:
                cols.append(col)
        output_df = output_df[cols]
        
        # 保存当前进度
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, sep='\t', index=False)
        
        # 打印统计信息
        self._log_analysis_stats(pheno_name, mic_values, sample_counts)
        self.logger.info(f"结果文件已保存至: {output_path}")

    def _log_analysis_stats(self, pheno_name: str, 
                          mic_values: np.ndarray, sample_counts: np.ndarray) -> None:
        """记录分析统计信息"""
        valid_mics = mic_values[~np.isnan(mic_values)]
        self.logger.info(f"\n{pheno_name} 计算结果统计:")
        self.logger.info(f"有效MIC值数量: {len(valid_mics)}")
        if len(valid_mics) > 0:
            self.logger.info(f"MIC值范围: [{np.min(valid_mics):.4f}, {np.max(valid_mics):.4f}]")
            self.logger.info(f"MIC值均值: {np.mean(valid_mics):.4f}")
            self.logger.info(f"MIC值中位数: {np.median(valid_mics):.4f}")
        self.logger.info(f"样本量范围: [{np.min(sample_counts)}, {np.max(sample_counts)}]")
        self.logger.info(f"平均样本量: {np.mean(sample_counts):.1f}")

def main() -> None:
    """主程序入口"""
    parser = argparse.ArgumentParser(description='GWAS MIC Analysis')
    parser.add_argument('--config', required=True, type=str,
                       help='配置文件目录路径')
    parser.add_argument('--phenotype', type=str, default=None,
                        help='仅分析指定的表型（可选）')
    parser.add_argument('--threads', type=int, default=None,
                        help='线程数，覆盖配置文件中的设置')
    parser.add_argument('--output', type=str, default=None,
                        help='结果输出路径，覆盖配置文件中的设置')
    
    args = parser.parse_args()
    
    # 配置根日志器 - 移除全局日志配置，避免与类中的日志配置冲突
    # 让GWASAnalyzer类管理自己的日志
    
    try:
        # 检查配置路径是否存在
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"配置路径不存在: {args.config}")
            
        # 日志警告级别设置
        warnings.simplefilter("ignore")
        
        # 创建分析器并运行
        analyzer = GWASAnalyzer(args.config)
        analyzer.analyze()
        analyzer.logger.info("分析完成!")
        
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except ConfigValidationError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"运行时错误: {str(e)}", file=sys.stderr)
        print("详细错误信息:", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
