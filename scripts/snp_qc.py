import subprocess
import logging
import sys
import re
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
import tempfile
import os
import time
import contextlib
import pandas as pd
import numpy as np

# 导入配置类
sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config, ConfigValidationError
from utils.data_loader import DataLoader, timer


@contextlib.contextmanager
def timer(description: str, logger: Optional[logging.Logger] = None):
    """
    Context manager for timing code execution
    
    Args:
        description: Description of the operation being timed
        logger: Logger to use for output
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        message = f"{description} took {elapsed:.2f} seconds"
        if logger:
            logger.info(message)
        else:
            print(message)

class SNPQualityControl:
    """SNP质量控制类"""
    
    def __init__(self, config_path: str, logger: Optional[logging.Logger] = None):
        """
        初始化SNP质量控制
        
        Args:
            config_path: 配置文件目录路径
        """
        # 使用传入的logger或创建新的logger
        self.logger = logger if logger is not None else self._setup_logger()
        
        # 初始化DataLoader，传递相同的logger
        self.data_loader = DataLoader(config_path, logger=self.logger)
        self.config = self.data_loader.config
        self.qc_config = self.config.preprocessing.snp_filtering

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('SNPQualityControl')
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def build_plink_command(self) -> List[str]:
        """构建PLINK命令"""
        input_prefix = self.data_loader.plink_prefix
        output_dir = Path(self.config.preprocessing.file_processing.output_directory)
        output_prefix = output_dir / self.qc_config.output_prefix

        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "plink",
            "--bfile", str(input_prefix),
            "--out", str(output_prefix),
            "--threads", str(self.qc_config.threads),
            "--memory", str(self.qc_config.memory)
        ]

        # 添加过滤参数
        if self.qc_config.geno['enable']:
            cmd.extend(["--geno", str(self.qc_config.geno['threshold'])])
        if self.qc_config.maf['enable']:
            cmd.extend(["--maf", str(self.qc_config.maf['threshold'])])
        if self.qc_config.hwe['enable']:
            cmd.extend(["--hwe", str(self.qc_config.hwe['threshold'])])
        if self.qc_config.mind['enable']:
            cmd.extend(["--mind", str(self.qc_config.mind['threshold'])])

        # 添加其他选项
        if self.qc_config.allow_no_sex:
            cmd.append("--allow-no-sex")
        if self.qc_config.autosome_only:
            cmd.append("--autosome")

        # 添加额外的命令
        if self.qc_config.extra_commands:
            cmd.extend(self.qc_config.extra_commands.split())

        # 添加make-bed以生成新的二进制文件
        cmd.append("--make-bed")

        return cmd

    def _count_snps_in_file(self, file_path: Path) -> int:
        """
        计算文件中SNP的数量
        
        Args:
            file_path: 包含SNP列表的文件路径
            
        Returns:
            SNP的数量
        """
        try:
            if not file_path.exists():
                return 0
            with open(file_path, 'r') as f:
                return sum(1 for _ in f)
        except Exception as e:
            self.logger.error(f"计算文件中SNP数量失败: {str(e)}")
            return 0

    def _parse_plink_log(self, log_file: Path) -> Dict:
        """
        解析PLINK日志文件以获取SNP统计信息
        
        Args:
            log_file: PLINK日志文件路径
            
        Returns:
            包含SNP统计信息的字典
        """
        stats = {
            'total_snps': 0,
            'filtered_snps': {
                'geno': 0,
                'maf': 0,
                'hwe': 0,
                'mind': 0
            },
            'remaining_snps': 0
        }
        
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
                
                # 获取总SNP数
                total_match = re.search(r'(\d+) variants loaded', log_content)
                if total_match:
                    stats['total_snps'] = int(total_match.group(1))
                
                # 获取各步骤过滤的SNP数
                geno_match = re.search(r'(\d+) variants removed due to missing genotype data', log_content)
                if geno_match and self.qc_config.geno['enable']:
                    stats['filtered_snps']['geno'] = int(geno_match.group(1))
                
                maf_match = re.search(r'(\d+) variants removed due to minor allele threshold', log_content)
                if maf_match and self.qc_config.maf['enable']:
                    stats['filtered_snps']['maf'] = int(maf_match.group(1))
                
                hwe_match = re.search(r'(\d+) variants removed due to Hardy-Weinberg exact test', log_content)
                if hwe_match and self.qc_config.hwe['enable']:
                    stats['filtered_snps']['hwe'] = int(hwe_match.group(1))
                
                # 获取剩余的SNP数量
                remaining_match = re.search(r'(\d+) variants and \d+ people pass filters and QC', log_content)
                if remaining_match:
                    stats['remaining_snps'] = int(remaining_match.group(1))
                
            return stats
        except Exception as e:
            self.logger.error(f"解析PLINK日志文件失败: {str(e)}")
            return stats

    def run_qc(self) -> Optional[Path]:
        """
        运行SNP质量控制
        
        Returns:
            Optional[Path]: 输出文件前缀的路径
        """
        try:
            cmd = self.build_plink_command()
            self.logger.info(f"执行命令: {' '.join(cmd)}")
            
            with timer("Running PLINK QC", self.logger):
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

            # 返回输出文件前缀
            output_prefix = Path(self.config.preprocessing.file_processing.output_directory) / \
                          self.qc_config.output_prefix
            
            # 解析日志文件获取SNP统计信息
            log_file = output_prefix.with_suffix('.log')
            stats = self._parse_plink_log(log_file)
            
            # 输出详细的统计信息
            self.logger.info(f"质控统计信息:")
            self.logger.info(f"  总SNP数量: {stats['total_snps']}")
            if self.qc_config.geno['enable']:
                self.logger.info(f"  基于缺失率(--geno)过滤的SNP数量: {stats['filtered_snps']['geno']}")
            if self.qc_config.maf['enable']:
                self.logger.info(f"  基于最小等位基因频率(--maf)过滤的SNP数量: {stats['filtered_snps']['maf']}")
            if self.qc_config.hwe['enable']:
                self.logger.info(f"  基于Hardy-Weinberg平衡(--hwe)过滤的SNP数量: {stats['filtered_snps']['hwe']}")
            if self.qc_config.mind['enable']:
                self.logger.info(f"  样本过滤导致的SNP数量变化: {stats['filtered_snps']['mind']}")
            
            total_filtered = sum(stats['filtered_snps'].values())
            self.logger.info(f"  总过滤SNP数量: {total_filtered} ({total_filtered/stats['total_snps']*100:.2f}%)")
            self.logger.info(f"  保留SNP数量: {stats['remaining_snps']} ({stats['remaining_snps']/stats['total_snps']*100:.2f}%)")
            
            self.logger.info("QC完成!")
            return output_prefix

        except subprocess.CalledProcessError as e:
            self.logger.error(f"PLINK执行失败: {e.stderr}")
            raise
        except Exception as e:
            self.logger.error(f"QC过程出错: {str(e)}")
            raise

    def run_ld_pruning(self, input_prefix: Path) -> Optional[Path]:
        """
        运行基于区域的LD pruning
        
        Args:
            input_prefix: 输入文件前缀路径
            
        Returns:
            Optional[Path]: LD pruning后的输出文件前缀
        """
        if not self.config.preprocessing.ld_pruning.enable:
            self.logger.info("LD pruning已禁用，跳过")
            return input_prefix

        try:
            output_dir = Path(self.config.preprocessing.file_processing.output_directory)
            gene_regions_file = Path(self.config.preprocessing.ld_pruning.gene_regions.get("bed_file", ""))

            if not gene_regions_file.exists():
                self.logger.warning(f"基因区域文件不存在: {gene_regions_file}，使用标准LD pruning")
                return self._run_standard_ld_pruning(input_prefix)

            # 创建临时目录保存中间文件
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                # 1. 从BIM文件获取SNP位置信息
                bim_file = f"{input_prefix}.bim"
                snp_df = self._load_snp_info(bim_file)
                total_snps = len(snp_df)
                self.logger.info(f"总SNP数量: {total_snps}")
                
                # 2. 从基因区域BED文件获取基因区域，并过滤染色体长度信息
                gene_regions = self._load_gene_regions_filtered(gene_regions_file)
                
                # 3. 使用DataLoader获取非编码区域
                
                # 4. 将SNP分类为基因区域和非基因区域
                genic_snps = temp_dir / "genic_snps.txt"
                nongenic_snps = temp_dir / "nongenic_snps.txt"
                self._classify_snps_by_region_enhanced(snp_df, gene_regions, genic_snps, nongenic_snps)
                
                # 获取分类后每个区域的SNP数量
                genic_count = self._count_snps_in_file(genic_snps)
                nongenic_count = self._count_snps_in_file(nongenic_snps)
                self.logger.info(f"基因区域SNP数量: {genic_count} ({genic_count/total_snps*100:.2f}%)")
                self.logger.info(f"非基因区域SNP数量: {nongenic_count} ({nongenic_count/total_snps*100:.2f}%)")
                
                # 5. 对基因区域SNPs进行LD pruning
                genic_output = temp_dir / "genic_pruned"
                genic_params = {
                    'window_size': self.config.preprocessing.ld_pruning.window_size,
                    'step_size': self.config.preprocessing.ld_pruning.step_size,
                    'r2_threshold': self.config.preprocessing.ld_pruning.gene_regions.get('genic_r2', 0.5)
                }
                self._run_plink_ld_pruning(
                    input_prefix,
                    genic_snps,
                    genic_output,
                    genic_params
                )

                # 计算基因区域LD pruning后的统计数据
                genic_kept = self._count_snps_in_file(genic_output.with_suffix('.prune.in'))
                genic_removed = self._count_snps_in_file(genic_output.with_suffix('.prune.out'))
                self.logger.info(f"基因区域LD pruning统计:")
                self.logger.info(f"  保留SNP数量: {genic_kept} ({genic_kept/genic_count*100:.2f}% 的基因区域SNP)")
                self.logger.info(f"  剔除SNP数量: {genic_removed} ({genic_removed/genic_count*100:.2f}% 的基因区域SNP)")

                # 6. 对非基因区域SNPs进行LD pruning
                nongenic_output = temp_dir / "nongenic_pruned"
                nongenic_params = {
                    'window_size': self.config.preprocessing.ld_pruning.window_size,
                    'step_size': self.config.preprocessing.ld_pruning.step_size,
                    'r2_threshold': self.config.preprocessing.ld_pruning.gene_regions.get('nongenic_r2', 0.1)
                }
                self._run_plink_ld_pruning(
                    input_prefix,
                    nongenic_snps,
                    nongenic_output,
                    nongenic_params
                )

                # 计算非基因区域LD pruning后的统计数据
                nongenic_kept = self._count_snps_in_file(nongenic_output.with_suffix('.prune.in'))
                nongenic_removed = self._count_snps_in_file(nongenic_output.with_suffix('.prune.out'))
                self.logger.info(f"非基因区域LD pruning统计:")
                self.logger.info(f"  保留SNP数量: {nongenic_kept} ({nongenic_kept/nongenic_count*100:.2f}% 的非基因区域SNP)")
                self.logger.info(f"  剔除SNP数量: {nongenic_removed} ({nongenic_removed/nongenic_count*100:.2f}% 的非基因区域SNP)")

                # 7. 合并基因和非基因区域的结果
                final_output = output_dir / f"{input_prefix.name}_ldpruned"
                self._merge_pruned_sets(
                    input_prefix,
                    genic_output.with_suffix('.prune.in'),
                    nongenic_output.with_suffix('.prune.in'),
                    final_output,
                    total_snps=total_snps,
                    genic_kept=genic_kept,
                    nongenic_kept=nongenic_kept
                )

                self.logger.info(f"区域特异性LD pruning完成，输出前缀: {final_output}")
                return final_output

        except Exception as e:
            self.logger.error(f"LD pruning失败: {str(e)}")
            import traceback
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            self.logger.warning("尝试使用标准LD pruning")
            return self._run_standard_ld_pruning(input_prefix)

    def _load_snp_info(self, bim_file: str) -> pd.DataFrame:
        """
        从BIM文件加载SNP信息
        
        Args:
            bim_file: BIM文件路径
            
        Returns:
            包含SNP信息的DataFrame
        """
        self.logger.info(f"从{bim_file}加载SNP信息")
        
        try:
            # 读取BIM文件
            snp_df = pd.read_csv(
                bim_file, 
                sep='\s+', 
                header=None,
                names=['chr', 'snp_id', 'genetic_dist', 'position', 'allele1', 'allele2']
            )
            self.logger.info(f"成功加载{len(snp_df)}个SNP")
            return snp_df
        except Exception as e:
            self.logger.error(f"加载SNP信息失败: {str(e)}")
            raise

    def _load_gene_regions_filtered(self, bed_file: Path) -> pd.DataFrame:
        """
        从BED文件加载基因区域信息，并过滤掉染色体长度记录
        
        Args:
            bed_file: BED文件路径
            
        Returns:
            包含基因区域信息的DataFrame
        """
        self.logger.info(f"从{bed_file}加载基因区域信息并过滤染色体长度记录")
        
        try:
            # 读取BED文件
            # BED格式标准列：chr, start, end, name, score, strand
            gene_df = pd.read_csv(
                bed_file, 
                sep='\t', 
                header=None,
                names=['chr', 'start', 'end', 'name', 'score', 'strand']
            )
            
            # 过滤掉染色体长度记录和其他非基因记录
            is_gene = gene_df['name'].str.contains('gene:', na=False)
            gene_df = gene_df[is_gene].copy()
            
            self.logger.info(f"成功加载{len(gene_df)}个基因区域（已过滤染色体长度记录）")
            return gene_df
        except Exception as e:
            self.logger.error(f"加载基因区域信息失败: {str(e)}")
            raise

    def _classify_snps_by_region_enhanced(self, 
                                       snp_df: pd.DataFrame, 
                                       gene_df: pd.DataFrame,
                                       noncoding_regions: Dict,
                                       genic_output: Path,
                                       nongenic_output: Path) -> None:
        """
        将SNP分类为基因区域和非基因区域(精简版)
        
        Args:
            snp_df: 包含SNP信息的DataFrame
            gene_df: 包含基因区域信息的DataFrame
            noncoding_regions: 非编码区域信息字典
            genic_output: 输出基因区域SNP的文件路径
            nongenic_output: 输出非基因区域SNP的文件路径
        """
        self.logger.info("将SNP分类为基因区域和非基因区域")
        
        try:
            # 创建染色体到基因区间的映射
            gene_intervals_by_chr = {}
            for _, gene in gene_df.iterrows():
                chr_id = str(gene['chr'])
                if chr_id not in gene_intervals_by_chr:
                    gene_intervals_by_chr[chr_id] = []
                gene_intervals_by_chr[chr_id].append((int(gene['start']), int(gene['end'])))
            
            # 对每个染色体的区间排序
            for chr_id in gene_intervals_by_chr:
                gene_intervals_by_chr[chr_id].sort()
                
            # 初始化分类结果
            genic_snps = []
            nongenic_snps = []
            
            # 处理每个SNP
            total_snps = len(snp_df)
            processed = 0
            batch_size = 100000
            
            for i in range(0, total_snps, batch_size):
                batch = snp_df.iloc[i:min(i+batch_size, total_snps)]
                
                for _, snp in batch.iterrows():
                    snp_chr = str(snp['chr'])
                    snp_pos = int(snp['position'])
                    snp_id = snp['snp_id']
                    
                    # 检查此SNP是否在任何基因区域内
                    in_gene = False
                    if snp_chr in gene_intervals_by_chr:
                        for start, end in gene_intervals_by_chr[snp_chr]:
                            if start <= snp_pos <= end:
                                in_gene = True
                                break
                    
                    # 根据分类保存SNP ID
                    if in_gene:
                        genic_snps.append(snp_id)
                    else:
                        nongenic_snps.append(snp_id)
                
                processed += len(batch)
                if processed % batch_size == 0 or processed == total_snps:
                    self.logger.info(f"已处理 {processed}/{total_snps} 个SNP")
            
            # 保存结果到文件
            with open(genic_output, 'w') as f:
                for snp_id in genic_snps:
                    f.write(f"{snp_id}\n")
            
            with open(nongenic_output, 'w') as f:
                for snp_id in nongenic_snps:
                    f.write(f"{snp_id}\n")
            
            self.logger.info(f"SNP分类完成: {len(genic_snps)}个基因区域SNP, {len(nongenic_snps)}个非基因区域SNP")
            
            # 添加统计信息
            if len(genic_snps) + len(nongenic_snps) > 0:
                genic_percent = len(genic_snps) / (len(genic_snps) + len(nongenic_snps)) * 100
                self.logger.info(f"基因区域SNP占比: {genic_percent:.2f}%")
                self.logger.info(f"非基因区域SNP占比: {100 - genic_percent:.2f}%")
        
        except Exception as e:
            self.logger.error(f"SNP分类失败: {str(e)}")
            self._create_empty_files([genic_output, nongenic_output])
            raise

    def _create_empty_files(self, file_paths: List[Path]) -> None:
        """创建空文件，用于错误恢复"""
        for file_path in file_paths:
            with open(file_path, 'w') as f:
                pass
            self.logger.warning(f"创建空文件: {file_path}")

    def _run_plink_ld_pruning(self, input_prefix: Path, snp_list: Path, 
                             output_prefix: Path, params: Dict[str, Union[float, int]]):
        """执行PLINK LD pruning"""
        # 先计算输入SNP数量
        input_snps_count = self._count_snps_in_file(snp_list)
        if input_snps_count == 0:
            self.logger.warning(f"输入SNP列表为空: {snp_list}")
            # 创建空的输出文件
            with open(output_prefix.with_suffix('.prune.in'), 'w') as f:
                pass
            with open(output_prefix.with_suffix('.prune.out'), 'w') as f:
                pass
            return
        
        cmd = [
            "plink",
            "--bfile", str(input_prefix),
            "--extract", str(snp_list),
            "--indep-pairwise",
            str(params['window_size']),
            str(params['step_size']),
            str(params['r2_threshold']),
            "--out", str(output_prefix)
        ]
        
        self.logger.info(f"执行LD pruning: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 计算LD pruning后保留和剔除的SNP数量
            kept_snps = self._count_snps_in_file(output_prefix.with_suffix('.prune.in'))
            removed_snps = self._count_snps_in_file(output_prefix.with_suffix('.prune.out'))
            
            if kept_snps + removed_snps > 0:
                self.logger.info(f"LD pruning结果: 保留{kept_snps}个SNP ({kept_snps/(kept_snps+removed_snps)*100:.2f}%), "
                               f"剔除{removed_snps}个SNP ({removed_snps/(kept_snps+removed_snps)*100:.2f}%)")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"LD pruning执行失败: {e.stderr}")
            # 确保输出文件存在
            self._create_empty_files([
                output_prefix.with_suffix('.prune.in'),
                output_prefix.with_suffix('.prune.out')
            ])

    def _merge_pruned_sets(self, input_prefix: Path, genic_prune: Path, 
                          nongenic_prune: Path, output_prefix: Path,
                          total_snps: int = 0, genic_kept: int = 0, nongenic_kept: int = 0):
        """合并两个区域的pruned SNPs集合"""
        # 检查prune文件是否存在且不为空
        genic_exists = genic_prune.exists() and genic_prune.stat().st_size > 0
        nongenic_exists = nongenic_prune.exists() and nongenic_prune.stat().st_size > 0
        
        if not genic_exists and not nongenic_exists:
            self.logger.warning("基因区域和非基因区域prune文件均不存在或为空，使用标准LD pruning")
            return self._run_standard_ld_pruning(input_prefix)
        
        # 合并SNP列表
        combined_snps = output_prefix.with_suffix('.snplist')
        combined_count = 0
        with open(combined_snps, 'w') as outf:
            if genic_exists:
                self.logger.info(f"合并基因区域SNP: {genic_prune}")
                with open(genic_prune) as inf:
                    outf.write(inf.read())
                    combined_count += genic_kept
            
            if nongenic_exists:
                self.logger.info(f"合并非基因区域SNP: {nongenic_prune}")
                with open(nongenic_prune) as inf:
                    outf.write(inf.read())
                    combined_count += nongenic_kept

        # 输出合并后的统计信息
        if total_snps > 0:
            self.logger.info(f"LD pruning后合并统计:")
            self.logger.info(f"  总保留SNP数量: {combined_count} ({combined_count/total_snps*100:.2f}% 的原始SNP)")
            self.logger.info(f"  总剔除SNP数量: {total_snps - combined_count} ({(total_snps - combined_count)/total_snps*100:.2f}% 的原始SNP)")

        # 创建最终数据集
        cmd = [
            "plink",
            "--bfile", str(input_prefix),
            "--extract", str(combined_snps),
            "--make-bed",
            "--out", str(output_prefix)
        ]
        
        self.logger.info(f"合并pruned SNP集: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            self.logger.info("SNP集合并完成")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"SNP集合并失败: {e.stderr}")
            raise

    def _run_standard_ld_pruning(self, input_prefix: Path) -> Optional[Path]:
        """执行标准的LD pruning"""
        try:
            output_dir = Path(self.config.preprocessing.file_processing.output_directory)
            output_prefix = output_dir / f"{input_prefix.name}_standard_ldpruned"
            
            # 先计算初始SNP数量
            bim_file = Path(f"{input_prefix}.bim")
            initial_snp_count = 0
            if bim_file.exists():
                with open(bim_file, 'r') as f:
                    initial_snp_count = sum(1 for _ in f)
                self.logger.info(f"LD pruning前总SNP数量: {initial_snp_count}")
            
            cmd = [
                "plink",
                "--bfile", str(input_prefix),
                "--indep-pairwise",
                str(self.config.preprocessing.ld_pruning.window_size),
                str(self.config.preprocessing.ld_pruning.step_size),
                str(self.config.preprocessing.ld_pruning.default_r2),
                "--out", str(output_prefix)
            ]
            
            self.logger.info(f"执行标准LD pruning命令: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # 计算LD pruning后的SNP统计
            kept_snps = self._count_snps_in_file(output_prefix.with_suffix('.prune.in'))
            removed_snps = self._count_snps_in_file(output_prefix.with_suffix('.prune.out'))
            
            if initial_snp_count > 0:
                kept_percent = kept_snps / initial_snp_count * 100
                removed_percent = removed_snps / initial_snp_count * 100
                self.logger.info(f"标准LD pruning统计:")
                self.logger.info(f"  保留SNP数量: {kept_snps} ({kept_percent:.2f}% 的原始SNP)")
                self.logger.info(f"  剔除SNP数量: {removed_snps} ({removed_percent:.2f}% 的原始SNP)")
            
            # 提取保留的SNPs
            cmd = [
                "plink",
                "--bfile", str(input_prefix),
                "--extract", f"{output_prefix}.prune.in",
                "--make-bed",
                "--out", str(output_prefix)
            ]
            
            self.logger.info(f"提取保留SNPs: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            return output_prefix
            
        except Exception as e:
            self.logger.error(f"标准LD pruning失败: {str(e)}")
            return input_prefix

def main():
    """主程序入口"""
    import argparse
    parser = argparse.ArgumentParser(description="SNP Quality Control")
    parser.add_argument("--config", required=True, help="配置文件目录路径")
    parser.add_argument("--log-level", default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    
    args = parser.parse_args()
    
    # 修复日志格式错误：从 levellevel 改为 levelname
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    
    logger = logging.getLogger('SNPQualityControl')
    
    try:
        # 初始化SNP质量控制
        qc = SNPQualityControl(args.config, logger=logger)
        
        # 运行基本QC
        output_prefix = qc.run_qc()
        
        # 如果启用了LD pruning，执行LD pruning
        if qc.config.preprocessing.ld_pruning.enable:
            final_prefix = qc.run_ld_pruning(output_prefix)
            logger.info(f"LD pruning完成，最终输出文件前缀: {final_prefix}")
        else:
            logger.info(f"QC完成，输出文件前缀: {output_prefix}")
        
    except Exception as e:
        logger.error(f"错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
