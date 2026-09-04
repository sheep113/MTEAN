from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import sys

# 确保可以导入Config类
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import Config, ConfigValidationError, MICAnalysisConfig

class ConfigManager:
    """配置管理器，处理所有与配置相关的操作"""
    
    def __init__(self, config_path: Union[str, Path], logger: Optional[logging.Logger] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
            logger: 可选的日志记录器
        """
        self.logger = logger or logging.getLogger("ConfigManager")
        self.config_path = Path(config_path)
        
        try:
            # 加载配置文件
            eval_path = self.config_path / "evaluation_config.json"
            eval_path_str = str(eval_path) if eval_path.exists() else None
            
            self.config = Config.from_json(
                str(self.config_path / "preprocessing_config.json"),
                str(self.config_path / "model_config.json"),
                eval_path_str
            )
            self.logger.info("配置文件加载成功")
        except ConfigValidationError as e:
            self.logger.error(f"配置加载失败: {e}")
            raise
        except Exception as e:
            self.logger.error(f"配置加载时发生未知错误: {e}")
            raise
    
    def get_phenotype_path(self) -> Path:
        """获取表型文件路径，支持相对路径"""
        path_str = self.config.preprocessing.file_processing.input_Phenfile
        path = Path(path_str)
        
        # 如果不是绝对路径，则从input_directory中查找
        if not path.is_absolute():
            base_dir = Path(self.config.preprocessing.file_processing.input_directory)
            path = base_dir / path
            
        return path
    
    def get_plink_prefix(self) -> Path:
        """获取PLINK文件前缀，支持相对路径"""
        path_str = self.config.preprocessing.file_processing.input_plinkfile
        path = Path(path_str)
        
        # 如果不是绝对路径，则从input_directory中查找
        if not path.is_absolute():
            base_dir = Path(self.config.preprocessing.file_processing.input_directory)
            path = base_dir / path
            
        return path
        
    def get_reference_genome_path(self) -> Path:
        """获取参考基因组文件路径，支持相对路径"""
        path_str = self.config.preprocessing.file_processing.input_refergenome
        path = Path(path_str)
        
        # 如果不是绝对路径，则从input_directory中查找
        base_dir = Path(self.config.preprocessing.file_processing.input_directory)
        path = base_dir / path
            
        return path
    
    def get_output_path(self) -> Path:
        """获取输出文件路径"""
        base_dir = Path(self.config.preprocessing.file_processing.output_directory)
        output_file = self.config.preprocessing.file_processing.output_file
        
        # 确保输出目录存在
        base_dir.mkdir(parents=True, exist_ok=True)
        
        return base_dir / output_file
    
    def get_attention_params(self, transformer_name: str = None, block_index: int = 0) -> Dict[str, Any]:
        """
        获取指定转换器的注意力参数
        
        Args:
            transformer_name: 转换器名称，如果提供则根据名称查找，否则使用block_index
            block_index: 区块索引，默认为0
            
        Returns:
            注意力参数字典
        """
        blocks = self.config.model.GFI_FormerBLOCKS.blocks
        
        if transformer_name:
            # 根据名称查找变换器
            block = next((b for b in blocks if b.name == transformer_name), None)
            if not block:
                self.logger.warning(f"未找到名称为{transformer_name}的转换器，使用索引{block_index}")
                block = blocks[block_index] if block_index < len(blocks) else blocks[0]
        else:
            # 使用索引查找
            block = blocks[block_index] if block_index < len(blocks) else blocks[0]
            
        attention = block.encoder.attention
            
        return {
            'num_heads': attention.num_heads,
            'd_attention': attention.d_attention,
            'dropout_rate': attention.dropout_rate,
            'temperature': attention.temperature,
            'type': attention.type
        }
        
    def get_pooling_config(self, transformer_name: str = None, block_index: int = 0) -> Dict[str, Any]:
        """
        获取指定转换器的池化配置
        
        Args:
            transformer_name: 转换器名称，如果提供则根据名称查找，否则使用block_index
            block_index: 区块索引，默认为0
            
        Returns:
            池化配置字典
        """
        blocks = self.config.model.GFI_FormerBLOCKS.blocks
        
        if transformer_name:
            # 根据名称查找变换器
            block = next((b for b in blocks if b.name == transformer_name), None)
            if not block:
                self.logger.warning(f"未找到名称为{transformer_name}的转换器，使用索引{block_index}")
                block = blocks[block_index] if block_index < len(blocks) else blocks[0]
        else:
            # 使用索引查找
            block = blocks[block_index] if block_index < len(blocks) else blocks[0]
            
        pooling = block.pooling
            
        result = {
            'type': pooling.type
        }
        
        if pooling.type == 'self_attention':
            result.update({
                'num_query_vectors': pooling.num_query_vectors,
                'query_dim': pooling.query_dim
            })
            
        return result
    
    def get_transformer_blocks(self) -> List[Dict[str, Any]]:
        """
        获取所有转换器区块的配置
        
        Returns:
            转换器区块配置列表
        """
        blocks = []
        for block in self.config.model.GFI_FormerBLOCKS.blocks:
            blocks.append({
                'name': block.name,
                'encoder': {
                    'num_layers': block.encoder.num_layers,
                    'attention': {
                        'type': block.encoder.attention.type,
                        'num_heads': block.encoder.attention.num_heads,
                        'd_attention': block.encoder.attention.d_attention,
                        'dropout_rate': block.encoder.attention.dropout_rate,
                        'temperature': block.encoder.attention.temperature
                    },
                    'ff_dim': block.encoder.ff_dim,
                    'layer_dropout': block.encoder.layer_dropout,
                    'add_norm': block.encoder.add_norm
                },
                'decoder': {
                    'num_layers': block.decoder.num_layers,
                    'attention': {
                        'type': block.decoder.attention.type,
                        'num_heads': block.decoder.attention.num_heads,
                        'd_attention': block.decoder.attention.d_attention,
                        'dropout_rate': block.decoder.attention.dropout_rate,
                        'temperature': block.decoder.attention.temperature
                    },
                    'ff_dim': block.decoder.ff_dim,
                    'layer_dropout': block.decoder.layer_dropout,
                    'add_norm': block.decoder.add_norm
                },
                'pooling': {
                    'type': block.pooling.type,
                    'num_query_vectors': block.pooling.num_query_vectors,
                    'query_dim': block.pooling.query_dim
                }
            })
        return blocks
    
    def get_block_by_name(self, name: str) -> Dict[str, Any]:
        """
        根据名称获取特定转换器区块的配置
        
        Args:
            name: 区块名称
            
        Returns:
            区块配置字典，如果未找到则返回None
        """
        for block in self.config.model.GFI_FormerBLOCKS.blocks:
            if block.name == name:
                return {
                    'name': block.name,
                    'encoder': {
                        'num_layers': block.encoder.num_layers,
                        'attention': {
                            'type': block.encoder.attention.type,
                            'num_heads': block.encoder.attention.num_heads,
                            'd_attention': block.encoder.attention.d_attention,
                            'dropout_rate': block.encoder.attention.dropout_rate,
                            'temperature': block.encoder.attention.temperature
                        },
                        'ff_dim': block.encoder.ff_dim,
                        'layer_dropout': block.encoder.layer_dropout,
                        'add_norm': block.encoder.add_norm
                    },
                    'decoder': {
                        'num_layers': block.decoder.num_layers,
                        'attention': {
                            'type': block.decoder.attention.type,
                            'num_heads': block.decoder.attention.num_heads,
                            'd_attention': block.decoder.attention.d_attention,
                            'dropout_rate': block.decoder.attention.dropout_rate,
                            'temperature': block.decoder.attention.temperature
                        },
                        'ff_dim': block.decoder.ff_dim,
                        'layer_dropout': block.decoder.layer_dropout,
                        'add_norm': block.decoder.add_norm
                    },
                    'pooling': {
                        'type': block.pooling.type,
                        'num_query_vectors': block.pooling.num_query_vectors,
                        'query_dim': block.pooling.query_dim
                    }
                }
        return None
    
    def validate_paths(self) -> None:
        """验证输入路径"""
        files_to_check = [
            (self.get_phenotype_path(), "表型文件"),
            (Path(f"{self.get_plink_prefix()}.bed"), "PLINK bed文件"),
            (Path(f"{self.get_plink_prefix()}.bim"), "PLINK bim文件"),
            (Path(f"{self.get_plink_prefix()}.fam"), "PLINK fam文件")
        ]
        
        # 检查参考基因组文件，如果配置中指定了该文件
        if hasattr(self.config.preprocessing.file_processing, 'input_refergenome') and self.config.preprocessing.file_processing.input_refergenome:
            files_to_check.append((self.get_reference_genome_path(), "参考基因组文件"))
        
        for file_path, file_type in files_to_check:
            if not file_path.exists():
                self.logger.warning(f"{file_type}不存在: {file_path}")
                
    def check_config_version(self):
        """检查配置版本兼容性"""
        expected_version = "1.0.0"
        actual_version = getattr(self.config.model, "version", None)
        
        if actual_version != expected_version:
            self.logger.warning(f"配置版本不匹配: 期望 {expected_version}, 实际 {actual_version}")
            
    def check_input_files(self) -> None:
        """
        检查输入文件是否存在，并报告文件大小和行数等统计信息
        """
        files_to_check = [
            (self.get_phenotype_path(), "表型文件"),
            (Path(f"{self.get_plink_prefix()}.bed"), "PLINK bed文件"),
            (Path(f"{self.get_plink_prefix()}.bim"), "PLINK bim文件"),
            (Path(f"{self.get_plink_prefix()}.fam"), "PLINK fam文件")
        ]
        
        # 检查参考基因组文件，如果配置中指定了该文件
        if hasattr(self.config.preprocessing.file_processing, 'input_refergenome') and self.config.preprocessing.file_processing.input_refergenome:
            files_to_check.append((self.get_reference_genome_path(), "参考基因组文件"))
        
        # 避免循环导入
        class FileNotFoundError(Exception):
            """文件未找到错误"""
            pass
        
        for file_path, file_type in files_to_check:
            if not file_path.exists():
                raise FileNotFoundError(f"{file_type}不存在: {file_path}")
            else:
                # 报告文件大小
                file_size = file_path.stat().st_size
                size_mb = file_size / (1024 * 1024)
                self.logger.info(f"{file_type} ({file_path}) 大小: {size_mb:.2f} MB")
                
                # 尝试计算行数但避免读取过大的文件
                if file_size < 100 * 1024 * 1024 and file_path.suffix not in ['.bed']:  # 跳过大文件和二进制文件
                    try:
                        with open(file_path, 'r') as f:
                            line_count = sum(1 for _ in f)
                        self.logger.info(f"{file_type} 行数: {line_count}")
                    except:
                        pass  # 忽略二进制文件无法读取的错误
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        获取评估配置
        
        Returns:
            评估配置字典，如果未配置评估则返回None
        """
        if not hasattr(self.config, 'evaluation') or self.config.evaluation is None:
            self.logger.warning("未找到评估配置")
            return None
        
        return self._config_to_dict(self.config.evaluation)
    
    def get_metrics_config(self) -> Dict[str, Any]:
        """
        获取评估指标配置
        
        Returns:
            评估指标配置字典，如果未配置评估则返回None
        """
        eval_config = self.get_evaluation_config()
        return eval_config.get('metrics') if eval_config else None
    
    def get_regression_metrics_config(self) -> Dict[str, Any]:
        """
        获取回归评估指标配置
        
        Returns:
            回归评估指标配置字典，如果未配置评估则返回None
        """
        eval_config = self.get_evaluation_config()
        return eval_config.get('regression') if eval_config else None
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """
        获取可视化配置
        
        Returns:
            可视化配置字典，如果未配置评估则返回None
        """
        eval_config = self.get_evaluation_config()
        return eval_config.get('visualization') if eval_config else None
    
    def get_interpretability_config(self) -> Dict[str, Any]:
        """
        获取可解释性配置
        
        Returns:
            可解释性配置字典，如果未配置评估则返回None
        """
        eval_config = self.get_evaluation_config()
        return eval_config.get('interpretability') if eval_config else None
    
    def get_partition_config(self) -> Dict[str, Any]:
        """
        获取分区配置
        
        Returns:
            分区配置字典，如果未配置分区则返回默认值
        """
        # 检查 preprocessing.model_input.partition 是否存在
        if (hasattr(self.config.preprocessing, 'model_input') and 
            self.config.preprocessing.model_input and 
            hasattr(self.config.preprocessing.model_input, 'partition')):
            partition = self.config.preprocessing.model_input.partition
            return {
                'enable': partition.enable,
                'max_size': partition.max_size,
                'method': partition.method,
                'min_size': partition.min_size
            }
        # 如果配置不存在，返回默认值
        return {
            'enable': True,
            'max_size': 1000,
            'method': 'adaptive',
            'min_size': 100
        }
    
    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        """
        将配置对象转换为字典
        
        Args:
            config: 配置对象
            
        Returns:
            配置字典
        """
        if hasattr(config, '__dict__'):
            return {
                k: self._config_to_dict(v)
                for k, v in config.__dict__.items()
                if not k.startswith('_')
            }
        elif isinstance(config, (list, tuple)):
            return [self._config_to_dict(x) for x in config]
        elif isinstance(config, dict):
            return {k: self._config_to_dict(v) for k, v in config.items()}
        elif isinstance(config, Path):
            return str(config)
        return config
