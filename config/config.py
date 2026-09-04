from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Callable, ClassVar, Final, List
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import wraps, lru_cache
import threading
from abc import ABC, abstractmethod

# 常量定义
VALID_FORMATS: Final = frozenset({'plink', 'vcf', 'bed'})
VALID_PRECISIONS: Final = frozenset({'float16', 'float32'})
VALID_ATTENTION_TYPES: Final = frozenset({'standard', 'probabilistic', 'probabilistic_cross'})
VALID_POOLING_TYPES: Final = frozenset({'max', 'mean', 'cls', 'self_attention'})

class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class ConfigValidator(ABC):
    @abstractmethod
    def validate(self) -> None:
        pass

def validate_config(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        instance = func(*args, **kwargs)
        instance.validate()
        return instance
    return wrapper

class ConfigValidationError(Exception):
    pass

@dataclass(frozen=True)
class BaseConfig(ConfigValidator):
    def _validate_field(self, field_name: str, valid_values: set, error_msg: str = None) -> None:
        value = getattr(self, field_name)
        if value not in valid_values:
            msg = error_msg or f"Invalid {field_name}: {value}. Must be one of {valid_values}"
            raise ConfigValidationError(msg)

    def validate(self) -> None:
        pass

@dataclass(frozen=True)
class FileProcessingConfig(BaseConfig):
    input_format: str
    input_directory: Path
    input_plinkfile: Path
    input_refergenome: Path
    input_Phenfile: Path
    output_directory: Path
    output_file: str
    n_jobs: int
    Samples_batch_size: int
    SNPs_batch_size: int
    
    def validate(self) -> None:
        self._validate_field('input_format', VALID_FORMATS)
        # 只检查输入目录是否存在，输出目录在运行时会自动创建
        if not Path(self.input_directory).exists():
            raise ConfigValidationError(f"输入目录不存在: {self.input_directory}")

@dataclass(frozen=True)
class SNPFilteringConfig(BaseConfig):
    geno: Dict[str, bool | float]
    maf: Dict[str, bool | float]
    hwe: Dict[str, bool | float]
    mind: Dict[str, bool | float]
    output_prefix: str
    threads: int
    memory: int
    allow_no_sex: bool
    autosome_only: bool
    extra_commands: str

    def validate(self) -> None:
        """验证SNP过滤配置"""
        if self.threads <= 0:
            raise ConfigValidationError("threads must be positive")
        if self.memory <= 0:
            raise ConfigValidationError("memory must be positive")
        for key in ['geno', 'maf', 'hwe', 'mind']:
            config = getattr(self, key)
            if config['enable'] and not 0 <= config['threshold'] <= 1:
                raise ConfigValidationError(f"Invalid {key} threshold: {config['threshold']}")

@dataclass(frozen=True)
class MissingValueHandlingConfig(BaseConfig):
    """配置缺失值处理方式"""
    enable: bool = False
    method: Optional[str] = 'mode' # 允许的方法: 'mode', 'mean', 或 None (不填充)

    def validate(self) -> None:
        """验证缺失值处理配置"""
        if self.enable and self.method not in ['mode', 'mean']:
            raise ConfigValidationError(f"无效的缺失值填充方法: {self.method}. 必须是 'mode' 或 'mean'.")

@dataclass(frozen=True)
class LDPruningConfig(BaseConfig):
    enable: bool
    window_size: int
    step_size: int
    default_r2: float
    gene_regions: Dict[str, bool | str | float]

@dataclass(frozen=True)
class MICAnalysisConfig(BaseConfig):
    enable: bool
    alpha: float
    c: int
    num_threads: int
    chunk_size: int
    min_samples: int
    MIC_output_file: str

    def validate(self) -> None:
        if self.enable:
            if not 0 < self.alpha <= 1:
                raise ConfigValidationError(f"Invalid alpha value: {self.alpha}")
            if self.c <= 0:
                raise ConfigValidationError(f"Invalid c value: {self.c}")
            if self.num_threads <= 0:
                raise ConfigValidationError(f"Invalid num_threads value: {self.num_threads}")
            if self.chunk_size <= 0:
                raise ConfigValidationError(f"Invalid chunk_size value: {self.chunk_size}")
            if self.min_samples <= 0:
                raise ConfigValidationError(f"Invalid min_samples value: {self.min_samples}")

@dataclass(frozen=True)
class CrossValidationConfig(BaseConfig):
    enable: bool
    n_splits: int
    shuffle: bool
    cv_random_seed: int

@dataclass(frozen=True)
class DataSplitConfig(BaseConfig):
    enable: bool
    train_ratio: float
    valid_ratio: float
    test_ratio: float
    random_seed: int
    stratify: bool
    cross_validation: CrossValidationConfig

@dataclass(frozen=True)
class PreprocessingConfig(BaseConfig):
    # 没有默认值的字段放在前面
    file_processing: FileProcessingConfig
    snp_filtering: SNPFilteringConfig
    ld_pruning: LDPruningConfig
    mic_analysis: MICAnalysisConfig
    data_split: DataSplitConfig
    
    # 有默认值的字段放在后面
    missing_value_handling: MissingValueHandlingConfig = field(default_factory=MissingValueHandlingConfig) 
    model_input: Optional[ModelInputConfig] = None  

    @staticmethod
    def _create_preprocessing_config(config_dict: dict) -> PreprocessingConfig:
        # --- 修改: 处理 missing_value_handling ---
        missing_handling_config = MissingValueHandlingConfig() # 使用默认值
        if 'missing_value_handling' in config_dict:
            missing_handling_config = MissingValueHandlingConfig(**config_dict['missing_value_handling'])
        # --- 结束修改 ---

        # --- 修改: 处理 model_input ---
        model_input = None
        if 'model_input' in config_dict and config_dict['model_input']: # 检查是否非空
            try:
                partition_config = PartitionConfig(**config_dict['model_input']['partition'])
                model_input = ModelInputConfig(partition=partition_config)
            except KeyError as e:
                raise ConfigValidationError(f"Model input config missing key: {e}")
            except TypeError as e:
                raise ConfigValidationError(f"Model input config type error: {e}")
        # --- 结束修改 ---

        return PreprocessingConfig(
            file_processing=FileProcessingConfig(**config_dict['file_processing']),
            snp_filtering=SNPFilteringConfig(**config_dict['snp_filtering']),
            ld_pruning=LDPruningConfig(**config_dict['ld_pruning']), # 确保顺序正确
            mic_analysis=MICAnalysisConfig(**config_dict['mic_analysis']), # 确保顺序正确
            data_split=DataSplitConfig( # 确保顺序正确
                **{k: v for k, v in config_dict['data_split'].items() 
                   if k != 'cross_validation'},
                cross_validation=CrossValidationConfig(
                    **config_dict['data_split']['cross_validation'])
            ),
            missing_value_handling=missing_handling_config, # 使用解析后的配置
            model_input=model_input # 使用解析后的配置
        )

    def validate(self) -> None:
        """验证预处理配置"""
        super().validate() # 调用基类验证（如果需要）
        self.file_processing.validate()
        self.snp_filtering.validate()
        self.ld_pruning.validate() # 确保顺序正确
        self.mic_analysis.validate() # 确保顺序正确
        self.data_split.validate() # 确保顺序正确
        self.missing_value_handling.validate() # 添加验证调用
        if self.model_input:
            self.model_input.validate()

@dataclass(frozen=True)
class EmbeddingConfig(BaseConfig):
    dim: int
    dropout_rate: float
    position_encoding: bool = False
    normalization: bool = True

@dataclass(frozen=True)
class AttentionConfig(BaseConfig):
    type: str
    num_heads: int
    d_attention: int  # 从 d_v 改为 d_attention
    dropout_rate: float
    temperature: float = 1.0  # 新增 temperature 参数，默认值为1.0

    def validate(self) -> None:
        self._validate_field('type', VALID_ATTENTION_TYPES)
        if self.num_heads <= 0:
            raise ConfigValidationError("num_heads must be positive")
        if self.d_attention <= 0:
            raise ConfigValidationError("d_attention must be positive")

@dataclass(frozen=True)
class EncoderConfig(BaseConfig):
    num_layers: int
    attention: AttentionConfig
    ff_dim: int
    layer_dropout: float
    add_norm: bool = True  # 新增 add_norm 参数，默认为True

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'EncoderConfig':
        attention_config = AttentionConfig(
            type=config['attention']['type'],
            num_heads=config['attention']['num_heads'],
            d_attention=config['attention']['d_attention'],
            dropout_rate=config['attention']['dropout_rate'],
            temperature=config['attention'].get('temperature', 1.0)
        )
        return cls(
            num_layers=config['num_layers'],
            attention=attention_config,
            ff_dim=config['ff_dim'],
            layer_dropout=config['layer_dropout'],
            add_norm=config.get('add_norm', True)
        )

@dataclass(frozen=True)
class DecoderConfig(BaseConfig):
    num_layers: int
    attention: AttentionConfig
    ff_dim: int
    layer_dropout: float
    add_norm: bool = True  # 新增 add_norm 参数，默认为True

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'DecoderConfig':
        attention_config = AttentionConfig(
            type=config['attention']['type'],
            num_heads=config['attention']['num_heads'],
            d_attention=config['attention']['d_attention'],
            dropout_rate=config['attention']['dropout_rate'],
            temperature=config['attention'].get('temperature', 1.0)
        )
        return cls(
            num_layers=config['num_layers'],
            attention=attention_config,
            ff_dim=config['ff_dim'],
            layer_dropout=config['layer_dropout'],
            add_norm=config.get('add_norm', True)
        )

@dataclass(frozen=True)
class PoolingConfig(BaseConfig):
    type: str
    num_query_vectors: Optional[int] = None
    query_dim: Optional[int] = None

    def validate(self) -> None:
        self._validate_field('type', VALID_POOLING_TYPES)
        if self.type == 'self_attention':
            if not self.num_query_vectors or self.num_query_vectors <= 0:
                raise ConfigValidationError("num_query_vectors must be positive for self_attention pooling")
            if not self.query_dim or self.query_dim <= 0:
                raise ConfigValidationError("query_dim must be positive for self_attention pooling")

@dataclass(frozen=True)
class TransformerBlockConfig(BaseConfig):
    name: str
    encoder: EncoderConfig
    decoder: DecoderConfig
    pooling: PoolingConfig

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'TransformerBlockConfig':
        return cls(
            name=config['name'],
            encoder=EncoderConfig.from_dict(config['encoder']),
            decoder=DecoderConfig.from_dict(config['decoder']),
            pooling=PoolingConfig(
                type=config['pooling']['type'],
                num_query_vectors=config['pooling'].get('num_query_vectors'),
                query_dim=config['pooling'].get('query_dim')
            )
        )

@dataclass(frozen=True)
class GFIFormerBlocksConfig(BaseConfig):
    num_blocks: int
    blocks: List[TransformerBlockConfig]

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'GFIFormerBlocksConfig':
        blocks = []
        for block_dict in config_dict['blocks']:
            blocks.append(TransformerBlockConfig.from_dict(block_dict))
        
        return cls(
            num_blocks=config_dict['num_blocks'],
            blocks=blocks
        )

    def validate(self) -> None:
        if self.num_blocks != len(self.blocks):
            raise ConfigValidationError(f"Number of blocks ({self.num_blocks}) doesn't match actual blocks count ({len(self.blocks)})")
        
        for block in self.blocks:
            block.validate()

@dataclass(frozen=True)
class OptimizerConfig(BaseConfig):
    type: str
    beta1: float
    beta2: float
    epsilon: float

@dataclass(frozen=True)
class SchedulerConfig(BaseConfig):
    type: str
    warmup_steps: int

@dataclass(frozen=True)
class MixedPrecisionConfig(BaseConfig):
    enabled: bool
    dtype: str
    dynamic_scaling: bool

@dataclass(frozen=True)
class TrainingConfig(BaseConfig):
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    mixed_precision: MixedPrecisionConfig

@dataclass(frozen=True)
class RegularizationConfig(BaseConfig):
    weight_decay: float
    gradient_clip_norm: float

@dataclass(frozen=True)
class PreprocessConfig(BaseConfig):
    snp_encoding: Dict[str, int] = field(default_factory=lambda: {
        "AA": 0, "AT": 1, "TA": 1, "AC": 2, "CA": 2,
        "AG": 3, "GA": 3, "TT": 4, "TC": 5, "CT": 5,
        "TG": 6, "GT": 6, "CC": 7, "CG": 8, "GC": 8, "GG": 9
    })
    chr_encoding: str = "numeric"
    pos_encoding: str = "normalized"
    region_indicator: Dict[str, int] = field(default_factory=lambda: {
        "coding": 0,
        "non_coding": 1
    })

@dataclass(frozen=True)
class OutputLayerConfig(BaseConfig):
    phenotype_dim: int
    phenotype_name: list[str]
    hidden_dims: list[int]
    activation: str
    dropout_rate: float

@dataclass(frozen=True)
class PrimaryLossConfig(BaseConfig):
    type: str
    label_smoothing: float
    reduction: str

@dataclass(frozen=True)
class AuxiliaryLossesConfig(BaseConfig):
    distribution_kl: dict[str, bool | float]
    l1_regularization: dict[str, bool | float]

@dataclass(frozen=True)
class LossConfig(BaseConfig):
    primary_loss: PrimaryLossConfig
    auxiliary_losses: AuxiliaryLossesConfig

@dataclass(frozen=True)
class PhenotypeConfig(BaseConfig):
    distribution: str
    normalize: bool
    scaling_method: str

@dataclass(frozen=True)
class VisualizationAttentionMapsConfig(BaseConfig):
    enabled: bool
    top_k: int
    threshold: float
    aggregation: str

@dataclass(frozen=True)
class VisualizationEmbeddingsConfig(BaseConfig):
    enabled: bool
    method: str
    perplexity: int
    n_components: int

@dataclass(frozen=True)
class VisualizationConfig(BaseConfig):
    confusion_matrix: bool
    roc_curve: bool
    precision_recall_curve: bool
    feature_importance: bool
    attention_maps: VisualizationAttentionMapsConfig
    embeddings: VisualizationEmbeddingsConfig

@dataclass(frozen=True)
class OutputConfig(BaseConfig):
    save_predictions: bool
    save_attention_weights: bool
    save_feature_importance: bool
    output_format: list[str]

@dataclass(frozen=True)
class ShapConfig(BaseConfig):
    enabled: bool
    n_samples: int
    background_samples: int

@dataclass(frozen=True)
class ImportantSNPsConfig(BaseConfig):
    enabled: bool
    top_k: int
    method: str

@dataclass(frozen=True)
class InterpretabilityConfig(BaseConfig):
    shap: ShapConfig
    important_snps: ImportantSNPsConfig

@dataclass(frozen=True)
class PermutationTestConfig(BaseConfig):
    enabled: bool
    n_permutations: int
    significance_level: float

@dataclass(frozen=True)
class SignificanceTestsConfig(BaseConfig):
    permutation_test: PermutationTestConfig

@dataclass(frozen=True)
class EvalCrossValidationConfig(BaseConfig):
    enabled: bool
    n_splits: int
    shuffle: bool
    stratify: bool

@dataclass(frozen=True)
class MetricsConfig(BaseConfig):
    accuracy: bool
    precision: bool
    recall: bool
    f1_score: bool
    auc_roc: bool
    confusion_matrix: bool
    class_report: bool

@dataclass(frozen=True)
class RegressionConfig(BaseConfig):
    enabled: bool
    mse: bool
    rmse: bool
    mae: bool
    r2_score: bool
    explained_variance: bool
    median_absolute_error: bool

@dataclass(frozen=True)
class EvaluationConfig(BaseConfig):
    metrics: MetricsConfig
    regression: RegressionConfig
    visualization: VisualizationConfig
    output: OutputConfig
    interpretability: InterpretabilityConfig
    significance_tests: SignificanceTestsConfig
    cross_validation: EvalCrossValidationConfig

@dataclass(frozen=True)
class ModelConfig(BaseConfig):
    random_seed: int
    precision: str
    embedding: EmbeddingConfig
    GFI_FormerBLOCKS: GFIFormerBlocksConfig
    output_layer: OutputLayerConfig
    loss_config: LossConfig
    phenotype: PhenotypeConfig
    training: Dict[str, Any]
    regularization: Dict[str, Any]
    version: str = "1.0.0"  # 所有默认参数必须在非默认参数之后

    def __post_init__(self):
        object.__setattr__(self, 'precision', self.precision.lower())
        self._validate_field('precision', VALID_PRECISIONS)

    @property
    def snp_transformer(self) -> TransformerBlockConfig:
        """向后兼容:返回第一个变换器作为SNP变换器"""
        return self.GFI_FormerBLOCKS.blocks[0] if self.GFI_FormerBLOCKS.blocks else None

    @property
    def gene_transformer(self) -> TransformerBlockConfig:
        """向后兼容:返回第二个变换器作为基因变换器"""
        return self.GFI_FormerBLOCKS.blocks[1] if len(self.GFI_FormerBLOCKS.blocks) > 1 else None

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """从字典创建ModelConfig"""
        embedding_config = EmbeddingConfig(
            dim=config_dict['embedding']['dim'],
            dropout_rate=config_dict['embedding']['dropout_rate'],
            position_encoding=config_dict['embedding'].get('position_encoding', False),
            normalization=config_dict['embedding'].get('normalization', True)
        )
        
        # 使用新的GFI_FormerBLOCKS配置
        gfi_blocks = GFIFormerBlocksConfig.from_dict(config_dict['GFI_FormerBLOCKS'])
        
        output_layer = OutputLayerConfig(**config_dict['output_layer'])
        
        loss_config = LossConfig(
            primary_loss=PrimaryLossConfig(**config_dict['loss_config']['primary_loss']),
            auxiliary_losses=AuxiliaryLossesConfig(**config_dict['loss_config']['auxiliary_losses'])
        )
        
        phenotype = PhenotypeConfig(**config_dict['phenotype'])
        
        return cls(
            random_seed=config_dict.get('random_seed', 42),
            precision=config_dict['precision'],
            embedding=embedding_config,
            GFI_FormerBLOCKS=gfi_blocks,
            output_layer=output_layer,
            loss_config=loss_config,
            phenotype=phenotype,
            training=config_dict['training'],
            regularization=config_dict['regularization'],
            version=config_dict.get('version', '1.0.0')
        )

class Config(metaclass=SingletonMeta):
    _instance = None
    _cache: ClassVar[Dict[str, Any]] = {}
    
    def __init__(self, preprocessing: PreprocessingConfig, model: ModelConfig, evaluation: Optional[EvaluationConfig] = None):
        self.preprocessing = preprocessing
        self.model = model
        self.evaluation = evaluation  # 添加评估配置
        self._logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

    def _setup_logging(self) -> None:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    @classmethod
    @lru_cache(maxsize=1)
    @validate_config
    def from_json(cls, preprocess_path: str, model_path: str, eval_path: Optional[str] = None) -> 'Config':
        """
        从JSON文件加载配置，使用LRU缓存避免重复加载
        """
        try:
            config = cls._load_from_cache(preprocess_path, model_path, eval_path)
            if config:
                return config

            with ThreadPoolExecutor() as executor:
                paths = [preprocess_path, model_path]
                if eval_path:
                    paths.append(eval_path)

                futures = {
                    executor.submit(cls._load_json, path): path
                    for path in paths
                }

                configs = {}
                for future in futures:
                    path = futures[future]
                    try:
                        configs[path] = future.result()
                    except Exception as e:
                        # Use standard logging here as instance logger is not available
                        logging.error(f"Failed to load {path}: {str(e)}")
                        raise ConfigValidationError(f"Failed to load {path}: {str(e)}")

            preprocessing = cls._create_preprocessing_config(configs[preprocess_path])
            model = cls._create_model_config(configs[model_path])

            evaluation = None
            if eval_path and eval_path in configs:
                evaluation = cls._create_evaluation_config(configs[eval_path])

            config = cls(preprocessing=preprocessing, model=model, evaluation=evaluation)
            cls._cache_config(preprocess_path, model_path, eval_path, config)
            return config

        except Exception as e:
            # Use standard logging here as instance logger is not available
            logging.error(f"Configuration loading failed: {str(e)}")
            raise

    @classmethod
    def _load_from_cache(cls, preprocess_path: str, model_path: str, eval_path: Optional[str] = None) -> Optional['Config']:
        cache_key = f"{preprocess_path}:{model_path}"
        if eval_path:
            cache_key += f":{eval_path}"
        return cls._cache.get(cache_key)

    @classmethod
    def _cache_config(cls, preprocess_path: str, model_path: str, eval_path: Optional[str], config: 'Config') -> None:
        cache_key = f"{preprocess_path}:{model_path}"
        if eval_path:
            cache_key += f":{eval_path}"
        cls._cache[cache_key] = config

    @staticmethod
    def _load_json(path: str) -> dict:
        try:
            with Path(path).open('r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigValidationError(f"Invalid JSON in {path}: {str(e)}")
        except Exception as e:
            raise ConfigValidationError(f"Failed to read {path}: {str(e)}")

    @staticmethod
    def _create_preprocessing_config(config_dict: dict) -> PreprocessingConfig:
        return PreprocessingConfig._create_preprocessing_config(config_dict)

    @staticmethod
    def _create_model_config(config_dict: dict) -> ModelConfig:
        return ModelConfig.from_dict(config_dict)

    @staticmethod
    def _create_evaluation_config(config_dict: dict) -> EvaluationConfig:
        """创建评估配置"""
        metrics = MetricsConfig(**config_dict['metrics'])
        regression = RegressionConfig(**config_dict['regression'])
        
        visualization_attention_maps = VisualizationAttentionMapsConfig(**config_dict['visualization']['attention_maps'])
        visualization_embeddings = VisualizationEmbeddingsConfig(**config_dict['visualization']['embeddings'])
        
        visualization = VisualizationConfig(
            confusion_matrix=config_dict['visualization']['confusion_matrix'],
            roc_curve=config_dict['visualization']['roc_curve'],
            precision_recall_curve=config_dict['visualization']['precision_recall_curve'],
            feature_importance=config_dict['visualization']['feature_importance'],
            attention_maps=visualization_attention_maps,
            embeddings=visualization_embeddings
        )
        
        output = OutputConfig(**config_dict['output'])
        
        shap = ShapConfig(**config_dict['interpretability']['shap'])
        important_snps = ImportantSNPsConfig(**config_dict['interpretability']['important_snps'])
        interpretability = InterpretabilityConfig(
            shap=shap,
            important_snps=important_snps
        )
        
        permutation_test = PermutationTestConfig(**config_dict['significance_tests']['permutation_test'])
        significance_tests = SignificanceTestsConfig(
            permutation_test=permutation_test
        )
        
        cross_validation = EvalCrossValidationConfig(**config_dict['cross_validation'])
        
        return EvaluationConfig(
            metrics=metrics,
            regression=regression,
            visualization=visualization,
            output=output,
            interpretability=interpretability,
            significance_tests=significance_tests,
            cross_validation=cross_validation
        )

    def validate(self) -> None:
        """使用策略模式进行配置验证"""
        validators = [
            self._validate_split_ratios,
            self._validate_transformer_config,
            self._validate_training_config,
            self._validate_data_paths
        ]
        
        for validator in validators:
            try:
                validator()
            except Exception as e:
                self._logger.error(f"Validation failed: {str(e)}")
                raise

    def _validate_data_paths(self) -> None:
        """验证数据路径的有效性"""
        # 只检查输入目录存在，输出目录会自动创建
        input_path = Path(self.preprocessing.file_processing.input_directory)
        if not input_path.exists():
            raise ConfigValidationError(f"输入目录不存在: {input_path}")

    def _validate_split_ratios(self) -> None:
        ratios = [
            self.preprocessing.data_split.train_ratio,
            self.preprocessing.data_split.valid_ratio,
            self.preprocessing.data_split.test_ratio
        ]
        if not abs(sum(ratios) - 1.0) < 1e-6:
            raise ConfigValidationError("Data split ratios must sum to 1.0")

    def _validate_transformer_config(self) -> None:
        """更新验证方法以支持GFI_FormerBLOCKS结构"""
        for block in self.model.GFI_FormerBLOCKS.blocks:
            if block.encoder.attention.type not in VALID_ATTENTION_TYPES:
                raise ConfigValidationError(
                    f"{block.name} transformer attention type must be one of {VALID_ATTENTION_TYPES}")
            if block.pooling.type not in VALID_POOLING_TYPES:
                raise ConfigValidationError(
                    f"{block.name} transformer pooling must be one of {VALID_POOLING_TYPES}")

    def _validate_training_config(self) -> None:
        training = self.model.training
        if training['batch_size'] <= 0:
            raise ConfigValidationError("Batch size must be positive")
        if training['epochs'] <= 0:
            raise ConfigValidationError("Number of epochs must be positive")
        if training['learning_rate'] <= 0:
            raise ConfigValidationError("Learning rate must be positive")

    def save(self, preprocess_path: str, model_path: str) -> None:
        """线程安全地保存配置"""
        try:
            preprocess_dict = self._to_dict(self.preprocessing)
            model_dict = self._to_dict(self.model)

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._save_json, preprocess_path, preprocess_dict),
                    executor.submit(self._save_json, model_path, model_dict)
                ]
                
                for future in futures:
                    future.result()
                
            self._logger.info("Configuration saved successfully")
        except Exception as e:
            self._logger.error(f"Failed to save configuration: {str(e)}")
            raise

    @staticmethod
    def _save_json(path: str, data: dict) -> None:
        with Path(path).open('w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def _to_dict(obj: Any) -> dict:
        """改进的序列化方法"""
        if hasattr(obj, '__dict__'):
            return {
                k: Config._to_dict(v)
                for k, v in obj.__dict__.items()
                if not k.startswith('_')
            }
        elif isinstance(obj, (list, tuple)):
            return [Config._to_dict(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: Config._to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, Path):
            return str(obj)
        return obj

if __name__ == "__main__":
    try:
        config = Config.from_json(
            "config/preprocessing_config.json",
            "config/model_config.json",
            "config/evaluation_config.json"  # 添加评估配置路径
        )
        config.validate()
        print("Configuration loaded and validated successfully")
    except Exception as e:
        logging.error(f"Configuration error: {str(e)}")
        raise

@dataclass(frozen=True)
class PartitionConfig(BaseConfig):
    enable: bool
    max_size: int
    method: str
    min_size: int

@dataclass(frozen=True)
class ModelInputConfig(BaseConfig):
    partition: PartitionConfig
