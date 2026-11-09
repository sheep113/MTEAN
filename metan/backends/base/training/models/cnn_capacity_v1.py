# Modified for METAN portable release 2026-11-09.
"""Opt-in CNN filter multiplicity; all historical signed sources stay untouched.

The existing Shared model supplies the unchanged attention/MoE/output core.
Only modules whose shape depends on per-path CNN channels are reconstructed.
The C1 route does not reconstruct anything or consume additional random draws.
"""
from copy import deepcopy
from pathlib import Path
import hashlib

from torch import nn

from .METAN import METAN
from .layers import CNNFeatureExtractor, GlobalContextLayerNorm
from .carp_embedding_layout_v2 import V2
from .task_layout_v2 import (
    TaskLayoutV2METAN, normalize_task_config, source_signature,
    resolve_training_model as resolve_original_model,
)
from .task_layout_v2_layers import (
    ARCHITECTURE_VERSION as ORIGINAL_VERSION,
    TargetAwarePositionFirstEmbedding,
)
from training.utils.config_utils import load_config

CAPACITY_VERSION = "growth500_cnn_capacity_v1"
CHANNEL_KEY = "cnn_path_output_channels"
INITIALIZATION_POLICY = "original_shared_core_then_shape_dependent_modules_v1"


def normalize_capacity_config(config):
    cfg = deepcopy(config)
    if cfg.get("layout_architecture_version") != CAPACITY_VERSION:
        raise ValueError("CNN capacity requires its explicit implementation version")
    e = cfg.get("embedding", {})
    channels = e.get(CHANNEL_KEY)
    if type(channels) is not int or channels not in (1, 2, 3):
        raise ValueError("cnn_path_output_channels must explicitly be 1, 2, or 3")
    if cfg.get("task_sharing_mode") != "shared" or e.get("cnn_feature_groups") != 2:
        raise ValueError("CNN capacity v1 requires Shared with exactly two CNN groups")
    if (e.get("CNN_Kernel") != [3, 5, 7, 9, 11]
            or e.get("CNN_Dilation") != [1, 2, 3]
            or e.get("CNN_layers") != 1):
        raise ValueError("Keep all 15 original single-layer kernel/dilation paths")
    cfg["layout_architecture_version"] = ORIGINAL_VERSION
    cfg = normalize_task_config(cfg)
    cfg["layout_architecture_version"] = CAPACITY_VERSION
    blocks = cfg["GFI_FormerBLOCKS"]
    if blocks.get("num_blocks") != 1 or len(blocks["blocks"]) != 1:
        raise ValueError("CNN capacity v1 requires the validated single GFI block")
    block = blocks["blocks"][0]
    attention = block["encoder"]["attention"]
    moe = block["decoder"]["MOE"]
    if (attention.get("hidden_dims") != 64 or attention.get("num_heads") != 4
            or moe.get("num_experts") != 2 or moe.get("experts_dims") != 64):
        raise ValueError("Do not change attention dimensions or MoE capacity")
    return cfg


class CNNCapacityEmbedding(TargetAwarePositionFirstEmbedding):
    """Reuse validated feature extraction, V2 pooling, auxiliary and PE methods."""

    def __init__(self, input_dim, config, target_dim, cnn_feature_groups,
                 path_channels, gradient_checkpointing_config=None):
        if path_channels == 1:
            super().__init__(input_dim, config, target_dim, cnn_feature_groups,
                             gradient_checkpointing_config)
            return
        if path_channels not in (2, 3) or cnn_feature_groups != 2:
            raise ValueError("Invalid CNN capacity embedding contract")
        # Narrow, versioned constructor: inherited forward/layout math is unchanged.
        nn.Module.__init__(self)
        self.target_dim = target_dim
        self.cnn_feature_groups = cnn_feature_groups
        self.phenotype_dim = cnn_feature_groups
        self.Block_length = config["Block_length"]
        self._validate_config()
        self.activation_fn = self._get_activation(config.get("activation", "gelu"))
        gc = gradient_checkpointing_config or config.get("gradient_checkpointing", {})
        self.gc_enabled = gc.get("enabled", False)
        self.use_cnn_checkpointing = self.gc_enabled and gc.get("embedding_cnn_extractors", False)
        self.use_pooling_checkpointing = self.gc_enabled and gc.get("embedding_block_pooling", False)
        self.use_reentrant = gc.get("use_reentrant", False)
        self.feature_extractors = nn.ModuleList([
            CNNFeatureExtractor(input_dim, config["CNN_Kernel"], config["CNN_Dilation"],
                                self.activation_fn, config["CNN_layers"],
                                feature_channels=path_channels)
            for _ in range(cnn_feature_groups)
        ])
        self.num_cnn_features_per_phenotype = self.feature_extractors[0].output_channels
        self.num_cnn_features_total = cnn_feature_groups * self.num_cnn_features_per_phenotype
        self.block_pooling = None
        self.ln_before_pooling = None
        self.ln_after_pooling = None
        self.ln_before_aux_ffn_no_pool = None
        dropout = config.get("dropout_rate", .1)
        self._setup_block_pooling(config, dropout, config.get("pooling_alpha", 1.5))
        self.ln_after_pooling = nn.LayerNorm(self.num_cnn_features_per_phenotype)
        self._setup_aux_networks(config.get("aux_ffn_expansion_factor", 2))
        self.dropout = nn.Dropout(dropout)
        self.position_encoding = config.get("position_encoding", False)
        self._setup_positional_encoding(config)


def capacity_source_signature():
    result = source_signature()  # Includes the unchanged legacy source hash checks.
    here = Path(__file__).resolve()
    result[here.name] = hashlib.sha256(here.read_bytes()).hexdigest()
    return result


class CNNCapacityMETAN(TaskLayoutV2METAN):
    def __init__(self, config_path=None, config=None, optimizer_config=None,
                 scheduler_config=None):
        if config is None:
            if config_path is None:
                raise ValueError("A model config is required")
            config = load_config(config_path)
        cfg = normalize_capacity_config(config)
        self._capacity_channels = cfg["embedding"][CHANNEL_KEY]
        self._capacity_constructing = True
        # Reuse the original complete model and training/loss methods. Its private
        # construction config uses the original version only while being built.
        template = deepcopy(cfg)
        template["layout_architecture_version"] = ORIGINAL_VERSION
        super().__init__(config=template, optimizer_config=optimizer_config,
                         scheduler_config=scheduler_config)
        if self._capacity_channels != 1:
            e = cfg["embedding"]
            self.model.embedding = CNNCapacityEmbedding(
                e["input_dims"], e, self.phenotype_dim, 2,
                self._capacity_channels, cfg.get("gradient_checkpointing", {}))
            width = self.model.embedding.num_cnn_features_total
            gfi = self.model.gfi_former
            gfi.input_dim = width
            gfi.pre_block_sequence_norms[0] = nn.InstanceNorm1d(width, affine=True)
            block = gfi.gfi_blocks[0]
            block.input_dim = width
            block.norm_input = GlobalContextLayerNorm(width)
            block.projection_from_input = nn.Sequential(
                nn.Linear(width, 4 * width), nn.GELU(),
                nn.Linear(4 * width, block.encoder_dim))
        self.hparams.config["layout_architecture_version"] = CAPACITY_VERSION
        self._capacity_constructing = False
        self._capacity_sources = capacity_source_signature()
        self.task_layout_signature()

    def task_layout_signature(self):
        signature = super().task_layout_signature()
        if self._capacity_constructing:
            return signature
        e = self.model.embedding
        channels = self._capacity_channels
        width = 15 * channels
        if (e.num_cnn_features_per_phenotype != width
                or e.num_cnn_features_total != 2 * width
                or e.block_pooling.num_experts != 2
                or e.block_pooling.expert_dim != width):
            raise RuntimeError("CNN/Embedding/Pooling width mismatch")
        for extractor in e.feature_extractors:
            if len(extractor.layers) != 1 or len(extractor.layers[0]) != 15:
                raise RuntimeError("CNN kernel/dilation path count mismatch")
            observed = []
            for branch in extractor.layers[0]:
                conv, norm = branch[0], branch[1]
                if (conv.in_channels != 10 or conv.out_channels != channels
                        or norm.num_features != channels):
                    raise RuntimeError("Per-path CNN channel contract mismatch")
                observed.append((conv.kernel_size[0], conv.dilation[0]))
            if observed != [(k, d) for k in [3, 5, 7, 9, 11] for d in [1, 2, 3]]:
                raise RuntimeError("Kernel/dilation combinations changed")
        block = self.model.gfi_former.gfi_blocks[0]
        if (block.input_dim != 2 * width
                or block.projection_from_input[0].in_features != 2 * width
                or block.projection_from_input[0].out_features != 8 * width
                or block.projection_from_input[2].out_features != 64
                or block.num_experts != 2 or block.experts_dims != 64):
            raise RuntimeError("GFI projection/MoE width contract mismatch")
        signature.update(schema_version=3, implementation_version=CAPACITY_VERSION,
                         cnn_path_output_channels=channels,
                         initialization_policy=INITIALIZATION_POLICY)
        for branch in signature["branches"]:
            branch.update(cnn_path_output_channels=channels, paths_per_group=15,
                          gfi_input_width=2 * width, gfi_projection_hidden_width=8 * width,
                          attention_hidden_width=64, attention_heads=4,
                          moe_experts=2, moe_expert_width=64)
        return signature

    def on_save_checkpoint(self, checkpoint):
        METAN.on_save_checkpoint(self, checkpoint)
        checkpoint["metan_architecture"] = self.task_layout_signature()
        checkpoint["embedding_layout_version"] = V2
        checkpoint["cnn_capacity_version"] = CAPACITY_VERSION
        checkpoint["layout_provenance"] = {
            "sources": self._capacity_sources,
            "implementation_version": CAPACITY_VERSION,
            "model_config": deepcopy(dict(self.hparams.config))}

    def on_load_checkpoint(self, checkpoint):
        self._layout_checkpoint_verified = False
        if (checkpoint.get("embedding_layout_version") != V2
                or checkpoint.get("cnn_capacity_version") != CAPACITY_VERSION
                or checkpoint.get("metan_architecture") != self.task_layout_signature()):
            raise RuntimeError("CNN capacity/layout/trait/width semantic mismatch")
        proof = checkpoint.get("layout_provenance", {})
        if (proof.get("sources") != capacity_source_signature()
                or proof.get("implementation_version") != CAPACITY_VERSION
                or proof.get("model_config") != dict(self.hparams.config)
                or checkpoint.get("hyper_parameters", {}).get("config") != dict(self.hparams.config)):
            raise RuntimeError("CNN capacity source/config provenance mismatch")
        normalized = dict(checkpoint)
        normalized["metan_architecture"] = self.model.architecture_signature()
        METAN.on_load_checkpoint(self, normalized)
        self._layout_checkpoint_verified = True


def resolve_training_model(config, legacy_class):
    """Explicit alternate selector; old selectors reject the new version tag."""
    if config.get("layout_architecture_version") == CAPACITY_VERSION:
        normalize_capacity_config(config)
        return CNNCapacityMETAN
    if CHANNEL_KEY in config.get("embedding", {}):
        raise ValueError("A CNN channel request requires the explicit capacity version")
    return resolve_original_model(config, legacy_class)
