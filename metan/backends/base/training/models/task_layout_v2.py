# Modified for METAN portable release 2026-11-09.
"""Versioned growth500 LayoutV2 task adapter; historical adapters remain intact."""
from copy import deepcopy
from pathlib import Path
import hashlib

from .METAN import METAN
from .growth500_layout_v2 import resolve_training_model as resolve_historical_model
from .carp_embedding_layout_v2 import V2, LEGACY, BASE_LAYER_SHA256
from .task_layout_v2_layers import ARCHITECTURE_VERSION, TargetAwarePositionFirstEmbedding
from training.utils.config_utils import load_config, resolve_phenotype_config


def normalize_task_config(config):
    cfg = deepcopy(config)
    if cfg.get("embedding_layout_version") != V2:
        raise ValueError("Task LayoutV2 requires explicit position_first_v2")
    version = cfg.setdefault("layout_architecture_version", ARCHITECTURE_VERSION)
    if version != ARCHITECTURE_VERSION:
        raise ValueError("Unknown layout_architecture_version")
    names, _ = resolve_phenotype_config(cfg)
    allowed = cfg.get("layout_allowed_traits")
    mode = cfg.get("task_sharing_mode", "shared")
    if allowed is None or mode not in ("shared", "independent"):
        raise ValueError("Unsupported species/task-sharing contract")
    if (mode == "independent" and names != allowed) or (
            mode == "shared" and names != allowed and not (len(names) == 1 and names[0] in allowed)):
        raise ValueError("Unsupported trait names/order for growth500 LayoutV2")
    e = cfg["embedding"]
    expected = 1 if mode == "independent" else 2
    groups = e.setdefault("cnn_feature_groups", expected)
    if type(groups) is not int or groups != expected:
        raise ValueError("Shared requires two CNN groups; Independent requires one per branch")
    if (e.get("input_type", "").lower() != "snp" or e.get("input_dims") != 10
            or e.get("Block_length") != 32
            or len(e.get("CNN_Kernel", [])) * len(e.get("CNN_Dilation", [])) != 15):
        raise ValueError("Expected 10-channel, 32-SNP, 15-feature CNN contract")
    return cfg


def source_signature():
    folder = Path(__file__).parent
    names = ("task_layout_v2.py", "task_layout_v2_layers.py", "model.py", "METAN.py",
             "layers.py", "pooling.py", "carp_embedding_layout_v2.py")
    result = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in names}
    if result["layers.py"] != BASE_LAYER_SHA256:
        raise RuntimeError("Validated Embedding base source changed")
    return result


class TaskLayoutV2METAN(METAN):
    def __init__(self, config_path=None, config=None, optimizer_config=None, scheduler_config=None):
        if config is None:
            if config_path is None:
                raise ValueError("A model config is required")
            config = load_config(config_path)
        cfg = normalize_task_config(config)
        self._layout_checkpoint_verified = False
        super().__init__(config=cfg, optimizer_config=optimizer_config, scheduler_config=scheduler_config)
        self.embedding_layout_version = V2
        self.layout_species = cfg["layout_species"]
        if self.model.context_input_dim or self.model.pca_enabled:
            raise ValueError("growth500 task LayoutV2 does not permit PCA/context")
        self._layout_sources = source_signature()
        self.task_layout_signature()  # Validate the actual constructed branches.

    def task_layout_signature(self):
        branches = list(self.model.task_branches) if self.task_sharing_mode == "independent" else [self.model]
        descriptions = []
        for branch in branches:
            e = branch.embedding
            if not isinstance(e, TargetAwarePositionFirstEmbedding):
                raise RuntimeError("A task branch did not construct the versioned V2 embedding")
            descriptions.append({
                "layout": V2, "traits": list(branch.phenotype_names),
                "output_dim": branch.phenotype_dim,
                "cnn_groups": e.cnn_feature_groups,
                "cnn_features_per_group": e.num_cnn_features_per_phenotype,
                "embedding_width": e.num_cnn_features_total,
                "block_length": e.Block_length})
        return {"schema_version": 2, "implementation_version": ARCHITECTURE_VERSION,
                "embedding_layout_version": V2, "layout_species": self.layout_species,
                "task_sharing_mode": self.task_sharing_mode,
                "phenotype_names": list(self.phenotype_names), "output_dim": self.phenotype_dim,
                "branches": descriptions}

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["metan_architecture"] = self.task_layout_signature()
        checkpoint["embedding_layout_version"] = V2
        checkpoint["layout_provenance"] = {"sources": self._layout_sources,
                                           "model_config": deepcopy(dict(self.hparams.config))}

    def compute_loss(self, model_outputs, y_true, raw_features=None):
        if y_true.ndim != 2 or y_true.shape != model_outputs["final_pred"].shape:
            raise ValueError("Task LayoutV2 loss requires exactly the selected [batch, targets] labels")
        return super().compute_loss(model_outputs, y_true, raw_features=raw_features)

    def on_load_checkpoint(self, checkpoint):
        self._layout_checkpoint_verified = False
        if (checkpoint.get("embedding_layout_version") != V2
                or checkpoint.get("metan_architecture") != self.task_layout_signature()):
            raise RuntimeError("Checkpoint layout/task/trait/CNN/output/branch semantic mismatch")
        proof = checkpoint.get("layout_provenance", {})
        if (proof.get("sources") != source_signature()
                or proof.get("model_config") != dict(self.hparams.config)
                or checkpoint.get("hyper_parameters", {}).get("config") != dict(self.hparams.config)):
            raise RuntimeError("Checkpoint LayoutV2 source/config provenance mismatch")
        normalized = dict(checkpoint)
        normalized["metan_architecture"] = self.model.architecture_signature()
        super().on_load_checkpoint(normalized)
        self._layout_checkpoint_verified = True

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not self._layout_checkpoint_verified or not strict:
            raise RuntimeError("Task LayoutV2 requires a verified semantic Checkpoint and strict loading")
        try:
            return super().load_state_dict(state_dict, strict=True, assign=assign)
        finally:
            self._layout_checkpoint_verified = False

    def load_pretrained_checkpoint(self, checkpoint):
        # Require the full Lightning state: raw model weights cannot assert layout semantics.
        if "state_dict" not in checkpoint:
            raise RuntimeError("Task LayoutV2 requires a complete semantic Lightning Checkpoint")
        self.on_load_checkpoint(checkpoint)
        return self.load_state_dict(checkpoint["state_dict"], strict=True)


def resolve_training_model(config, legacy_class):
    version = config.get("layout_architecture_version")
    if version is not None and version != ARCHITECTURE_VERSION:
        raise ValueError("Unknown layout_architecture_version")
    layout = config.get("embedding_layout_version", LEGACY)
    if layout != V2:
        if version is not None or "cnn_feature_groups" in config.get("embedding", {}):
            raise ValueError("CNN-group decoupling requires explicit versioned LayoutV2")
        return resolve_historical_model(config, legacy_class)
    names, _ = resolve_phenotype_config(deepcopy(config))
    if version is not None or config.get("task_sharing_mode", "shared") == "independent" or len(names) == 1:
        normalize_task_config(config)
        return TaskLayoutV2METAN
    if "cnn_feature_groups" in config.get("embedding", {}):
        raise ValueError("Explicit CNN groups require layout_architecture_version")
    return resolve_historical_model(config, legacy_class)
