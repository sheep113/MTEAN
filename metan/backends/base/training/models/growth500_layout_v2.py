# Modified for METAN portable release 2026-11-09.
"""Explicit Shared LayoutV2 adapter for the two growth-stratified500 datasets.
Reuses the previously validated PositionFirstEmbedding; legacy and Independent defaults are untouched.
"""
from pathlib import Path
import hashlib
from .METAN import METAN as LegacyMETAN
from .carp_embedding_layout_v2 import PositionFirstEmbedding, LEGACY, V2, BASE_LAYER_SHA256
from .layers import EmbeddingLayer_onlySNP
POSITION_SOURCE_SHA256 = "ef4269b1019c778689dc524578bf1f7739f717cffa20d7b69d9c6b463669f939"

class Growth500LayoutMETAN(LegacyMETAN):
    def __init__(self, config_path=None, config=None, optimizer_config=None, scheduler_config=None):
        super().__init__(config_path=config_path, config=config, optimizer_config=optimizer_config, scheduler_config=scheduler_config)
        cfg=self.hparams.config
        self.embedding_layout_version=cfg.get("embedding_layout_version", LEGACY)
        self._layout_checkpoint_verified=False
        self.layout_species=cfg.get("layout_species")
        if self.embedding_layout_version!=V2:
            raise ValueError("This opt-in adapter requires explicit position_first_v2")
        if self.phenotype_names!=config.get("layout_allowed_traits") or self.task_sharing_mode!="shared":
            raise ValueError("LayoutV2 species/trait/shared contract mismatch")
        if self.model.context_input_dim or self.model.pca_enabled:
            raise ValueError("These500 experiments do not permit PCA/context")
        folder=Path(__file__).parent
        if hashlib.sha256((folder/"layers.py").read_bytes()).hexdigest()!=BASE_LAYER_SHA256 or hashlib.sha256((folder/"carp_embedding_layout_v2.py").read_bytes()).hexdigest()!=POSITION_SOURCE_SHA256:
            raise RuntimeError("Validated Embedding base/overlay source hash mismatch")
        e=self.model.embedding
        if type(e) is not EmbeddingLayer_onlySNP or e.phenotype_dim!=2 or e.Block_length!=int(cfg["embedding"]["Block_length"]) or e.num_cnn_features_per_phenotype!=15:
            raise ValueError("Unexpected two-CNN-group Embedding contract")
        e.__class__=PositionFirstEmbedding

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["embedding_layout_version"]=V2
        checkpoint["metan_architecture"]["embedding_layout_version"]=V2
        checkpoint["layout_species"]=self.layout_species
        checkpoint["layout_provenance"]={"adapter_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"position_source_sha256":POSITION_SOURCE_SHA256,"base_layers_sha256":BASE_LAYER_SHA256,"phenotype_names":list(self.phenotype_names),"model_config":dict(self.hparams.config)}

    def on_load_checkpoint(self, checkpoint):
        signature=dict(checkpoint.get("metan_architecture") or {})
        saved=checkpoint.get("embedding_layout_version",LEGACY)
        recorded=signature.pop("embedding_layout_version",saved)
        cfg=checkpoint.get("hyper_parameters",{}).get("config",{})
        proof=checkpoint.get("layout_provenance",{})
        if saved!=V2 or recorded!=V2 or cfg.get("embedding_layout_version",LEGACY)!=V2 or checkpoint.get("layout_species")!=self.layout_species:
            raise RuntimeError("Checkpoint Embedding layout/species semantic mismatch")
        if proof.get("adapter_sha256")!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or proof.get("position_source_sha256")!=POSITION_SOURCE_SHA256 or proof.get("base_layers_sha256")!=BASE_LAYER_SHA256:
            raise RuntimeError("Checkpoint LayoutV2 source provenance mismatch")
        if proof.get("phenotype_names")!=self.phenotype_names or proof.get("model_config")!=dict(self.hparams.config):
            raise RuntimeError("Checkpoint trait/config metadata mismatch")
        normalized=dict(checkpoint);normalized["metan_architecture"]=signature
        super().on_load_checkpoint(normalized)
        self._layout_checkpoint_verified=True

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if not self._layout_checkpoint_verified:
            raise RuntimeError("Raw state_dict lacks layout semantics; load a verified Checkpoint")
        if not strict:raise RuntimeError("LayoutV2 requires strict Checkpoint loading")
        return super().load_state_dict(state_dict,strict=True,assign=assign)


def resolve_training_model(config, legacy_class):
    layout=config.get("embedding_layout_version",LEGACY)
    if layout==LEGACY:
        return legacy_class
    if layout!=V2:
        raise ValueError("Unknown embedding_layout_version: "+repr(layout))
    if config.get("task_sharing_mode","shared")!="shared":
        raise ValueError("The500 LayoutV2 adapter does not alter Independent mode")
    return Growth500LayoutMETAN
