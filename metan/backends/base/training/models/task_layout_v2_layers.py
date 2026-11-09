# Modified for METAN portable release 2026-11-09.
"""Opt-in target/CNN decoupling over the validated position-first embedding."""
from torch import nn
from .carp_embedding_layout_v2 import PositionFirstEmbedding, V2

ARCHITECTURE_VERSION = "growth500_tasks_v1"


class TargetAwarePositionFirstEmbedding(PositionFirstEmbedding):
    """CNN groups describe features; target_dim describes auxiliary predictions."""

    def __init__(self, input_dim, config, target_dim, cnn_feature_groups,
                 gradient_checkpointing_config=None):
        self.target_dim = target_dim
        self.cnn_feature_groups = cnn_feature_groups
        super().__init__(input_dim, config, phenotype_dim=cnn_feature_groups,
                         gradient_checkpointing_config=gradient_checkpointing_config)

    def _setup_aux_networks(self, expansion):
        if self.target_dim == self.cnn_feature_groups:
            # Preserve the original parameter names, initialization order and math.
            return super()._setup_aux_networks(expansion)
        if self.target_dim != 1 or self.cnn_feature_groups != 2:
            raise ValueError("Unsupported target/CNN-group contract")
        width = self.num_cnn_features_total
        self.target_aux_ffn = nn.Sequential(
            nn.Linear(width, int(width * expansion)), nn.Tanh(),
            nn.Linear(int(width * expansion), self.target_dim))

    def _compute_aux_loss(self, features, batch_size, num_blocks):
        if self.target_dim == self.cnn_feature_groups:
            return super()._compute_aux_loss(features, batch_size, num_blocks)
        grouped = features.reshape(batch_size, num_blocks, self.num_cnn_features_total)
        return self.target_aux_ffn(grouped).mean(dim=1)


def create_task_embedding(config, target_dim, gradient_checkpointing_config):
    if (config.get("layout_architecture_version") != ARCHITECTURE_VERSION
            or config.get("embedding_layout_version") != V2):
        raise ValueError("Task-aware embedding requires explicit versioned LayoutV2")
    embedding = config["embedding"]
    groups = embedding.get("cnn_feature_groups")
    if type(groups) is not int or groups not in (1, 2):
        raise ValueError("Explicit cnn_feature_groups must be 1 or 2")
    return TargetAwarePositionFirstEmbedding(
        embedding["input_dims"], embedding, target_dim, groups,
        gradient_checkpointing_config)
