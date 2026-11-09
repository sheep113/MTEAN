# Modified for METAN portable release 2026-11-09.
"""Opt-in Carp-only Embedding layout v2. No shared source or Pooling edits.
The overridden method is the historical method with exactly two token-order changes.
Parameters are initialized by the original constructor before the class transition.
"""
from pathlib import Path
import hashlib
from .METAN import METAN as LegacyMETAN
from .layers import EmbeddingLayer_onlySNP, Tensor, Optional, Tuple, rearrange, checkpoint, F, torch
LEGACY = "legacy_group_first"
V2 = "position_first_v2"
BASE_LAYER_SHA256 = "42a8df21df2a89b26123cd6d9cc7337eba17de6846a77e2066f7b8d5ae7567e4"
class PositionFirstEmbedding(EmbeddingLayer_onlySNP):
    def _process_with_pooling(
        self,
        cnn_output: Tensor,
        batch_size: int,
        E: int,
        C: int,
        L: int
    ) -> Tuple[Tensor, int, Optional[Tensor]]:
        """
        将 SNP 序列划分为长度 L 的完整 block。
        若尾部存在不足 L 的 SNP，则保留为最后一个 remainder block。

        例如:
            Seq = 28804, L = 32
            -> 900 个长度 32 的完整 block
            -> 1 个长度 4 的 remainder block
            -> 总计 901 个 block

        模型计算过程中不删除 SNP，也不对 remainder block 做输入 padding。
        仅在 eval 导出 pooling weights 时，将 remainder 权重尾部补 0 到 L，
        以保持统一的 HDF5 张量形状。
        """
        seq_len = cnn_output.size(2)

        if seq_len <= 0:
            raise ValueError(
                f"Invalid sequence length for block pooling: {seq_len}"
            )

        num_full_blocks = seq_len // L
        remainder = seq_len % L
        num_blocks = num_full_blocks + (1 if remainder > 0 else 0)

        if self.block_pooling is None:
            raise RuntimeError(
                "Block_length > 1 but block_pooling is None."
            )

        pooled_chunks = []
        weight_chunks = []
        weights_complete = not self.training


        def run_pooling(
            blocked_tensor: Tensor,
            num_chunk_blocks: int
        ):
            """对一批同长度 block 执行 LN + pooling。"""

            normed_input = self.ln_before_pooling(
                blocked_tensor,
                batch_size,
                num_chunk_blocks
            )

            actual_weights = None

            if self.training and self.use_pooling_checkpointing:
                pooled_tuple = checkpoint(
                    self.block_pooling,
                    normed_input,
                    None,
                    use_reentrant=self.use_reentrant,
                    preserve_rng_state=True
                )
                pooled = pooled_tuple[0]

            else:
                pooled, actual_weights = self.block_pooling(
                    normed_input,
                    None
                )

            return pooled, actual_weights


        # ====================================================
        # A. 完整 block
        # ====================================================
        full_length = num_full_blocks * L

        if num_full_blocks > 0:
            full_part = cnn_output[:, :, :full_length]

            # Explicit v2: position first, CNN groups interleaved; public Pooling unchanged.
            full_blocked = rearrange(
                full_part,
                'b (e c) (n l) -> (b n) (l e) c',
                e=E,
                c=C,
                n=num_full_blocks,
                l=L
            )

            full_pooled, full_weights = run_pooling(
                full_blocked,
                num_full_blocks
            )

            full_pooled = rearrange(
                full_pooled,
                '(b n) e c -> b n e c',
                b=batch_size,
                n=num_full_blocks
            )

            pooled_chunks.append(full_pooled)

            if not self.training:
                if full_weights is None:
                    weights_complete = False
                else:
                    full_weights = rearrange(
                        full_weights,
                        '(b n) e l -> b n e l',
                        b=batch_size,
                        n=num_full_blocks,
                        l=L
                    )
                    weight_chunks.append(full_weights)


        # ====================================================
        # B. remainder block
        # ====================================================
        if remainder > 0:
            remainder_part = cnn_output[:, :, full_length:]

            # [B, E*C, R] -> [B, E*R, C]
            remainder_blocked = rearrange(
                remainder_part,
                'b (e c) r -> b (r e) c',
                e=E,
                c=C,
                r=remainder
            )

            remainder_pooled, remainder_weights = run_pooling(
                remainder_blocked,
                1
            )

            # [B, E, C] -> [B, 1, E, C]
            remainder_pooled = remainder_pooled.unsqueeze(1)
            pooled_chunks.append(remainder_pooled)

            if not self.training:
                if remainder_weights is None:
                    weights_complete = False
                else:
                    # 实际模型只计算 R 个 SNP。
                    # 这里仅为导出统一形状补 0 到长度 L。
                    # [B,E,R] -> [B,E,L]
                    remainder_weights = F.pad(
                        remainder_weights,
                        (0, L - remainder),
                        mode='constant',
                        value=0.0
                    )

                    # [B,E,L] -> [B,1,E,L]
                    weight_chunks.append(
                        remainder_weights.unsqueeze(1)
                    )


        # ====================================================
        # C. 合并完整 block 和 remainder block
        # ====================================================
        pooled_output = torch.cat(
            pooled_chunks,
            dim=1
        )
        # [B, N, E, C]

        pooled_output = rearrange(
            pooled_output,
            'b n e c -> (b n) e c'
        )

        # 与原实现一致：pooling 后进行 LayerNorm
        if self.ln_after_pooling is not None:
            pooled_output = self.ln_after_pooling(
                pooled_output
            )

        normed_features = rearrange(
            pooled_output,
            '(b n) e c -> (b n e) c',
            b=batch_size,
            n=num_blocks
        )


        # ====================================================
        # D. eval 模式下统一 pooling weights
        # ====================================================
        embedding_pooling_weights_to_return = None

        if (
            not self.training
            and weights_complete
            and len(weight_chunks) == len(pooled_chunks)
        ):
            all_weights = torch.cat(
                weight_chunks,
                dim=1
            )
            # [B, N, E, L]

            embedding_pooling_weights_to_return = rearrange(
                all_weights,
                'b n e l -> (b n) e l'
            )
            # 保持原 predict/save 接口:
            # [B*N, E, L]

        return (
            normed_features,
            num_blocks,
            embedding_pooling_weights_to_return
        )

class CarpLayoutMETAN(LegacyMETAN):
    def __init__(self, config_path=None, config=None, optimizer_config=None, scheduler_config=None):
        super().__init__(config_path=config_path, config=config, optimizer_config=optimizer_config, scheduler_config=scheduler_config)
        cfg=self.hparams.config
        self.embedding_layout_version=cfg.get("embedding_layout_version",LEGACY)
        if self.embedding_layout_version not in [LEGACY,V2]:
            raise ValueError("Unknown Embedding layout: "+str(self.embedding_layout_version))
        self._layout_checkpoint_verified=False
        if self.embedding_layout_version==V2:
            if self.task_sharing_mode!="shared" or self.phenotype_names!=cfg.get("layout_allowed_traits"):
                raise ValueError("position_first_v2 requires the configured shared trait order")
            original=Path(__file__).with_name("layers.py")
            if hashlib.sha256(original.read_bytes()).hexdigest()!=BASE_LAYER_SHA256:
                raise RuntimeError("Unreviewed base Embedding source; refusing layout overlay")
            e=self.model.embedding
            if type(e) is not EmbeddingLayer_onlySNP or e.phenotype_dim!=2 or e.Block_length!=32 or e.num_cnn_features_per_phenotype!=15:
                raise ValueError("Unexpected Embedding contract for this isolated experiment")
            # No constructor, parameter registration, RNG call, or parameter copy here.
            e.__class__=PositionFirstEmbedding

    def on_save_checkpoint(self,checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["embedding_layout_version"]=self.embedding_layout_version
        checkpoint["metan_architecture"]["embedding_layout_version"]=self.embedding_layout_version
        checkpoint["layout_provenance"]={
            "scope":"configured position-first overlay",
            "implementation":__file__,
            "implementation_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "base_layers_sha256":BASE_LAYER_SHA256,
            "task_sharing_mode":self.task_sharing_mode,
            "phenotype_names":list(self.phenotype_names),
            "model_config":dict(self.hparams.config)}
        # Full model/optimizer/scheduler configs are already stored in hyper_parameters.

    def on_load_checkpoint(self,checkpoint):
        saved=checkpoint.get("embedding_layout_version",LEGACY)
        signature=dict(checkpoint.get("metan_architecture") or {})
        recorded=signature.pop("embedding_layout_version",saved)
        cfg=checkpoint.get("hyper_parameters",{}).get("config",{})
        if saved!=self.embedding_layout_version or recorded!=saved or cfg.get("embedding_layout_version",LEGACY)!=saved:
            raise RuntimeError("Checkpoint Embedding layout semantic mismatch")
        if saved==V2:
            proof=checkpoint.get("layout_provenance",{})
            if proof.get("implementation_sha256")!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
                raise RuntimeError("Unverified layout implementation source hash")
            if proof.get("task_sharing_mode")!="shared" or proof.get("phenotype_names")!=["survival","SL"]:
                raise RuntimeError("Layout checkpoint task/trait metadata mismatch")
        normalized=dict(checkpoint)
        if signature:normalized["metan_architecture"]=signature
        super().on_load_checkpoint(normalized)
        self._layout_checkpoint_verified=True

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if self.embedding_layout_version==V2 and not self._layout_checkpoint_verified:
            raise RuntimeError("Raw state_dict lacks layout semantics; use verified load_from_checkpoint/load_pretrained_checkpoint")
        if not strict:
            raise RuntimeError("This layout experiment requires strict checkpoint loading")
        return super().load_state_dict(state_dict,strict=True,assign=assign)
