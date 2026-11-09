# Modified for METAN portable release 2026-11-09.
from __future__ import annotations
"""
DNA序列分析模型的主要实现 (模型结构部分)
"""
import torch
import torch.nn as nn
from training.utils.context_utils import resolve_context_config
import torch.nn.functional as F
import warnings
import logging
import math
from torch.utils.checkpoint import checkpoint
from torch import Tensor
from typing import Optional, Dict, List, Any, Tuple, Union, Literal
from functools import partial
from dataclasses import dataclass
from torch.jit import Final
from einops import rearrange
import numpy as np

import os
import json
from pathlib import Path
import yaml # 导入 yaml

# --- 修改：导入需要的层 ---
from .layers import (
    EmbeddingLayer,
    EmbeddingLayer_onlySNP, # <--- 添加
    OutputLayer,
    ExpertChoiceMoE,
    GlobalContextLayerNorm # <--- 添加
)
# --- 结束修改 ---
from .attention_types import create_attention_from_config
from .cross_attentions_types import create_crossattention_from_config
from .pooling import create_pooling_from_config, BasePooling
from training.utils.config_utils import resolve_phenotype_config, validate_task_sharing_config


class GFIFormerBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 block_config: Dict[str, Any],
                 phenotype_dim: int = 1,
                 random_seed: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.block_config = block_config

        self.phenotype_dim = int(phenotype_dim)

        if self.phenotype_dim <= 0:
            raise ValueError(
                f"phenotype_dim must be > 0, got "
                f"{self.phenotype_dim}"
            )

        self.name = block_config.get("name", "GFI_Block")
        self.context_length = block_config.get("context_length")
        if self.context_length is None:
            raise ValueError(f"Block '{self.name}': 'context_length' is required in block config.")

        encoder_config = block_config.get("encoder", {})
        decoder_config = block_config.get("decoder", {})
        moe_config = decoder_config.get("MOE", {})
        pooling_config = decoder_config.get("pooling", {})

        encoder_attention_config = encoder_config.get("attention", {})
        self.encoder_dim = encoder_attention_config.get("hidden_dims")
        if self.encoder_dim is None:
            raise ValueError(f"Block '{self.name}': Encoder attention config requires 'hidden_dims'.")

        # 1. Input projection: Changed from InstanceNorm1d to GlobalContextLayerNorm
        self.norm_input = GlobalContextLayerNorm(self.input_dim) # MODIFIED
        input_ffn_hidden_dim = self.input_dim * 4
        self.projection_from_input = nn.Sequential(
            nn.Linear(self.input_dim, input_ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(input_ffn_hidden_dim, self.encoder_dim)
        )

        # Encoder part (structure remains, uses self.encoder_dim)
        self.ffn_expansion_factor = encoder_config.get("ffn_expansion_factor", 4)
        self.ffn_sharing = encoder_config.get("ffn_sharing", False)
        self.encoder_num_layers = encoder_config.get("num_layers", 1)
        self.encoder_layers = nn.ModuleList()
        shared_ffn = None
        if self.ffn_sharing:
             hidden_dim = self.encoder_dim * self.ffn_expansion_factor
             shared_ffn = nn.Sequential(
                 nn.Linear(self.encoder_dim, hidden_dim),
                 nn.GELU(),
                 nn.Dropout(encoder_config.get("dropout_rate", 0.1)),
                 nn.Linear(hidden_dim, self.encoder_dim),
                 nn.Dropout(encoder_config.get("dropout_rate", 0.1))
             )
        for i in range(self.encoder_num_layers):
             self_attention_config = encoder_config.get("attention", {}).copy()
             self_attention_config["hidden_dims"] = self.encoder_dim
             self_attention = create_attention_from_config(self_attention_config, random_seed)
             ffn = shared_ffn if self.ffn_sharing else nn.Sequential(
                 nn.Linear(self.encoder_dim, self.encoder_dim * self.ffn_expansion_factor),
                 nn.GELU(),
                 nn.Dropout(encoder_config.get("dropout_rate", 0.1)),
                 nn.Linear(self.encoder_dim * self.ffn_expansion_factor, self.encoder_dim),
                 nn.Dropout(encoder_config.get("dropout_rate", 0.1))
             )
             self.encoder_layers.append(nn.ModuleList([
                 GlobalContextLayerNorm(self.encoder_dim), # norm1
                 self_attention,
                 GlobalContextLayerNorm(self.encoder_dim), # norm2
                 ffn
             ]))

        # Decoder part
        self.norm_cross_attn = GlobalContextLayerNorm(self.encoder_dim)
        cross_attention_config = decoder_config.get("cross_attention", {})
        if not cross_attention_config:
             warnings.warn(f"Block '{self.name}': Cross-attention config not found, using default.")
             cross_attention_config = {"type": "standard", "num_heads": 8, "head_dims": 64}
        self.v_projection_source = cross_attention_config.get("v_projection_source", "encoder")
        self.cross_attention = create_crossattention_from_config(cross_attention_config, self.encoder_dim)

        if not moe_config:
            raise ValueError(f"Block '{self.name}': MOE config is required.")
        self.num_experts = moe_config.get("num_experts")
        self.experts_dims = moe_config.get("experts_dims") # D_moe
        if self.num_experts is None or self.experts_dims is None:
            raise ValueError(f"Block '{self.name}': MOE config requires 'num_experts' and 'experts_dims'.")

        self.projection_to_moe_input = nn.Linear(self.encoder_dim, self.num_experts * self.experts_dims)
        self.norm_moe_input = GlobalContextLayerNorm(self.experts_dims)
        self.moe = ExpertChoiceMoE(moe_config)

        # 2. Pooling: Add LayerNorm before pooling, remove BatchNorm1d after pooling
        self.norm_before_pooling = GlobalContextLayerNorm(self.experts_dims) # Added LN before pooling

        if not pooling_config:
             warnings.warn(f"Block '{self.name}': Pooling config not found, using default average.")
             pooling_config = {"type": "average"}
        pooling_config_copy = pooling_config.copy()
        pooling_config_copy["num_heads"] = self.num_experts      # 让池化的 num_heads 跟随 MOE 的专家数
        pooling_config_copy["head_dims"] = self.experts_dims      # 专家维度也用 MOE 的
        if "num_experts" in pooling_config_copy: del pooling_config_copy["num_experts"]
        if "expert_dim" in pooling_config_copy: del pooling_config_copy["expert_dim"]
        self.pooling: BasePooling = create_pooling_from_config(pooling_config_copy)

        # 3. Auxiliary Supervision FFN
        aux_ffn_expansion_factor = moe_config.get("aux_ffn_expansion_factor", 4)
        aux_hidden_dim = self.experts_dims * aux_ffn_expansion_factor
        self.norm_before_aux_ffn = GlobalContextLayerNorm(self.experts_dims)
        if self.num_experts == 1:
            self.aux_loss_ffn = nn.Sequential(
                nn.Linear(self.experts_dims, aux_hidden_dim),
                nn.Tanh(),
                nn.Linear(aux_hidden_dim, 1)
            )
        else:
            self.aux_loss_ffns = nn.ModuleList()
            for _ in range(self.num_experts):
                self.aux_loss_ffns.append(
                    nn.Sequential(
                        nn.Linear(self.experts_dims, aux_hidden_dim),
                        nn.Tanh(),
                        nn.Linear(aux_hidden_dim, 1)
                    )
                )
        if hasattr(self, 'aux_loss_projection_layer'):
            del self.aux_loss_projection_layer
        if hasattr(self, 'aux_loss_projection_layers'):
            del self.aux_loss_projection_layers

        self.block_output_dim = (
            self.num_experts * self.experts_dims
        )

        # ====================================================
        # Dynamic phenotype auxiliary head
        #
        # 原代码的辅助输出数与 num_experts 绑定。
        # 现在改为真正输出 phenotype_dim 个性状。
        #
        # 保留旧 aux_loss_ffn(s)，用于旧 checkpoint
        # state_dict 兼容；新训练使用 task_aux_head。
        # ====================================================

        self.task_aux_head = nn.Sequential(
            nn.Linear(
                self.block_output_dim,
                aux_hidden_dim
            ),
            nn.Tanh(),
            nn.Linear(
                aux_hidden_dim,
                self.phenotype_dim
            )
        )

        gradient_checkpointing_config = block_config.get("gradient_checkpointing", {})
        self.use_gradient_checkpointing = gradient_checkpointing_config.get("enabled", True)
        self.use_encoder_block_checkpointing = gradient_checkpointing_config.get("encoder_block", True)
        self.use_self_attention_checkpointing = gradient_checkpointing_config.get("self_attention", False)
        self.use_ffn_checkpointing = gradient_checkpointing_config.get("ffn", False)
        self.use_cross_attention_checkpointing = gradient_checkpointing_config.get("cross_attention", False)
        self.use_moe_checkpointing = gradient_checkpointing_config.get("moe", False)
        self.use_reentrant = gradient_checkpointing_config.get("use_reentrant", False)

    def _encoder_layer_forward(self, x: torch.Tensor, layer_modules: List[nn.Module], mask: Optional[torch.Tensor], batch_size, num_blocks):
        norm1, self_attention, norm2, ffn = layer_modules
        residual = x
        x_norm1 = norm1(x, batch_size, num_blocks, mask=mask)  # 传入mask
        attn_output = None
        if self.training and self.use_gradient_checkpointing and self.use_self_attention_checkpointing:
             attn_output = checkpoint(lambda q, k, v, m: self_attention(q, k, v, mask=m)[0], x_norm1, x_norm1, x_norm1, mask, use_reentrant=self.use_reentrant)
        else:
             attn_output, _ = self_attention(x_norm1, x_norm1, x_norm1, mask=mask)
        x = residual + attn_output
        residual = x
        x_norm2 = norm2(x, batch_size, num_blocks, mask=mask)  # 传入mask
        ffn_output = None
        if self.training and self.use_gradient_checkpointing and self.use_ffn_checkpointing:
             ffn_output = checkpoint(ffn, x_norm2, use_reentrant=self.use_reentrant)
        else:
             ffn_output = ffn(x_norm2)
        x = residual + ffn_output
        return x

    def _cross_attention_forward_for_checkpoint(self, query, key, value, mask):
        output, _ = self.cross_attention(query, key, value, attn_mask=mask)
        return output

    def _moe_forward_for_checkpoint(self, x_reshaped, mask, batch_size, num_blocks):
        normed_for_moe = self.norm_moe_input(x_reshaped, batch_size, num_blocks, mask=mask)  # 传入mask
        return self.moe(normed_for_moe, attention_mask=mask)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, batch_size: Optional[int] = None, num_blocks: Optional[int] = None, token_coverage: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Optional[torch.Tensor]]]:
        batch_size_eff, seq_len_block, _ = x.shape 

        if batch_size is None or num_blocks is None:
            raise ValueError("GFIFormerBlock.forward now requires true_batch_size and num_total_blocks parameters from GFIFormer")
        
        x_norm = self.norm_input(x, batch_size=batch_size, num_blocks=num_blocks, mask=mask)
        projected_input = self.projection_from_input(x_norm)

        encoder_output = projected_input
        for layer_idx, layer_modules in enumerate(self.encoder_layers):
             if self.training and self.use_gradient_checkpointing and self.use_encoder_block_checkpointing:
                 encoder_output = checkpoint(
                     self._encoder_layer_forward,
                     encoder_output,
                     layer_modules,
                     mask,
                     batch_size,
                     num_blocks,
                     use_reentrant=self.use_reentrant,
                     preserve_rng_state=True 
                 )
             else:
                 encoder_output = self._encoder_layer_forward(encoder_output, layer_modules, mask=mask, batch_size=batch_size, num_blocks=num_blocks)
        
        residual1 = encoder_output
        normed_encoder_output = self.norm_cross_attn(encoder_output, batch_size=batch_size, num_blocks=num_blocks, mask=mask)
        
        value_source_for_cross_attn = encoder_output
        if self.v_projection_source != "encoder":
            value_source_for_cross_attn = projected_input

        normed_value_source = self.norm_cross_attn(value_source_for_cross_attn, batch_size=batch_size, num_blocks=num_blocks, mask=mask)

        cross_output = None
        cross_attn_weights = None
        cross_attn_mask_for_call = mask 
        if self.training and self.use_gradient_checkpointing and self.use_cross_attention_checkpointing:
             cross_output = checkpoint(
                 self._cross_attention_forward_for_checkpoint,
                 normed_encoder_output,
                 normed_encoder_output,
                 normed_value_source,
                 cross_attn_mask_for_call,
                 use_reentrant=self.use_reentrant,
                 preserve_rng_state=True
             )
        else:
             cross_output, cross_attn_weights = self.cross_attention(
                 query=normed_encoder_output,
                 key=normed_encoder_output, 
                 value=normed_value_source,
                 attn_mask=cross_attn_mask_for_call 
             )
        cross_output_with_residual = residual1 + cross_output

        projected_for_moe = self.projection_to_moe_input(cross_output_with_residual)
        residual2_unshaped = projected_for_moe
        reshaped_for_moe = projected_for_moe.view(batch_size_eff, seq_len_block * self.num_experts, self.experts_dims)

        expanded_mask_for_moe = None
        if mask is not None:
            expanded_mask_for_moe = mask.repeat_interleave(self.num_experts, dim=1)

        moe_output_raw = None
        if self.training and self.use_gradient_checkpointing and self.use_moe_checkpointing:
            moe_output_raw = checkpoint(
                self._moe_forward_for_checkpoint,
                reshaped_for_moe,
                expanded_mask_for_moe,
                batch_size,
                num_blocks,
                use_reentrant=self.use_reentrant,
                preserve_rng_state=True
            )
        else:
            normed_for_moe = self.norm_moe_input(reshaped_for_moe, batch_size=batch_size, num_blocks=num_blocks, mask=expanded_mask_for_moe)
            moe_output_raw = self.moe(normed_for_moe, attention_mask=expanded_mask_for_moe)

        residual2 = residual2_unshaped.view(batch_size_eff, seq_len_block * self.num_experts, self.experts_dims)
        moe_output_with_residual = residual2 + moe_output_raw

        normed_moe_output = self.norm_before_pooling(moe_output_with_residual, batch_size=batch_size, num_blocks=num_blocks, mask=expanded_mask_for_moe)
        
        pooling_return = self.pooling(
            expert_output=normed_moe_output, 
            attention_mask=mask,
            token_coverage=token_coverage
        )
        pooled_output = pooling_return[0]

        aux_loss_projections = None

        if self.training:

            # pooled_output:
            # [B_effective, num_experts, experts_dims]
            #
            # flatten:
            # [B_effective, num_experts * experts_dims]
            #
            # task_aux_head:
            # [B_effective, phenotype_dim]
            #
            # final DS shape:
            # [B_effective, phenotype_dim, 1]

            aux_input = pooled_output.reshape(
                pooled_output.shape[0],
                self.block_output_dim
            )

            aux_loss_projections = (
                self.task_aux_head(aux_input)
                .unsqueeze(-1)
            )

        output_features = pooled_output

        weights_dict = {}
        if not self.training:
             weights_dict['cross_attention'] = cross_attn_weights
             weights_dict['pooling'] = pooling_return[1] if len(pooling_return) > 1 else None

        return output_features, aux_loss_projections, weights_dict


class GFIFormer(nn.Module):
    """
    多个 GFIFormer Block 串联构成的 GFIFormer 模型
    """
    def __init__(self, input_dim: int, config: Dict[str, Any], random_seed: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.config = config

        (
            self.phenotype_names,
            self.phenotype_dim
        ) = resolve_phenotype_config(
            self.config
        )

        blocks_config = config.get("GFI_FormerBLOCKS", {})
        self.num_blocks = blocks_config.get("num_blocks", 1) # MODIFIED: Changed from num_gfi_blocks to num_blocks
        blocks = blocks_config.get("blocks", [])

        if len(blocks) < self.num_blocks: # MODIFIED: Changed from num_gfi_blocks to num_blocks
            raise ValueError(f"配置文件中只提供了{len(blocks)}个GFI块配置，但需要{self.num_blocks}个") # MODIFIED

        gradient_checkpointing_config = config.get("gradient_checkpointing", {})
        self.use_gradient_checkpointing = gradient_checkpointing_config.get("enabled", True)
        self.use_gfi_block_checkpointing = gradient_checkpointing_config.get("gfi_block", True)
        self.use_reentrant = gradient_checkpointing_config.get("use_reentrant", False)

        self.gfi_blocks = nn.ModuleList()
        self.pre_block_sequence_norms = nn.ModuleList() # 新增: 用于块前序列InstanceNorm
        current_block_input_dim = self.input_dim
        self.block_output_dims = [] 

        for i in range(self.num_blocks): # MODIFIED: Changed from num_gfi_blocks to num_blocks
            block_config = blocks[i]
            block_config["gradient_checkpointing"] = gradient_checkpointing_config

            self.pre_block_sequence_norms.append(
                nn.InstanceNorm1d(current_block_input_dim, affine=True)
            )

            gfi_block = GFIFormerBlock(
                input_dim=current_block_input_dim,
                block_config=block_config,
                phenotype_dim=self.phenotype_dim,
                random_seed=random_seed
            )
            self.gfi_blocks.append(gfi_block)
            current_block_input_dim = gfi_block.block_output_dim
            self.block_output_dims.append(current_block_input_dim)

        self.final_output_dim = current_block_input_dim 

        print(f"梯度检查点启用状态: {self.use_gradient_checkpointing}")
        print(f"GFIFormer 最终输出维度 (E*D_moe_last): {self.final_output_dim}")

    def _forward_block(self, block_input):
        block, x, mask, batch_size, num_blocks, coverage = block_input
        return block(x, mask, batch_size, num_blocks, coverage)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, token_coverage: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]], List[Dict[str, Optional[torch.Tensor]]]]:
        all_block_features: List[torch.Tensor] = []
        all_aux_projections: List[Optional[torch.Tensor]] = []
        attention_weights_list: List[Dict[str, Optional[torch.Tensor]]] = []

        true_batch_size, current_seq_len, current_dim = x.shape
        current_input = x
        current_mask = mask
        current_coverage = token_coverage if token_coverage is not None else x.new_ones(current_seq_len)
        assert current_coverage.shape == (current_seq_len,) and bool((current_coverage > 0).all())

        for i, gfi_block in enumerate(self.gfi_blocks):
            context_length = gfi_block.context_length
            
            if current_input.numel() > 0 and current_input.size(1) > 1:
                current_input_permuted = current_input.permute(0, 2, 1)
                normed_input_permuted = self.pre_block_sequence_norms[i](current_input_permuted)
                current_input_normed = normed_input_permuted.permute(0, 2, 1)
            else:
                current_input_normed = current_input
            
            remainder = current_seq_len % context_length
            num_full_blocks = current_seq_len // context_length
            num_blocks_for_this_layer = num_full_blocks + (1 if remainder > 0 else 0)

            parallel_mask = None
            padded_input = None
            
            if remainder > 0:
                padding_length = context_length - remainder
                padding_tensor = torch.zeros((true_batch_size, padding_length, current_dim),
                                            device=current_input_normed.device, dtype=current_input_normed.dtype)
                padded_input = torch.cat([current_input_normed, padding_tensor], dim=1)
                del padding_tensor

                target_mask_shape = (true_batch_size, num_blocks_for_this_layer, context_length)
                if current_mask is not None:
                    if current_mask.dtype != torch.bool:
                        current_mask = current_mask.bool()
                    
                    if current_mask.shape[1] != current_seq_len:
                         pass

                    parallel_mask_unshaped = torch.zeros(target_mask_shape, dtype=torch.bool, device=current_mask.device)
                    if num_full_blocks > 0:
                        parallel_mask_unshaped[:, :num_full_blocks, :] = current_mask[:, :num_full_blocks * context_length].reshape(
                            true_batch_size, num_full_blocks, context_length
                        )
                    parallel_mask_unshaped[:, num_full_blocks, :remainder] = current_mask[:, num_full_blocks * context_length : num_full_blocks * context_length + remainder]
                    parallel_mask = parallel_mask_unshaped.reshape(-1, context_length)
                    del parallel_mask_unshaped
                else:
                    parallel_mask_unshaped = torch.zeros(target_mask_shape, dtype=torch.bool, device=padded_input.device)
                    if num_full_blocks > 0:
                        parallel_mask_unshaped[:, :num_full_blocks, :] = True
                    parallel_mask_unshaped[:, num_full_blocks, :remainder] = True
                    parallel_mask = parallel_mask_unshaped.reshape(-1, context_length)
                    del parallel_mask_unshaped
            else:
                padded_input = current_input_normed
                if current_mask is not None:
                    if current_mask.dtype != torch.bool:
                        current_mask = current_mask.bool()
                    parallel_mask = current_mask.reshape(-1, context_length)
            
            # parallel_input = padded_input.view(-1, context_length, current_dim)
            parallel_input = padded_input.contiguous().view(-1, context_length, current_dim)
            padded_coverage = F.pad(current_coverage, (0, num_blocks_for_this_layer * context_length - current_seq_len))
            block_coverage = padded_coverage.view(num_blocks_for_this_layer, context_length)
            parallel_coverage = block_coverage.unsqueeze(0).expand(true_batch_size, -1, -1).reshape(-1, context_length)
            del padded_input

            block_output_tuple = None
            if self.use_gradient_checkpointing and self.use_gfi_block_checkpointing and self.training:
                block_output_tuple = torch.utils.checkpoint.checkpoint(
                    self._forward_block,
                    (gfi_block, parallel_input, parallel_mask, true_batch_size, num_blocks_for_this_layer, parallel_coverage),
                    preserve_rng_state=not self.use_reentrant,
                    use_reentrant=self.use_reentrant
                )
            else:
                block_output_tuple = gfi_block(parallel_input, parallel_mask, true_batch_size, num_blocks_for_this_layer, parallel_coverage)

            block_output_features, aux_projections_flat, attention_weights = block_output_tuple
            
            reshaped_features = block_output_features.view(
                true_batch_size, num_blocks_for_this_layer, gfi_block.num_experts, gfi_block.experts_dims
            )
            next_input = rearrange(reshaped_features, 'b n e d -> b n (e d)')

            all_block_features.append(next_input)
            all_aux_projections.append(aux_projections_flat) 

            current_input = next_input
            current_coverage = block_coverage.sum(dim=1)
            current_seq_len = num_blocks_for_this_layer
            current_dim = current_input.shape[-1]
            if num_blocks_for_this_layer > 0:
                current_mask = torch.ones((true_batch_size, num_blocks_for_this_layer), dtype=torch.bool, device=current_input.device)
            else:
                current_mask = torch.empty((true_batch_size, 0), dtype=torch.bool, device=current_input.device)


            del parallel_input
            if parallel_mask is not None:
                del parallel_mask
            del block_output_features 

            if not self.training and attention_weights:
                 attention_weights_list.append(attention_weights)

        if self.training or not attention_weights_list:
            attention_weights_list = [{} for _ in self.gfi_blocks]

        self.last_output_coverage = current_coverage
        return all_block_features, all_aux_projections, attention_weights_list


class METANModel(nn.Module):
    """
    封装 Embedding, GFIFormer, 和 OutputLayer 的完整模型结构。
    """
    def __init__(self, config: Dict[str, Any], random_seed: Optional[int] = None):
        super().__init__()
        self.config = config
        self.task_sharing_mode = validate_task_sharing_config(config)
        if self.task_sharing_mode == "independent":
            _initialize_independent_tasks(self, config, random_seed)
            return

        embedding_config = config.get("embedding", {})
        output_config = config.get("output_layer", {})
        loss_config = config.get("loss_config", {})
        gradient_checkpointing_config = config.get("gradient_checkpointing", {})

        (
            self.phenotype_names,
            self.phenotype_dim
        ) = resolve_phenotype_config(
            self.config
        )

        input_type = embedding_config.get("input_type", "SNPwithPOS").lower()
        self.embedding = None
        embedding_output_dim = 0
        if input_type == "snpwithpos":
            self.embedding = EmbeddingLayer(
                input_dim=15, 
                config=embedding_config, 
                phenotype_dim=self.phenotype_dim,
                gradient_checkpointing_config=gradient_checkpointing_config
            )
            embedding_output_dim = self.embedding.num_cnn_features_total
        elif input_type == "snp":
            snp_input_dim = embedding_config.get("input_dims")
            if snp_input_dim is None:
                raise ValueError("Embedding input_type 'SNP' requires 'input_dims' in config.")
            if "layout_architecture_version" in config:
                # Explicit opt-in only: keep historical construction/RNG unchanged.
                from .task_layout_v2_layers import create_task_embedding
                self.embedding = create_task_embedding(
                    config, self.phenotype_dim, gradient_checkpointing_config)
            else:
                self.embedding = EmbeddingLayer_onlySNP(
                    input_dim=snp_input_dim,
                    config=embedding_config,
                    phenotype_dim=self.phenotype_dim,
                    gradient_checkpointing_config=gradient_checkpointing_config
                )
            embedding_output_dim = self.embedding.num_cnn_features_total
        else:
            raise ValueError(f"Unsupported embedding input_type: {input_type}")

        # 如果配置中未提供 GFI_FormerBLOCKS 或 num_blocks 为 0，则跳过创建 GFIFormer，
        # 直接用 embedding 输出维度 作为 OutputLayer 的输入维度。
        blocks_config = config.get("GFI_FormerBLOCKS", None)
        create_gfi = False
        if blocks_config and isinstance(blocks_config, dict):
            try:
                num_blocks_cfg = int(blocks_config.get("num_blocks", 0))
                if num_blocks_cfg > 0:
                    create_gfi = True
            except Exception:
                create_gfi = True

        if create_gfi:
            self.gfi_former = GFIFormer(input_dim=embedding_output_dim, config=config, random_seed=random_seed)
            last_gfi_output_dim = self.gfi_former.final_output_dim
        else:
            self.gfi_former = None
            last_gfi_output_dim = embedding_output_dim

        # === PCA COVARIATES BEGIN ===
        # === GENERIC CONTEXT INPUTS ===
        self.context_spec = resolve_context_config(
            config
        )

        self.context_input_dim = int(
            self.context_spec["total_dim"]
        )

        self.context_late_fusion = bool(
            self.context_spec["late_fusion"]
        )

        conditioning_cfg = self.context_spec[
            "conditioning"
        ]

        self.context_conditioning_enabled = bool(
            conditioning_cfg.get(
                "enabled",
                False
            )
        )

        self.context_conditioner = None

        if self.context_conditioning_enabled:

            hidden_dim = int(
                conditioning_cfg.get(
                    "hidden_dim",
                    32
                )
            )

            if hidden_dim <= 0:
                raise ValueError(
                    "context conditioning "
                    "hidden_dim must be > 0"
                )

            # Context-conditioned FiLM on post-embedding
            # representation.
            self.context_conditioner = nn.Sequential(
                nn.LayerNorm(
                    self.context_input_dim
                ),
                nn.Linear(
                    self.context_input_dim,
                    hidden_dim
                ),
                nn.GELU(),
                nn.Linear(
                    hidden_dim,
                    2 * embedding_output_dim
                )
            )

            # Identity initialization:
            # gamma = 0
            # beta  = 0
            if bool(
                conditioning_cfg.get(
                    "init_identity",
                    True
                )
            ):
                nn.init.zeros_(
                    self.context_conditioner[-1].weight
                )
                nn.init.zeros_(
                    self.context_conditioner[-1].bias
                )

        output_input_dim = last_gfi_output_dim

        if (
            self.context_late_fusion
            and self.context_input_dim > 0
        ):
            output_input_dim += self.context_input_dim

        self.output_layer = OutputLayer(
            input_dim=output_input_dim,
            config=output_config
        )

        # Legacy aliases so previous PCA5 evaluation scripts
        # and checkpoints keep working.
        pca_source = self.context_spec[
            "sources"
        ].get(
            "pca",
            {}
        )

        self.pca_enabled = bool(
            pca_source.get(
                "enabled",
                False
            )
        )

        self.pca_n_components = int(
            pca_source.get(
                "input_dim",
                0
            )
        )

    def _snp_token_coverage(self, x, embedded):
        length = int(self.embedding.Block_length)
        count = embedded.shape[1]
        coverage = embedded.new_full((count,), float(length))
        coverage[-1] = x.shape[1] - (count - 1) * length
        assert bool((coverage > 0).all())
        return coverage

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        context_features: Optional[torch.Tensor] = None,
        pca_covariates: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        if self.task_sharing_mode == "independent":
            return _forward_independent_tasks(self, x, mask, context_features, pca_covariates)

        embedding_outputs = self.embedding(x)
        embed_features = embedding_outputs[0]

        # === Context-conditioned FiLM forward ===

        # Old PCA API becomes an alias of generic context.
        if (
            context_features is None
            and pca_covariates is not None
        ):
            context_features = pca_covariates

        context_required = (
            self.context_input_dim > 0
            and (
                self.context_late_fusion
                or self.context_conditioning_enabled
            )
        )

        if context_required:

            if context_features is None:
                raise ValueError(
                    "Context inputs are enabled but "
                    "context_features were not provided."
                )

            if context_features.ndim != 2:
                raise ValueError(
                    f"Expected context [B,C], got "
                    f"{context_features.shape}"
                )

            if (
                context_features.shape[0]
                != embed_features.shape[0]
            ):
                raise ValueError(
                    f"Context batch mismatch: "
                    f"{context_features.shape[0]} vs "
                    f"{embed_features.shape[0]}"
                )

            if (
                context_features.shape[1]
                != self.context_input_dim
            ):
                raise ValueError(
                    f"Expected context dim "
                    f"{self.context_input_dim}, got "
                    f"{context_features.shape[1]}"
                )

            context_features = context_features.to(
                device=embed_features.device,
                dtype=embed_features.dtype
            )

        # FiLM occurs BEFORE GFIFormer.
        # Therefore GFI attention/MoE/pooling receives
        # a context-conditioned representation.
        if self.context_conditioning_enabled:

            film = self.context_conditioner(
                context_features
            )

            gamma, beta = film.chunk(
                2,
                dim=-1
            )

            gamma = torch.tanh(gamma)

            embed_features = (
                embed_features
                * (
                    1.0
                    + gamma.unsqueeze(1)
                )
                + beta.unsqueeze(1)
            )

        embed_aux_proj = embedding_outputs[1]
        emb_pool_weights_from_layer = None
        num_emb_blocks_from_layer = 1

        if len(embedding_outputs) > 2:
            emb_pool_weights_from_layer = embedding_outputs[2]
        if len(embedding_outputs) > 3:
            num_emb_blocks_from_layer = embedding_outputs[3]

        if self.gfi_former is None:
            gfi_block_features_list: List[torch.Tensor] = []
            gfi_aux_proj_list: List[Optional[torch.Tensor]] = []
            attention_weights_list_raw: List[Dict[str, Optional[torch.Tensor]]] = [{} for _ in range(0)]
        else:
            gfi_block_features_list, gfi_aux_proj_list, attention_weights_list_raw = self.gfi_former(embed_features, mask, token_coverage=self._snp_token_coverage(x, embed_features))
 
        if not gfi_block_features_list:
            if (
                self.gfi_former is not None
                and self.gfi_former.num_blocks > 0
            ):
                raise RuntimeError(
                    "GFIFormer configured but returned "
                    "no block features."
                )

            output_layer_input = embed_features
        else:
             output_layer_input = gfi_block_features_list[-1] 
        
        if output_layer_input.ndim > 2:
            coverage = self.gfi_former.last_output_coverage.to(dtype=output_layer_input.dtype)
            output_layer_input = (output_layer_input * (coverage / coverage.sum())[None, :, None]).sum(dim=1)

        # === PCA COVARIATES BEGIN ===
        # === GENERIC CONTEXT LATE FUSION ===
        if (
            self.context_late_fusion
            and self.context_input_dim > 0
        ):
            output_layer_input = torch.cat(
                [
                    output_layer_input,
                    context_features
                ],
                dim=-1
            )

        final_pred = self.output_layer(output_layer_input)

        outputs_dict = {
            'final_pred': final_pred,
            'embed_features': embed_features,
            'gfi_block_features': gfi_block_features_list,
            'embed_aux_proj': embed_aux_proj,
            'gfi_aux_projections': gfi_aux_proj_list,
            'attention_weights_full': attention_weights_list_raw if not self.training else [{} for _ in range(getattr(self.gfi_former, 'num_blocks', 0))],
        }

        if not self.training:
            outputs_dict['embedding_pooling_weights'] = emb_pool_weights_from_layer
            outputs_dict['num_embedding_blocks'] = num_emb_blocks_from_layer
        
        return outputs_dict

    @property
    def num_gfi_blocks(self):
        if self.task_sharing_mode == "independent":
            return self.task_branches[0].num_gfi_blocks
        return getattr(self.gfi_former, "num_blocks", 0)

    def architecture_signature(self):
        return {"schema_version": 1, "task_sharing_mode": self.task_sharing_mode,
                "phenotype_names": list(self.phenotype_names),
                "independent_branch_definition": "single_phenotype_full_network" if self.task_sharing_mode == "independent" else None}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        assert_task_checkpoint_mode(state_dict, self.task_sharing_mode)
        if assign:
            return super().load_state_dict(state_dict, strict=strict, assign=True)
        return super().load_state_dict(state_dict, strict=strict)


def assert_task_checkpoint_mode(state_dict, expected_mode, prefix=""):
    observed = "independent" if any(str(key).startswith(prefix + "task_branches.") for key in state_dict) else "shared"
    if observed != expected_mode:
        raise RuntimeError("Checkpoint architecture mismatch: saved=" + observed + ", requested=" + expected_mode
                           + ". Shared and independent weights cannot be silently interchanged (including strict=False).")


def _initialize_independent_tasks(model, config, random_seed):
    from copy import deepcopy
    model.phenotype_names, model.phenotype_dim = resolve_phenotype_config(config)
    model.task_branches = nn.ModuleList()
    for trait in model.phenotype_names:
        branch_config = deepcopy(config)
        branch_config["task_sharing_mode"] = "shared"
        branch_config["output_layer"]["phenotype_name"] = [trait]
        branch_config["output_layer"]["phenotype_dim"] = 1
        # Original SNP embedding width depends on phenotype_dim. Each branch is a
        # complete original single-phenotype network; report actual capacity changes.
        model.task_branches.append(METANModel(branch_config, random_seed=random_seed))
    first = model.task_branches[0]
    model.context_spec = deepcopy(first.context_spec)
    model.context_input_dim = first.context_input_dim
    model.pca_enabled = first.pca_enabled
    model.pca_n_components = first.pca_n_components


def _forward_independent_tasks(model, x, mask, context_features, pca_covariates):
    outputs = [branch(x, mask, context_features=context_features, pca_covariates=pca_covariates)
               for branch in model.task_branches]
    final_pred = torch.cat([out["final_pred"] for out in outputs], dim=1)
    # No fictitious shared features: auxiliary interfaces are explicitly restricted.
    return {"final_pred": final_pred, "task_outputs": outputs,
            "task_sharing_mode": "independent", "phenotype_names": list(model.phenotype_names),
            "embed_features": None, "gfi_block_features": [], "embed_aux_proj": None,
            "gfi_aux_projections": [], "attention_weights_full": []}
