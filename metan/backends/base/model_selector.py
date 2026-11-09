# Modified for METAN portable release 2026-11-09.
"""Configurable capacity and PCA late-fusion computation."""
from copy import deepcopy
from pathlib import Path
import hashlib
from training.models.cnn_capacity_v1 import CNNCapacityMETAN, capacity_source_signature
from training.models.layers import OutputLayer
from training.utils.context_utils import resolve_context_config

VERSION='METAN_pca_late_fusion_v1'


class PCACapacityMETAN(CNNCapacityMETAN):
 def __init__(self,config_path=None,config=None,optimizer_config=None,scheduler_config=None):
  if config is None:
   import json
   config=json.loads(Path(config_path).read_text())
  full=deepcopy(config);pc=full.get('pca_covariates',{})
  if full.get('boundary_handling_version')!='valid_SNP_coverage_v1':raise ValueError('Explicit boundary coverage version required')
  if (full.get('pca_aux_implementation_version')!=VERSION or pc.get('enabled') is not True
      or type(pc.get('n_components')) is not int or pc['n_components'] < 1
      or pc.get('fusion')!='concat' or full.get('context_inputs') is not None
      or full['embedding'].get('cnn_path_output_channels')!=3
      or not full['output_layer']['phenotype_name']):
   raise ValueError('Explicit C3 and positive PCA dimension required')
  binding=full.get('pca_asset_binding',{})
  required={'run_id','seed','fold','n_components','CV_SHA256','fit_IDs_SHA256','validation_IDs_SHA256','asset_manifest_SHA256'}
  if set(binding)!=required or binding['n_components']!=pc['n_components']:
   raise ValueError('PCA asset/Fold binding missing or inconsistent')
  self._pca_building=True
  template=deepcopy(full);template.pop('pca_covariates');template.pop('pca_asset_binding')
  super().__init__(config=template,optimizer_config=optimizer_config,scheduler_config=scheduler_config)
  # Reuse the exact public context resolution, forward concatenation and OutputLayer.
  # The existing signed adapter stays unchanged and still builds/verifies the entire C3 core.
  b=self.model;b.context_spec=resolve_context_config(full)
  b.context_input_dim=b.context_spec['total_dim'];b.context_late_fusion=True
  b.context_conditioning_enabled=False;b.context_conditioner=None
  b.pca_enabled=True;b.pca_n_components=pc['n_components']
  b.output_layer=OutputLayer(input_dim=b.gfi_former.final_output_dim+b.context_input_dim,config=full['output_layer'])
  self.hparams.config['pca_covariates']=deepcopy(pc)
  self.hparams.config['pca_asset_binding']=deepcopy(binding)
  self.hparams.config['pca_aux_implementation_version']=VERSION
  b.config=self.hparams.config
  self._pca_building=False
  self.task_layout_signature()

 def task_layout_signature(self):
  sig=super().task_layout_signature()
  if self._pca_building:return sig
  b=self.model;k=int(self.hparams.config['pca_covariates']['n_components'])
  if (not b.pca_enabled or b.context_conditioning_enabled or not b.context_late_fusion
      or b.context_input_dim!=k or b.output_layer.input_dim!=128+k
      or b.gfi_former.final_output_dim!=128):
   raise RuntimeError('C3/PCA late fusion shape or semantic mismatch')
  sig.update(pca_aux_implementation_version=VERSION,pca_components=k,
             pca_fusion='valid_SNP_coverage_weighted_GFI_concat_before_OutputLayer',boundary_handling_version='valid_SNP_coverage_v1',
             output_input_dim=128+k,pca_asset_binding=deepcopy(self.hparams.config['pca_asset_binding']))
  return sig

 def on_save_checkpoint(self,checkpoint):
  super().on_save_checkpoint(checkpoint)
  checkpoint['pca_aux_provenance']={'implementation_version':VERSION,
    'adapter_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'asset_binding':deepcopy(self.hparams.config['pca_asset_binding'])}

 def on_load_checkpoint(self,checkpoint):
  proof=checkpoint.get('pca_aux_provenance',{})
  if proof!={'implementation_version':VERSION,
     'adapter_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'asset_binding':deepcopy(self.hparams.config['pca_asset_binding'])}:
   raise RuntimeError('PCA implementation/Fold/component/asset mismatch')
  return super().on_load_checkpoint(checkpoint)

def resolve_training_model(config,legacy=None):
 if config.get('pca_aux_implementation_version')!=VERSION:
  raise ValueError('METAN PCA implementation version does not match the configured architecture')
 return PCACapacityMETAN
