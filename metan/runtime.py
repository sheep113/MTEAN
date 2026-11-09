"""Config-driven model assembly with versioned source and checkpoint verification."""
import hashlib
import importlib
import json
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
_TEMP=None
_BACKEND=None

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            h.update(block)
    return h.hexdigest()

def activate(spec):
    global _TEMP,_BACKEND
    backend=spec['backend']
    if backend not in ('legacy','position_first'):
        raise ValueError('Unknown embedding layout backend')
    if _BACKEND is not None:
        if backend!=_BACKEND:
            raise RuntimeError('Run different embedding layouts in separate Python processes')
        return Path(_TEMP.name)
    manifest=json.loads((ROOT/'metan/source_manifest.json').read_text())
    for name,digest in manifest.items():
        if sha256(ROOT/name)!=digest:
            raise RuntimeError('METAN source integrity mismatch: '+name)
    _TEMP=tempfile.TemporaryDirectory(prefix='metan_runtime_')
    work=Path(_TEMP.name)
    base=ROOT/'metan/backends/base'
    paths=list(base.rglob('*.py'))
    if backend=='legacy':
        overlay=ROOT/'metan/backends/legacy'
        paths+=list(overlay.rglob('*.py'))
    for source in paths:
        anchor=base if source.is_relative_to(base) else overlay
        target=work/source.relative_to(anchor)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes())
    if 'training.models.model' in sys.modules:
        raise RuntimeError('METAN backend must be activated before importing model modules')
    sys.path[:0]=[str(work),str(work/'training')]
    sys.dont_write_bytecode=True
    _BACKEND=backend
    return work

def build(spec,config,optimizer_config=None,scheduler_config=None):
    activate(spec)
    if spec['backend']=='legacy':
        cls=importlib.import_module('training.models.METAN').METAN
    else:
        cls=importlib.import_module('model_selector').resolve_training_model(config)
    model=cls(config=config,optimizer_config=optimizer_config,scheduler_config=scheduler_config)
    observed=sum(p.numel() for p in model.parameters())
    if observed!=spec['parameters']:
        raise ValueError(f'Expected {spec["parameters"]} parameters, observed {observed}')
    return model

def load_checkpoint(spec,path,expected_sha256=None):
    import torch
    if expected_sha256 and sha256(path)!=expected_sha256:
        raise ValueError('Checkpoint SHA256 mismatch')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    proof=checkpoint.get('metan_release')
    manifest=json.loads((ROOT/'metan/source_manifest.json').read_text())
    if not proof or proof.get('source_manifest')!=manifest:
        raise ValueError('Checkpoint does not belong to this versioned METAN release')
    if proof['backend']!=spec['backend'] or proof['traits']!=spec['traits']:
        raise ValueError('Checkpoint layout or trait order mismatch')
    config=checkpoint.get('model_config') or checkpoint['hyper_parameters']['config']
    tc=checkpoint.get('training_config')
    optimizer=tc['optimizer'] if tc else checkpoint['hyper_parameters'].get('optimizer_config')
    scheduler=tc['scheduler'] if tc else checkpoint['hyper_parameters'].get('scheduler_config')
    model=build(spec,config,optimizer,scheduler)
    if 'model_state_dict' in checkpoint:
        state=checkpoint['model_state_dict']
    else:
        model.on_load_checkpoint(checkpoint)
        state=checkpoint['state_dict']
    result=model.load_state_dict(state,strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError('Strict checkpoint loading failed')
    if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError('Nonfinite checkpoint tensor')
    return model.eval(),checkpoint
