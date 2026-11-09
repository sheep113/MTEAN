"""Load one complete experiment definition, with paths relative to repository root."""
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def load_config(path):
    path=Path(path).resolve()
    spec=json.loads(path.read_text(encoding='utf8'))
    for key in ['model','training','baseline_config']:
        spec[key+'_path']=str((path.parent/spec[key]).resolve())
    for key in ['data','pca_root','cv_root']:
        spec[key]=str((ROOT/spec[key]).resolve())
    evaluation=spec.get('evaluation',{})
    if evaluation.get('data'):
        evaluation['data']=str((ROOT/evaluation['data']).resolve())
    spec['config_path']=str(path)
    return copy.deepcopy(spec)
