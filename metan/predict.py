"""Evaluate one frozen model on an explicitly supplied HDF5 and matching PCA CSV."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--sha256')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--pca-csv',type=Path,required=True,help='Scores transformed using the checkpoint training-fold PCA; never refit on evaluation data')
    p.add_argument('--normalization',type=Path,help='Required if historical checkpoint does not contain target mean/std')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--describe',action='store_true')
    a=p.parse_args()
    from metan.config import load_config
    spec=load_config(a.config)
    if a.describe:print(json.dumps({'species':spec['name'],'inference_only':True}));return
    import h5py
    import numpy as np
    import pandas as pd
    import torch
    from metan.runtime import load_checkpoint,sha256
    from metan.metrics import regression_metrics
    a.out.mkdir(parents=True,exist_ok=False)
    m,cp=load_checkpoint(spec,a.checkpoint,a.sha256)
    config=cp.get('model_config') or cp['hyper_parameters']['config']
    traits=config['output_layer']['phenotype_name'];k=config['pca_covariates']['n_components']
    norm=cp.get('normalization') or cp.get('release_provenance',{}).get('normalization') or cp.get('pipeline_provenance',{}).get('normalization')
    if a.normalization:norm=json.loads(a.normalization.read_text())
    if not norm or 'mean' not in norm or 'std' not in norm:raise ValueError('Checkpoint-specific training target normalization is required')
    def decode(values):return [v.decode() if isinstance(v,bytes) else str(v) for v in values]
    with h5py.File(a.data,'r') as h:
        ids=decode(h['sample_ids'][:]);names=decode(h['phenotype_names'][:]);raw=h['phenotypes'][:]
        assert len(ids)==len(set(ids)) and set(traits)<=set(names)
        genotype=h['features/genotype_features'][:]
    assert genotype.ndim==3 and genotype.shape[1]==len(ids) and genotype.shape[2]==10
    genotype=np.transpose(genotype,(1,0,2)).astype(np.float32)
    targets=np.asarray(raw)[:,[names.index(t) for t in traits]]
    pcs=pd.read_csv(a.pca_csv,dtype={'sample_id':str}).set_index('sample_id')
    assert pcs.index.is_unique and set(ids)<=set(pcs.index)
    cov=pcs.loc[ids,[f'PC{i}' for i in range(1,k+1)]].to_numpy(np.float32)
    assert np.isfinite(cov).all() and np.isfinite(genotype).all()
    if not torch.cuda.is_available():raise RuntimeError('Formal FlashAttention inference requires CUDA')
    torch.backends.cudnn.enabled=spec['cudnn'];m.cuda().eval();parts=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(ids),32):
            x=torch.from_numpy(genotype[start:start+32]).cuda();c=torch.from_numpy(cov[start:start+32]).cuda()
            parts.append(m(x,pca_covariates=c)['final_pred'].float().cpu().numpy())
    normalized=np.concatenate(parts);pred=normalized.astype(float)*np.asarray(norm['std'])+np.asarray(norm['mean'])
    frame={'sample_id':ids};metrics=[]
    for j,t in enumerate(traits):
        frame['true_'+t]=targets[:,j];frame['pred_'+t]=pred[:,j]
        metrics.append({'trait':t,**regression_metrics(targets[:,j],pred[:,j])})
    pd.DataFrame(frame).to_csv(a.out/'predictions.csv',index=False,float_format='%.17g')
    pd.DataFrame(metrics).to_csv(a.out/'metrics.csv',index=False)
    (a.out/'evaluation_provenance.json').write_text(json.dumps({'checkpoint_SHA256':sha256(a.checkpoint),'data_SHA256':sha256(a.data),'pca_scores_SHA256':sha256(a.pca_csv),'strict_reload':True,'PCA_fitted_during_evaluation':False,'inference_only':True},indent=2))


if __name__=='__main__':main()
