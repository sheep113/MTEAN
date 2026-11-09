"""Validate release integrity, fixed CV membership, HDF5 axes and PCA bindings."""
import argparse
import hashlib
import json
from pathlib import Path

from metan.config import ROOT,load_config
from metan.runtime import sha256

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--pca-root',type=Path)
    args=parser.parse_args()
    spec=load_config(args.config)
    manifest=json.loads((ROOT/'metan/source_manifest.json').read_text())
    for relative,expected in manifest.items():
        if sha256(ROOT/relative)!=expected:raise ValueError('Source checksum mismatch: '+relative)
    import h5py
    import numpy as np
    import pandas as pd
    source=args.data or Path(spec['data'])
    if not source.is_file():raise FileNotFoundError('Configured data are required from their custodian: '+str(source))
    with h5py.File(source,'r') as h:
        ids=[x.decode() if isinstance(x,bytes) else str(x) for x in h['sample_ids'][:]]
        if len(ids)!=spec['development_n'] or len(set(ids))!=len(ids):raise ValueError('Development sample count or uniqueness mismatch')
        if h['features/genotype_features'].shape!=(spec['SNPs'],len(ids),10):raise ValueError('HDF5 feature axes mismatch')
    pca=args.pca_root or Path(spec['pca_root'])
    for seed in spec['seeds']:
        csv=Path(spec['cv_root'])/f'cv_splits_{seed}.csv'
        frame=pd.read_csv(csv,dtype={'sample_id':str})
        if set(frame.fold)!=set(spec['folds']):raise ValueError('Missing or unexpected CV folds')
        validation=[]
        for fold in spec['folds']:
            block=frame[frame.fold==fold]
            training=block.loc[block.split=='train','sample_id'].tolist()
            val=block.loc[block.split=='val','sample_id'].tolist()
            if len(training)!=len(set(training)) or len(val)!=len(set(val)) or set(training)&set(val) or set(training+val)!=set(ids):raise ValueError('Invalid fixed CV membership')
            validation+=val
            asset=pca/f'seed_{seed}'/f'fold_{fold}'
            meta=json.loads((asset/spec['pca_metadata']).read_text())
            if spec['pca_metadata']=='asset_manifest.json':
                if meta['fit_IDs']!=training or meta['validation_IDs']!=val or meta['CV_SHA256']!=sha256(csv) or meta['n_components']!=spec['PCA']:raise ValueError('PCA/CV binding mismatch')
                for name,digest in meta['files_SHA256'].items():
                    if sha256(asset/name)!=digest:raise ValueError('PCA asset checksum mismatch')
            else:
                if meta['cv_sha256']!=sha256(csv) or meta['Test_used_or_transformed']:raise ValueError('PCA metadata mismatch')
            scores=pd.read_csv(asset/'pca_scores.csv',dtype={'sample_id':str}).set_index('sample_id')
            if not scores.index.is_unique or not set(ids)<=set(scores.index):raise ValueError('Missing or duplicate PCA sample IDs')
            if not np.isfinite(scores.loc[ids,[f'PC{i}' for i in range(1,spec['PCA']+1)]].to_numpy()).all():raise ValueError('Nonfinite PCA score')
        if len(validation)!=len(ids) or set(validation)!=set(ids):raise ValueError('Validation folds do not cover each individual once')
    print(json.dumps({'experiment':spec['name'],'status':'PASS','samples':len(ids),'SNPs':spec['SNPs'],'folds':len(spec['seeds'])*len(spec['folds']),'training_started':False}))
