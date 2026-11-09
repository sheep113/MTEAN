# Modified for METAN portable release 2026-11-09.
"""Training-fold PCA preparation without raw matrix files or Test transforms."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path('${METAN_ROOT}')
PREFIX = ROOT / 'data/blackcarp499/pca_plink_unique/blackcarp_unique'
PYTHON = '${METAN_ENV}/bin/python'
PLINK = '${METAN_ENV}/bin/plink'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)

def sample_mapping():
    with h5py.File(ROOT/'data/blackcarp499/processed/blackcarp_allele10_trainval.h5', 'r') as f:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['sample_ids'][:]]
    assert len(ids) == len(set(ids)) == 424
    fam = [line.split()[:2] for line in Path(str(PREFIX)+'.fam').read_text().splitlines()]
    aliases = {}
    for i, (fid, iid) in enumerate(fam):
        for value in {iid, fid+'_'+iid, fid+':'+iid, fid+'-'+iid}:
            aliases.setdefault(value, []).append(i)
    indices = []
    for sid in ids:
        hits = aliases.get(sid, [])
        assert len(hits) == 1, (sid, hits)
        indices.append(hits[0])
    assert len(set(indices)) == 424
    return ids, fam, np.array(indices)

def snp_indices(names):
    wanted = set(names)
    assert len(wanted) == len(names)
    found = []
    ordered = []
    with Path(str(PREFIX)+'.bim').open() as f:
        for i, line in enumerate(f):
            name = line.split()[1]
            if name in wanted:
                found.append(i)
                ordered.append(name)
    assert len(found) == len(wanted), 'Missing or non-unique pruned SNP'
    return np.array(found), ordered

def read_bed(marker_indices, sample_indices, n_fam):
    bed = Path(str(PREFIX)+'.bed')
    with bed.open('rb') as f:
        assert f.read(3) == b'\x6c\x1b\x01', 'BED must be SNP-major'
    stride = (n_fam+3)//4
    assert (bed.stat().st_size-3) % stride == 0
    data = np.memmap(bed, mode='r', dtype=np.uint8, offset=3, shape=((bed.stat().st_size-3)//stride, stride))
    # PLINK1 --recode A counts BIM allele 1. Verify against PLINK below.
    lookup = np.array([2., np.nan, 1., 0.], dtype=np.float32)
    result = np.empty((len(sample_indices), len(marker_indices)), dtype=np.float32)
    for begin in range(0, len(marker_indices), 8192):
        rows = data[marker_indices[begin:begin+8192]]
        codes = (rows[:, sample_indices//4] >> (2*(sample_indices%4))) & 3
        result[:, begin:begin+len(rows)] = lookup[codes].T
    del data
    assert np.isin(result[np.isfinite(result)], [0.,1.,2.]).all()
    return result

def verify_bed(run):
    ids, fam, indices = sample_mapping()
    audit = run/'audit'/'bed_decode_verification.json'
    if audit.exists():
        result = json.loads(audit.read_text())
        assert result['status'] == 'success'
        return
    tmp = run/'tmp'/'bed_verification'
    tmp.mkdir()
    with Path(str(PREFIX)+'.bim').open() as f:
        names = [next(f).split()[1] for _ in range(16)]
    (tmp/'snps.txt').write_text('\n'.join(names)+'\n')
    (tmp/'samples.keep').write_text(''.join(' '.join(fam[i])+'\n' for i in indices[:24]))
    cmd = [PLINK,'--chr-set','24','no-x','no-y','no-xy','no-mt','--bfile',str(PREFIX),
           '--extract',str(tmp/'snps.txt'),'--keep',str(tmp/'samples.keep'),'--recode','A',
           '--threads','1','--out',str(tmp/'check')]
    result = subprocess.run(cmd, capture_output=True, text=True)
    (run/'audit'/'bed_verification_plink.log').write_text(result.stdout+'\n'+result.stderr)
    assert result.returncode == 0, 'Small PLINK compatibility check failed'
    raw = pd.read_csv(tmp/'check.raw',sep=r'\s+',dtype={'FID':str,'IID':str})
    raw_rows = {(str(row.FID),str(row.IID)):i for i,row in raw.iterrows()}
    observed = raw.iloc[[raw_rows[tuple(fam[i])] for i in indices[:24]],6:].to_numpy(dtype=np.float32)
    decoded = read_bed(np.arange(16),indices[:24],len(fam))
    assert np.array_equal(observed,decoded,equal_nan=True), 'BED dosage differs from original PLINK1 export'
    assert len(raw)==24 and observed.shape==(24,16)
    files = [tmp/name for name in ['snps.txt','samples.keep','check.raw','check.log','check.nosex'] if (tmp/name).exists()]
    save_json(audit,dict(status='success',trainval_only=True,samples=24,SNPs=16,dosage='BIM allele1: BED 00=2,01=NA,10=1,11=0',
                         command=cmd,temporary_created=[str(x) for x in files]))
    # Exact inventory, no glob deletion. Verify ownership/boundary and reject symlinks.
    cleaned=[]
    for path in files:
        assert not path.is_symlink() and path.resolve().parent == tmp.resolve()
        assert tmp.resolve().is_relative_to((run/'tmp').resolve())
        path.unlink()
        cleaned.append(str(path))
    tmp.rmdir()
    save_json(run/'audit'/'temporary_cleanup.json',dict(cleaned=cleaned,other_files_deleted=False))
    print('BED_DECODE_VERIFIED',flush=True)

def prepare(run, dimension, seed, fold, attempt_index=1):
    verify_bed(run)
    output = run/'pca'/f'PCA{dimension}'/f'attempt_{attempt_index}'/f'seed_{seed}'/f'fold_{fold}'
    complete = output/'COMPLETE.json'
    if complete.exists():
        info=json.loads(complete.read_text())
        for name, digest in info['files_sha256'].items():
            assert sha(output/name)==digest
        return output
    output.mkdir(parents=True,exist_ok=False)
    ids, fam, indices = sample_mapping()
    original = ROOT/'data/blackcarp499/pca_covariates'/f'seed_{seed}'/f'fold_{fold}'
    names = original.joinpath('pruned_snps.txt').read_text().splitlines()
    marker_indices, ordered_names = snp_indices(names)
    cvfile=ROOT/f'data/blackcarp499/cv_splits_{seed}.csv'
    cv = pd.read_csv(cvfile,dtype={'sample_id':str})
    current=cv[cv.fold==fold]
    train_ids=current.loc[current.split=='train','sample_id'].tolist()
    val_ids=current.loc[current.split=='val','sample_id'].tolist()
    assert not set(train_ids)&set(val_ids) and set(train_ids+val_ids)==set(ids)
    train_idx=np.array([ids.index(s) for s in train_ids])
    start=time.monotonic()
    X=read_bed(marker_indices,indices,len(fam))
    print('PCA_INPUT',dimension,seed,fold,X.shape,flush=True)
    imputer=SimpleImputer(strategy='mean')
    X_train_i=imputer.fit_transform(X[train_idx])
    X_all_i=imputer.transform(X)
    del X
    scaler=StandardScaler()
    X_train_s=scaler.fit_transform(X_train_i)
    X_all_s=scaler.transform(X_all_i)
    del X_train_i, X_all_i
    old=pd.read_csv(original/'pca_scores.csv',dtype={'sample_id':str}).set_index('sample_id').loc[ids]
    old5=old[[f'PC{i}' for i in range(1,6)]].to_numpy()
    # Reproduce original PCA5 once per Seed/Fold to verify BED direction, SNP order and preprocessing.
    reference=run/'audit'/f'pca5_reproduction_s{seed}_f{fold}.json'
    if not reference.exists():
        pca5=PCA(n_components=5,svd_solver='randomized',random_state=seed)
        pca5.fit(X_train_s)
        reproduced=pca5.transform(X_all_s)
        corr=[float(np.corrcoef(old5[:,i],reproduced[:,i])[0,1]) for i in range(5)]
        relative_rmse=(np.sqrt(np.mean((old5-reproduced)**2,axis=0))/np.std(old5,axis=0)).tolist()
        save_json(reference,dict(seed=seed,fold=fold,signed_correlations=corr,relative_RMSE=relative_rmse,
                                tolerance=dict(min_signed_correlation=.99999,max_relative_RMSE=.002),
                                original_scores_sha256=sha(original/'pca_scores.csv'),
                                passed=bool(min(corr)>.99999 and max(relative_rmse)<.002),
                                test_transformed=False))
        assert min(corr)>.99999 and max(relative_rmse)<.002, 'Original PCA5 reproduction mismatch; no training allowed'
        del pca5,reproduced
    else:
        assert json.loads(reference.read_text())['passed']
    pca=PCA(n_components=dimension,svd_solver='randomized',random_state=seed)
    pca.fit(X_train_s)
    scores=pca.transform(X_all_s)
    assert scores.shape==(424,dimension) and np.isfinite(scores).all()
    splits={sid:'train' for sid in train_ids};splits.update({sid:'val' for sid in val_ids})
    result=pd.DataFrame(dict(sample_id=ids,split=[splits[sid] for sid in ids]))
    for i in range(dimension):result[f'PC{i+1}']=scores[:,i]
    result.to_csv(output/'pca_scores.csv',index=False)
    joblib.dump(dict(imputer=imputer,scaler=scaler,pca=pca,training_sample_ids=train_ids,
                     SNP_list_path=str(original/'pruned_snps.txt'),SNP_order='BIM order',sample_order=ids),
                output/'fitted_preprocessing.joblib',compress=3)
    comparisons=[]
    for i in range(5):
        comparisons.append(dict(component=i+1,signed_correlation_to_original_PCA5=float(np.corrcoef(old5[:,i],scores[:,i])[0,1]),
                                absolute_correlation_to_original_PCA5=float(abs(np.corrcoef(old5[:,i],scores[:,i])[0,1]))))
    save_json(output/'metadata.json',dict(seed=seed,fold=fold,n_components=dimension,train_n=len(train_ids),val_n=len(val_ids),
              test_n=0,pca_fit_samples='fold_train_only',imputer_fit_samples='fold_train_only',scaler_fit_samples='fold_train_only',
              original_pruned_SNPs=len(names),retained_SNPs=int(X_train_s.shape[1]),
              original_pruned_SNP_list=str(original/'pruned_snps.txt'),pruned_SNP_sha256=sha(original/'pruned_snps.txt'),
              BIM_order_sha256=hashlib.sha256(('\n'.join(ordered_names)+'\n').encode()).hexdigest(),
              cv_sha256=sha(cvfile),algorithm=pca.get_params(),explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
              comparisons_to_PCA5=comparisons,fit_seconds=time.monotonic()-start,
              independent_fit_per_dimension=True,Test_used_or_transformed=False,
              note='Signed/numerical differences from independently randomized PCA5 are recorded; no sign alignment or shared PCA20 truncation.'))
    save_json(complete,dict(status='success',files_sha256={name:sha(output/name) for name in ['pca_scores.csv','metadata.json','fitted_preprocessing.joblib']}))
    print('PCA_COMPLETE',dimension,seed,fold,flush=True)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True);p.add_argument('--verify-only',action='store_true')
    p.add_argument('--dimension',type=int);p.add_argument('--seed',type=int);p.add_argument('--fold',type=int)
    p.add_argument('--attempt-index',type=int,default=1)
    a=p.parse_args()
    if a.verify_only:verify_bed(Path(a.run))
    else:
        assert a.dimension in [8,10,15,20]
        prepare(Path(a.run),a.dimension,a.seed,a.fold,a.attempt_index)
