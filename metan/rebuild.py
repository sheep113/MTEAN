"""Rebuild a public 500-individual input from its source data and fixed IDs."""
import argparse
import json
from pathlib import Path
import shutil

from metan.config import ROOT,load_config
from metan.prepare import PAIR_CHANNEL,decode

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    spec=load_config(args.config)
    cfg=spec.get('raw_input')
    if cfg is None:raise ValueError('The configured raw input is private and supplied by its custodian')
    import h5py
    import numpy as np
    import pandas as pd
    source=ROOT/cfg['root']
    ids=[line.strip() for line in (ROOT/cfg['selected_ids']).read_text().splitlines() if line.strip()]
    if len(ids)!=len(set(ids)) or len(ids)!=spec['development_n']:raise ValueError('Selected sample IDs mismatch')
    meta=pd.read_csv(ROOT/cfg['metadata'],sep='\t',dtype=str).set_index(cfg['marker_column'])
    if not meta.index.is_unique:raise ValueError('Duplicate marker IDs')
    if cfg['genotype_format']=='tabular_dosage':
        frame=pd.read_csv(source/cfg['genotype_file'],sep=r'\s+',dtype={cfg['sample_id_column']:str}).set_index(cfg['sample_id_column'])
        markers=frame.columns.astype(str).tolist()
        if not frame.index.is_unique:raise ValueError('Duplicate raw genotype sample IDs')
        dosage=frame.loc[ids].to_numpy(float)
        dosage=np.where(dosage==cfg['missing_code'],np.nan,dosage)
    elif cfg['genotype_format']=='blup_compact':
        markers=meta.index.astype(str).tolist()
        wanted=set(ids);records={}
        with (source/cfg['genotype_file']).open() as stream:
            for line in stream:
                fields=line.split()
                if not fields:continue
                sid=fields[0]
                if sid in wanted:
                    if sid in records:raise ValueError('Duplicate raw genotype sample ID')
                    text=''.join(fields[1:])
                    if len(text)!=len(markers):raise ValueError('Raw genotype sequence length mismatch')
                    records[sid]=np.frombuffer(text.encode('ascii'),dtype=np.uint8).astype(np.int16)-ord('0')
        if set(records)!=wanted:raise ValueError('Selected sample missing from raw genotypes')
        dosage=np.stack([records[sid] for sid in ids]).astype(float)
        dosage=np.where(dosage==cfg['missing_code'],np.nan,dosage)
    else:raise ValueError('Unknown raw genotype format')
    if set(markers)!=set(meta.index) or len(markers)!=spec['SNPs']:raise ValueError('Raw genotype and marker annotation mismatch')
    if not np.isin(dosage[np.isfinite(dosage)],[0,1,2]).all():raise ValueError('Unexpected genotype code')
    meta=meta.loc[markers]
    a1=meta[cfg['allele1_column']].str.upper().tolist()
    a2=meta[cfg['allele2_column']].str.upper().tolist()
    wanted_columns=[cfg['sample_id_column']]+spec['traits']
    pheno=pd.read_csv(source/cfg['phenotype_file'],sep=cfg.get('phenotype_separator',r'\s+'),usecols=wanted_columns,index_col=False,dtype={cfg['sample_id_column']:str}).set_index(cfg['sample_id_column'])
    if not pheno.index.is_unique:raise ValueError('Duplicate phenotype sample IDs')
    y=pheno.loc[ids,spec['traits']].to_numpy(np.float32)
    if not np.isfinite(y).all():raise ValueError('Missing selected target phenotype')
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'processed').mkdir()
    with h5py.File(args.out/'processed/all_500_allele10.h5','x') as out:
        text=h5py.string_dtype('utf8')
        for name,values in [('sample_ids',ids),('snp_ids',markers),('phenotype_names',spec['traits']),('allele1',a1),('allele2',a2)]:out.create_dataset(name,data=np.asarray(values,dtype=text))
        out.create_dataset('phenotypes',data=y)
        out.create_dataset('phenotypes_na_mask',data=np.zeros(y.shape,dtype=bool))
        genotype=out.create_dataset('features/genotype_features',shape=(len(markers),len(ids),10),dtype=np.float32,compression='gzip')
        for start in range(0,len(markers),512):
            stop=min(start+512,len(markers));block=np.zeros((stop-start,len(ids),10),dtype=np.float32)
            for j in range(start,stop):
                if a1[j]==a2[j]:raise ValueError('Marker allele annotations are not biallelic')
                for value,pair in enumerate([a1[j]*2,a1[j]+a2[j],a2[j]*2]):
                    block[j-start,dosage[:,j]==value,PAIR_CHANNEL[tuple(sorted(pair))]]=1
            genotype[start:stop]=block
    for seed in spec['seeds']:shutil.copyfile(Path(spec['cv_root'])/f'cv_splits_{seed}.csv',args.out/f'cv_splits_{seed}.csv')
    shutil.copytree(spec['pca_root'],args.out/'pca')
    (args.out/'rebuild.json').write_text(json.dumps({'selected_sample_count':len(ids),'SNP_count':len(markers),'traits':spec['traits'],'new_sampling':False,'PCA_refitted':False,'input_layout':'SNP x individual x channel'},indent=2))
