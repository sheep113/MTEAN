"""Export model-aligned dosage and phenotypes for baseline reproduction."""
import argparse
from pathlib import Path
import json

from metan.config import load_config

CHANNELS=['AA','AT','AC','AG','TT','TC','TG','CC','CG','GG']
PAIR_CHANNEL={tuple(sorted(pair)):i for i,pair in enumerate(CHANNELS)}

def decode(values):
    return [v.decode() if isinstance(v,bytes) else str(v) for v in values]

def export_dosage(source,destination):
    import h5py
    import numpy as np
    import pandas as pd
    with h5py.File(source,'r') as h:
        ids=decode(h['sample_ids'][:]);snps=decode(h['snp_ids'][:])
        traits=decode(h['phenotype_names'][:])
        phenotype=h['phenotypes'][:]
        if 'allele1' not in h or 'allele2' not in h:
            raise ValueError('Original marker-specific allele1/allele2 annotations are required')
        a1=decode(h['allele1'][:]);a2=decode(h['allele2'][:])
        x=h['features/genotype_features'][:]
    if len(set(ids))!=len(ids) or len(set(snps))!=len(snps):raise ValueError('Duplicate sample or SNP ID')
    if x.shape!=(len(snps),len(ids),10):raise ValueError('Unexpected SNP array axes')
    if not np.isin(x,[0,1]).all() or not np.isin(x.sum(2),[0,1]).all():raise ValueError('Expected one-hot or all-zero missing SNPs')
    dosage=np.full((len(ids),len(snps)),np.nan)
    for j,(left,right) in enumerate(zip(a1,a2)):
        if left==right:raise ValueError('Non-biallelic marker')
        for value,pair in enumerate([left+left,left+right,right+right]):
            channel=PAIR_CHANNEL[tuple(sorted(pair))]
            dosage[x[j,:,channel]==1,j]=value
    pd.DataFrame(dosage,index=ids,columns=snps).rename_axis('sample_id').to_csv(destination/'dosage.csv',float_format='%.17g')
    frame=pd.DataFrame(phenotype,columns=traits);frame.insert(0,'sample_id',ids)
    frame.to_csv(destination/'phenotypes.csv',index=False,float_format='%.17g')
    (destination/'sample_ids.txt').write_text('\n'.join(ids)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    spec=load_config(args.config)
    args.out.mkdir(parents=True,exist_ok=False)
    export_dosage(args.data or Path(spec['data']),args.out)
    (args.out/'preparation.json').write_text(json.dumps({'new_sampling':False,'PCA_refitted':False,'channel_order':CHANNELS,'dosage_orientation':'0=allele1 homozygote,1=heterozygote,2=allele2 homozygote'},indent=2))
