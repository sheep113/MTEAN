"""Fit the three baseline models using configured parameters and the fixed CV IDs."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from metan.config import ROOT,load_config
from metan.metrics import regression_metrics

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--genotype',type=Path,required=True,help='Original 0/1/2 dosage CSV, sample_id followed by SNPs')
    parser.add_argument('--phenotype',type=Path,required=True,help='Original-unit CSV, sample_id followed by configured traits')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--models',nargs='+',choices=['SVR','LightGBM','GBLUP'],default=['SVR','LightGBM','GBLUP'])
    parser.add_argument('--holdout-ids',type=Path,help='Required for configured holdout evaluation')
    parser.add_argument('--rscript',default='Rscript')
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    spec=load_config(args.config)
    params=json.loads(Path(spec['baseline_config_path']).read_text())
    if args.describe:
        print(json.dumps({'experiment':spec['name'],'models':args.models,'fits_started':False,'settings':params}))
        return
    import numpy as np
    import pandas as pd
    import joblib
    dosage=pd.read_csv(args.genotype,dtype={'sample_id':str}).set_index('sample_id')
    phenotype=pd.read_csv(args.phenotype,dtype={'sample_id':str}).set_index('sample_id')
    if not dosage.index.is_unique or not phenotype.index.is_unique or not dosage.columns.is_unique:
        raise ValueError('Duplicate sample or SNP IDs')
    if dosage.shape[1]!=spec['SNPs']:
        raise ValueError('SNP count differs from configured baseline input')
    if not np.isin(dosage.to_numpy()[np.isfinite(dosage.to_numpy())],[0,1,2]).all():
        raise ValueError('Dosage must contain only 0/1/2 or missing values')
    holdout=None
    if spec['evaluation']['mode']=='holdout':
        if args.holdout_ids is None:raise ValueError('The original holdout sample IDs are required')
        holdout=[line.strip() for line in args.holdout_ids.read_text().splitlines() if line.strip()]
        if len(holdout)!=len(set(holdout)) or len(holdout)!=spec['test_n']:
            raise ValueError('Holdout sample count or uniqueness mismatch')
    args.out.mkdir(parents=True,exist_ok=False)
    state={'status':'RUNNING','completed':0,'expected':len(spec['seeds'])*len(spec['folds'])*len(args.models)*len(spec['traits']),'started':time.time()}
    def save():
        temp=args.out/'status.json.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(args.out/'status.json')
    rows=[]
    save()
    try:
        for seed in spec['seeds']:
            cv=pd.read_csv(Path(spec['cv_root'])/f'cv_splits_{seed}.csv',dtype={'sample_id':str})
            for fold in spec['folds']:
                block=cv[cv.fold==fold]
                training=block.loc[block.split=='train','sample_id'].tolist()
                validation=block.loc[block.split=='val','sample_id'].tolist()
                evaluation=holdout if holdout is not None else validation
                if set(training)&set(evaluation) or len(training)!=len(set(training)):
                    raise ValueError('Overlapping or duplicate training/evaluation IDs')
                x=dosage.loc[training].to_numpy(float)
                v=dosage.loc[evaluation].to_numpy(float)
                median=np.nanmedian(x,axis=0);median=np.where(np.isnan(median),0.,median)
                xt=np.where(np.isnan(x),median,x);xv=np.where(np.isnan(v),median,v)
                for trait in spec['traits']:
                    y=phenotype.loc[training,trait].to_numpy(float)
                    truth=phenotype.loc[evaluation,trait].to_numpy(float)
                    if not np.isfinite(y).all() or not np.isfinite(truth).all():raise ValueError('Nonfinite phenotype')
                    for name in args.models:
                        target=args.out/f'{name}_{trait}_seed{seed}_fold{fold}'
                        target.mkdir()
                        state['current']={'model':name,'trait':trait,'seed':seed,'fold':fold};save()
                        if name=='GBLUP':
                            pd.DataFrame(x,index=training,columns=dosage.columns).rename_axis('sample_id').to_csv(target/'train_dosage.csv')
                            pd.DataFrame(v,index=evaluation,columns=dosage.columns).rename_axis('sample_id').to_csv(target/'eval_dosage.csv')
                            pd.DataFrame({'sample_id':training,'phenotype':y}).to_csv(target/'train_phenotype.csv',index=False)
                            request={'directory':str(target.resolve()),'settings':params['GBLUP']}
                            (target/'request.json').write_text(json.dumps(request))
                            with (target/'fit.log').open('w') as log:
                                subprocess.run([args.rscript,str(ROOT/'baselines/gblup.R'),str((target/'request.json').resolve())],check=True,stdout=log,stderr=subprocess.STDOUT)
                            result=pd.read_csv(target/'fitted_predictions.csv',dtype={'sample_id':str}).set_index('sample_id')
                            if result.index.tolist()!=evaluation:raise ValueError('GBLUP sample order mismatch')
                            prediction=result.prediction.to_numpy(float)
                        else:
                            effective=params['effective_parameters'][str(seed)][name]
                            if name=='SVR':
                                from sklearn.svm import SVR
                                model=SVR(**effective)
                            else:
                                from lightgbm import LGBMRegressor
                                model=LGBMRegressor(**effective)
                            np.random.seed(seed)
                            model.fit(xt,y);prediction=model.predict(xv)
                            joblib.dump({'model':model,'training_median':median,'SNP_order':list(dosage.columns)},target/'model.joblib')
                        pd.DataFrame({'sample_id':evaluation,'y_true':truth,'y_pred':prediction}).to_csv(target/'predictions.csv',index=False,float_format='%.17g')
                        row={'model':name,'trait':trait,'seed':seed,'fold':fold,**regression_metrics(truth,prediction)}
                        rows.append(row);pd.DataFrame(rows).to_csv(args.out/'fold_metrics.csv',index=False)
                        state['completed']+=1;save()
        from metan.summarize import summarize
        summarize(args.out/'fold_metrics.csv',args.out/'summary.csv')
        state.update(status='COMPLETED',current=None,finished=time.time());save()
    except BaseException as error:
        state.update(status='FAILED',error=repr(error),finished=time.time());save();raise
