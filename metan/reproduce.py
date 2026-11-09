"""Run the complete configured training, evaluation and baseline workflow."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from metan.config import ROOT,load_config

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rscript',default='Rscript')
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    spec=load_config(args.config)
    if args.describe:
        print(json.dumps({'experiment':spec['name'],'training_fits':len(spec['seeds'])*len(spec['folds']),'checkpoint_traits':spec['traits'],'evaluation':spec['evaluation'],'baseline_models':['SVR','LightGBM','GBLUP'],'started':False}))
        return
    command=[sys.executable,'-B','-m','metan']
    config=str(args.config.resolve())
    # Validate required data and PCA files before reserving the new run directory.
    subprocess.run(command+['check','--config',config],check=True)
    subprocess.run([args.rscript,'-e',"stopifnot(requireNamespace('asreml',quietly=TRUE),requireNamespace('ASRgenomics',quietly=TRUE),requireNamespace('jsonlite',quietly=TRUE),requireNamespace('data.table',quietly=TRUE))"],check=True)
    if spec['evaluation']['mode']=='holdout':
        required=[Path(spec['evaluation']['data'])]
        required.extend(ROOT/value for value in spec['baseline_inputs'].values())
        required.extend(ROOT/spec['evaluation']['pca_root']/f'seed_{seed}'/f'fold_{fold}'/'pca_scores.csv' for seed in spec['seeds'] for fold in spec['folds'])
        for path in required:
            if not path.is_file():raise FileNotFoundError('Private evaluation input required: '+str(path))
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    state={'status':'RUNNING','stage':'training','started':time.time()}
    def save():
        temporary=args.out/'status.json.tmp';temporary.write_text(json.dumps(state,indent=2));temporary.replace(args.out/'status.json')
    def run(stage,arguments):
        state['stage']=stage;save()
        with (args.out/(stage+'.log')).open('x') as log:
            subprocess.run(command+arguments,check=True,stdout=log,stderr=subprocess.STDOUT)
    save()
    try:
        run('METAN_training',['cv','--config',config,'--out',str(args.out/'METAN')])
        import pandas as pd
        metrics=[]
        if spec['evaluation']['mode']=='validation':
            for seed in spec['seeds']:
                for fold in spec['folds']:
                    metrics.append(pd.read_csv(args.out/'METAN'/f'seed{seed}_fold{fold}'/'fold_metrics.csv'))
        else:
            for seed in spec['seeds']:
                for fold in spec['folds']:
                    task=args.out/'METAN'/f'seed{seed}_fold{fold}'
                    pca=ROOT/spec['evaluation']['pca_root']/f'seed_{seed}'/f'fold_{fold}'/'pca_scores.csv'
                    for trait in spec['traits']:
                        output=args.out/f'holdout_{trait}_seed{seed}_fold{fold}'
                        run(output.name,['predict','--config',config,'--checkpoint',str(task/'checkpoints'/f'{trait}_best.ckpt'),'--data',spec['evaluation']['data'],'--pca-csv',str(pca),'--out',str(output)])
                        frame=pd.read_csv(output/'metrics.csv')
                        frame=frame[frame.trait==trait].copy();frame['seed']=seed;frame['fold']=fold
                        metrics.append(frame)
        metan_metrics=pd.concat(metrics,ignore_index=True);metan_metrics['model']='METAN'
        metan_metrics.to_csv(args.out/'METAN_fold_metrics.csv',index=False)
        if spec['evaluation']['mode']=='validation':
            run('baseline_inputs',['prepare','--config',config,'--out',str(args.out/'baseline_inputs')])
            genotype=args.out/'baseline_inputs/dosage.csv';phenotype=args.out/'baseline_inputs/phenotypes.csv'
            extra=[]
        else:
            genotype=ROOT/spec['baseline_inputs']['genotype'];phenotype=ROOT/spec['baseline_inputs']['phenotype']
            extra=['--holdout-ids',str(ROOT/spec['baseline_inputs']['holdout_ids'])]
        run('baseline_fits',['baselines','--config',config,'--genotype',str(genotype),'--phenotype',str(phenotype),'--out',str(args.out/'baselines'),'--rscript',args.rscript]+extra)
        baseline_metrics=pd.read_csv(args.out/'baselines/fold_metrics.csv')
        combined=pd.concat([metan_metrics,baseline_metrics],ignore_index=True)
        combined.to_csv(args.out/'all_model_fold_metrics.csv',index=False)
        from metan.summarize import summarize
        summarize(args.out/'all_model_fold_metrics.csv',args.out/'all_model_summary.csv')
        state.update(status='COMPLETED',stage='finished',finished=time.time());save()
    except BaseException as error:
        state.update(status='FAILED',error=repr(error),finished=time.time());save();raise
