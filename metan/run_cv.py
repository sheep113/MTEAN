"""Run the configured folds sequentially and retain explicit task/exit status."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from metan.config import load_config

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--pca-root',type=Path)
    parser.add_argument('--describe',action='store_true')
    args=parser.parse_args()
    spec=load_config(args.config)
    jobs=[(seed,fold) for seed in spec['seeds'] for fold in spec['folds']]
    if args.describe:
        print(json.dumps({'experiment':spec['name'],'jobs':jobs,'training_started':False}))
        return
    args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    state={'status':'RUNNING','PID':os.getpid(),'completed':0,'expected':len(jobs),'tasks':[],'started':time.time()}
    def save():
        temp=args.out/'status.json.tmp'
        temp.write_text(json.dumps(state,indent=2),encoding='utf8')
        temp.replace(args.out/'status.json')
    save()
    try:
        for seed,fold in jobs:
            task=args.out/f'seed{seed}_fold{fold}'
            state['current']={'seed':seed,'fold':fold,'directory':str(task)}
            save()
            command=[sys.executable,'-B','-m','metan','train','--config',str(args.config.resolve()),'--seed',str(seed),'--fold',str(fold),'--out',str(task)]
            for key in ['data','pca_root']:
                value=getattr(args,key)
                if value:command+=['--'+key.replace('_','-'),str(value.resolve())]
            with (args.out/f'seed{seed}_fold{fold}.log').open('x',encoding='utf8') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            record={'seed':seed,'fold':fold,'exit_code':result.returncode}
            state['tasks'].append(record)
            if result.returncode:
                raise RuntimeError(f'Fold exited with code {result.returncode}')
            child=json.loads((task/'status.json').read_text())
            if child['status']!='TRAINING_COMPLETED_PENDING_INDEPENDENT_VERIFICATION':
                raise RuntimeError('Fold did not finish strict checkpoint verification')
            state['completed']+=1
            save()
        state.update(status='COMPLETED',current=None,finished=time.time())
        save()
    except BaseException as error:
        state.update(status='FAILED',error=repr(error),unfinished=len(jobs)-state['completed'],finished=time.time())
        save()
        raise
