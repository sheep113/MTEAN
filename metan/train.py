"""Portable single-fold entry. Mathematical models and final optimization settings are frozen.

This entry replaces server paths/process orchestration, not model computations.
Historical training managers are archived under training/<species>/historical_protocol.
Never run on an original experiment output directory.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data', type=Path, help='Processed training/development HDF5, not the independent test file')
    p.add_argument('--pca-root', type=Path, help='Existing fold-fitted PCA scores and binding metadata')
    p.add_argument('--seed', type=int, choices=[1,2,3], required=True)
    p.add_argument('--fold', type=int, choices=range(5), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--describe', action='store_true')
    a = p.parse_args()
    from metan.config import load_config
    spec = load_config(a.config)
    a.data = a.data or Path(spec['data'])
    a.pca_root = a.pca_root or Path(spec['pca_root'])
    if a.describe:
        print(json.dumps({'species':spec['name'], 'settings':spec, 'training_started':False}))
        return
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, EarlyStopping
    from pytorch_lightning.loggers.logger import Logger
    from metan.runtime import activate, build, sha256
    from metan.metrics import regression_metrics
    activate(spec)
    from training.data.datamodule_onlySNP import METANDataModule_onlySNP
    import training.data.datamodule_onlySNP as dm_module

    a.data = a.data.resolve();a.pca_root = a.pca_root.resolve();a.out = a.out.resolve()
    if a.out.exists():
        raise FileExistsError('A fresh output directory is required')
    reference_cv = Path(spec['cv_root'])/f'cv_splits_{a.seed}.csv'
    cv_path = a.data.parent.parent/f'cv_splits_{a.seed}.csv'
    if not cv_path.is_file():
        raise FileNotFoundError('Exact fixed CV file required; no random fallback is allowed')
    if sha256(cv_path)!=sha256(reference_cv):
        raise ValueError('Relocated CV file differs from the configured fixed split')
    cv = pd.read_csv(cv_path, dtype={'sample_id':str})
    block = cv[cv['fold'] == a.fold]
    train_ids = block.loc[block['split']=='train','sample_id'].tolist()
    val_ids = block.loc[block['split']=='val','sample_id'].tolist()
    assert train_ids and val_ids and len(set(train_ids+val_ids)) == len(train_ids)+len(val_ids)
    mc = json.loads(Path(spec['model_path']).read_text())
    tc = yaml.safe_load(Path(spec['training_path']).read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    status_path = a.out/'status.json'
    def dump(path, value):
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value, indent=2, default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)), encoding='utf-8')
        temporary.replace(path)
    status = {'status':'RUNNING','species':spec['name'],'seed':a.seed,'fold':a.fold,'PID':os.getpid(),'started':time.time(),'release_entry':'0.1.0-rc1'}
    dump(status_path,status)
    try:
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.backends.cudnn.enabled = spec['cudnn']
        pl.seed_everything(a.seed, workers=True)
        mc['pca_covariates']['scores_root'] = str(a.pca_root)
        if spec['checkpoint_format'] != 'plain':
            mc['random_seed'] = a.seed
        tc['training']['random_seed'] = a.seed
        tc['training']['cv_fold_idx'] = a.fold
        tc['data']['data_path'] = str(a.data)
        tc['checkpoint']['dirpath'] = str(a.out/'checkpoints')
        tc['logging']['save_dir'] = str(a.out/'logs')
        tc['logging']['experiment_name'] = f"{spec['name']}_s{a.seed}_f{a.fold}"
        mc['gradient_checkpointing'] = copy.deepcopy(tc['gradient_checkpointing'])
        mc.setdefault('logging',{}).update(tc['logging'])
        covdir = a.pca_root/f'seed_{a.seed}'/f'fold_{a.fold}'
        if spec['checkpoint_format'] != 'plain':
            asset = json.loads((covdir/'asset_manifest.json').read_text())
            assert asset['fit_IDs'] == train_ids and asset['validation_IDs'] == val_ids
            assert asset['CV_SHA256'] == sha256(cv_path) and asset['n_components'] == spec['PCA']
            for name,h in asset['files_SHA256'].items():
                assert sha256(covdir/name) == h
            mc['pca_asset_binding'] = {'run_id':a.out.name,'seed':a.seed,'fold':a.fold,'n_components':spec['PCA'],
                'CV_SHA256':sha256(cv_path),'fit_IDs_SHA256':hashlib.sha256(('\n'.join(train_ids)+'\n').encode()).hexdigest(),
                'validation_IDs_SHA256':hashlib.sha256(('\n'.join(val_ids)+'\n').encode()).hexdigest(),
                'asset_manifest_SHA256':sha256(covdir/'asset_manifest.json')}
        else:
            asset = json.loads((covdir/'metadata.json').read_text())
            assert asset['seed'] == a.seed and asset['fold'] == a.fold
            assert asset['cv_sha256'] == sha256(cv_path) and not asset['Test_used_or_transformed']
            fitted = __import__('joblib').load(covdir/'fitted_preprocessing.joblib')
            assert fitted['training_sample_ids'] == train_ids
        def refuse(*args, **kwargs):
            raise RuntimeError('Random KFold fallback is disabled')
        dm_module.KFold = refuse
        dm = METANDataModule_onlySNP(a.data,tc,model_config=mc,phenotype_names=spec['traits'],seed=a.seed)
        dm.setup('fit')
        ids = [v.decode() if isinstance(v,bytes) else str(v) for v in dm.dataset.sample_ids]
        assert [ids[i] for i in dm.train_indices] == train_ids
        assert [ids[i] for i in dm.val_indices] == val_ids
        assert len(dm.test_indices) == 0 and len(ids) == spec['development_n']
        assert dm.dataset.genotype_data.shape == (spec['development_n'],spec['SNPs'],10)
        raw = dm.dataset.phenotypes
        mean = raw[dm.train_indices].mean(0);std = raw[dm.train_indices].std(0)
        assert np.isfinite(std).all() and (std>0).all()
        np.testing.assert_array_equal(dm.dataset._phenotype_stats['mean'],mean)
        np.testing.assert_array_equal(dm.dataset._phenotype_stats['std'],std)
        model = build(spec,mc,tc['optimizer'],tc['scheduler'])
        assert model.primary_loss_type == 'mse'
        dump(a.out/'model_config.json',mc);dump(a.out/'training_config.json',tc)
        norm = {'mean':mean.tolist(),'std':std.tolist(),'traits':spec['traits']}
        dump(a.out/'normalization.json',norm)
        binding = {'CV_SHA256':sha256(cv_path),'data_SHA256':sha256(a.data),'PCA_metadata_SHA256':sha256(covdir/('metadata.json' if spec['checkpoint_format']=='plain' else 'asset_manifest.json'))}
        class QuietLogger(Logger):
            @property
            def name(self):return 'epoch_only'
            @property
            def version(self):return 'release_v1'
            def log_hyperparams(self,*args,**kwargs):pass
            def log_metrics(self,*args,**kwargs):pass
            def save(self):pass
            def finalize(self,*args,**kwargs):pass
        best = {t:None for t in spec['traits']};history=[]
        class Audit(Callback):
            def on_train_epoch_start(self,tr,m):self.total=0.;self.n=0
            def on_train_batch_end(self,tr,m,out,batch,index):
                loss = out['loss'] if isinstance(out,dict) else out
                assert torch.isfinite(loss).all(), 'Nonfinite loss'
                n = len(batch['phenotype']);self.total+=float(loss.detach().cpu())*n;self.n+=n
            def on_after_backward(self,tr,m):
                grads = [v.grad for v in m.parameters() if v.grad is not None]
                assert grads and all(torch.isfinite(v).all() for v in grads), 'Nonfinite gradient'
            def on_validation_epoch_start(self,tr,m):self.outputs=[]
            def on_validation_batch_end(self,tr,m,out,batch,index,dataloader_idx=0):
                if not tr.sanity_checking:self.outputs.append(out['preds'].detach().float().cpu())
            def on_validation_end(self,tr,m):
                if tr.sanity_checking:return
                ep=int(tr.current_epoch);prediction=torch.cat(self.outputs).numpy()
                assert prediction.shape==(len(val_ids),len(spec['traits'])) and np.isfinite(prediction).all()
                row={'epoch':ep,'train_loss':self.total/self.n,'learning_rate':tr.optimizers[0].param_groups[0]['lr'],'optimizer_steps':int(tr.global_step)}
                for j,t in enumerate(spec['traits']):
                    score=float(tr.callback_metrics['val_pearson_'+t].detach().cpu())
                    assert np.isfinite(score)
                    row[t+'_validation_Pearson']=score
                    if best[t] is None or score>best[t]['score']:
                        best[t]={'epoch':ep,'score':score}
                        checkpoint=a.out/'checkpoints'/f'{t}_best.ckpt';checkpoint.parent.mkdir(exist_ok=True)
                        if spec['checkpoint_format']=='plain':
                            payload={'model_state_dict':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},
                                'model_config':copy.deepcopy(mc),'training_config':copy.deepcopy(tc),'normalization':norm,'selection':{'trait':t,**best[t]},'binding':binding}
                        else:
                            payload={'state_dict':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},'hyper_parameters':copy.deepcopy(dict(m.hparams)),
                                'pytorch-lightning_version':pl.__version__,'epoch':ep,'global_step':tr.global_step,'normalization':norm}
                            m.on_save_checkpoint(payload)
                        payload['release_provenance']={'species':spec['name'],'seed':a.seed,'fold':a.fold,'binding':binding,'normalization':norm,'optimization_settings_preserved':True}
                        payload['metan_release']={'source_manifest':json.loads((REPO/'metan/source_manifest.json').read_text()),'backend':spec['backend'],'traits':spec['traits']}
                        tmp=checkpoint.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(checkpoint)
                        rawpred=prediction.astype(float)*std.astype(float)+mean.astype(float)
                        frame={'sample_id':val_ids}
                        for q,trait in enumerate(spec['traits']):frame['true_'+trait]=raw[dm.val_indices,q];frame['pred_'+trait]=rawpred[:,q]
                        pd.DataFrame(frame).to_csv(a.out/f'validation_{t}_best.csv',index=False,float_format='%.17g')
                history.append(row);pd.DataFrame(history).to_csv(a.out/'epoch_metrics.csv',index=False)
                status.update(epochs_completed=len(history),best=best,current=row);dump(status_path,status)
                if spec['early_stopping_rule']=='all_traits_strict' and all(ep-v['epoch']>=tc['early_stopping']['patience'] for v in best.values()):tr.should_stop=True
        callbacks=[Audit()]
        if spec['checkpoint_format']!='plain':
            es=tc['early_stopping'];callbacks.append(EarlyStopping(monitor=es['monitor'],mode=es['mode'],patience=es['patience'],min_delta=es['min_delta']))
        tr=tc['training'];torch.set_float32_matmul_precision(tr['tensor_cores'])
        trainer=pl.Trainer(max_epochs=tr['max_epochs'],accelerator=tr['accelerator'],devices=tr['devices'],precision=tr['precision'],
            accumulate_grad_batches=tr['accumulate_grad_batches'],gradient_clip_val=tr['gradient_clip_val'],val_check_interval=tr['val_check_interval'],
            log_every_n_steps=tc['logging']['log_every_n_steps'],callbacks=callbacks,logger=QuietLogger(),enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False)
        trainer.fit(model,datamodule=dm)
        assert not trainer.interrupted and all(best.values())
        assert trainer.lr_scheduler_configs[0].interval=='step'
        assert trainer.lr_scheduler_configs[0].scheduler.T_max==tc['training']['max_epochs']*len(dm.train_dataloader())
        from metan.runtime import load_checkpoint
        metric_rows=[]
        for t in spec['traits']:
            loaded,payload=load_checkpoint(spec,a.out/'checkpoints'/f'{t}_best.ckpt')
            frame=pd.read_csv(a.out/f'validation_{t}_best.csv',float_precision='round_trip')
            assert frame.sample_id.astype(str).tolist()==val_ids
            metric_rows.append({'species':spec['name'],'seed':a.seed,'fold':a.fold,'trait':t,**regression_metrics(frame['true_'+t],frame['pred_'+t])})
        pd.DataFrame(metric_rows).to_csv(a.out/'fold_metrics.csv',index=False)
        status.update(status='TRAINING_COMPLETED_PENDING_INDEPENDENT_VERIFICATION',finished=time.time(),checkpoint_count=len(best))
        dump(status_path,status)
    except BaseException as exc:
        import traceback
        status.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),finished=time.time());dump(status_path,status)
        raise


if __name__=='__main__':
    main()
