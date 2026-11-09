"""Aggregate per-evaluation metrics, including fold-wise RMSE, with sample SD."""
def summarize(source,destination):
    import pandas as pd
    frame=pd.read_csv(source)
    keys=[name for name in ['model','trait'] if name in frame]
    if frame.duplicated(keys+['seed','fold']).any():raise ValueError('Duplicate model/trait/seed/fold records')
    metrics=['Pearson','MSE','RMSE','MAE','R2','prediction_SD_ratio']
    rows=[]
    for values,group in frame.groupby(keys):
        if len(group)!=15 or group.groupby('seed').size().tolist()!=[5,5,5]:raise ValueError('A complete three-seed five-fold table is required')
        values=values if isinstance(values,tuple) else (values,)
        base=dict(zip(keys,values))
        for label,subset in [('overall',group)]+[(f'seed{seed}',part) for seed,part in group.groupby('seed')]:
            for metric in metrics:
                rows.append({**base,'scope':label,'metric':metric,'n':len(subset),'mean':subset[metric].mean(),'SD':subset[metric].std(ddof=1)})
    pd.DataFrame(rows).to_csv(destination,index=False)
