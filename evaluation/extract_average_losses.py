import glob
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_losses(event_file):
    acc = EventAccumulator(event_file)
    acc.Reload()
    tags = acc.Tags()['scalars']
    train_tag = None
    val_tag = None
    for t in ['train_loss_epoch', 'train_loss']:
        if t in tags: train_tag = t; break
    for t in ['val_loss', 'val_loss_epoch']:
        if t in tags: val_tag = t; break
    if train_tag is None or val_tag is None:
        return None, None, None
    train_df = pd.DataFrame(acc.Scalars(train_tag))[['step','value']].rename(columns={'value':'train_loss'})
    val_df = pd.DataFrame(acc.Scalars(val_tag))[['step','value']].rename(columns={'value':'val_loss'})
    merged = pd.merge(train_df, val_df, on='step', how='inner')
    return merged['step'].values, merged['train_loss'].values, merged['val_loss'].values

def collect_average(pattern):
    all_train = []
    all_val = []
    for seed in [1,2,3]:
        for fold in [0,1,2,3,4]:
            files = glob.glob(pattern.format(seed=seed, fold=fold))
            if not files:
                continue
            epochs, train_loss, val_loss = extract_losses(files[0])
            if epochs is not None and len(epochs) > 0:
                all_train.append(train_loss)
                all_val.append(val_loss)
    if not all_train:
        return None, None, None
    min_len = min(len(x) for x in all_train)
    train_arr = np.array([x[:min_len] for x in all_train])
    val_arr = np.array([x[:min_len] for x in all_val])
    avg_train = train_arr.mean(axis=0)
    avg_val = val_arr.mean(axis=0)
    return np.arange(min_len), avg_train, avg_val

# 预训练
pat_pre = "logs0/DNAWhisper_finetune_cv/blackcarp_{seed}/fold_{fold}/events.out.tfevents.*"
epochs, avg_train_pre, avg_val_pre = collect_average(pat_pre)
if epochs is not None:
    pd.DataFrame({'epoch':epochs,'train_loss':avg_train_pre,'val_loss':avg_val_pre}).to_csv('evaluation/blackcarp/test_results/avg_pretrain_loss.csv', index=False)
    print("平均预训练损失已保存")

# 非预训练
pat_nopre = "logs0/DNAWhisper_finetune_cv_nopretrain/blackcarp_{seed}/fold_{fold}/events.out.tfevents.*"
epochs, avg_train_nopre, avg_val_nopre = collect_average(pat_nopre)
if epochs is not None:
    pd.DataFrame({'epoch':epochs,'train_loss':avg_train_nopre,'val_loss':avg_val_nopre}).to_csv('evaluation/blackcarp/test_results/avg_nopretrain_loss.csv', index=False)
    print("平均非预训练损失已保存")
