import glob
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_losses(event_file):
    accumulator = EventAccumulator(event_file)
    accumulator.Reload()
    tags = accumulator.Tags()['scalars']
    print(f"事件文件: {event_file}")
    print(f"可用 tags: {tags}")

    epochs = []
    train_loss = []
    val_loss = []

    # 提取训练损失（epoch级别优先）
    train_tag = None
    for tag in ['train_loss_epoch', 'train_loss', 'train_loss_step']:
        if tag in tags:
            train_tag = tag
            data = accumulator.Scalars(tag)
            epochs = [item.step for item in data]
            train_loss = [item.value for item in data]
            print(f"使用训练损失 tag: {tag}, 长度={len(train_loss)}")
            break

    # 提取验证损失（epoch级别优先）
    val_tag = None
    for tag in ['val_loss_epoch', 'val_loss', 'val_pearson_corr_epoch']:
        if tag in tags:
            val_tag = tag
            data = accumulator.Scalars(tag)
            val_loss = [item.value for item in data]
            if not epochs:
                epochs = [item.step for item in data]
            print(f"使用验证损失 tag: {tag}, 长度={len(val_loss)}")
            break

    # 确保长度一致：以最短的为准进行截断
    if train_loss and val_loss:
        min_len = min(len(epochs), len(train_loss), len(val_loss))
        epochs = epochs[:min_len]
        train_loss = train_loss[:min_len]
        val_loss = val_loss[:min_len]
        print(f"截断到公共长度: {min_len}")
    elif train_loss and not val_loss:
        # 只有训练损失
        min_len = min(len(epochs), len(train_loss))
        epochs = epochs[:min_len]
        train_loss = train_loss[:min_len]
        val_loss = [np.nan] * min_len
    elif val_loss and not train_loss:
        min_len = min(len(epochs), len(val_loss))
        epochs = epochs[:min_len]
        train_loss = [np.nan] * min_len
        val_loss = val_loss[:min_len]
    else:
        print("未找到损失数据")
        return [], [], []

    return epochs, train_loss, val_loss

# 示例：种子1 fold0
seed = 1
fold = 0
pretrain_pattern = f"logs0/DNAWhisper_finetune_cv/blackcarp_{seed}/fold_{fold}/events.out.tfevents.*"
nopretrain_pattern = f"logs0/DNAWhisper_finetune_cv_nopretrain/blackcarp_{seed}/fold_{fold}/events.out.tfevents.*"

pretrain_files = sorted(glob.glob(pretrain_pattern))
nopretrain_files = sorted(glob.glob(nopretrain_pattern))

if pretrain_files:
    epochs, train_loss, val_loss = extract_losses(pretrain_files[0])
    if epochs:
        pd.DataFrame({'epoch': epochs, 'train_loss': train_loss, 'val_loss': val_loss}).to_csv('pretrain_loss_fold0.csv', index=False)
        print("已保存 pretrain_loss_fold0.csv")
    else:
        print("预训练事件文件中没有损失数据")
else:
    print("未找到预训练事件文件")

if nopretrain_files:
    epochs, train_loss, val_loss = extract_losses(nopretrain_files[0])
    if epochs:
        pd.DataFrame({'epoch': epochs, 'train_loss': train_loss, 'val_loss': val_loss}).to_csv('nopretrain_loss_fold0.csv', index=False)
        print("已保存 nopretrain_loss_fold0.csv")
    else:
        print("非预训练事件文件中没有损失数据")
else:
    print("未找到非预训练事件文件")
