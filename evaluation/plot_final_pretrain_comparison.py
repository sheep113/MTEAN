#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

result_dir = "evaluation/blackcarp/test_results"

# 读取原始预测记录
pretrain_raw = pd.read_csv(os.path.join(result_dir, "pretrain_all_splits.csv"))
nopretrain_raw = pd.read_csv(os.path.join(result_dir, "nopretrain_all_splits.csv"))

# 多种子多折平均：按 split 和 sample_id 分组
def average_predictions(df):
    avg = df.groupby(['split','sample_id'], as_index=False).agg(
        BW_true=('BW_true','mean'),
        BW_pred=('BW_pred','mean'),
        LE_true=('LE_true','mean'),
        LE_pred=('LE_pred','mean')
    )
    return avg

pretrain_avg = average_predictions(pretrain_raw)
nopretrain_avg = average_predictions(nopretrain_raw)

# 计算残差
for df in [pretrain_avg, nopretrain_avg]:
    df['BW_residual'] = df['BW_pred'] - df['BW_true']
    df['LE_residual'] = df['LE_pred'] - df['LE_true']

# 可选：标准化残差（使用各自训练集的标准差）
std_bw = pretrain_avg[pretrain_avg['split']=='train']['BW_residual'].std()
std_le = pretrain_avg[pretrain_avg['split']=='train']['LE_residual'].std()
for df in [pretrain_avg, nopretrain_avg]:
    df['BW_resid_scaled'] = df['BW_residual'] / std_bw
    df['LE_resid_scaled'] = df['LE_residual'] / std_le

# 颜色
pretrain_color = '#1f77b4'
nopretrain_color = '#ff7f0e'

# 读取平均损失
try:
    pretrain_loss = pd.read_csv(os.path.join(result_dir, "avg_pretrain_loss.csv"))
    nopretrain_loss = pd.read_csv(os.path.join(result_dir, "avg_nopretrain_loss.csv"))
    loss_available = True
except:
    loss_available = False

fig, axes = plt.subplots(3, 2, figsize=(14, 16))

# (a) 验证集残差箱线图（标准化）
ax = axes[0,0]
for i, pheno in enumerate(['BW','LE']):
    data_pre = pretrain_avg[pretrain_avg['split']=='val'][f'{pheno}_resid_scaled']
    data_nopre = nopretrain_avg[nopretrain_avg['split']=='val'][f'{pheno}_resid_scaled']
    positions = [i*2, i*2+1]
    bp = ax.boxplot([data_pre, data_nopre], positions=positions, patch_artist=True)
    bp['boxes'][0].set_facecolor(pretrain_color)
    bp['boxes'][1].set_facecolor(nopretrain_color)
ax.set_xticks([0.5, 2.5])
ax.set_xticklabels(['BW','LE'])
ax.set_ylabel('Standardized validation residual')
ax.set_title('(a) Validation residual boxplot')
ax.legend([bp['boxes'][0], bp['boxes'][1]], ['Pretrained','Non-pretrained'])

# (b) 验证集残差密度图（标准化）
ax = axes[0,1]
for pheno in ['BW','LE']:
    sns.kdeplot(pretrain_avg[pretrain_avg['split']=='val'][f'{pheno}_resid_scaled'], color=pretrain_color, ax=ax, linestyle='-', label=f'{pheno} Pretrained')
    sns.kdeplot(nopretrain_avg[nopretrain_avg['split']=='val'][f'{pheno}_resid_scaled'], color=nopretrain_color, ax=ax, linestyle='--', label=f'{pheno} Non-pretrained')
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_title('(b) Validation residual density')
ax.legend()

# (c) 训练损失曲线
ax = axes[1,0]
if loss_available:
    ax.plot(pretrain_loss['train_loss'], color=pretrain_color, label='Pretrained')
    ax.plot(nopretrain_loss['train_loss'], color=nopretrain_color, label='Non-pretrained')
else:
    ax.text(0.5,0.5,'Loss data not found', ha='center')
ax.set_xlabel('Epoch')
ax.set_ylabel('Training loss')
ax.set_title('(c) Training loss')
ax.legend()

# (d) 测试集残差箱线图（标准化）
ax = axes[2,0]
for i, pheno in enumerate(['BW','LE']):
    data_pre = pretrain_avg[pretrain_avg['split']=='test'][f'{pheno}_resid_scaled']
    data_nopre = nopretrain_avg[nopretrain_avg['split']=='test'][f'{pheno}_resid_scaled']
    positions = [i*2, i*2+1]
    bp = ax.boxplot([data_pre, data_nopre], positions=positions, patch_artist=True)
    bp['boxes'][0].set_facecolor(pretrain_color)
    bp['boxes'][1].set_facecolor(nopretrain_color)
ax.set_xticks([0.5, 2.5])
ax.set_xticklabels(['BW','LE'])
ax.set_ylabel('Standardized test residual')
ax.set_title('(d) Test residual boxplot')
ax.legend([bp['boxes'][0], bp['boxes'][1]], ['Pretrained','Non-pretrained'])

# (e) 测试集残差密度图（标准化）
ax = axes[2,1]
for pheno in ['BW','LE']:
    sns.kdeplot(pretrain_avg[pretrain_avg['split']=='test'][f'{pheno}_resid_scaled'], color=pretrain_color, ax=ax, linestyle='-', label=f'{pheno} Pretrained')
    sns.kdeplot(nopretrain_avg[nopretrain_avg['split']=='test'][f'{pheno}_resid_scaled'], color=nopretrain_color, ax=ax, linestyle='--', label=f'{pheno} Non-pretrained')
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_title('(e) Test residual density')
ax.legend()

# (f) 验证损失曲线
ax = axes[1,1]
if loss_available:
    ax.plot(pretrain_loss['val_loss'], color=pretrain_color, label='Pretrained')
    ax.plot(nopretrain_loss['val_loss'], color=nopretrain_color, label='Non-pretrained')
else:
    ax.text(0.5,0.5,'Val loss data not found', ha='center')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation loss')
ax.set_title('(f) Validation loss')
ax.legend()

plt.tight_layout()
out_path = os.path.join(result_dir, "pretrain_vs_nopretrain_multi_seed_average.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"图像已保存: {out_path}")
