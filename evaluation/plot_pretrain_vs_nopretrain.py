#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置路径
result_dir = "evaluation/blackcarp/test_results"

# 读取预测结果
pretrain_df = pd.read_csv(os.path.join(result_dir, "pretrain_all_splits.csv"))
nopretrain_df = pd.read_csv(os.path.join(result_dir, "nopretrain_all_splits.csv"))

# 计算残差（预测值 - 真实值）
for df in [pretrain_df, nopretrain_df]:
    df['BW_residual'] = df['BW_pred'] - df['BW_true']
    df['LE_residual'] = df['LE_pred'] - df['LE_true']

# 颜色
pretrain_color = '#1f77b4'
nopretrain_color = '#ff7f0e'

fig, axes = plt.subplots(3, 2, figsize=(14, 16))

# (a) 验证集残差箱线图
ax = axes[0,0]
for i, pheno in enumerate(['BW','LE']):
    data_pre = pretrain_df[pretrain_df['split']=='val'][f'{pheno}_residual']
    data_nopre = nopretrain_df[nopretrain_df['split']=='val'][f'{pheno}_residual']
    positions = [i*2, i*2+1]
    bp = ax.boxplot([data_pre, data_nopre], positions=positions, patch_artist=True)
    bp['boxes'][0].set_facecolor(pretrain_color)
    bp['boxes'][1].set_facecolor(nopretrain_color)
ax.set_xticks([0.5, 2.5])
ax.set_xticklabels(['BW','LE'])
ax.set_ylabel('Validation residual')
ax.set_title('(a) Validation residual boxplot')
ax.legend([bp['boxes'][0], bp['boxes'][1]], ['Pretrained','Non-pretrained'])

# (b) 验证集残差密度直方图
ax = axes[0,1]
for pheno in ['BW','LE']:
    sns.kdeplot(pretrain_df[pretrain_df['split']=='val'][f'{pheno}_residual'], color=pretrain_color, ax=ax, linestyle='-', label=f'{pheno} Pretrained')
    sns.kdeplot(nopretrain_df[nopretrain_df['split']=='val'][f'{pheno}_residual'], color=nopretrain_color, ax=ax, linestyle='--', label=f'{pheno} Non-pretrained')
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_title('(b) Validation residual density')
ax.legend()

# (c) 训练损失曲线
ax = axes[1,0]
pretrain_loss_file = os.path.join(result_dir, "pretrain_loss_fold0.csv")
nopretrain_loss_file = os.path.join(result_dir, "nopretrain_loss_fold0.csv")
if os.path.exists(pretrain_loss_file) and os.path.exists(nopretrain_loss_file):
    pretrain_loss_df = pd.read_csv(pretrain_loss_file)
    nopretrain_loss_df = pd.read_csv(nopretrain_loss_file)
    ax.plot(pretrain_loss_df['train_loss'], color=pretrain_color, label='Pretrained')
    ax.plot(nopretrain_loss_df['train_loss'], color=nopretrain_color, label='Non-pretrained')
else:
    ax.text(0.5,0.5,'Loss data not found', ha='center')
ax.set_xlabel('Epoch')
ax.set_ylabel('Training loss')
ax.set_title('(c) Training loss')
ax.legend()

# (d) 测试集残差箱线图
ax = axes[2,0]
for i, pheno in enumerate(['BW','LE']):
    data_pre = pretrain_df[pretrain_df['split']=='test'][f'{pheno}_residual']
    data_nopre = nopretrain_df[nopretrain_df['split']=='test'][f'{pheno}_residual']
    positions = [i*2, i*2+1]
    bp = ax.boxplot([data_pre, data_nopre], positions=positions, patch_artist=True)
    bp['boxes'][0].set_facecolor(pretrain_color)
    bp['boxes'][1].set_facecolor(nopretrain_color)
ax.set_xticks([0.5, 2.5])
ax.set_xticklabels(['BW','LE'])
ax.set_ylabel('Test residual')
ax.set_title('(d) Test residual boxplot')
ax.legend([bp['boxes'][0], bp['boxes'][1]], ['Pretrained','Non-pretrained'])

# (e) 测试集残差密度直方图
ax = axes[2,1]
for pheno in ['BW','LE']:
    sns.kdeplot(pretrain_df[pretrain_df['split']=='test'][f'{pheno}_residual'], color=pretrain_color, ax=ax, linestyle='-', label=f'{pheno} Pretrained')
    sns.kdeplot(nopretrain_df[nopretrain_df['split']=='test'][f'{pheno}_residual'], color=nopretrain_color, ax=ax, linestyle='--', label=f'{pheno} Non-pretrained')
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_title('(e) Test residual density')
ax.legend()

# (f) 验证损失曲线
ax = axes[1,1]
if os.path.exists(pretrain_loss_file) and os.path.exists(nopretrain_loss_file):
    ax.plot(pretrain_loss_df['val_loss'], color=pretrain_color, label='Pretrained')
    ax.plot(nopretrain_loss_df['val_loss'], color=nopretrain_color, label='Non-pretrained')
else:
    ax.text(0.5,0.5,'Val loss data not found', ha='center')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation loss')
ax.set_title('(f) Validation loss')
ax.legend()

plt.tight_layout()
output_path = os.path.join(result_dir, "pretrain_vs_nopretrain_blackcarp.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"图像已保存到: {output_path}")
