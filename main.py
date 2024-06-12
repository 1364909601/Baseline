# 导入必要库
import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier


# 读取训练集标签
train_label = pd.read_csv('training_data/train_label.csv', encoding='gb2312')
print(train_label.head(10))

# 读取数据集
train_features = []

# 对训练集个体进行遍历
for sid in os.listdir('./training_data/'):
  if '.csv' in sid:
    continue

# 获取训练集三类观测数据
df_acc = pd.read_csv(f'./training_data/{sid}/ACC.csv')
df_gsr = pd.read_csv(f'./training_data/{sid}/GSR.csv')
df_ppg = pd.read_csv(f'./training_data/{sid}/PPG.csv')

# 按照时间维度拼接三类观测数据
df = pd.concat([df_acc, df_gsr.iloc[::2, :-1].reset_index(drop=True), df_ppg.iloc[:, :-1]], axis=1)

df['GSR'] = df['GSR'].round(4)
df['GSR'] = df['GSR'].replace(0.0061, np.nan)

print(sid, df['recording_time'].min(), df['recording_time'].max())
label = train_label.set_index('文件名').loc[sid].values[0]
for idx in range(df.shape[0] // 10000):
  df_batch = df.iloc[idx*10000: (idx+1)*10000]
  feat = manual_feature(df_batch)
  feat = [sid] + feat + [label]
  train_features.append(feat)
  
