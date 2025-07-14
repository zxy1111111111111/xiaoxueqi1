# 特征处理
import pandas as pd

def get_feature(df):
    X = pd.get_dummies(df[['Variety', 'Package']], columns=['Variety', 'Package'])
    y = df['Avg Price']
    return X, y