# 数据分析
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    df = df.dropna(subset=['Low Price', 'High Price'])
    df['Avg Price'] = (df['Low Price'] + df['High Price']) / 2
    return df[['Variety', 'Package', 'Low Price', 'High Price', 'Avg Price']], df['Avg Price']

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)