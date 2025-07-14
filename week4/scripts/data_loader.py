# 数据加载
import pandas as pd
from pathlib import Path

def load_data():
    return pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "us-pumpkins.csv")