# 读取数据函数化
import pandas as pd
from pathlib import Path

def load_pumpkin_data() -> pd.DataFrame:
    file_path = Path(__file__).resolve().parent.parent / "data" / "us-pumpkins.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{file_path}")
    return pd.read_csv(file_path)