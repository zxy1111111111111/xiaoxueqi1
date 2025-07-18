# 模型
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

def build_model(model_type, params=None):
    params = params or {}
    if model_type == "RandomForestRegressor":
        return RandomForestRegressor(**params)
    elif model_type == "LinearRegression":
        return LinearRegression(**params)
    elif model_type == "LGBMRegressor":
        return LGBMRegressor(**params)
    elif model_type == "XGBRegressor":
        return XGBRegressor(**params)
    else:
        raise ValueError("Unsupported model type")