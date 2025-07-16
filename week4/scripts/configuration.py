# 配置
conf = {
    "RandomForest": {
        "type": "RandomForestRegressor",
        "params": {"n_estimators": 100, "random_state": 42}
    },
    "LinearRegression": {
        "type": "LinearRegression",
        "params": {}
    },
    "LGBM": {
        "type": "LGBMRegressor",
        "params": {"n_estimators": 100, "random_state": 42}
    },
    "XGBoost": {
        "type": "XGBRegressor",
        "params": {"n_estimators": 100, "random_state": 42}
    }
}