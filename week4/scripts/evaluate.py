# 评估
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, train_test_split

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def get_regression_model_performance(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred


def evaluate_rf(model, X, y):
    """随机森林评估 + 可视化"""
    from scripts.rf_viz import visualize_rf_tree

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mse, r2, y_pred = get_regression_model_performance(model, X_train, X_test, y_train, y_test)
    print(f"[RandomForest] MSE: {mse:.2f}, R²: {r2:.2f}")

    # 生成可视化
    visualize_rf_tree(model, X, tree_idx=0, max_depth=3)


def evaluate_lr(model, X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mse, r2, y_pred = get_regression_model_performance(model, X_train, X_test, y_train, y_test)
    print(f"[LinearRegression] MSE: {mse:.2f}, R²: {r2:.2f}")

    new = pd.DataFrame({
        'Variety_HOWDEN TYPE': [1],
        'Package_24 inch bins': [1]
    })
    new = new.reindex(columns=X.columns, fill_value=0)
    pred = model.predict(new)[0]
    print(f"[LinearRegression] 示例预测价格: ${pred:.2f}")


def evaluate_lgbm(model, X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.set_params(verbose=-1)
    mse, r2, y_pred = get_regression_model_performance(model, X_train, X_test, y_train, y_test)
    print(f"[LGBM] MSE: {mse:.2f}, R²: {r2:.2f}")


def evaluate_xgb(model, X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.set_params(verbosity=0)
    mse, r2, y_pred = get_regression_model_performance(model, X_train, X_test, y_train, y_test)
    print(f"[XGBoost] MSE: {mse:.2f}, R²: {r2:.2f}")