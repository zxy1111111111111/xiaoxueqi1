# 评估
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_regression_model_performance(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

def evaluate_rf(model, X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mse, r2, y_pred = get_regression_model_performance(model, X_train, X_test, y_train, y_test)
    print(f"[RandomForest] MSE: {mse:.2f}, R²: {r2:.2f}")

    # 特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        fi = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        fi.head(15).plot(kind='barh')
        plt.title("Top 15 Feature Importances")
        save_path = Path(__file__).resolve().parent.parent / "images" / "rf_feature_importance.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    # 实际 vs 预测
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    save_path = Path(__file__).resolve().parent.parent / "images" / "rf_actual_vs_predicted.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    # 学习曲线
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 5))
    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), label='CV')
    plt.xlabel("Training examples")
    plt.ylabel("R²")
    plt.title("Learning Curve")
    save_path = Path(__file__).resolve().parent.parent / "images" / "rf_learning_curve.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

from pathlib import Path

def evaluate_lr(model, X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mse, r2, y_pred = get_regression_model_performance(model, X_train, X_test, y_train, y_test)
    print(f"[LinearRegression] MSE: {mse:.2f}, R²: {r2:.2f}")

    # 示例预测
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
    mse, r2, y_pred = get_regression_model_performance(model, X_train, X_test, y_train, y_test)
    print(f"[LGBM] MSE: {mse:.2f}, R²: {r2:.2f}")

def evaluate_xgb(model, X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mse, r2, y_pred = get_regression_model_performance(model, X_train, X_test, y_train, y_test)
    print(f"[XGBoost] MSE: {mse:.2f}, R²: {r2:.2f}")