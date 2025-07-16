# 主函数
import warnings
import matplotlib
from scripts.io import load_pumpkin_data
from scripts.data_analysis import clean_data
from scripts.feature_processing import get_feature
from scripts.configuration import conf
from scripts.evaluate import evaluate_rf, evaluate_lr, evaluate_lgbm, evaluate_xgb
from scripts.model import build_model
from scripts.rf_viz import visualize_rf_tree

warnings.filterwarnings("ignore")
matplotlib.use('Agg')


def main():
    df = load_pumpkin_data()
    X, y = clean_data(df)
    X_feat, y_feat = get_feature(X)

    # 随机森林
    rf_model = build_model("RandomForestRegressor", conf["RandomForest"]["params"])
    evaluate_rf(rf_model, X_feat, y_feat)
    visualize_rf_tree(rf_model, X_feat, tree_idx=0, max_depth=3)

    # 线性回归
    lr_model = build_model("LinearRegression", conf["LinearRegression"]["params"])
    evaluate_lr(lr_model, X_feat, y_feat)

    # LightGBM
    lgb_model = build_model("LGBMRegressor", conf["LGBM"]["params"])
    evaluate_lgbm(lgb_model, X_feat, y_feat)

    # XGBoost
    xgb_model = build_model("XGBRegressor", conf["XGBoost"]["params"])
    evaluate_xgb(xgb_model, X_feat, y_feat)


if __name__ == '__main__':
    main()