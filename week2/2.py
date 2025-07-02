# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建images文件夹
if not os.path.exists('images'):
    os.makedirs('images')
# 读取数据

data = pd.read_csv('US-pumpkins.csv')

# 填充缺失值
data.fillna(method='ffill', inplace=True)

# 转换日期格式
try:
    data['Date'] = pd.to_datetime(data['Date'])
except Exception as e:
    print(f"日期转换错误: {e}")
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y', errors='coerce')

# 重新处理日期字段，确保转换成功
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['DayOfYear'] = data['Date'].dt.dayofyear

# 特征工程：对分类变量进行编码
label_encoders = {}
for col in ['City Name', 'Type', 'Package', 'Variety', 'Color']:
    le = LabelEncoder()
    data[col] = data[col].astype(str)
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 特殊处理 'Item Size' 列，将其映射到数值
size_mapping = {
    'sml': 0,
    'med': 1,
    'xlge': 2,
    'jbo': 3,
    'med-lge': 4,
    'lge': 5,
    'exjbo': 6,
    'xtra': 7
}
data['Item Size'] = data['Item Size'].map(size_mapping).fillna(-1).astype(int)

# 选择特征和目标变量
features = ['City Name', 'Type', 'Package', 'Variety', 'Year', 'Month', 'DayOfYear', 'Item Size', 'Color']
X = data[features]
y = data['Low Price']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 预测和评估
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方误差 (MSE): {mse}")
print(f"决定系数 (R²): {r2}")

# 特征重要性
importances = model.feature_importances_
feature_importance = pd.Series(importances, index=features)
feature_importance = feature_importance.sort_values(ascending=False)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title('特征重要性')
plt.xlabel('重要性')
plt.ylabel('特征')
plt.savefig('images/feature_importance.png')
plt.show()

# 预测结果可视化
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.xlabel('实际最低价格')
plt.ylabel('预测最低价格')
plt.title('实际值 vs 预测值')
plt.savefig('images/actual_vs_predicted.png')
plt.show()

# 输出前5个预测结果与实际结果的对比
comparison = pd.DataFrame({'实际值': y_test, '预测值': y_pred})
print(comparison.head())

# 绘制实际价格与预测价格的分布图
plt.figure(figsize=(10, 6))
sns.histplot(y_test, bins=30, color='blue', alpha=0.5, label='实际值')
sns.histplot(y_pred, bins=30, color='red', alpha=0.5, label='预测值')
plt.title('实际价格与预测价格分布')
plt.xlabel('价格')
plt.ylabel('频次')
plt.legend()
plt.savefig('images/price_distribution.png')
plt.show()

# 绘制学习曲线
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    model, X_train_scaled, y_train, train_sizes=np.linspace(.1, 1.0, 5), cv=5
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='训练评分')
plt.plot(train_sizes, test_scores_mean, 'o-', color='red', label='交叉验证评分')
plt.title('学习曲线')
plt.xlabel('训练样本数')
plt.ylabel('评分')
plt.legend()
plt.savefig('images/learning_curve.png')
plt.show()