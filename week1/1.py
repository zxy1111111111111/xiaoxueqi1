import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 创建images文件夹
if not os.path.exists('images'):
    os.makedirs('images')

# 读取数据
data = pd.read_csv('nigerian-songs.csv')

# 选择用于聚类的特征
features = ['danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'tempo']

# 数据预处理
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

# 使用 K-Means 进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_features)
data['cluster'] = kmeans.labels_

# 计算轮廓系数
silhouette_avg = silhouette_score(scaled_features, kmeans.labels_)
print(f"平均轮廓系数: {silhouette_avg:.4f}")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 可视化聚类结果
plt.figure(figsize=(10, 7))
scatter = plt.scatter(data['acousticness'], data['energy'], c=data['cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.xlabel('原声性')
plt.ylabel('能量')
plt.title('音乐聚类结果')
plt.savefig('images/音乐聚类结果.png')
plt.show()

# 绘制箱型图（特征分布）
for feature in features:
    plt.figure(figsize=(10, 7))
    sns.boxplot(x='cluster', y=feature, data=data)
    plt.title(f'{feature} 在不同聚类中的分布')
    plt.xlabel('聚类')
    plt.ylabel(feature)
    plt.savefig(f'images/{feature}_箱型图.png')
    plt.show()

# 绘制聚类中心
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)

for feature in features:
    plt.figure(figsize=(10, 7))
    sns.barplot(x=np.arange(3), y=cluster_centers_df[feature], palette='viridis')
    plt.title(f'{feature} 聚类中心')
    plt.xlabel('聚类')
    plt.ylabel(feature)
    plt.xticks(np.arange(3), labels=['聚类 0', '聚类 1', '聚类 2'])
    plt.savefig(f'images/{feature}_聚类中心.png')
    plt.show()

# 绘制相关性热图
plt.figure(figsize=(12, 8))
corr = data[features + ['cluster']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('特征相关性热图')
plt.savefig('images/特征相关性热图.png')
plt.show()

# 输出聚类结果的描述性统计
cluster_stats = data.groupby('cluster')[features].agg(['mean', 'std', 'count'])
print(cluster_stats)

# 输出每个聚类中的歌曲数量
cluster_counts = data['cluster'].value_counts()
print("\n各聚类歌曲数量:")
print(cluster_counts)

# 输出每个聚类中的前几首歌曲
print("\n各聚类样本歌曲:")
for cluster in data['cluster'].unique():
    print(f"\n聚类 {cluster}:")
    print(data[data['cluster'] == cluster][['name', 'artist', 'artist_top_genre']].head())