# 音乐聚类分析

## 项目简介
使用 K-Means 聚类算法对尼日利亚音乐数据集进行分析，探索不同类型音乐的特征差异，并使用轮廓系数评估聚类质量。

## 数据集
数据集包含以下列：
- `name`: 歌曲名称
- `album`: 专辑名称
- `artist`: 艺术家名称
- `artist_top_genre`: 艺术家主要音乐流派
- `release_date`: 发布日期
- `length`: 歌曲长度
- `popularity`: 流行度
- `danceability`: 舞蹈性
- `acousticness`: 原声性
- `energy`: 能量
- `instrumentalness`: 器乐性
- `liveness`: 现场感
- `loudness`: 响度
- `speechiness`: 语音性
- `tempo`: 节奏
- `time_signature`: 拍号

## 分析方法
使用 K-Means 聚类算法对以下特征进行聚类分析：
- `danceability`
- `energy`
- `loudness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `tempo`

## 代码分析

### 数据预处理
```python
# 读取数据
data = pd.read_csv('nigerian-songs.csv')

# 选择用于聚类的特征
features = ['danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'tempo']

# 数据预处理
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])
```
- 使用 `StandardScaler` 对特征数据进行标准化处理，使每个特征具有零均值和单位方差。

### K-Means 聚类
```python
# 使用 K-Means 进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_features)
data['cluster'] = kmeans.labels_
```
- 使用 K-Means 算法对数据进行聚类，设置聚类数量为 3。
- 将聚类结果存储在数据框的 `cluster` 列中。

### 轮廓系数计算
```python
# 计算轮廓系数
silhouette_avg = silhouette_score(scaled_features, kmeans.labels_)
print(f"平均轮廓系数: {silhouette_avg:.4f}")
```
- 使用 `silhouette_score` 计算聚类的轮廓系数，评估聚类效果。

### 可视化
```python
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
```
- 设置中文字体以支持中文显示。
- 绘制散点图展示聚类结果，使用 `acousticness` 和 `energy` 作为坐标轴。

### 特征分布箱型图
```python
# 绘制箱型图（特征分布）
for feature in features:
    plt.figure(figsize=(10, 7))
    sns.boxplot(x='cluster', y=feature, data=data)
    plt.title(f'{feature} 在不同聚类中的分布')
    plt.xlabel('聚类')
    plt.ylabel(feature)
    plt.savefig(f'images/{feature}_箱型图.png')
    plt.show()
```
- 为每个特征绘制箱型图，展示其在不同聚类中的分布情况。

### 聚类中心图
```python
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
```
- 计算并绘制每个特征的聚类中心值，使用柱状图展示。

### 特征相关性热图
```python
# 绘制相关性热图
plt.figure(figsize=(12, 8))
corr = data[features + ['cluster']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('特征相关性热图')
plt.savefig('images/特征相关性热图.png')
plt.show()
```
- 计算特征之间的相关性，并使用热图可视化。

## 分析结果

### 聚类结果可视化

#### 音乐聚类结果
![音乐聚类结果](images/音乐聚类结果.png)
展示 `原声性` 和 `能量` 的关系及其聚类分布。从图中可以看出，不同聚类在原声性和能量上的分布差异明显。

#### 特征分布箱型图
- **Danceability**
  ![Danceability 箱型图](images/danceability_箱型图.png)
  各聚类的舞蹈性分布存在差异，聚类 1 的舞蹈性整体较高。

- **Energy**
  ![Energy 箱型图](images/energy_箱型图.png)
  聚类 1 的能量最高，聚类 2 的能量最低，这表明聚类 1 的音乐更为高能。

- **Loudness**
  ![Loudness 箱型图](images/loudness_箱型图.png)
  聚类 0 的响度最低，聚类 1 的响度最高，这显示了不同聚类在响度上的显著差异。

- **Acousticness**
  ![Acousticness 箱型图](images/acousticness_箱型图.png)
  聚类 2 的原声性较高，而聚类 1 的原声性较低，表明聚类 2 更多包含原声音乐。

- **Instrumentalness**
  ![Instrumentalness 箱型图](images/instrumentalness_箱型图.png)
  各聚类的器乐性分布较为集中，但聚类 2 的器乐性略高。

- **Liveness**
  ![Liveness 箱型图](images/liveness_箱型图.png)
  聚类 0 的现场感较高，聚类 1 和 2 的现场感相对较低。

- **Tempo**
  ![Tempo 箱型图](images/tempo_箱型图.png)
  聚类 0 的节奏最快，聚类 2 的节奏最慢，显示出不同聚类在节奏上的明显差异。

#### 聚类中心图
- **Danceability**
  ![Danceability 聚类中心](images/danceability_聚类中心.png)
  聚类 1 的舞蹈性中心值最高，聚类 2 次之，聚类 0 最低。

- **Energy**
  ![Energy 聚类中心](images/energy_聚类中心.png)
  聚类 0 和 1 的能量中心值较高，聚类 2 最低。

- **Loudness**
  ![Loudness 聚类中心](images/loudness_聚类中心.png)
  聚类 1 的响度中心值最高，聚类 0 次之，聚类 2 最低。

- **Acousticness**
  ![Acousticness 聚类中心](images/acousticness_聚类中心.png)
  聚类 2 的原声性中心值最高，聚类 0 次之，聚类 1 最低。

- **Instrumentalness**
  ![Instrumentalness 聚类中心](images/instrumentalness_聚类中心.png)
  聚类 2 的器乐性中心值最高，聚类 0 和 1 较低。

- **Liveness**
  ![Liveness 聚类中心](images/liveness_聚类中心.png)
  聚类 0 的现场感中心值最高，聚类 2 次之，聚类 1 最低。

- **Tempo**
  ![Tempo 聚类中心](images/tempo_聚类中心.png)
  聚类 0 的节奏中心值最高，聚类 1 次之，聚类 2 最低。

#### 特征相关性热图
![特征相关性热图](images/特征相关性热图.png)
展示特征之间的相关性。可以看到能量和响度之间有较高的正相关性（0.73），而能量和原声性之间有较高的负相关性（-0.29）。

### 结果解读

#### 聚类特征分析
1. **聚类 0**：
   - **特征描述**：能量和舞蹈性相对较高，原声性和器乐性适中，响度较低。
   - **样本歌曲**：`Sparky` by Cruel Santino, `LITT!` by AYLØ.

2. **聚类 1**：
   - **特征描述**：能量和舞蹈性最高，响度较高，原声性和器乐性较低。
   - **样本歌曲**：`shuga rush` by Odunsi (The Engine), `Alté` by DRB Lasgidi.

3. **聚类 2**：
   - **特征描述**：能量和舞蹈性适中，原声性和器乐性较高，响度最低。
   - **样本歌曲**：`luv in a mosh` by Odunsi (The Engine), `City on Lights!` by AYLØ.

#### 轮廓系数结果
- 平均轮廓系数为 **0.2220**，表明聚类效果一般。可以尝试调整聚类数量或选择不同的特征以提高聚类质量。
