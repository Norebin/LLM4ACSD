import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


# 数据预处理：拼接所有文件夹的数据并处理 NaN
def preprocess_data(root_dir, file_type='class'):
    all_data = []
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            csv_file = os.path.join(folder_path, f'{file_type}.csv')
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                # 提取指标列
                if file_type == 'class':
                    metric_cols = [col for col in df.columns if col not in ['file', 'class', 'type']]
                else:  # method
                    metric_cols = [col for col in df.columns if
                                   col not in ['file', 'class', 'method', 'constructor', 'line', 'hasJavaDoc']]
                all_data.append(df[metric_cols])
    # 拼接所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    # 处理 NaN
    for col in combined_df.columns:
        if combined_df[col].dtype == 'int64':  # 离散整数用众数填充
            combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])
        else:  # 浮点数用中位数填充
            combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])
    # 标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_df)
    return pd.DataFrame(scaled_data, columns=combined_df.columns)


# 方法 1：方差筛选
def variance_filter(data, threshold=0.05):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(data)
    selected_features = data.columns[selector.get_support()].tolist()
    return selected_features, None  # 无重要性分数


# 方法 2：PCA 降维
def pca_filter(data, variance_ratio=0.95):
    pca = PCA()
    pca.fit(data)
    # 选择解释 variance_ratio 的成分数
    n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= variance_ratio) + 1
    loadings = np.abs(pca.components_[:n_components])  # 加载系数绝对值
    importance_scores = np.mean(loadings, axis=0)  # 每列特征的平均贡献
    threshold = np.percentile(importance_scores, 50)  # 保留前 50%
    selected_features = data.columns[importance_scores > threshold].tolist()
    return selected_features, importance_scores


# 方法 3：特征重要性（K-Means + 随机森林）
def importance_filter(data, n_clusters=None):
    if n_clusters is None:
        n_clusters = int(np.sqrt(data.shape[0]))  # 默认簇数
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pseudo_labels = kmeans.fit_predict(data)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(data, pseudo_labels)
    importance_scores = rf.feature_importances_
    threshold = np.percentile(importance_scores, 50)  # 保留前 50%
    selected_features = data.columns[importance_scores > threshold].tolist()
    return selected_features, importance_scores


# 综合筛选结果
def combine_results(data, methods):
    results = {}
    for method_name, (func, params) in methods.items():
        selected_features, scores = func(data, **params)
        results[method_name] = {'features': selected_features, 'scores': scores}
    # 取所有方法的交集作为最终保留特征
    all_features = [set(results[m]['features']) for m in results]
    union_features = set.intersection(*all_features)
    # 计算平均重要性分数（仅对有分数的 PCA 和 RF 方法）
    feature_scores = {}
    for feature in union_features:
        scores = []
        if results['pca']['scores'] is not None:
            idx = data.columns.get_loc(feature)
            scores.append(results['pca']['scores'][idx])
        # if results['importance']['scores'] is not None:
        #     idx = data.columns.get_loc(feature)
        #     scores.append(results['importance']['scores'][idx])
        feature_scores[feature] = np.mean(scores) if scores else 0.0
    return list(union_features), feature_scores


# 主函数
def main(root_dir='data'):
    # 处理 class.csv 和 method.csv
    for file_type in ['class', 'method']:
        print(f"\n处理 {file_type}.csv 数据...")
        # 预处理数据
        data = preprocess_data(root_dir, file_type)
        print(f"数据维度: {data.shape}")
        # 定义筛选方法
        methods = {
            'variance': (variance_filter, {'threshold': 0.01}),
            'pca': (pca_filter, {'variance_ratio': 0.95}),
            # 'importance': (importance_filter, {'n_clusters': None})
        }
        # 综合筛选
        selected_features, feature_scores = combine_results(data, methods)
        # 输出结果
        print(f"{file_type}.csv 推荐保留的特征:")
        print("特征名\t重要性分数")
        for feature in selected_features:
            score = feature_scores.get(feature, 0.0)
            print(f"{feature}\t{score:.4f}")


if __name__ == "__main__":
    root_dir = '/model/data/Research/CK/jsoup'  # 修改为你的数据目录
    main(root_dir)