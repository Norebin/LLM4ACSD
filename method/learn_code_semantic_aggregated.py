"""
使用大模型中间层hidden_states表示的代码语义，训练一个二分类模型，并计算预测的指标。

*** 聚合训练版本 ***
本脚本会将具有相同前缀的多个项目数据(.pt文件)合并在一起，进行统一的训练和测试，
旨在评估模型在更泛化、更多样的数据集上的表现。

实验目标:
1.  对比不同预训练模型（如 CodeBERT, GraphCodeBERT, UniXcoder等）提取的代码语义特征对下游任务的影响。
2.  探究模型不同层级（深、中、浅层）的hidden_states在表征代码语义时的效果差异。
3.  比较不同分类模型（如简单的神经网络、传统机器学习方法）在基于代码语义进行分类时的性能。

数据来源：
-   特征数据位于 `./semantic_features` 目录下，以`.pt`文件格式存储。
-   文件名约定：`实验组前缀_项目名.pt` (例如: lora_deepseek_stirling.pt)
-   本脚本会将 `实验组前缀` 相同的文件聚合在一起。

结果输出：
-   实验结果将以表格形式呈现，清晰展示不同模型、不同特征、不同分类器下的性能指标（Precision, Accuracy, Recall, F1, MCC）。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, matthews_corrcoef
from abc import ABC, abstractmethod
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# --- 1. 配置区域 ---
FEATURE_DIR = '/model/lxj/LLM_comment_generate/temp/semantic_features'
RESULTS_FILE = '/model/lxj/LLM_comment_generate/temp/hidden_states_results_aggregated6.csv' # 修改了输出文件名
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-2


# --- 2. 数据加载模块 ---
def load_data(file_path):
    """
    从.pt文件中加载特征和标签。

    Args:
        file_path (str): .pt文件的路径。

    Returns:
        tuple: (metadata, features_dict, labels)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    data = torch.load(file_path, map_location=DEVICE)
    return data['metadata'], data['features'], data['labels']


# --- 3. 分类模型定义 (模块化) ---
class BaseModel(ABC):
    """分类模型的抽象基类，确保所有模型都有统一的接口。"""
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def train_model(self, X_train, y_train):
        pass

    @abstractmethod
    def predict_model(self, X_test):
        pass

    def __str__(self):
        return self.name

class MLPClassifier(BaseModel, nn.Module):
    """一个简单的多层感知机分类器。"""
    def __init__(self, input_dim, hidden_dim=1024):
        BaseModel.__init__(self, "MLP")
        nn.Module.__init__(self)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
        self.to(DEVICE)

    def forward(self, x):
        return self.network(x)

    def train_model(self, X_train, y_train):
        """训练逻辑"""
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCELoss()

        self.train()
        for epoch in range(EPOCHS):
            for features, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(features)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
    
    def predict_model(self, X_test):
        """预测逻辑"""
        self.eval()
        with torch.no_grad():
            outputs = self(X_test)
            predicted = (outputs > 0.5).squeeze().cpu().numpy()
        return predicted

# 在这里可以添加更多模型，例如：
# from sklearn.linear_model import LogisticRegression
# class SklearnModel(BaseModel):
#     ...

# --- 4. 训练与评估模块 ---
def calculate_metrics(y_true, y_pred):
    """计算并返回所有需要的指标。"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

def run_experiment(model_name, experiment_group, feature_type, classifier_name, X, Y, x, y):
    """
    运行单次实验（训练、评估、返回结果）。

    Args:
        model_name (str): 预训练模型的名称。
        experiment_group (str): 实验组的名称 (文件前缀).
        feature_type (str): 特征提取的类型。
        classifier_name (str): 分类器的名称。
        X (torch.Tensor): 特征数据。
        y (torch.Tensor): 标签数据。

    Returns:
        dict: 本次实验的结果。
    """
    # X_train, y_train = X, Y
    # X_test, y_test = x, y
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # 根据分类器名称选择模型
    if classifier_name == 'MLP':
        input_dim = X_train.shape[1]
        classifier = MLPClassifier(input_dim=input_dim)
        classifier.train_model(X_train, y_train)
        predictions = classifier.predict_model(X_test)
    # elif classifier_name == 'LogisticRegression':
    #     # 这里可以实例化和训练 sklearn 模型
    #     pass
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
        
    y_test_np = y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test
    metrics = calculate_metrics(y_test_np, predictions)
    
    result = {
        'Pretrained Model': model_name,
        'Experiment Group': experiment_group,
        'Feature Type': feature_type,
        'Classifier': classifier_name,
        **metrics
    }
    return result

# --- 5. 主流程 ---
def main():
    """
    主函数，编排整个实验流程。
    此版本将具有相同前缀的多个项目数据合并在一起进行训练和测试。
    """
    print("--- Starting Aggregated Experiments ---")
    if not os.path.exists(FEATURE_DIR):
        print(f"Error: Feature directory not found at '{FEATURE_DIR}'.")
        print("Please ensure your .pt feature files are located there.")
        return

    feature_files = [f for f in os.listdir(FEATURE_DIR) if f.endswith('.pt')]
    if not feature_files:
        print(f"No .pt files found in '{FEATURE_DIR}'.")
        return

    # 按文件名前缀对文件进行分组
    # e.g., 'lora_deepseek_code_semantic_stirling.pt' -> prefix 'lora_deepseek_code_semantic'
    file_groups = defaultdict(list)
    for file_name in feature_files:
        # 稳健地提取前缀
        parts = os.path.splitext(file_name)[0].split('_')
        if len(parts) > 1:
            prefix = '_'.join(parts[0:4])
            file_groups[prefix].append(file_name)
        else:
            # 如果文件名中没有下划线，则将整个文件名（无扩展名）作为组名
            file_groups[parts[0]].append(file_name)

    print(f"Found {len(file_groups)} file group(s): {list(file_groups.keys())}")
    all_results = []
    classifiers_to_test = ['MLP']

    for group_prefix, files_in_group in file_groups.items():
        print(f"\nProcessing group: '{group_prefix}' ({len(files_in_group)} file(s))")

        # 分别聚合训练和测试数据
        train_features = defaultdict(list)
        train_labels_list = []
        test_features = defaultdict(list)
        test_labels_list = []
        model_name_from_meta = "Unknown"  # 假定组内模型名称一致

        for file_name in files_in_group:
            print(f"  - Loading file: {file_name}")
            file_path = os.path.join(FEATURE_DIR, file_name)
            try:
                metadata, features_dict, labels = load_data(file_path)
                model_name_from_meta = metadata.get('model_name', model_name_from_meta)

                # 根据文件名判断是训练集还是测试集
                if 'train' in file_name:
                    for feature_type, features in features_dict.items():
                        train_features[feature_type].append(features)
                    train_labels_list.append(labels)
                else:
                    for feature_type, features in features_dict.items():
                        test_features[feature_type].append(features)
                    test_labels_list.append(labels)

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue

        if not train_labels_list or not test_labels_list:
            print(f"Warning: Missing train or test data for group '{group_prefix}'. Skipping.")
            continue

        # 将列表中的张量合并成一个大张量
        final_train_labels = torch.cat(train_labels_list, dim=0)
        final_test_labels = torch.cat(test_labels_list, dim=0)
        
        final_train_features_dict = {
            ft: torch.cat(tensors, dim=0) for ft, tensors in train_features.items() if tensors
        }
        final_test_features_dict = {
            ft: torch.cat(tensors, dim=0) for ft, tensors in test_features.items() if tensors
        }

        # 确保训练和测试数据有共同的特征类型
        common_feature_types = set(final_train_features_dict.keys()) & set(final_test_features_dict.keys())

        if not common_feature_types:
            print(f"Warning: No common feature types between train and test sets for group '{group_prefix}'. Skipping.")
            continue

        # 对聚合后的数据进行实验
        for feature_type in sorted(list(common_feature_types)):
            X_train = final_train_features_dict[feature_type].to(DEVICE)
            y_train = final_train_labels.to(DEVICE)
            X_test = final_test_features_dict[feature_type].to(DEVICE)
            y_test = final_test_labels.to(DEVICE)

            print(f"\n  - Testing aggregated feature type: '{feature_type}'")
            print(f"    Train size: {X_train.shape}, Test size: {X_test.shape}")

            for classifier_name in classifiers_to_test:
                print(f"    - Using classifier: {classifier_name}")
                
                result = run_experiment(model_name_from_meta, group_prefix, feature_type, classifier_name, X_train, y_train, X_test, y_test)
                all_results.append(result)
                print(f"      Metrics: {result}")

    if not all_results:
        print("\nNo experiments were successfully run. No results to display.")
        return

    # 将结果保存到DataFrame并输出
    results_df = pd.DataFrame(all_results)
    print("\n\n--- Aggregated Experiment Results ---")
    print(results_df.to_string())
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == '__main__':
    main() 