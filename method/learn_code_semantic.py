"""
使用大模型中间层hidden_states表示的代码语义，训练一个二分类模型，并计算预测的指标。

本脚本旨在实现以下实验目标：
1.  对比不同预训练模型（如 CodeBERT, GraphCodeBERT, UniXcoder等）提取的代码语义特征对下游任务的影响。
2.  探究模型不同层级（深、中、浅层）的hidden_states在表征代码语义时的效果差异。
3.  比较不同分类模型（如简单的神经网络、传统机器学习方法）在基于代码语义进行分类时的性能。

数据来源：
-   特征数据位于 `./semantic_features` 目录下，以`.pt`文件格式存储。
-   每个文件包含了从特定模型、特定代码项目上提取的特征。
-   文件名约定：`微调方式-模型-测试项目.pt`
-   `.pt` 文件内容: 一个字典，包含 'metadata', 'features', 'labels'。
    -   'features' 是一个字典，key为特征类型（如 'shallow_avg_pooling', 'last_token'），value是对应的特征张量。

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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# --- 1. 配置区域 ---
FEATURE_DIR = '/model/lxj/LLM_comment_generate/temp/semantic_features'
RESULTS_FILE = '/model/lxj/LLM_comment_generate/temp/hidden_states_`results.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


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
    def __init__(self, input_dim, hidden_dim=248):
        BaseModel.__init__(self, "MLP")
        nn.Module.__init__(self)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
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

def run_experiment(model_name, project_name, feature_type, classifier_name, X, Y, x, y):
    """
    运行单次实验（训练、评估、返回结果）。

    Args:
        model_name (str): 预训练模型的名称。
        project_name (str): 测试项目的名称。
        feature_type (str): 特征提取的类型。
        classifier_name (str): 分类器的名称。
        X (torch.Tensor): 特征数据。
        y (torch.Tensor): 标签数据。

    Returns:
        dict: 本次实验的结果。
    """
    X_train = X
    Y_train = Y
    X_test = x
    Y_test = y

    # 根据分类器名称选择模型
    if classifier_name == 'MLP':
        input_dim = X_train.shape[1]
        classifier = MLPClassifier(input_dim=input_dim)
        classifier.train_model(X_train, Y_train)
        predictions = classifier.predict_model(X_test)
    # elif classifier_name == 'LogisticRegression':
    #     # 这里可以实例化和训练 sklearn 模型
    #     pass
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
        
    y_test_np = Y_test.cpu().numpy()
    metrics = calculate_metrics(y_test_np, predictions)
    
    result = {
        'Pretrained Model': model_name,
        'Project': project_name,
        'Feature Type': feature_type,
        'Classifier': classifier_name,
        **metrics
    }
    return result

# --- 5. 主流程 ---
def main():
    """
    主函数，编排整个实验流程。
    """
    print("Starting experiments...")
    if not os.path.exists(FEATURE_DIR):
        print(f"Error: Feature directory not found at '{FEATURE_DIR}'.")
        print("Please ensure your .pt feature files are located there.")
        return

    feature_files = [f for f in os.listdir(FEATURE_DIR) if f.endswith('train.pt')]
    if not feature_files:
        print(f"No .pt files found in '{FEATURE_DIR}'.")
        return

    all_results = []
    
    # 定义要进行实验的分类器列表
    classifiers_to_test = ['MLP'] #, 'LogisticRegression']

    for file_name in feature_files:
        print(f"\nProcessing file: {file_name}...")
        # 从文件名解析项目名称，假设格式为 '..._项目名.pt'
        # 例如 'lora_deepseek_code_semantic_stirling.pt' -> 'stirling'
        project_name = os.path.splitext(file_name)[0].split('_')[-2]
        train_file_path = os.path.join(FEATURE_DIR, file_name)
        test_file_path = os.path.join(FEATURE_DIR, file_name.replace('_train', ''))
        
        try:
            metadata_train, features_dict_train, labels_train = load_data(train_file_path)
            metadata_test, features_dict_test, labels_test = load_data(test_file_path)
            model_name = metadata_train.get('model_name', 'Unknown')
            
            for feature_type, features_train in features_dict_train.items():
                if feature_type not in features_dict_test:
                    print(f"  - Skipping feature type {feature_type}: not found in test data.")
                    continue
            
                features_test = features_dict_test[feature_type]

                print(f"  - Testing feature type: {feature_type}")
                for classifier_name in classifiers_to_test:
                    print(f"    - Using classifier: {classifier_name}")
                    
                    # 确保特征和标签在同一设备上
                    features_train = features_train.to(DEVICE)
                    labels_train = labels_train.to(DEVICE)
                    features_test = features_test.to(DEVICE)
                    labels_test = labels_test.to(DEVICE)

                    result = run_experiment(model_name, project_name, feature_type, classifier_name, features_train, labels_train, features_test, labels_test)
                    all_results.append(result)
                    print(f"      Metrics: {result}")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            continue
            
    if not all_results:
        print("\nNo experiments were successfully run. No results to display.")
        return

    # 将结果保存到DataFrame并输出
    results_df = pd.DataFrame(all_results)
    print("\n--- Experiment Results ---")
    print(results_df)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == '__main__':
    main()


