import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
import networkx as nx
from tqdm import tqdm
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN

# 根据您的graphml文件分析得出的所有可能的数值型节点特征
# 这样可以保证即使某些节点缺少某些属性，特征向量的维度也是一致的
NODE_FEATURE_KEYS = [
    "cbo","cboModified","fanin","fanout","wmc","dit","noc","rfc","lcom","lcom*","tcc","lcc","totalMethodsQty","staticMethodsQty","publicMethodsQty","privateMethodsQty","protectedMethodsQty","defaultMethodsQty","visibleMethodsQty","abstractMethodsQty","finalMethodsQty","synchronizedMethodsQty","totalFieldsQty","staticFieldsQty","publicFieldsQty","privateFieldsQty","protectedFieldsQty","defaultFieldsQty","finalFieldsQty","synchronizedFieldsQty","nosi","loc","returnQty","loopQty","comparisonsQty","tryCatchQty","parenthesizedExpsQty","stringLiteralsQty","numbersQty","assignmentsQty","mathOperationsQty","variablesQty","maxNestedBlocksQty","anonymousClassesQty","innerClassesQty","lambdasQty","uniqueWordsQty","modifiers","logStatementsQty",
    "line","cbo","cboModified","fanin","fanout","wmc","rfc","loc","returnsQty","variablesQty","parametersQty","methodsInvokedQty","methodsInvokedLocalQty","methodsInvokedIndirectLocalQty","loopQty","comparisonsQty","tryCatchQty","parenthesizedExpsQty","stringLiteralsQty","numbersQty","assignmentsQty","mathOperationsQty","maxNestedBlocksQty","anonymousClassesQty","innerClassesQty","lambdasQty","uniqueWordsQty","modifiers","logStatementsQty"
]

def parse_graphml(graphml_path):
    """使用networkx加载并解析graphml文件"""
    try:
        # networkx可以很好地处理graphml中data key的嵌套
        graph_nx = nx.read_graphml(graphml_path)
        return graph_nx
    except Exception as e:
        print(f"Error parsing {graphml_path}: {e}")
        return None

def extract_features(graph_nx):
    """从networkx图中提取节点特征和边索引"""
    # 建立从node_id到整数索引的映射，因为PyG需要从0开始的连续索引
    node_mapping = {node_id: i for i, node_id in enumerate(graph_nx.nodes())}
    # 提取节点特征
    x = []
    for node_id, node_data in graph_nx.nodes(data=True):
        features = []
        for key in NODE_FEATURE_KEYS:
            # 如果节点有该属性，则使用其值；否则用0填充
            # 确保所有值都是数值型
            value = node_data.get(key, 0)
            try:
                v = float(value)
                if pd.isna(v) or (isinstance(v, float) and (v != v)):  # 检查nan
                    v = 0.0
                features.append(v)
            except (ValueError, TypeError):
                features.append(0.0) # 如果转换失败，也填充0
        x.append(features)
    
    x = torch.tensor(x, dtype=torch.float)

    # 提取边索引
    edge_index = []
    for u, v in graph_nx.edges():
        edge_index.append([node_mapping[u], node_mapping[v]])
    
    # PyG需要shape为[2, num_edges]的edge_index
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return x, edge_index

class CodeSmellDataset(Dataset):
    def __init__(self, csv_dir, graph_root_dir, split='train', project=None, transform=None, pre_transform=None):
        """
        Args:
            csv_dir (string): 存放所有csv文件的目录.
            graph_root_dir (string): 存放所有项目GraphML文件的根目录.
            split (string): 'train' or 'test'，用于筛选数据.
            project (string, optional): 指定要加载的项目，例如'cassandra'。如果为None，则加载所有项目的数据。
        """
        super().__init__(None, transform, pre_transform)
        self.graph_root_dir = graph_root_dir
        self.split = split
        self.project = project
        
        # 1. 加载所有CSV文件或指定项目的CSV文件
        if project is None:
            # 加载所有CSV文件并合并
            all_csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('_Split.csv')]
            df_list = [pd.read_csv(f) for f in all_csv_files]
            self.smell_data = pd.concat(df_list, ignore_index=True)
        else:
            # 只加载指定项目的CSV文件
            csv_file = os.path.join(csv_dir, f"{project}_Split.csv")
            if os.path.exists(csv_file):
                self.smell_data = pd.read_csv(csv_file)
            else:
                raise FileNotFoundError(f"CSV文件不存在: {csv_file}")
        
        # 2. 根据split筛选数据
        self.smell_data = self.smell_data[self.smell_data['dataset_split'] == self.split].reset_index(drop=True)
        
        # 3. 预先检查哪些graphml文件存在，避免在__getitem__中处理缺失文件
        project_name = project if project else "所有项目"
        print(f"检查{project_name}在{self.split}拆分中存在的图文件...")
        self.existing_indices = []
        for idx, row in tqdm(self.smell_data.iterrows(), total=self.smell_data.shape[0]):
            graph_path = self._get_graph_path(row)
            if os.path.exists(graph_path):
                self.existing_indices.append(idx)
        
        print(f"找到 {len(self.existing_indices)}/{len(self.smell_data)} 可用的图，用于'{self.split}'拆分。")
        self.smell_data = self.smell_data.iloc[self.existing_indices].reset_index(drop=True)

        self.graphs = []
        for idx, row in tqdm(self.smell_data.iterrows(), total=self.smell_data.shape[0], desc='预处理图'):
            graph_path = self._get_graph_path(row)
            graph_nx = parse_graphml(graph_path)
            if graph_nx is None or len(graph_nx.nodes) == 0:
                self.graphs.append(None)
                continue
            x, edge_index = extract_features(graph_nx)
            y = torch.tensor([row['actionable']], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, y=y)
            self.graphs.append(data)

    def _get_graph_path(self, row):
        # 构建graphml文件的完整路径
        # 路径格式: /path/to/graphml_root/{Project}/{unique_id}.graphml
        return os.path.join(self.graph_root_dir, row['Project'], f"{row['unique_id']}.graphml")

    def len(self):
        return len(self.smell_data)

    def get(self, idx):
        return self.graphs[idx]

class ImbalanceHandler:
    """处理数据不平衡的类，提供各种重采样方法"""
    
    @staticmethod
    def get_available_methods():
        """返回所有可用的平衡方法"""
        return [
            'none',  # 不进行平衡处理
            'random_oversampling',  # 随机过采样
            'random_undersampling',  # 随机欠采样
            'smote',  # SMOTE过采样
            'adasyn',  # ADASYN过采样
            'tomek_links',  # Tomek Links欠采样
            'near_miss',  # NearMiss欠采样
            'smote_tomek',  # SMOTE + Tomek Links混合采样
            'smote_enn'  # SMOTE + ENN混合采样
        ]
    
    @staticmethod
    def apply_resampling(X, y, method='none', random_state=42):
        """
        对数据应用重采样方法
        
        参数:
        X: 特征矩阵
        y: 标签数组
        method: 重采样方法
        random_state: 随机种子
        
        返回:
        X_resampled, y_resampled: 重采样后的特征和标签
        """
        if method == 'none' or len(np.unique(y)) <= 1:
            return X, y
            
        try:
            if method == 'random_oversampling':
                # 随机过采样
                # 获取少数类和多数类的索引
                minority_class = np.argmin(np.bincount(y))
                majority_class = np.argmax(np.bincount(y))
                
                # 分别获取少数类和多数类的样本
                X_minority = X[y == minority_class]
                y_minority = y[y == minority_class]
                X_majority = X[y == majority_class]
                y_majority = y[y == majority_class]
                
                # 对少数类进行过采样，使其数量与多数类相同
                X_minority_resampled, y_minority_resampled = resample(
                    X_minority, y_minority,
                    replace=True,
                    n_samples=len(X_majority),
                    random_state=random_state
                )
                
                # 合并过采样后的少数类与多数类
                X_resampled = np.vstack((X_majority, X_minority_resampled))
                y_resampled = np.hstack((y_majority, y_minority_resampled))
                
                return X_resampled, y_resampled
                
            elif method == 'random_undersampling':
                # 随机欠采样
                rus = RandomUnderSampler(random_state=random_state)
                return rus.fit_resample(X, y)
                
            elif method == 'smote':
                # SMOTE过采样
                smote = SMOTE(random_state=random_state)
                return smote.fit_resample(X, y)
                
            elif method == 'adasyn':
                # ADASYN过采样
                adasyn = ADASYN(random_state=random_state)
                return adasyn.fit_resample(X, y)
                
            elif method == 'tomek_links':
                # Tomek Links欠采样
                tomek = TomekLinks()
                return tomek.fit_resample(X, y)
                
            elif method == 'near_miss':
                # NearMiss欠采样
                nm = NearMiss(version=3)  # 使用NearMiss-3版本
                return nm.fit_resample(X, y)
                
            elif method == 'smote_tomek':
                # SMOTE + Tomek Links混合采样
                smt = SMOTETomek(random_state=random_state)
                return smt.fit_resample(X, y)
                
            elif method == 'smote_enn':
                # SMOTE + ENN混合采样
                smote_enn = SMOTEENN(random_state=random_state)
                return smote_enn.fit_resample(X, y)
                
            else:
                print(f"未知的重采样方法: {method}，不进行处理")
                return X, y
                
        except Exception as e:
            print(f"应用重采样方法 {method} 时出错: {e}")
            print("返回原始数据")
            return X, y


class BalancedCodeSmellDataset(Dataset):
    """带有数据平衡处理的代码气味数据集"""
    
    def __init__(self, base_dataset, balance_method='none', random_state=42):
        """
        初始化平衡处理后的数据集
        
        参数:
        base_dataset: CodeSmellDataset实例
        balance_method: 平衡方法，默认为'none'（不进行处理）
        random_state: 随机种子
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.balance_method = balance_method
        self.random_state = random_state
        
        # 如果不需要平衡处理，直接使用原始数据集
        if balance_method == 'none':
            self.balanced_indices = list(range(len(base_dataset)))
            self.graphs = base_dataset.graphs
            return
            
        # 收集所有非空图的索引和标签
        valid_indices = []
        labels = []
        for i in range(len(base_dataset)):
            data = base_dataset.get(i)
            if data is not None:
                valid_indices.append(i)
                labels.append(data.y.item())
        
        if len(valid_indices) == 0:
            print("警告：没有有效的图数据")
            self.balanced_indices = []
            self.graphs = []
            return
            
        # 转换为NumPy数组以便于处理
        indices_array = np.array(valid_indices).reshape(-1, 1)  # 转为列向量
        labels_array = np.array(labels)
        
        # 应用重采样方法
        print(f"应用平衡方法 '{balance_method}' 到数据集...")
        print(f"原始类别分布：{np.bincount(labels_array)}")
        
        try:
            # 对索引进行重采样（而不是特征）
            resampled_indices, _ = ImbalanceHandler.apply_resampling(
                indices_array, labels_array,
                method=balance_method,
                random_state=random_state
            )
            
            # 展平并转换为列表
            self.balanced_indices = resampled_indices.flatten().tolist()
            
            # 收集重采样后的图数据
            self.graphs = [base_dataset.get(idx) for idx in self.balanced_indices]
            
            # 统计平衡后的标签分布
            balanced_labels = [base_dataset.get(idx).y.item() for idx in self.balanced_indices]
            print(f"平衡后类别分布：{np.bincount(balanced_labels)}")
            
        except Exception as e:
            print(f"平衡数据时出错: {e}")
            print("使用原始数据集")
            self.balanced_indices = valid_indices
            self.graphs = [base_dataset.get(idx) for idx in valid_indices]
    
    def len(self):
        return len(self.balanced_indices)
        
    def get(self, idx):
        if idx >= len(self.graphs):
            return None
        return self.graphs[idx]

if __name__ == "__main__":
    # 1. 加载数据集
    print("Loading training data...")
    train_dataset = CodeSmellDataset(csv_dir='/model/lxj/actionableSmell', graph_root_dir='/model/data/R-SE/tools/graphgen', split='train')
    print("Loaded")
    # 测试 get 方法
    print("Testing get(0)...")
    sample = train_dataset.get(0)
    print(sample)