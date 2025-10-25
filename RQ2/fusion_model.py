"""
融合代码语义特征与图结构特征进行代码气味可操作性分类

本脚本实现了一个融合模型，将大模型提取的代码语义特征与图神经网络提取的代码结构特征进行深度融合，
用于代码气味的可操作性（Actionability）分类。

主要功能：
1. 加载并处理两种特征：从PT文件加载语义特征，从GraphML文件构建并提取图结构特征
2. 使用多种注意力机制实现特征融合
3. 利用融合特征进行二分类任务
4. 评估模型性能并输出多种评估指标

作者: 
日期: 2025-07-18
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import networkx as nx
import argparse
import fusion_implement
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score

from torch_geometric.nn import GINConv, BatchNorm, global_mean_pool, global_max_pool
import torch_geometric.data as geo_data
from torch_geometric.loader import DataLoader as PyGDataLoader # 导入PyG的DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# --- 配置区域 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
EPOCHS = 100
SEMANTIC_FEATURE_TYPE = 'deep_avg_pooling'  # 根据需求可更改
GRAPH_HIDDEN_DIM = 128
FUSION_HIDDEN_DIM = 256
DROPOUT_RATE = 0.4

# 设置随机种子，保证结果可复现
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# --- 数据处理模块 ---

def parse_graphml(graphml_path):
    """加载并解析graphml文件"""
    try:
        graph_nx = nx.read_graphml(graphml_path)
        return graph_nx
    except Exception as e:
        print(f"Error parsing {graphml_path}: {e}")
        return None

def extract_node_features(graph_nx):
    """从networkx图中提取节点特征"""
    # 这里定义要提取的特征
    feature_keys = [
        "cbo","cboModified","fanin","fanout","wmc","dit","noc","rfc","lcom","lcom*","tcc","lcc","totalMethodsQty","staticMethodsQty","publicMethodsQty","privateMethodsQty","protectedMethodsQty","defaultMethodsQty","visibleMethodsQty","abstractMethodsQty","finalMethodsQty","synchronizedMethodsQty","totalFieldsQty","staticFieldsQty","publicFieldsQty","privateFieldsQty","protectedFieldsQty","defaultFieldsQty","finalFieldsQty","synchronizedFieldsQty","nosi","loc","returnQty","loopQty","comparisonsQty","tryCatchQty","parenthesizedExpsQty","stringLiteralsQty","numbersQty","assignmentsQty","mathOperationsQty","variablesQty","maxNestedBlocksQty","anonymousClassesQty","innerClassesQty","lambdasQty","uniqueWordsQty","modifiers","logStatementsQty",
        "line","cbo","cboModified","fanin","fanout","wmc","rfc","loc","returnsQty","variablesQty","parametersQty","methodsInvokedQty","methodsInvokedLocalQty","methodsInvokedIndirectLocalQty","loopQty","comparisonsQty","tryCatchQty","parenthesizedExpsQty","stringLiteralsQty","numbersQty","assignmentsQty","mathOperationsQty","maxNestedBlocksQty","anonymousClassesQty","innerClassesQty","lambdasQty","uniqueWordsQty","modifiers","logStatementsQty"
    ]
    
    # 提取节点特征
    node_features = []
    for _, node_data in graph_nx.nodes(data=True):
        features = []
        for key in feature_keys:
            # 如果节点有该属性则使用，否则填充0
            value = node_data.get(key, 0)
            try:
                v = float(value)
                if pd.isna(v) or (isinstance(v, float) and (v != v)):  # 检查nan
                    v = 0.0
                features.append(v)
            except (ValueError, TypeError):
                features.append(0.0)
        node_features.append(features)
    
    # 如果没有节点，返回一个默认的特征向量
    if not node_features:
        return torch.zeros((1, len(feature_keys)), dtype=torch.float)
    
    return torch.tensor(node_features, dtype=torch.float)

def extract_edges(graph_nx):
    """从networkx图中提取边索引"""
    # 构建节点ID到索引的映射
    node_mapping = {node: idx for idx, node in enumerate(graph_nx.nodes())}
    
    # 提取边索引
    edge_index = []
    for u, v in graph_nx.edges():
        edge_index.append([node_mapping[u], node_mapping[v]])
    
    # 如果没有边，返回一个空的边索引
    if not edge_index:
        return torch.zeros((2, 0), dtype=torch.long)
    
    # PyG需要shape为[2, num_edges]的edge_index
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def create_pyg_data(graph_nx):
    """将networkx图转换为PyG的Data对象"""
    if not graph_nx or len(graph_nx.nodes()) == 0:
        # 获取feature_keys的长度，用于创建合适维度的空特征
        from inspect import getsource
        feature_keys_len = len(extract_node_features.__code__.co_consts[1])
        
        # 处理空图的情况
        x = torch.zeros((1, feature_keys_len), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return geo_data.Data(x=x, edge_index=edge_index)
    
    x = extract_node_features(graph_nx)
    edge_index = extract_edges(graph_nx)
    return geo_data.Data(x=x, edge_index=edge_index)

def load_semantic_features(file_path):
    """从.pt文件中加载语义特征"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    data = torch.load(file_path, map_location=DEVICE)
    metadata = data.get('metadata', {})
    features = data.get('features', {}).get(SEMANTIC_FEATURE_TYPE, None)
    
    if features is None:
        raise ValueError(f"Feature type '{SEMANTIC_FEATURE_TYPE}' not found in file")
    
    # 注意：这里没有直接使用labels，因为我们需要根据unique_ids来匹配CSV中的标签
    unique_ids = metadata.get('unique_ids', [])
    
    return features, unique_ids

def load_csv_data(csv_path):
    """加载CSV文件中的标签和项目分类信息"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    # 保留必要的列：unique_id, actionable, dataset_split
    return df[['unique_id', 'actionable', 'dataset_split']]


class FusionDataset(Dataset):
    """融合数据集类，结合语义特征和图特征"""
    
    def __init__(self, project_name, model_name, semantic_feature_dir, csv_dir, graph_root_dir, split='train'):
        """
        初始化融合数据集
        
        Args:
            project_name (str): 项目名称，如"cassandra"
            model_name (str)：模型名称
            semantic_feature_dir (str): 语义特征文件目录
            csv_dir (str): CSV文件目录
            graph_root_dir (str): 图文件根目录
            split (str): 'train'或'test'
        """
        self.project_name = project_name
        self.split = split
        self.graph_root_dir = graph_root_dir
        self.semantic_dim = None  # 将在加载数据后设置
        
        # 1. 加载CSV数据
        csv_path = os.path.join(csv_dir, f"{project_name}_Split.csv")
        self.df = load_csv_data(csv_path)
        
        # 2. 只保留指定split的数据
        self.df = self.df[self.df['dataset_split'] == split].reset_index(drop=True)
        
        # 3. 加载语义特征
        # 查找匹配的特征文件
        if split:
            feature_files = [f for f in os.listdir(semantic_feature_dir) 
                            if f.endswith(f'{project_name}_{split}.pt') and model_name in f]
        # else:
        #     feature_files = [f for f in os.listdir(semantic_feature_dir) 
        #                     if f.endswith(f'{project_name}.pt') and model_name in f]
        
        if not feature_files:
            raise FileNotFoundError(f"No semantic feature file found for project: {project_name}")
            
        # 使用找到的第一个文件
        semantic_path = os.path.join(semantic_feature_dir, feature_files[0])
        self.semantic_features, self.unique_ids = load_semantic_features(semantic_path)
        
        # 自动获取语义特征的维度
        self.semantic_dim = self.semantic_features.shape[1] if len(self.semantic_features.shape) >= 2 else 0
        print(f"Semantic feature dimension automatically detected: {self.semantic_dim}")
        
        # 4. 建立unique_id到索引的映射
        self.id_to_index = {id_: idx for idx, id_ in enumerate(self.unique_ids)}
        
        # 5. 筛选出同时存在于CSV和特征中的数据，并确保特征与DataFrame对齐
        
        # 筛选DataFrame，只保留那些在语义特征和图文件中都存在的样本
        valid_df_indices = []
        valid_semantic_indices = []
        
        for idx, row in self.df.iterrows():
            unique_id = row['unique_id']
            graph_path = os.path.join(graph_root_dir, project_name, f"{unique_id}.graphml")
            
            if unique_id in self.id_to_index and os.path.exists(graph_path):
                valid_df_indices.append(idx)
                valid_semantic_indices.append(self.id_to_index[unique_id])
        
        self.df = self.df.iloc[valid_df_indices].reset_index(drop=True)
        
        # 根据筛选后的顺序，重新排列语义特征张量，确保一一对应
        self.semantic_features = self.semantic_features[valid_semantic_indices]
        
        # 更新id_to_index，现在它将是简单的顺序索引
        self.id_to_index = {id_: idx for idx, id_ in enumerate(self.df['unique_id'])}
        
        print(f"Found {len(self.df)} valid samples for {project_name} ({split})")
        
        # 预加载图数据以提高效率
        self.graph_data_cache = {}
        for unique_id in tqdm(self.df['unique_id'], desc=f"Loading graphs for {project_name}"):
            self.graph_data_cache[unique_id] = self._load_graph(unique_id)
    
    def _load_graph(self, unique_id):
        """加载单个图数据"""
        graph_path = os.path.join(self.graph_root_dir, self.project_name, f"{unique_id}.graphml")
        graph_nx = parse_graphml(graph_path)
        if graph_nx is None:
            return None
        return create_pyg_data(graph_nx)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        unique_id = row['unique_id']
        label = row['actionable']
        
        # 获取语义特征
        semantic_idx = self.id_to_index[unique_id]
        semantic_feature = self.semantic_features[semantic_idx]
        
        # 获取图数据 (从缓存中)
        graph_data = self.graph_data_cache.get(unique_id)
        if graph_data is None:
            # 如果缓存中没有，尝试加载
            graph_data = self._load_graph(unique_id)
            if graph_data is None:
                # ...创建空图的代码...
                pass
        
        # 只将标签附加到Data对象
        graph_data.y = torch.tensor(label, dtype=torch.long)
        graph_data.unique_id = unique_id
        # 返回图数据和语义特征作为元组
        return graph_data, semantic_feature

def custom_collate(batch):
    """
    自定义的collate函数，分别处理图数据和语义特征
    """
    graphs = [item[0] for item in batch]
    semantic_features = torch.stack([item[1] for item in batch])
    
    # 使用PyG的Batch.from_data_list函数处理图数据
    batched_graphs = geo_data.Batch.from_data_list(graphs)
    
    return batched_graphs, semantic_features
# --- 模型定义 ---

class GINModel(nn.Module):
    """GIN图神经网络模型，用于提取图结构特征"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # 记录输入维度
        self.input_dim = input_dim
        
        # GIN层
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        
        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        )
        
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(output_dim)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 如果batch为None，说明只有一个图，手动创建batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 图卷积层
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        
        # 图池化，获取图级表示
        x = global_mean_pool(x, batch)
        
        return x


class FusionClassifier(nn.Module):
    """融合分类器，将语义特征和图特征融合后进行分类"""
    
    # 类变量，用于存储图节点特征的维度
    gin_input_dim = 78  # 默认值，将在运行时根据实际数据更新
    
    def __init__(self, semantic_dim, graph_dim, fusion_dim, fusion_type='self_attention'):
        super().__init__()
        
        # GIN模型，用于图特征提取
        self.gin = GINModel(input_dim=self.gin_input_dim, hidden_dim=GRAPH_HIDDEN_DIM, output_dim=graph_dim)
        
        # 融合层
        if fusion_type == 'simple':
            self.fusion = fusion_implement.SimpleFusion(semantic_dim, graph_dim, fusion_dim)
        elif fusion_type == 'self_attention':
            self.fusion = fusion_implement.SelfAttentionFusion(semantic_dim, graph_dim, fusion_dim)
        elif fusion_type == 'cross_attention':
            self.fusion = fusion_implement.CrossAttentionFusion(semantic_dim, graph_dim, fusion_dim)
        elif fusion_type == 'gate':
            self.fusion = fusion_implement.GateFusion(semantic_dim, graph_dim, fusion_dim)
        elif fusion_type == 'multihead':
            self.fusion = fusion_implement.MultiHeadAttentionFusion(semantic_dim, graph_dim, fusion_dim)
        elif fusion_type == 'balance':
            self.fusion = fusion_implement.BalancedFusion(semantic_dim, graph_dim, fusion_dim)
        elif fusion_type == 'dynamic':
            self.fusion = fusion_implement.DynamicWeightedFusion(semantic_dim, graph_dim, fusion_dim)
        elif fusion_type == 'progress':
            self.fusion = fusion_implement.ProgressiveFusion(semantic_dim, graph_dim, fusion_dim)
        elif fusion_type == 'enhance':
            self.fusion = fusion_implement.EnhancedCrossAttention(semantic_dim, graph_dim, fusion_dim)
        elif fusion_type == 'FiLM':
            self.fusion = fusion_implement.FiLMFusion(semantic_dim, graph_dim, fusion_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
            
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(fusion_dim // 2, 2)  # 二分类
        )

    def forward(self, semantic_feature, graph_data):
        # 检查semantic_feature是否为批次列表（由follow_batch生成）
        if hasattr(graph_data, 'semantic_feature_batch'):
            # 提取batch中的所有semantic_feature
            semantic_feature = graph_data.semantic_feature
        
        # graph_data现在是一个PyG的Batch对象，可以直接传递给GIN
        graph_feature = self.gin(graph_data)
        
        # 特征融合
        fused_feature = self.fusion(semantic_feature, graph_feature)
        
        # 分类
        logits = self.classifier(fused_feature)
        return logits


# --- 训练与评估函数 ---

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_graph, batch_semantic in tqdm(dataloader, desc="Training", leave=False):
        # 将两种数据都移动到目标设备
        batch_graph = batch_graph.to(device)
        batch_semantic = batch_semantic.to(device)
        
        optimizer.zero_grad()
        # 从Batch对象中提取语义特征和图数据
        outputs = model(batch_semantic, batch_graph)
        loss = criterion(outputs, batch_graph.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_semantic.size(0)
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_graph.y.cpu().numpy())
    
    
    # 计算指标
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = total_loss / len(dataloader.dataset)
    
    return metrics

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_unique_ids = []
    
    with torch.no_grad():
        for batch_graph, batch_semantic in tqdm(dataloader, desc="Evaluating", leave=False):
            batch_graph = batch_graph.to(device)
            batch_semantic = batch_semantic.to(device)
            
            outputs = model(batch_semantic, batch_graph)
            loss = criterion(outputs, batch_graph.y)
            
            total_loss += loss.item() * batch_semantic.size(0)
            
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_graph.y.cpu().numpy())
            all_unique_ids.extend(batch_graph.unique_id)
    
    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 添加类别统计
    positive_count = np.sum(all_labels)
    negative_count = len(all_labels) - positive_count
    predicted_positive = np.sum(all_preds)
    print(f"类别分布 - 实际: 正例={positive_count}, 负例={negative_count} | 预测: 正例={predicted_positive}, 负例={len(all_preds)-predicted_positive}")
    # # 监控每个样本的预测概率分布
    # probs_mean = np.mean(all_probs)
    # probs_std = np.std(all_probs)
    # print(f"预测概率分布: 均值={probs_mean:.4f}, 标准差={probs_std:.4f}")

    # 计算指标
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(dataloader.dataset)
    
    # 计算额外的指标
    metrics['auc'] = roc_auc_score(all_labels, all_probs)
    metrics['ap'] = average_precision_score(all_labels, all_probs)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵:")
    print(cm)

    predictions_df = pd.DataFrame({
        'unique_id': all_unique_ids,
        'true_label': all_labels,
        'predicted_label': all_preds
    })

    return metrics, predictions_df

def calculate_metrics(y_true, y_pred):
    """计算并返回评估指标"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }


# --- 主函数 ---

def main():
    parser = argparse.ArgumentParser(description='融合代码语义特征与图结构特征进行分类')
    parser.add_argument('--semantic_dir', type=str, default='/model/lxj/LLM_comment_generate/temp/semantic_features', 
                        help='语义特征目录')
    parser.add_argument('--csv_dir', type=str, default='/model/lxj/actionableSmell', 
                        help='CSV文件目录')
    parser.add_argument('--graph_dir', type=str, default='/model/data/R-SE/tools/graphgen', 
                        help='图文件根目录')
    parser.add_argument('--project', type=str, default='jsoup', 
                    help='要处理的项目名称，如"cassandra"。如all_projects，将加载所有项目。')
    parser.add_argument('--model_name', type=str, default='lora_phi4', 
                        help='要处理的模型名称')
    parser.add_argument('--fusion_type', type=str, default='progress', 
                        help='特征融合方式')
    parser.add_argument('--semantic_dim', type=int, default=None, 
                        help='语义特征维度 (如不指定，将自动从数据中检测)')
    parser.add_argument('--graph_dim', type=int, default=64, 
                        help='图特征维度 (GIN输出维度)')
    parser.add_argument('--fusion_dim', type=int, default=256, 
                        help='融合特征维度')
    parser.add_argument('--epochs', type=int, default=EPOCHS, 
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, 
                        help='批量大小')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, 
                        help='学习率')
    parser.add_argument('--output_dir', type=str, default='/model/lxj/LLM4ACS/RQ2/results1', 
                        help='结果输出目录')
    parser.add_argument('--early_stopping_patience', type=int, default=5, 
                    help='早停耐心值：连续多少个epoch指标没有改善则停止训练')
    parser.add_argument('--early_stopping_delta', type=float, default=0.001, 
                        help='早停阈值：指标改善需要超过该值才算有效改善')
    parser.add_argument('--early_stopping_metric', type=str, default='mcc', 
                        choices=['f1', 'accuracy', 'loss', 'auc', 'ap', 'mcc'],
                        help='早停监控指标')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    if args.project == 'all_projects':
        # 查找所有可用的项目
        csv_files = [f for f in os.listdir(args.csv_dir) if f.endswith("_Split.csv")]
        projects = [f.split("_Split.csv")[0] for f in csv_files]
        print(f"No project specified. Loading data for all {len(projects)} projects: {', '.join(projects)}")
    else:
        projects = [args.project]
        print(f"Loading data for specified project: {args.project}")

    # 创建用于存储所有数据的列表
    all_train_data = []
    all_test_data = []
    # 遍历加载每个项目的数据
    for project in projects:
        print(f"Loading datasets for project: {project}")
        try:
            # 加载训练数据
            train_dataset = FusionDataset(
                project_name=project,
                model_name=args.model_name,
                semantic_feature_dir=args.semantic_dir,
                csv_dir=args.csv_dir,
                graph_root_dir=args.graph_dir,
                split='train'
            )
            all_train_data.append(train_dataset)
            
            # 加载测试数据
            test_dataset = FusionDataset(
                project_name=project,
                model_name=args.model_name,
                semantic_feature_dir=args.semantic_dir,
                csv_dir=args.csv_dir,
                graph_root_dir=args.graph_dir,
                split='test'
            )
            all_test_data.append(test_dataset)
            
            print(f"Successfully loaded {project} - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        except Exception as e:
            print(f"Error loading data for project {project}: {e}")

    # 创建多项目组合数据集类
    class CombinedDataset(Dataset):
        def __init__(self, datasets):
            self.datasets = datasets
            self.dataset_lengths = [len(ds) for ds in datasets]
            self.cumulative_lengths = [0] + list(np.cumsum(self.dataset_lengths))
            self.semantic_dim = datasets[0].semantic_dim if datasets else 0
            
        def __len__(self):
            return sum(self.dataset_lengths)
        
        def __getitem__(self, idx):
            # 确定 idx 属于哪个数据集
            dataset_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
            local_idx = idx - self.cumulative_lengths[dataset_idx]
            return self.datasets[dataset_idx][local_idx]

    # 合并数据集
    if all_train_data and all_test_data:
        train_dataset = CombinedDataset(all_train_data)
        test_dataset = CombinedDataset(all_test_data)
        print(f"Combined dataset created. Total train: {len(train_dataset)}, Total test: {len(test_dataset)}")
    else:
        raise ValueError("No valid data was loaded. Check your data paths and projects.")

    # 创建数据加载器
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate
    )

    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate
    )
    
    print(f"Dataset loaded. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # 获取语义特征的维度
    semantic_dim = args.semantic_dim if args.semantic_dim is not None else train_dataset.semantic_dim
    
    # 获取图特征的输入维度，用于GIN模型的初始化
    # 从DataLoader中取一个样本批次来确定维度
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty. Check your data paths and splits.")
    sample_batch = next(iter(train_loader))
    node_feature_dim = sample_batch[0].x.shape[1]
    print(f"Graph node feature dimension detected: {node_feature_dim}")
    
    # 更新GINModel中的input_dim
    FusionClassifier.gin_input_dim = node_feature_dim
    
    if args.fusion_type is None:
        # fusion_type = ['simple','self_attention','cross_attention','gate','multihead','balance','dynamic','progress','enhance','FiLM']
        fusion_type = ['self_attention','cross_attention','gate','balance','dynamic','progress','enhance','FiLM']
    else:
        fusion_type = [args.fusion_type]
    for fusion in fusion_type:
        # 初始化模型
        model = FusionClassifier(
            semantic_dim=semantic_dim,
            graph_dim=args.graph_dim,
            fusion_dim=args.fusion_dim,
            fusion_type=fusion
        ).to(DEVICE)

        print(f"Model initialized. Using fusion type: {fusion}")
        print(f"Semantic dimension: {semantic_dim}, Graph dimension: {args.graph_dim}, Fusion dimension: {args.fusion_dim}")
        print(f"Running on device: {DEVICE}")

        # 定义损失函数和优化器
        # 添加类权重处理不平衡问题
        train_labels = np.array([data[0].y.item() for data in train_dataset])
        class_counts = np.bincount(train_labels)
        total_samples = len(train_labels)
        class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) for count in class_counts]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 使用更小的学习率
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

        # 添加学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        
        # 训练循环
        best_metric_value = 0.0
        best_f1 = 0.0
        results = []
        no_improve_count = 0
        monitor_better = lambda current, best: current > best
        if args.early_stopping_metric == 'loss':
            monitor_better = lambda current, best: current < best
            best_metric_value = float('inf')  # 对于loss，初始设为无穷大
        print(f"早停机制：监控指标 '{args.early_stopping_metric}'，耐心值 {args.early_stopping_patience}，阈值 {args.early_stopping_delta}")

        
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            
            # 训练
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, Pre: {train_metrics['precision']:.4f},"
                f"F1: {train_metrics['f1']:.4f}, MCC: {train_metrics['mcc']:.4f}")
            
            # 评估
            test_metrics, last_predictions_df = evaluate(model, test_loader, criterion, DEVICE)
            print(f"Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}, Pre: {test_metrics['precision']:.4f},"
                f"Rec: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}, MCC: {test_metrics['mcc']:.4f}, "
                f"AUC: {test_metrics['auc']:.4f}, AP: {test_metrics['ap']:.4f}")
            
            # 记录结果
            results.append({
                'epoch': epoch,
                'fusion_type': fusion,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'train_f1': train_metrics['f1'],
                'test_loss': test_metrics['loss'],
                'test_accuracy': test_metrics['accuracy'],
                'test_f1': test_metrics['f1'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_mcc': test_metrics['mcc'],
                'test_auc': test_metrics.get('auc', 0),
                'test_ap': test_metrics.get('ap', 0)
            })
            # current_metric_value = test_metrics[args.early_stopping_metric]
            # # 早停逻辑
            # if monitor_better(current_metric_value, best_metric_value + args.early_stopping_delta):
            #     # 指标有显著改善
            #     print(f"指标 {args.early_stopping_metric} 改善: {best_metric_value:.4f} -> {current_metric_value:.4f}")
            #     best_metric_value = current_metric_value
            #     no_improve_count = 0  # 重置计数器
            # else:
            #     # 指标没有显著改善
            #     no_improve_count += 1
            #     print(f"指标 {args.early_stopping_metric} 没有改善，已连续 {no_improve_count}/{args.early_stopping_patience} 轮")
                
            # # 检查是否触发早停
            # if no_improve_count >= args.early_stopping_patience:
            #     print(f"早停触发！指标 {args.early_stopping_metric} 已连续 {args.early_stopping_patience} 轮没有改善")
            #     break
            current_metric_value = test_metrics[args.early_stopping_metric]
            # 实现早停机制，如果测试指标一直不变动就停止训练
            # 保存最佳F1模型（不受早停机制影响，始终保存最佳模型）
            # if test_metrics['f1'] > best_f1:
            #     best_f1 = test_metrics['f1']
            #     project_name = args.project if args.project != 'all_projects' else "all_projects"
            #     torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model_name}_{project_name}_{fusion}_best.pt"))
            #     print(f"New best model saved with F1: {best_f1:.4f}")

        # 保存结果
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(args.output_dir, f"{args.model_name}_{args.project}_{fusion}_results.csv"), index=False)
        print(f"Results saved to {os.path.join(args.output_dir, f'{args.model_name}_{args.project}_{fusion}_results.csv')}")
        if last_predictions_df is not None:
            pred_filename = os.path.join(args.output_dir, f"{args.model_name}_{args.project}_{fusion}_pre.csv")
            last_predictions_df.to_csv(pred_filename, index=False)
            print(f"Prediction details from last epoch saved to {pred_filename}")

if __name__ == "__main__":
    main()
