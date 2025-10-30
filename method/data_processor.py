import os
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphMLParser:
    """GraphML文件解析器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def parse_graphml(self, file_path: str) -> Optional[nx.Graph]:
        """解析GraphML文件为NetworkX图"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"GraphML文件不存在: {file_path}")
                return None
                
            G = nx.read_graphml(file_path)
            return G
        except Exception as e:
            logger.error(f"解析GraphML文件失败 {file_path}: {e}")
            return None
    
    def extract_node_features(self, G: nx.Graph) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        """提取节点特征 - 自动识别所有数字属性"""
        nodes = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # 自动收集所有数字属性
        all_numeric_attrs = set()
        for node in nodes:
            node_data = G.nodes[node]
            for attr, value in node_data.items():
                if self._is_numeric(value):
                    all_numeric_attrs.add(attr)
        
        numeric_attrs = sorted(list(all_numeric_attrs))
        logger.info(f"发现 {len(numeric_attrs)} 个数字属性: {numeric_attrs}")
        
        # 初始化特征矩阵
        feature_matrix = np.zeros((len(nodes), len(numeric_attrs)))
        
        for i, node in enumerate(nodes):
            node_data = G.nodes[node]
            for j, attr in enumerate(numeric_attrs):
                try:
                    value = node_data.get(attr, 0)
                    feature_matrix[i, j] = self._convert_to_numeric(value)
                except (ValueError, TypeError):
                    feature_matrix[i, j] = 0
        
        # 标准化特征
        if not self.is_fitted:
            feature_matrix = self.scaler.fit_transform(feature_matrix)
            self.is_fitted = True
        else:
            feature_matrix = self.scaler.transform(feature_matrix)
            
        return feature_matrix, node_to_idx, numeric_attrs
    
    def _is_numeric(self, value) -> bool:
        """判断值是否为数字"""
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            # 尝试转换为数字
            try:
                float(value)
                return True
            except ValueError:
                # 检查布尔值
                if value.lower() in ['true', 'false']:
                    return True
                return False
        return False
    
    def _convert_to_numeric(self, value) -> float:
        """将值转换为数字"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            if value.lower() == 'true':
                return 1.0
            elif value.lower() == 'false':
                return 0.0
            else:
                return float(value)
        return 0.0
    
    def find_smell_nodes(self, G: nx.Graph) -> List[str]:
        """在图中找到异味节点（is_smell=True）"""
        smell_nodes = []
        for node in G.nodes():
            node_data = G.nodes[node]
            is_smell = node_data.get('is_smell', False)
            
            # 处理不同的布尔值表示方式
            if isinstance(is_smell, str):
                is_smell = is_smell.lower() == 'true'
            elif isinstance(is_smell, bool):
                is_smell = is_smell
            else:
                is_smell = bool(is_smell)
            
            if is_smell:
                smell_nodes.append(node)
        
        return smell_nodes

class CodeSmellDataset(Dataset):
    """代码异味数据集"""
    
    def __init__(self, config: Config, csv_files: List[str], split: str = None):
        super().__init__()
        self.config = config
        self.split = split
        self.parser = GraphMLParser(config)
        
        # 加载所有CSV数据
        self.data_df = self.load_csv_data(csv_files)
        
        # 过滤特定split
        if split:
            self.data_df = self.data_df[self.data_df['dataset_split'] == split]
        
        # 构建图数据
        self.graph_data = []
        self.labels = []
        self.valid_indices = []
        
        self._build_dataset()
        
    def load_csv_data(self, csv_files: List[str]) -> pd.DataFrame:
        """加载CSV文件"""
        all_data = []
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                all_data.append(df)
            else:
                logger.warning(f"CSV文件不存在: {csv_file}")
        
        if not all_data:
            raise ValueError("没有找到有效的CSV文件")
            
        return pd.concat(all_data, ignore_index=True)
    
    def _build_dataset(self):
        """构建数据集"""
        logger.info(f"开始构建数据集，总样本数: {len(self.data_df)}")
        
        for idx, row in self.data_df.iterrows():
            try:
                # 构建GraphML文件路径
                project = row['Project']
                version = row['Version']
                unique_id = row['unique_id']
                
                # 根据unique_id构建GraphML文件路径
                graphml_path = self._construct_graphml_path(project, version, unique_id)
                
                if not graphml_path or not os.path.exists(graphml_path):
                    logger.debug(f"GraphML文件不存在: {graphml_path}")
                    continue
                
                # 解析图
                G = self.parser.parse_graphml(graphml_path)
                if G is None:
                    continue
                
                # 检查图中是否有异味节点
                smell_nodes = self.parser.find_smell_nodes(G)
                if not smell_nodes:
                    logger.debug(f"图中没有找到异味节点: {graphml_path}")
                    continue
                
                # 提取特征和构建PyTorch Geometric数据
                graph_data = self._build_pyg_data(G, smell_nodes, row)
                if graph_data is not None:
                    self.graph_data.append(graph_data)
                    self.labels.append(int(row['actionable']))
                    self.valid_indices.append(idx)
                    
            except Exception as e:
                logger.error(f"处理样本 {idx} 失败: {e}")
                continue
        
        logger.info(f"成功构建 {len(self.graph_data)} 个图样本")
    
    def _construct_graphml_path(self, project: str, version: str, unique_id: str) -> Optional[str]:
        """根据unique_id构建GraphML文件路径"""
        # 根据你的文件组织结构调整这个函数
        # 示例路径构建逻辑，假设GraphML文件以unique_id命名
        base_path = os.path.join(self.config.graphml_base_dir, project)
        
        # 可能的GraphML文件位置
        possible_paths = [
            os.path.join(base_path, f"{unique_id}.graphml"),
            os.path.join(base_path, version, f"{unique_id}.graphml"),
            os.path.join(base_path, "graphml", f"{unique_id}.graphml"),
            # 如果文件名规则不同，可以添加更多可能的路径
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _build_pyg_data(self, G: nx.Graph, smell_nodes: List[str], row: pd.Series) -> Optional[Data]:
        """构建PyTorch Geometric数据对象 - 图级别分类"""
        try:
            # 提取节点特征
            node_features, node_to_idx, feature_names = self.parser.extract_node_features(G)
            
            # 转换为PyTorch Geometric格式
            pyg_data = from_networkx(G)
            
            # 设置节点特征
            pyg_data.x = torch.FloatTensor(node_features)
            
            # 记录异味节点索引（用于可解释性分析）
            smell_indices = [node_to_idx[node] for node in smell_nodes if node in node_to_idx]
            pyg_data.smell_nodes = torch.LongTensor(smell_indices)
            
            # 添加图级别的元信息
            pyg_data.unique_id = row['unique_id']
            pyg_data.smell_type = row['Smell']
            pyg_data.project = row['Project']
            pyg_data.num_smell_nodes = len(smell_nodes)
            
            # 添加节点类型信息（用于异构图处理）
            node_types = []
            for node in G.nodes():
                node_data = G.nodes[node]
                node_type = node_data.get('type', 'unknown')
                if node_type == 'class':
                    node_types.append(0)
                elif node_type == 'method':
                    node_types.append(1)
                else:
                    node_types.append(2)  # unknown
            
            pyg_data.node_type = torch.LongTensor(node_types)
            
            return pyg_data
            
        except Exception as e:
            logger.error(f"构建PyG数据失败: {e}")
            return None
    
    def len(self):
        return len(self.graph_data)
    
    def get(self, idx):
        return self.graph_data[idx], self.labels[idx]

class DataProcessor:
    """数据处理主类"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_datasets(self) -> Tuple[CodeSmellDataset, CodeSmellDataset, CodeSmellDataset]:
        """加载训练、验证、测试数据集"""
        # 获取所有CSV文件
        csv_files = []
        for file in os.listdir(self.config.csv_data_dir):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(self.config.csv_data_dir, file))
        
        if not csv_files:
            raise ValueError(f"在 {self.config.csv_data_dir} 中没有找到CSV文件")
        
        # 创建数据集
        train_dataset = CodeSmellDataset(self.config, csv_files, split='train')
        val_dataset = CodeSmellDataset(self.config, csv_files, split='validation') 
        test_dataset = CodeSmellDataset(self.config, csv_files, split='test')
        
        return train_dataset, val_dataset, test_dataset
    
    def get_data_loaders(self, train_dataset, val_dataset, test_dataset):
        """获取数据加载器"""
        from torch_geometric.loader import DataLoader
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
