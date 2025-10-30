import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.nn import BatchNorm, LayerNorm
from typing import List, Dict, Any
from abc import ABC, abstractmethod

class BaseGNNModel(nn.Module, ABC):
    """GNN模型基类"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 dropout: float = 0.5, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 构建网络层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.build_layers()
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, output_dim)
        )
        
    @abstractmethod
    def build_layers(self):
        """构建图卷积层"""
        pass
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 图卷积
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 图级别预测：使用全局池化
        x = global_mean_pool(x, batch)
        
        # 分类
        x = self.classifier(x)
        return x

class GCNModel(BaseGNNModel):
    """GCN模型"""
    
    def build_layers(self):
        dims = [self.input_dim] + self.hidden_dims
        for i in range(len(dims) - 1):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # 不在最后一层添加BatchNorm
                self.batch_norms.append(BatchNorm(dims[i + 1]))

class GATModel(BaseGNNModel):
    """GAT模型"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout: float = 0.5, heads: int = 8, **kwargs):
        self.heads = heads
        super().__init__(input_dim, hidden_dims, output_dim, dropout, **kwargs)
    
    def build_layers(self):
        dims = [self.input_dim] + self.hidden_dims
        in_channels = self.input_dim
        
        # 中间层
        for i in range(len(self.hidden_dims) - 1):
            out_channels = self.hidden_dims[i]
            conv = GATConv(in_channels, out_channels, heads=self.heads, dropout=self.dropout)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(out_channels * self.heads))
            in_channels = out_channels * self.heads
            
        # 输出层
        out_channels = self.hidden_dims[-1]
        conv = GATConv(in_channels, out_channels, heads=1, concat=False, dropout=self.dropout)
        self.convs.append(conv)

class GINModel(BaseGNNModel):
    """GIN模型"""
    
    def build_layers(self):
        dims = [self.input_dim] + self.hidden_dims
        for i in range(len(dims) - 1):
            mlp = nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.Linear(dims[i + 1], dims[i + 1])
            )
            self.convs.append(GINConv(mlp))
            if i < len(dims) - 2:
                self.batch_norms.append(BatchNorm(dims[i + 1]))

class GraphSAGEModel(BaseGNNModel):
    """GraphSAGE模型"""
    
    def build_layers(self):
        dims = [self.input_dim] + self.hidden_dims
        for i in range(len(dims) - 1):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.batch_norms.append(BatchNorm(dims[i + 1]))

class EnhancedGNNModel(BaseGNNModel):
    """增强的GNN模型，包含注意力机制和残差连接"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout: float = 0.5, model_type: str = 'GCN', **kwargs):
        self.model_type = model_type
        super().__init__(input_dim, hidden_dims, output_dim, dropout, **kwargs)
        
        # 添加注意力权重
        self.attention = nn.MultiheadAttention(hidden_dims[-1], num_heads=4, dropout=dropout)
        
        # 残差连接
        self.residual_connections = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] == hidden_dims[i + 1]:
                self.residual_connections.append(nn.Identity())
            else:
                self.residual_connections.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
    
    def build_layers(self):
        dims = [self.input_dim] + self.hidden_dims
        
        for i in range(len(dims) - 1):
            if self.model_type == 'GCN':
                self.convs.append(GCNConv(dims[i], dims[i + 1]))
            elif self.model_type == 'GAT':
                self.convs.append(GATConv(dims[i], dims[i + 1], heads=1, dropout=self.dropout))
            elif self.model_type == 'GIN':
                mlp = nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.ReLU(),
                    nn.Linear(dims[i + 1], dims[i + 1])
                )
                self.convs.append(GINConv(mlp))
            elif self.model_type == 'SAGE':
                self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            
            if i < len(dims) - 2:
                self.batch_norms.append(BatchNorm(dims[i + 1]))
                
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 保存每层的输出用于残差连接
        intermediate_features = []
        
        # 图卷积层处理
        for i, conv in enumerate(self.convs):
            # 保存上一层的输出
            if i > 0:
                prev_x = x
                intermediate_features.append(x)
            
            # 应用图卷积
            x = conv(x, edge_index)
            
            # 应用BatchNorm
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            # 应用残差连接（除第一层外）
            if i > 0:
                residual = intermediate_features[-1]
                # 使用残差连接调整维度
                residual = self.residual_connections[i-1](residual)
                x = x + residual
            
            # 应用激活函数和Dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 图级别池化
        x = global_mean_pool(x, batch)
        
        # 应用注意力机制（需要调整维度以适应nn.MultiheadAttention）
        x_reshaped = x.unsqueeze(0)  # [N, D] -> [1, N, D]
        x_attended, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x = x + x_attended.squeeze(0)  # 注意力的残差连接
        
        # 分类
        x = self.classifier(x)
        return x

class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model(model_type: str, model_params: Dict[str, Any]) -> BaseGNNModel:
        """创建模型实例"""
        model_type = model_type.upper()
        
        if model_type == 'GCN':
            return GCNModel(**model_params)
        elif model_type == 'GAT':
            return GATModel(**model_params)
        elif model_type == 'GIN':
            return GINModel(**model_params)
        elif model_type == 'GRAPHSAGE':
            return GraphSAGEModel(**model_params)
        elif model_type == 'ENHANCED':
            return EnhancedGNNModel(**model_params)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """获取可用的模型类型"""
        return ['GCN', 'GAT', 'GIN', 'GraphSAGE', 'Enhanced']

# 模型参数配置
MODEL_CONFIGS = {
    'GCN': {
        'description': 'Graph Convolutional Network',
        'default_params': {'dropout': 0.5}
    },
    'GAT': {
        'description': 'Graph Attention Network',
        'default_params': {'dropout': 0.6, 'heads': 8}
    },
    'GIN': {
        'description': 'Graph Isomorphism Network',
        'default_params': {'dropout': 0.5}
    },
    'GRAPHSAGE': {
        'description': 'GraphSAGE',
        'default_params': {'dropout': 0.5}
    },
    'ENHANCED': {
        'description': 'Enhanced GNN with attention and residual connections',
        'default_params': {'dropout': 0.5, 'model_type': 'GCN'}
    }
}

def get_model_info(model_type: str) -> Dict[str, Any]:
    """获取模型信息"""
    return MODEL_CONFIGS.get(model_type.upper(), {})