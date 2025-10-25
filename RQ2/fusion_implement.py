import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import networkx as nx
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score

from torch_geometric.nn import GINConv, BatchNorm, global_mean_pool, global_max_pool
import torch_geometric.data as geo_data
from torch_geometric.loader import DataLoader as PyGDataLoader # 导入PyG的DataLoader

class BalancedFusion(nn.Module):
    def __init__(self, semantic_dim, graph_dim, output_dim):
        super().__init__()
        # 对高维语义特征进行降维
        self.semantic_compressor = nn.Sequential(
            nn.Linear(semantic_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),  # 增加归一化层
            nn.Dropout(0.3),
            nn.Linear(512, 128)  # 压缩到与图特征相近的维度
        )
        
        # 图特征映射
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, semantic_feature, graph_feature):
        sem_compressed = self.semantic_compressor(semantic_feature)
        graph_projected = self.graph_proj(graph_feature)
        combined = torch.cat([sem_compressed, graph_projected], dim=1)
        return self.fusion(combined)

class DynamicWeightedFusion(nn.Module):
    def __init__(self, semantic_dim, graph_dim, output_dim):
        super().__init__()
        # 语义特征压缩
        self.semantic_compressor = nn.Sequential(
            nn.Linear(semantic_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # 图特征处理
        self.graph_proj = nn.Linear(graph_dim, 256)
        
        # 权重生成网络
        self.weight_generator = nn.Sequential(
            nn.Linear(semantic_dim + graph_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)  # 确保权重和为1
        )
        
        self.out = nn.Linear(256, output_dim)
        
    def forward(self, semantic_feature, graph_feature):
        # 特征处理
        sem_feat = self.semantic_compressor(semantic_feature)
        graph_feat = self.graph_proj(graph_feature)
        
        # 生成动态权重
        combined_input = torch.cat([semantic_feature, graph_feature], dim=1)
        weights = self.weight_generator(combined_input)
        
        # 加权融合
        fused = weights[:, 0].unsqueeze(1) * sem_feat + weights[:, 1].unsqueeze(1) * graph_feat
        return self.out(fused)

class ProgressiveFusion(nn.Module):
    def __init__(self, semantic_dim, graph_dim, output_dim):
        super().__init__()
        # 第一阶段：单独处理特征
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 256)
        )
        
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        # 第二阶段：低层次融合
        self.fusion_layer1 = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # 第三阶段：高层次融合
        self.fusion_layer2 = nn.Linear(256 + 128 + 256, output_dim)
        
    def forward(self, semantic_feature, graph_feature):
        # 单独处理
        sem_encoded = self.semantic_encoder(semantic_feature)
        graph_encoded = self.graph_encoder(graph_feature)
        
        # 低层次融合
        first_fusion = self.fusion_layer1(torch.cat([sem_encoded, graph_encoded], dim=1))
        
        # 高层次融合（包含原始编码和低层次融合结果）
        final_fusion = self.fusion_layer2(
            torch.cat([sem_encoded, graph_encoded, first_fusion], dim=1)
        )
        
        return final_fusion
    
class EnhancedCrossAttention(nn.Module):
    def __init__(self, semantic_dim, graph_dim, output_dim):
        super().__init__()
        # 特征压缩和映射
        self.sem_encoder = nn.Sequential(
            nn.Linear(semantic_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256)
        )
        
        self.graph_encoder = nn.Linear(graph_dim, 256)
        
        # 多头注意力层
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=256, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, semantic_feature, graph_feature):
        # 编码
        sem_encoded = self.sem_encoder(semantic_feature)
        graph_encoded = self.graph_encoder(graph_feature)
        
        # 重塑为注意力输入格式
        sem_reshaped = sem_encoded.unsqueeze(1)  # [batch_size, 1, 256]
        graph_reshaped = graph_encoded.unsqueeze(1)  # [batch_size, 1, 256]
        
        # 交叉注意力：语义->图
        attn_output1, _ = self.multihead_attn(
            sem_reshaped, graph_reshaped, graph_reshaped
        )
        
        # 交叉注意力：图->语义
        attn_output2, _ = self.multihead_attn(
            graph_reshaped, sem_reshaped, sem_reshaped
        )
        
        # 结合注意力输出
        fused_attn = torch.cat([
            attn_output1.squeeze(1), 
            attn_output2.squeeze(1)
        ], dim=1)
        
        # 最终融合
        output = self.fusion_layer(fused_attn)
        return output
    
class FiLMFusion(nn.Module):
    def __init__(self, semantic_dim, graph_dim, output_dim):
        super().__init__()
        # 语义特征处理
        self.semantic_processor = nn.Sequential(
            nn.Linear(semantic_dim, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512,256)
        )
        
        # 图特征处理
        self.graph_processor = nn.Linear(graph_dim, 128)
        
        # FiLM参数生成器
        self.film_generator = nn.Sequential(
            nn.Linear(graph_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256*2)  # 生成gamma和beta
        )
        
        # 输出层
        self.out = nn.Linear(256 + 128, output_dim)
        
    def forward(self, semantic_feature, graph_feature):
        # 处理特征
        sem_processed = self.semantic_processor(semantic_feature)
        graph_processed = self.graph_processor(graph_feature)
        
        # 生成FiLM参数
        film_params = self.film_generator(graph_feature)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        # 应用FiLM调制
        modulated_sem = sem_processed * (1 + gamma) + beta
        
        # 融合并输出
        combined = torch.cat([modulated_sem, graph_processed], dim=1)
        return self.out(combined)
    
class SimpleFusion(nn.Module):
    """简单的特征拼接和全连接融合"""
    def __init__(self, semantic_dim, graph_dim, output_dim):
        super().__init__()
        self.semantic_proj = nn.Linear(semantic_dim, output_dim // 2)
        self.graph_proj = nn.Linear(graph_dim, output_dim // 2)
        self.bn_semantic = nn.BatchNorm1d(output_dim // 2)
        self.bn_graph = nn.BatchNorm1d(output_dim // 2)
        
    def forward(self, semantic_feature, graph_feature):
        semantic_proj = F.relu(self.bn_semantic(self.semantic_proj(semantic_feature)))
        graph_proj = F.relu(self.bn_graph(self.graph_proj(graph_feature)))
        return torch.cat([semantic_proj, graph_proj], dim=1)
    
class SelfAttentionFusion(nn.Module):
    """使用自注意力机制融合特征"""
    def __init__(self, semantic_dim, graph_dim, output_dim):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.graph_dim = graph_dim
        
        # 投影层，将两种特征映射到同一维度
        self.semantic_proj = nn.Linear(semantic_dim, output_dim)
        self.graph_proj = nn.Linear(graph_dim, output_dim)
        
        # 自注意力层
        self.query = nn.Linear(output_dim, output_dim)
        self.key = nn.Linear(output_dim, output_dim)
        self.value = nn.Linear(output_dim, output_dim)
        
        # 输出层
        self.out = nn.Linear(output_dim, output_dim)
        
    def forward(self, semantic_feature, graph_feature):
        # 投影到同一维度
        semantic_proj = self.semantic_proj(semantic_feature)
        graph_proj = self.graph_proj(graph_feature)
        
        # 拼接特征
        x = torch.stack([semantic_proj, graph_proj], dim=1)  # [batch_size, 2, output_dim]
        
        # 自注意力计算
        q = self.query(x)  # [batch_size, 2, output_dim]
        k = self.key(x)    # [batch_size, 2, output_dim]
        v = self.value(x)  # [batch_size, 2, output_dim]
        
        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.semantic_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, 2, 2]
        
        # 应用注意力权重
        out = torch.matmul(attn_weights, v)  # [batch_size, 2, output_dim]
        
        # 求和得到最终的融合特征
        fused_feature = out.sum(dim=1)  # [batch_size, output_dim]
        return self.out(fused_feature)


class CrossAttentionFusion(nn.Module):
    """使用交叉注意力机制融合特征"""
    
    def __init__(self, semantic_dim, graph_dim, output_dim):
        super().__init__()
        # 投影层
        self.semantic_proj = nn.Linear(semantic_dim, output_dim)
        self.graph_proj = nn.Linear(graph_dim, output_dim)
        
        # 注意力层 - 重新设计为简化版本
        self.semantic_attn = nn.Linear(output_dim, output_dim)
        self.graph_attn = nn.Linear(output_dim, output_dim)
        
        # 层归一化帮助训练稳定性
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        
        # 输出层
        self.out = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, semantic_feature, graph_feature):
        # 投影到同一维度空间
        semantic_proj = self.semantic_proj(semantic_feature)
        graph_proj = self.graph_proj(graph_feature)
        
        # 应用层归一化
        semantic_proj = self.layer_norm1(semantic_proj)
        graph_proj = self.layer_norm2(graph_proj)
        
        # 计算注意力权重 (使用缩放点积注意力)
        scale = float(semantic_proj.size(-1)) ** 0.5
        attn_weights = torch.matmul(semantic_proj, graph_proj.transpose(-2, -1)) / scale
        semantic_attn = F.softmax(attn_weights, dim=-1)
        graph_attn = F.softmax(attn_weights.transpose(-2, -1), dim=-1)
        
        # 应用注意力
        semantic_context = torch.matmul(semantic_attn, graph_proj)
        graph_context = torch.matmul(graph_attn, semantic_proj)
        
        # 添加残差连接
        semantic_output = semantic_proj + semantic_context
        graph_output = graph_proj + graph_context
        
        # 拼接并投影回原始维度
        fused = torch.cat([semantic_output, graph_output], dim=-1)
        output = self.out(fused)
        
        return output


class GateFusion(nn.Module):
    """使用门控机制融合特征"""
    
    def __init__(self, semantic_dim, graph_dim, output_dim):
        super().__init__()
        
        # 投影层
        self.semantic_proj = nn.Linear(semantic_dim, output_dim)
        self.graph_proj = nn.Linear(graph_dim, output_dim)
        
        # 门控层
        self.semantic_gate = nn.Linear(semantic_dim + graph_dim, output_dim)
        self.graph_gate = nn.Linear(semantic_dim + graph_dim, output_dim)
        
        # 输出层
        self.out = nn.Linear(output_dim, output_dim)
        
    def forward(self, semantic_feature, graph_feature):
        # 投影
        sem_proj = self.semantic_proj(semantic_feature)
        graph_proj = self.graph_proj(graph_feature)
        
        # 特征拼接
        combined = torch.cat([semantic_feature, graph_feature], dim=1)
        
        # 计算门控值
        sem_gate = torch.sigmoid(self.semantic_gate(combined))
        graph_gate = torch.sigmoid(self.graph_gate(combined))
        
        # 应用门控
        gated_sem = sem_proj * sem_gate
        gated_graph = graph_proj * graph_gate
        
        # 求和
        fused = gated_sem + gated_graph
        return self.out(fused)


class MultiHeadAttentionFusion(nn.Module):
    """使用多头注意力机制融合特征"""
    
    def __init__(self, semantic_dim, graph_dim, output_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # 投影层
        self.semantic_proj = nn.Linear(semantic_dim, output_dim)
        self.graph_proj = nn.Linear(graph_dim, output_dim)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(output_dim, num_heads, dropout=0.4)
        
        # 输出层
        self.out = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(0.4)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, semantic_feature, graph_feature):
        # 投影
        sem_proj = self.semantic_proj(semantic_feature)
        graph_proj = self.graph_proj(graph_feature)
        
        # 拼接特征 [batch_size, 2, output_dim]
        x = torch.stack([sem_proj, graph_proj], dim=0)
        
        # 应用多头注意力 (PyTorch的MultiheadAttention期望输入是[seq_len, batch_size, emb_dim])
        attn_output, _ = self.attention(x, x, x)
        
        # 残差连接和层归一化
        x = x + self.dropout(attn_output)
        x = self.layer_norm(x)
        
        # 求和得到融合特征 [batch_size, output_dim]
        fused_feature = x.sum(dim=0)
        
        return self.out(fused_feature)