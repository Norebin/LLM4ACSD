import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(filename="error.log", level=logging.ERROR, encoding="utf-8")

graphml_path = "/model/lxj/cfexplainer/code smell/data_preprocessing/org.apache.cassandra.db.compaction.unified.Controller#getNumShards.graphml"
# 读取 GraphML 文件
try:
    G = nx.read_graphml(graphml_path)
except Exception as e:
    print(f"读取 GraphML 文件失败: {e}")
    logging.error(f"读取 GraphML 文件失败: {e}", exc_info=True)
    exit(1)

# 可视化
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)  # 布局算法

# 绘制节点
node_colors = [attrs.get("color", "lightblue") for node, attrs in G.nodes(data=True)]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)

# 绘制边
nx.draw_networkx_edges(G, pos)

# 绘制节点标签
node_labels = {node: attrs.get("label", node) for node, attrs in G.nodes(data=True)}
nx.draw_networkx_labels(G, pos, labels=node_labels)

# 绘制边标签（例如权重）
edge_labels = {(source, target): attrs.get("weight", "") for source, target, attrs in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("GraphML 可视化")
plt.show()