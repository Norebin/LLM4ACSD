import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from typing import Dict, List, Set
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm

def create_graphml(nodes: List[str],
                   node_types: List[int],
                   edge_index: torch.Tensor,
                   edge_types: List[int],
                   node_to_idx: Dict[str, int],
                   output_name: str):
    """
    创建 GraphML 格式的图并保存
    """
    # 创建有向图
    G = nx.DiGraph()

    # 添加节点及其属性
    for idx, (node_name, node_type) in enumerate(zip(nodes, node_types)):
        G.add_node(node_name,
                   id=idx,
                   node_type=int(node_type),
                   label=node_name)

    # 添加边及其属性
    edge_index = edge_index.t().numpy()  # 转置并转换为numpy数组
    for i, (src, dst) in enumerate(edge_index):
        G.add_edge(nodes[src], nodes[dst],
                   edge_type=int(edge_types[i]))

    # 保存为 GraphML 格式
    output_file = "/model/LiangXJ/ClassDependency/" + output_name + ".graphml"
    nx.write_graphml(G, output_file)

    # 同时保存节点映射关系，方便后续使用
    output_json = "/model/LiangXJ/ClassDependency/" + output_name + ".json"
    with open(output_json, 'w') as f:
        json.dump(node_to_idx, f, indent=2)

    return G

def is_official_package(name: str) -> bool:
    """
    判断是否为官方包/类/方法
    """
    official_prefixes = {
        'java.',
        'javax.',
        'sun.',
        'com.sun.',
        'org.w3c.',
        'org.xml.',
        'org.omg.',
        'org.ietf.',
        'org.slf4j'
        'android.',
    }
    return any(name.startswith(prefix) for prefix in official_prefixes)


def xml_to_graph(xml_file):
    # 解析XML
    tree = ET.parse(xml_file)
    root = tree.getroot()
    if root is None:
        raise ValueError("No dependencies found in XML")
    # 创建节点和边的列表
    nodes = []  # 存储所有的节点
    edges = []  # 存储所有的边(source_idx, target_idx) -> edge_type
    node_types = []  # 存储节点类型 (0:package, 1:class, 2:feature)
    edge_types = []  # 存储边的类型 (0:package-class, 1:class-class, 2:class-feature, 3:feature-feature)
    # 创建节点名称到索引的映射
    node_to_idx = {}

    def add_node(node_name, node_type):
        if is_official_package(node_name):
            return None
        if node_name not in node_to_idx:
            node_to_idx[node_name] = len(nodes)
            nodes.append(node_name)
            node_types.append(node_type)
        return node_to_idx[node_name]

    unique_edges = {}

    def add_edge(source_idx, target_idx, edge_type):
        if source_idx is None or target_idx is None:
            return
        edge_key = (source_idx, target_idx)
        if edge_key not in unique_edges:
            unique_edges[edge_key] = edge_type

    # 找到dependencies下的所有class节点
    for package in tqdm(root.findall(".//package[@confirmed='yes']"), desc="Processing packages"):
        package_name = package.find("name").text
        if not package_name:
            continue
        package_idx = add_node(package_name, 0)  # 包节点
        if package_idx is None:
            continue
        for class_node in package.findall(".//class"):
            class_name = class_node.find("name").text
            if not class_name:  # 跳过空类名
                continue
            class_idx = add_node(class_name, 1)  # 类节点
            if class_idx is None:
                continue
            # 添加包到类的边
            add_edge(package_idx, class_idx, 0)

            # 处理出边（outbound）
            for outbound in class_node.findall('./outbound'):
                target_name = outbound.text
                if target_name:
                    if outbound.get('type') == 'class':
                        target_idx = add_node(target_name, 1)
                        if target_idx is None:
                            continue
                        add_edge(class_idx, target_idx, 1)
                    elif outbound.get('type') == 'feature':
                        target_idx = add_node(target_name, 2)
                        if target_idx is None:
                            continue
                        add_edge(class_idx, target_idx, 2)

            # 处理入边（inbound）
            for inbound in class_node.findall('./inbound'):
                source_name = inbound.text
                if source_name:
                    if inbound.get('type') == 'class':
                        source_idx = add_node(source_name, 1)
                        if source_idx is None:
                            continue
                        add_edge(source_idx, class_idx, 1)
                    elif inbound.get('type') == 'feature':
                        source_idx = add_node(source_name, 2)
                        if source_idx is None:
                            continue
                        add_edge(source_idx, class_idx, 2)

            # 处理特征（features）
            for feature in class_node.findall("./feature"):
                feature_name = feature.find("name").text
                feature_idx = add_node(feature_name, 2)
                if feature_idx is None:
                    continue
                # 添加类到边的特征
                add_edge(class_idx, feature_idx, 2)

                # 处理特征间的关系
                for outbound in feature.findall('./outbound'):
                    target_name = outbound.text
                    if target_name:
                        if outbound.get('type') == 'class':
                            target_idx = add_node(target_name, 1)
                            if target_idx is None:
                                continue
                            add_edge(feature_idx, target_idx, 2)
                        elif outbound.get('type') == 'feature':
                            target_idx = add_node(target_name, 2)
                            if target_idx is None:
                                continue
                            add_edge(feature_idx, target_idx, 3)

                for inbound in feature.findall('./inbound'):
                    source_name = inbound.text
                    if source_name:
                        if inbound.get('type') == 'class':
                            source_idx = add_node(source_name, 1)
                            if source_idx is None:
                                continue
                            add_edge(source_idx, feature_idx, 2)
                        elif inbound.get('type') == 'feature':
                            source_idx = add_node(source_name, 2)
                            if source_idx is None:
                                continue
                            add_edge(source_idx, feature_idx, 3)

    for (source, target), edge_type in unique_edges.items():
        edges.append([source, target])
        edge_types.append(edge_type)
    # 转换为PyG格式
    # x = torch.eye(len(nodes))  # 节点特征矩阵（使用one-hot编码）
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # edge_types = torch.tensor(edge_types, dtype=torch.long)
    node_type = torch.tensor(node_types, dtype=torch.long)
    # 创建PyG的Data对象
    # data = Data(
    #     x=x,
    #     edge_index=edge_index,
    #     edge_attr=edge_types,
    #     node_type=node_type,
    #     node_names=nodes  # 保存节点名称以便后续使用
    # )

    return nodes, node_types, edge_index, edge_types, node_to_idx


if __name__ == "__main__":

    nodes, node_types, edge_index, edge_types, node_to_idx = xml_to_graph('/model/LiangXJ/ClassDependency/jena.xml')
    graph = create_graphml(nodes, node_types, edge_index, edge_types, node_to_idx, "jena")

    # print(f"Number of nodes: {data.num_nodes}")
    # print(f"Number of edges: {data.num_edges}")
    # print(f"Node feature matrix shape: {data.x.shape}")
    # print(f"Number of node types: {len(torch.unique(data.node_type))}")
    # print(f"Number of edge types: {len(torch.unique(data.edge_attr))}")