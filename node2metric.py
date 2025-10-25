import os
import graph_tool.all as gt
import pandas as pd
import re
import networkx as nx
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm


class MethodMetricsProcessor:
    def __init__(self, method_metrics, method_numeric_cols):
        self.method_lookup = self._preprocess_methods(method_metrics, method_numeric_cols)
        self.method_numeric_cols = method_numeric_cols

    def _preprocess_methods(self, method_metrics, method_numeric_cols) -> Dict:
        """预处理方法信息，建立快速查找表"""
        method_lookup = {}
        for _, row in method_metrics.iterrows():
            method = row['method']
            class_name = row['class'].split('.')[-1]
            features = row[method_numeric_cols].values.tolist()
            # 解析并存储方法信息
            method_name, params = self._parse_csv_method(method)
            # 使用元组作为键以加快查找
            if method_name == '':
                key = (class_name, class_name, params)
            key = (class_name, method_name, params)
            method_lookup[key] = features

        return method_lookup

    def _parse_csv_method(self, method: str) -> Tuple[str, int]:
        """使用正则表达式解析CSV中的方法"""
        match = re.compile(r'(\w+)/\d+(?:\[(.*)])?').match(method)
        if not match:
            return '', 0
        method_name = match.group(1)
        params = len(match.group(2).split(',')) if match.group(2) else 0
        return method_name, params

    def _parse_node_label(self, label: str) -> Tuple[str, str, int]:
        """使用正则表达式解析节点标签"""
        match = re.compile(r'([\w.]+\.(\w+))\.([\w]+)\(([\w.,\s]+)\)').match(label)
        if not match:
            return '', '', 0

        class_name = match.group(2)
        method_name = match.group(3)
        params = len(match.group(4).split(',')) if match.group(4) else 0
        return class_name, method_name, params

    def get_method_features(self, name: str) -> List[float]:
        """获取方法特征"""
        class_name, method_name, params = self._parse_node_label(name)

        # 尝试直接匹配
        key = (class_name, method_name, params)
        if key in self.method_lookup:
            return self.method_lookup[key]

        # 如果没有完全匹配，查找最佳部分匹配
        best_match_features = None

        for (csv_class_name, csv_method_name, csv_params), features in self.method_lookup.items():
            if class_name == csv_class_name and method_name == csv_method_name:
                best_match_features = features

        return best_match_features if best_match_features is not None else [0] * len(self.method_numeric_cols)


def group_files(directory):
    """Group GraphML files by project name."""
    groups = defaultdict(list)
    for file in os.listdir(directory):
        if file.endswith('.graphml'):
            if 'org.apache.lucene' in file or 'org.tartarus' in file:
                groups['solr'].append(os.path.join(directory, file))
                continue
            if 'org.apache.tools' in file:
                groups['pig'].append(os.path.join(directory, file))
                continue
            project = next(name for name in ['cassandra', 'cayenne', 'cdt', 'cxf', 'jackrabbit', 'jena', 'mahout', 'pig', 'solr']
                           if name in file.lower())
            groups[project].append(os.path.join(directory, file))
    return groups


def process_graph(project, graphml_file,
                  class_features_lookup, class_split_names_lookup, class_numeric_cols,
                  method_processor,
                  output_dir):
    """Process single GraphML file and add metrics to nodes."""
    G = gt.load_graph(graphml_file)
    # Create property maps for metrics
    node_type = G.vertex_properties["node_type"]
    node_name = G.vertex_properties["label"]
    code_metric = G.new_vertex_property("vector<double>")
    G.vertex_properties['code_metric'] = code_metric
    vertex_filter = G.new_vertex_property('bool', True)
    # Process each vertex
    for v in G.vertices():
        v_type = node_type[v]
        name = node_name[v]

        if v_type == 0 and project in name : # package node
            code_metric[v] = [0] * class_numeric_cols
        else :
            vertex_filter[v] = False
        if v_type == 1:  # Class node
            if name in class_features_lookup:
                features = class_features_lookup[name]
            else:
                name_parts = name.split('.')
                last_name_part = name_parts[-1]
                max_match_length = 0
                best_match_feature = None
                for class_name, class_name_parts in class_split_names_lookup.items():
                    # 如果最后一个词匹配
                    if class_name_parts[-1] == last_name_part:
                        match_length = sum(1 for x, y in zip(reversed(name_parts), reversed(class_name_parts)) if x == y)
                        # 更新最佳匹配项
                        if match_length > max_match_length:
                            max_match_length = match_length
                            best_match_feature = class_features_lookup[class_name]
                if best_match_feature is None:
                    features = [0] * class_numeric_cols
                else:
                    features = best_match_feature
            # 存入新的节点属性'code_metric'
            print(features)
            if not any(features):
                vertex_filter[v] = False
            else:
                code_metric[v] = features
        elif v_type == 2:  # Method node
            features = method_processor.get_method_features(name)
            if not any(features):
                vertex_filter[v] = False
            else:
                code_metric[v] = features
            # def parse_csv_method(method):
            #     # 提取CSV中的方法名和参数列表
            #     method_name = method.split('/')[0]
            #     params = []
            #     if '[' in method:
            #         params = method.split('[')[1].split(']')[0].split(',')
            #     return method_name, params
            #
            # def parse_node_label(label):
            #     # 提取类名和方法签名
            #     class_name = '.'.join(label.split('(')[0].split('.')[:-1])
            #     method_name = label.split('(')[0].split('.')[-1]
            #     method_params = label.split('(')[-1].split(')')[0]
            #     return class_name, method_name, method_params
            #
            # class_name, method_name, method_params = parse_node_label(name)
            # max_match_length = 0
            # best_match_row = None
            # for _, r in method_metrics.iterrows():
            #     csv_method_name, csv_method_params = parse_csv_method(r['method'])
            #     # 检查方法名和参数列表是否匹配
            #     if method_name == csv_method_name:
            #         match_length = sum(1 for x, y in zip(method_params, csv_method_params) if x == y)
            #         # 更新最佳匹配项
            #         if match_length > max_match_length:
            #             max_match_length = match_length
            #             best_match_row = r
            # if best_match_row is not None:
            #     features = best_match_row[method_numeric_cols].values.tolist()[0]
            # else:
            #     # 如果还找不到，填充等长的全0
            #     features = [0] * len(method_numeric_cols)
            # # 存入新的节点属性'code_metric'
            # code_metric[v] = features

    # Save processed graph
    output_file = os.path.join(output_dir, f"{os.path.basename(graphml_file)}")
    G_filtered = gt.GraphView(G, vfilt=vertex_filter)
    G_new = G_filtered.copy()  # 创建一个新的图，只包含过滤后的顶点
    print(f"原图顶点数: {G.num_vertices()}")
    print(f"过滤后顶点数: {G_filtered.num_vertices()}")
    print(f"新图顶点数: {G_new.num_vertices()}")
    G_new.save(output_file)

def process_graph_networkx(project, graphml_file,
                  class_features_lookup, class_split_names_lookup, class_numeric_cols,
                  method_processor,
                  output_dir):
    G = nx.read_graphml(graphml_file)
    nodes_to_remove = []
    for node, data in tqdm(G.nodes(data=True), desc="Processing all nodes in this graph", total=len(G.nodes())):
        # 获取节点类型和名称
        v_type = data.get("node_type", None)
        name = data.get("label", None)
        if v_type == 0 and project in name:  # package node
            # 为 package 节点分配默认的特征
            features = [0] * class_numeric_cols
            G.nodes[node]['code_metric'] = ','.join(map(str, features))
        elif v_type == 0 and project not in name:
            nodes_to_remove.append(node)
        if v_type == 1:  # Class node
            if name in class_features_lookup:
                features = class_features_lookup[name]
            else:
                name_parts = name.split('.')
                last_name_part = name_parts[-1]
                max_match_length = 0
                best_match_feature = None
                # 查找最匹配的类特征
                for class_name, class_name_parts in class_split_names_lookup.items():
                    if class_name_parts[-1] == last_name_part:
                        match_length = sum(
                            1 for x, y in zip(reversed(name_parts), reversed(class_name_parts)) if x == y)
                        # 更新最佳匹配项
                        if match_length > max_match_length:
                            max_match_length = match_length
                            best_match_feature = class_features_lookup[class_name]
                if best_match_feature is None:
                    features = [0] * class_numeric_cols
                else:
                    features = best_match_feature
            if not any(features):
                nodes_to_remove.append(node)
            else:
                G.nodes[node]['code_metric'] = ','.join(map(str, features))
        elif v_type == 2:  # Method node
            features = method_processor.get_method_features(name)
            if not any(features):
                nodes_to_remove.append(node)
            else:
                G.nodes[node]['code_metric'] = ','.join(map(str, features))
    # Save processed graph
    output_file = os.path.join(output_dir, f"{os.path.basename(graphml_file)}")
    print(f"原图节点数量: {G.number_of_nodes()}")
    print(f"将要删除的节点数量: {len(nodes_to_remove)}")
    G.remove_nodes_from(nodes_to_remove)
    print(f"处理后节点数量: {G.number_of_nodes()}")
    nx.write_graphml(G, output_file)

def main():
    input_dir = "/model/LiangXJ/developCodeSmell"
    output_dir = "/model/LiangXJ/graph2metric"

    os.makedirs(output_dir, exist_ok=True)
    grouped_files = group_files(input_dir)

    for project, files in grouped_files.items():
        # Read metric files
        class_metrics = pd.read_csv(f"{output_dir}/{project}class.csv")
        class_metrics.fillna(0, inplace=True)
        method_metrics = pd.read_csv(f"{output_dir}/{project}method.csv")
        method_metrics.fillna(0, inplace=True)
        class_numeric_cols = class_metrics.select_dtypes(include=['int64', 'float64']).columns
        method_numeric_cols = method_metrics.select_dtypes(include=['int64', 'float64']).columns
        class_feature_len = len(class_numeric_cols)
        method_feature_len = len(method_numeric_cols)
        # 创建类名到特征的快速查找字典
        class_features_lookup = {}
        # 创建类名到分割后部分的查找字典
        class_split_names_lookup = {}
        # 预处理所有类名
        for idx, row in class_metrics.iterrows():
            class_name = row['class']
            features = row[class_numeric_cols].values.tolist()
            class_features_lookup[class_name] = features
            # 预处理分割的类名
            parts = class_name.split('.')
            class_split_names_lookup[class_name] = parts
        method_processor = MethodMetricsProcessor(method_metrics, method_numeric_cols)
        for file in tqdm(files, total=len(files), desc=f"Processing files from {project}"):
            process_graph_networkx(project, file,
                          class_features_lookup, class_split_names_lookup, class_feature_len,
                          method_processor,
                          output_dir)


if __name__ == "__main__":
    main()