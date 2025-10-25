import pandas as pd
import networkx as nx
import os
from collections import defaultdict
from typing import Set, Optional

from exceptiongroup import catch
from graph_tool.all import load_graph, GraphView, Graph


def load_xlsx(filepath):
    """
    加载xlsx文件
    返回DataFrame和按项目分类的class_names字典
    """
    df = pd.read_excel(filepath)

    # 根据class_name确定属于哪个项目
    project_classes = defaultdict(list)
    for class_name in df['class-name']:
        # 这里需要根据实际的命名规则来确定项目归属
        # 示例规则:根据包名前缀判断
        if 'cassandra' in class_name.lower():
            project_classes['cassandra'].append(class_name)
        if 'cxf' in class_name.lower():
            project_classes['cxf'].append(class_name)
        if 'jena' in class_name.lower():
            project_classes['jena'].append(class_name)
        if 'lucene' in class_name.lower() or 'solr' in class_name.lower() or 'tartarus' in class_name.lower():
            project_classes['solr'].append(class_name)
        if 'pig' in class_name.lower() or 'tools' in class_name.lower():
            project_classes['pig'].append(class_name)
        if 'cdt' in class_name.lower():
            project_classes['cdt'].append(class_name)
        if 'jackrabbit' in class_name.lower():
            project_classes['jackrabbit'].append(class_name)
        if 'mahout' in class_name.lower():
            project_classes['mahout'].append(class_name)
        if 'cayenne' in class_name.lower():
            project_classes['cayenne'].append(class_name)

    return df, project_classes


def extract_subgraph(G, class_name):
    """
    从依赖图中提取与指定类相关的子图
    返回子图的序列化表示
    """
    try:
        # 找到与class_name匹配的节点
        target_node = None
        for node in G.nodes():
            if class_name in str(G.nodes[node].get('label', '')):
                target_node = node
                break

        if target_node is None:
            return None

        # 提取相关节点
        # 获取所有与目标节点相关的前驱和后继节点 (可以设置深度限制)
        related_nodes = set()
        related_nodes.add(target_node)

        # 向前查找依赖这个类的节点
        predecessors = nx.descendants(G, target_node)
        related_nodes.update(predecessors)

        # 向后查找这个类依赖的节点
        successors = nx.ancestors(G, target_node)
        related_nodes.update(successors)

        # 提取子图
        subgraph = G.subgraph(related_nodes)

        # 将子图转换为可存储的格式
        # 这里使用简单的边列表表示,可以根据需要修改
        edges = list(subgraph.edges())
        return str(edges)

    except Exception as e:
        print(f"处理{class_name}时发生错误: {str(e)}")
        return None


def extract_related_subgraph(
        graph_file,
        target_class: str,
        output_dir: str = "subgraph.graphml",
        max_distance: Optional[int] = None
) -> nx.DiGraph:
    """
    提取与目标类相关的子图

    参数:
        graph_file: 输入的graphml文件路径
        target_class: 目标类名
        output_file: 输出的子图文件路径
        max_distance: 最大跳数限制，None表示不限制
    """
    # 读取原图
    # G = nx.read_graphml(graph_file)
    G = graph_file
    # 确保目标节点存在
    if target_class not in G.nodes():
        raise ValueError(f"Class {target_class} not found in the graph")

    def get_connected_nodes(node: str, max_dist: Optional[int] = None) -> Set[str]:
        """获取与指定节点相连的所有节点（包括入边和出边）"""
        connected = set()
        current_nodes = {node}
        visited = {node}
        distance = 0

        while current_nodes and (max_dist is None or distance < max_dist):
            next_nodes = set()
            for current in current_nodes:
                # 获取所有相邻节点（包括入边和出边）
                neighbors = set(G.successors(current)) | set(G.predecessors(current))
                next_nodes.update(neighbors - visited)

            visited.update(next_nodes)
            connected.update(next_nodes)
            current_nodes = next_nodes
            distance += 1

        return connected
    # 获取所有相关节点
    related_nodes = get_connected_nodes(target_class, max_distance)
    related_nodes.add(target_class)  # 确保包含目标节点

    # 创建子图
    subgraph = G.subgraph(related_nodes).copy()

    # 添加一些统计信息作为图的属性
    subgraph.graph['target_node'] = target_class
    subgraph.graph['node_count'] = len(subgraph)
    subgraph.graph['edge_count'] = subgraph.number_of_edges()

    # 保存子图
    output_file = output_dir + target_class + ".graphml"
    nx.write_graphml(subgraph, output_file)

    return subgraph

def create_subgraph(graph, target_class, output_dir, max_distance):
    # 加载 GraphML 图
    # 确保目标节点存在
    matches = []
    target_parts = target_class.split('.')
    target_node = None
    for v in graph.vertices():
        if graph.vp.label[v] == target_class:
            target_node = v
            break
    if target_node == None:
        for v in graph.vertices():
            node_label = graph.vp.label[v]
            node_parts = node_label.split('.')
            if node_parts[-1] == target_parts[-1] :
                target_node = v
                break
    def get_connected_nodes(graph, target_node, max_dist=None):
        """
        获取与指定节点相连的所有节点（包括入边和出边）
        """
        connected = set([target_node])
        current_nodes = set([target_node])
        visited = set([target_node])
        distance = 0

        while current_nodes and (max_dist is None or distance < max_dist):
            next_nodes = set()
            for node in current_nodes:
                # 获取入边和出边的邻居节点
                neighbors = set(
                    graph.get_out_neighbors(node)
                ) | set(graph.get_in_neighbors(node))
                next_nodes.update(neighbors - visited)

            visited.update(next_nodes)
            connected.update(next_nodes)
            current_nodes = next_nodes
            distance += 1

        return connected

    try :
        if target_node is None:
            raise ValueError(f"Class {target_class} not found in the graph")
        # 获取所有相关节点
        related_nodes = get_connected_nodes(graph, target_node, max_dist=max_distance)

        # 创建子图
        subgraph = GraphView(graph, vfilt=lambda v: v in related_nodes)

        # 将子图转换为完整图以支持保存
        full_subgraph = Graph(subgraph, prune=True)

        # 添加一些统计信息
        full_subgraph.graph_properties["target_node"] = full_subgraph.new_graph_property("string")
        full_subgraph.graph_properties["target_node"] = target_class
        full_subgraph.graph_properties["node_count"] = full_subgraph.new_graph_property("int")
        full_subgraph.graph_properties["node_count"] = len(related_nodes)
        full_subgraph.graph_properties["edge_count"] = full_subgraph.new_graph_property("int")
        full_subgraph.graph_properties["edge_count"] = full_subgraph.num_edges()

        # 保存子图
        output_file = f"{output_dir}/{target_class}.graphml"
        full_subgraph.save(output_file)
        return full_subgraph
    except ValueError as e:
        print(target_class)

def process_project(graphml_path, df, class_names):
    """
    处理单个项目的依赖图和相关类
    """
    print(f"正在处理 {graphml_path}")

    # 加载依赖图
    # G = nx.read_graphml(graphml_path)
    G = load_graph(graphml_path)
    # 处理该项目的所有相关类
    for class_name in class_names:
        # 在DataFrame中找到对应行的索引
        indices = df.index[df['class-name'] == class_name].tolist()

        if not indices:
            continue

        # 提取子图
        # subgraph = extract_related_subgraph(G, class_name, '/model/LiangXJ/developCodeSmell/',2)
        create_subgraph(G, class_name, '/model/LiangXJ/developCodeSmell',2)
        # 更新DataFrame
        # for idx in indices:
        #     df.at[idx, 'dependency_subgraph'] = subgraph
    return df


def main():
    # 加载xlsx文件
    xlsx_path = '/model/LiangXJ/developCodeSmell.xlsx'  # 替换为实际的文件路径
    df, project_classes = load_xlsx(xlsx_path)

    # 添加新列用于存储子图
    # df['dependency_subgraph'] = None

    # 处理每个项目
    graphml_dir = '/model/LiangXJ/ClassDependency'  # 替换为实际的目录路径
    for filename in os.listdir(graphml_dir):
        if filename.endswith('.graphml'):
            project_name = filename.split('.')[0]  # 假设文件名就是项目名

            # if project_name in project_classes:
            if project_name == 'jena':
                graphml_path = os.path.join(graphml_dir, filename)
                df = process_project(graphml_path, df, project_classes[project_name])

    # 保存结果
    # df.to_excel('output.xlsx', index=False)
    print("处理完成,结果已保存到output.xlsx")


if __name__ == "__main__":
    main()