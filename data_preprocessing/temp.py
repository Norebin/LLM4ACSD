import xml.etree.ElementTree as ET
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

class XMLDependencyGraph:
    def __init__(self, xml_file):
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()
        self.graph = nx.DiGraph()
        self.node_counter = 0
        
    def _get_node_id(self):
        self.node_counter += 1
        return self.node_counter
    
    def find_element(self, name):
        """在XML中查找指定名称的类或特征"""
        # 搜索类
        for cls in self.root.findall('.//class'):
            if cls.find('name').text == name:
                return cls
            # 搜索类中的特征
            for feature in cls.findall('feature'):
                if feature.find('name').text == name:
                    return feature
        # 单独搜索特征 (可能跨类)
        for feature in self.root.findall('.//feature'):
            if feature.find('name').text == name:
                return feature
        return None
    
    def extract_dependencies(self, start_name, max_hops=3):
        """提取指定节点的依赖关系，控制最大跳跃次数"""
        visited = set()
        queue = deque()
        
        start_node = self.find_element(start_name)
        if not start_node:
            print(f"未找到名称: {start_name}")
            return
        
        # 初始化队列和访问集合
        queue.append((start_node, 0))
        visited.add(start_name)
        
        while queue:
            current_node, current_hop = queue.popleft()
            if current_hop >= max_hops:
                continue
                
            node_name = current_node.find('name').text
            self._add_to_graph(node_name, current_node)
            
            # 处理outbound
            for outbound in current_node.findall('outbound'):
                target = outbound.text
                if target not in visited:
                    visited.add(target)
                    target_node = self.find_element(target)
                    if target_node:
                        queue.append((target_node, current_hop + 1))
                self.graph.add_edge(node_name, target, 
                                  type=outbound.get('type'),
                                  confirmed=outbound.get('confirmed'))
            
            # 处理inbound
            for inbound in current_node.findall('inbound'):
                source = inbound.text
                if source not in visited:
                    visited.add(source)
                    source_node = self.find_element(source)
                    if source_node:
                        queue.append((source_node, current_hop + 1))
                self.graph.add_edge(source, node_name,
                                  type=inbound.get('type'),
                                  confirmed=inbound.get('confirmed'))
    
    def _add_to_graph(self, name, xml_node):
        """将节点添加到图中"""
        if name not in self.graph:
            node_type = 'class' if xml_node.tag == 'class' else 'feature'
            confirmed = xml_node.get('confirmed', 'no')
            self.graph.add_node(name, type=node_type, confirmed=confirmed)
    
    def visualize(self, filename=None):
        """可视化图结构"""
        pos = nx.spring_layout(self.graph)
        
        # 根据节点类型设置颜色
        node_colors = []
        for node in self.graph.nodes():
            if self.graph.nodes[node]['type'] == 'class':
                node_colors.append('lightblue')
            else:
                node_colors.append('lightgreen')
        
        # 根据确认状态设置节点边框
        edge_colors = []
        for u, v in self.graph.edges():
            if self.graph.edges[u, v]['confirmed'] == 'yes':
                edge_colors.append('black')
            else:
                edge_colors.append('red')
        
        plt.figure(figsize=(12, 10))
        nx.draw(self.graph, pos, with_labels=True, 
               node_color=node_colors, edge_color=edge_colors,
               font_size=8, node_size=800, 
               arrowsize=20)
        
        # 添加图例
        plt.scatter([], [], c='lightblue', label='Class')
        plt.scatter([], [], c='lightgreen', label='Feature')
        plt.plot([], [], color='black', label='Confirmed Edge')
        plt.plot([], [], color='red', label='Unconfirmed Edge')
        plt.legend()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"图已保存为 {filename}")
        plt.show()
    
    def save_graph(self, filename):
        """保存图为文件"""
        nx.write_graphml(self.graph, filename)
        print(f"图已保存为 {filename} (GraphML格式)")

# 使用示例
if __name__ == "__main__":
    # 初始化解析器
    parser = XMLDependencyGraph("/model/LiangXJ/DependencyFinder-1.4.3/test.xml")
    
    # 输入要分析的节点名称
    start_node = input("请输入要分析的类名或方法名: ")
    
    # 提取依赖关系 (最大3次跳跃)
    parser.extract_dependencies(start_node, max_hops=3)
    
    # 可视化并保存
    parser.visualize(f"{start_node}_graph.png")
    parser.save_graph(f"{start_node}_graph.graphml")