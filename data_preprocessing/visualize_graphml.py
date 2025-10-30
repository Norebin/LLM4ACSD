import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import argparse
import json

class GraphMLVisualizer:
    """
    A comprehensive GraphML file visualizer with support for node attributes,
    different node types, and interactive visualization.
    """
    
    def __init__(self, graphml_file: str):
        """Initialize the visualizer with a GraphML file."""
        self.graphml_file = graphml_file
        self.graph = nx.read_graphml(graphml_file)
        self.node_colors = {
            'class': '#FF6B6B',
            'method': '#4ECDC4',
            'default': '#95E1D3'
        }
        self.edge_colors = {
            'belongs_to': '#FF9F43',
            'calls': '#5F27CD',
            'default': '#636e72'
        }
        
    def get_node_info(self, node_id: str) -> Dict:
        """Get comprehensive information about a node."""
        node_data = self.graph.nodes[node_id]
        return {
            'id': node_id,
            'label': node_data.get('label', node_id),
            'type': node_data.get('type', 'unknown'),
            'attributes': {k: v for k, v in node_data.items() if k not in ['label', 'type']}
        }
    
    def filter_nodes(self, 
                    node_type: Optional[str] = None,
                    min_attribute_value: Optional[Dict[str, float]] = None,
                    max_attribute_value: Optional[Dict[str, float]] = None) -> List[str]:
        """Filter nodes based on type and attribute values."""
        filtered_nodes = []
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            
            # Filter by type
            if node_type and node_data.get('type') != node_type:
                continue
                
            # Filter by attribute ranges
            skip_node = False
            if min_attribute_value:
                for attr, min_val in min_attribute_value.items():
                    if attr in node_data:
                        try:
                            if float(node_data[attr]) < min_val:
                                skip_node = True
                                break
                        except (ValueError, TypeError):
                            pass
            
            if max_attribute_value and not skip_node:
                for attr, max_val in max_attribute_value.items():
                    if attr in node_data:
                        try:
                            if float(node_data[attr]) > max_val:
                                skip_node = True
                                break
                        except (ValueError, TypeError):
                            pass
            
            if not skip_node:
                filtered_nodes.append(node_id)
                
        return filtered_nodes
    
    def create_matplotlib_visualization(self, 
                                      layout_algorithm: str = 'spring',
                                      node_size_attribute: str = 'wmc',
                                      show_labels: bool = True,
                                      figsize: Tuple[int, int] = (20, 16),
                                      filtered_nodes: Optional[List[str]] = None):
        """Create a matplotlib visualization of the graph."""
        
        # Create subgraph if filtering is applied
        if filtered_nodes:
            G = self.graph.subgraph(filtered_nodes)
        else:
            G = self.graph
            
        # Choose layout algorithm
        layout_functions = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'random': nx.random_layout,
            'shell': nx.shell_layout,
            'kamada_kawai': nx.kamada_kawai_layout
        }
        
        if layout_algorithm in layout_functions:
            pos = layout_functions[layout_algorithm](G, k=3, iterations=50)
        else:
            pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Set up the plot
        plt.figure(figsize=figsize)
        plt.title("GraphML Visualization - Cassandra Project Structure", fontsize=16, fontweight='bold')
        
        # Prepare node properties
        node_colors_list = []
        node_sizes_list = []
        
        for node in G.nodes():
            node_data = G.nodes[node]
            
            # Node color based on type
            node_type = node_data.get('type', 'default')
            node_colors_list.append(self.node_colors.get(node_type, self.node_colors['default']))
            
            # Node size based on attribute
            if node_size_attribute in node_data:
                try:
                    size_value = float(node_data[node_size_attribute])
                    # Normalize size (min 300, max 3000)
                    normalized_size = 300 + (size_value / 100) * 200
                    node_sizes_list.append(min(max(normalized_size, 300), 3000))
                except (ValueError, TypeError):
                    node_sizes_list.append(500)
            else:
                node_sizes_list.append(500)
        
        # Draw edges
        for edge_type, color in self.edge_colors.items():
            edges_of_type = [(u, v) for u, v, d in G.edges(data=True) 
                           if d.get('relation', 'default') == edge_type]
            if edges_of_type:
                nx.draw_networkx_edges(G, pos, edgelist=edges_of_type, 
                                     edge_color=color, alpha=0.6, width=1.5,
                                     arrows=True, arrowsize=20, arrowstyle='->')
        
        # Draw default edges
        default_edges = [(u, v) for u, v, d in G.edges(data=True) 
                        if d.get('relation') not in self.edge_colors]
        if default_edges:
            nx.draw_networkx_edges(G, pos, edgelist=default_edges, 
                                 edge_color=self.edge_colors['default'], 
                                 alpha=0.4, width=1, arrows=True, arrowsize=15)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, 
                             node_size=node_sizes_list, alpha=0.8, linewidths=2)
        
        # Draw labels if requested
        if show_labels:
            labels = {node: G.nodes[node].get('label', node.split('.')[-1]) 
                     for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        # Create legend
        legend_elements = []
        for node_type, color in self.node_colors.items():
            if node_type != 'default':
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=10, 
                                                label=f'{node_type.capitalize()} Node'))
        
        for edge_type, color in self.edge_colors.items():
            if edge_type != 'default':
                legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=3, 
                                                label=f'{edge_type.replace("_", " ").title()} Relationship'))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()
    
    def create_plotly_visualization(self, 
                                  layout_algorithm: str = 'spring',
                                  node_size_attribute: str = 'wmc',
                                  color_attribute: str = 'cbo',
                                  filtered_nodes: Optional[List[str]] = None):
        """Create an interactive plotly visualization."""
        
        # Create subgraph if filtering is applied
        if filtered_nodes:
            G = self.graph.subgraph(filtered_nodes)
        else:
            G = self.graph
        
        # Generate layout
        if layout_algorithm == 'spring':
            pos = nx.spring_layout(G, k=3, iterations=50)
        elif layout_algorithm == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_sizes = []
        node_colors_vals = []
        node_symbols = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            label = node_data.get('label', node.split('.')[-1])
            node_type = node_data.get('type', 'unknown')
            
            # Create hover text with key attributes
            hover_text = f"<b>{label}</b><br>"
            hover_text += f"Type: {node_type}<br>"
            hover_text += f"ID: {node}<br>"
            
            # Add key metrics
            key_attributes = ['cbo', 'wmc', 'loc', 'fanin', 'fanout', 'rfc', 'lcom']
            for attr in key_attributes:
                if attr in node_data:
                    hover_text += f"{attr.upper()}: {node_data[attr]}<br>"
            
            node_info.append(hover_text)
            node_text.append(label)
            
            # Node size based on attribute
            if node_size_attribute in node_data:
                try:
                    size_value = float(node_data[node_size_attribute])
                    normalized_size = 20 + (size_value / 50) * 30
                    node_sizes.append(min(max(normalized_size, 10), 60))
                except (ValueError, TypeError):
                    node_sizes.append(20)
            else:
                node_sizes.append(20)
            
            # Node color based on attribute
            if color_attribute in node_data:
                try:
                    node_colors_vals.append(float(node_data[color_attribute]))
                except (ValueError, TypeError):
                    node_colors_vals.append(0)
            else:
                node_colors_vals.append(0)
            
            # Node symbol based on type
            if node_type == 'class':
                node_symbols.append('square')
            elif node_type == 'method':
                node_symbols.append('circle')
            else:
                node_symbols.append('diamond')
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_data = G.edges[edge]
            relation = edge_data.get('relation', 'unknown')
            edge_info.append(f"{edge[0]} → {edge[1]}<br>Relation: {relation}")
        
        # Create edge trace
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=1, color='#888'),
                               hoverinfo='none',
                               mode='lines')
        
        # Create node trace
        node_trace = go.Scatter(x=node_x, y=node_y,
                               mode='markers+text',
                               hovertemplate='%{customdata}<extra></extra>',
                               customdata=node_info,
                               text=node_text,
                               textposition="middle center",
                               textfont=dict(size=8),
                               marker=dict(size=node_sizes,
                                         color=node_colors_vals,
                                         colorscale='Viridis',
                                         showscale=True,
                                         colorbar=dict(title=f"{color_attribute.upper()}<br>Values"),
                                         symbol=node_symbols,
                                         line=dict(width=2, color='white')))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(text='Interactive GraphML Visualization - Cassandra Project',
                                     x=0.5, xanchor='center'),
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Python code visualization using NetworkX and Plotly",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'))
        
        return fig
    
    def generate_statistics(self) -> Dict:
        """Generate statistics about the graph."""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': {},
            'edge_types': {},
            'key_metrics': {}
        }
        
        # Count node types
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'unknown')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # Count edge types
        for edge in self.graph.edges():
            edge_type = self.graph.edges[edge].get('relation', 'unknown')
            stats['edge_types'][edge_type] = stats['edge_types'].get(edge_type, 0) + 1
        
        # Calculate key metrics
        numeric_attributes = ['cbo', 'wmc', 'loc', 'fanin', 'fanout']
        for attr in numeric_attributes:
            values = []
            for node in self.graph.nodes():
                if attr in self.graph.nodes[node]:
                    try:
                        values.append(float(self.graph.nodes[node][attr]))
                    except (ValueError, TypeError):
                        pass
            
            if values:
                stats['key_metrics'][attr] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return stats
    
    def export_data(self, output_file: str, format: str = 'json'):
        """Export graph data to various formats."""
        if format.lower() == 'json':
            data = {
                'nodes': [],
                'edges': [],
                'statistics': self.generate_statistics()
            }
            
            for node in self.graph.nodes():
                node_data = dict(self.graph.nodes[node])
                node_data['id'] = node
                data['nodes'].append(node_data)
            
            for edge in self.graph.edges():
                edge_data = dict(self.graph.edges[edge])
                edge_data['source'] = edge[0]
                edge_data['target'] = edge[1]
                data['edges'].append(edge_data)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        elif format.lower() == 'csv':
            # Export nodes to CSV
            nodes_df = pd.DataFrame([dict(self.graph.nodes[node], id=node) 
                                   for node in self.graph.nodes()])
            nodes_df.to_csv(output_file.replace('.csv', '_nodes.csv'), index=False)
            
            # Export edges to CSV
            edges_df = pd.DataFrame([dict(self.graph.edges[edge], 
                                        source=edge[0], target=edge[1]) 
                                   for edge in self.graph.edges()])
            edges_df.to_csv(output_file.replace('.csv', '_edges.csv'), index=False)


def main():
    """Main function to run the visualizer."""
    parser = argparse.ArgumentParser(description='Visualize GraphML files with node attributes')
    parser.add_argument('--graphml_file', default='/model/data/R-SE/tools/graphgen/cassandra/cassandra_4399.graphml')
    parser.add_argument('--layout', choices=['spring', 'circular', 'random', 'kamada_kawai'], 
                       default='spring', help='Layout algorithm')
    parser.add_argument('--output', default='/model/data/R-SE/cassandra_4399.svg')
    parser.add_argument('--format', choices=['html', 'png', 'svg', 'json', 'csv'], 
                       default='svg', help='Output format')
    parser.add_argument('--interactive', action='store_true', 
                       help='Create interactive plotly visualization')
    parser.add_argument('--node-type-filter', help='Filter nodes by type (class/method)')
    parser.add_argument('--min-wmc', type=float, help='Minimum WMC value for filtering')
    parser.add_argument('--max-wmc', type=float, help='Maximum WMC value for filtering')
    parser.add_argument('--size-attribute', default='wmc', help='Attribute for node sizing')
    parser.add_argument('--color-attribute', default='cbo', help='Attribute for node coloring')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = GraphMLVisualizer(args.graphml_file)
    
    # Apply filters
    filter_kwargs = {}
    if args.node_type_filter:
        filter_kwargs['node_type'] = args.node_type_filter
    if args.min_wmc is not None:
        filter_kwargs['min_attribute_value'] = {'wmc': args.min_wmc}
    if args.max_wmc is not None:
        filter_kwargs['max_attribute_value'] = {'wmc': args.max_wmc}
    
    filtered_nodes = None
    if filter_kwargs:
        filtered_nodes = visualizer.filter_nodes(**filter_kwargs)
        print(f"Filtered to {len(filtered_nodes)} nodes")
    
    # Generate statistics
    stats = visualizer.generate_statistics()
    print("\nGraph Statistics:")
    print(f"Total Nodes: {stats['total_nodes']}")
    print(f"Total Edges: {stats['total_edges']}")
    print("Node Types:", stats['node_types'])
    print("Edge Types:", stats['edge_types'])
    
    # Create visualization
    if args.interactive:
        fig = visualizer.create_plotly_visualization(
            layout_algorithm=args.layout,
            node_size_attribute=args.size_attribute,
            color_attribute=args.color_attribute,
            filtered_nodes=filtered_nodes
        )
        
        if args.output:
            if args.format == 'html':
                fig.write_html(args.output)
            else:
                fig.write_image(args.output)
        else:
            fig.show()
    else:
        fig = visualizer.create_matplotlib_visualization(
            layout_algorithm=args.layout,
            node_size_attribute=args.size_attribute,
            filtered_nodes=filtered_nodes
        )
        
        if args.output:
            fig.savefig(args.output, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    # Export data if requested
    if args.format in ['json', 'csv']:
        output_file = args.output or f"graph_data.{args.format}"
        visualizer.export_data(output_file, args.format)
        print(f"Data exported to {output_file}")


if __name__ == "__main__":
    main()
