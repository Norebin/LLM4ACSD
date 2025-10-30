import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency
def summarize_best_mcc(directory_path='/model/lxj/LLM4ACS/RQ2/results1'):
    """
    Finds the record with the best 'test_mcc' from each '_results.csv' file
    in a directory and compiles them into a summary table.

    Args:
        directory_path (str): The path to the directory containing the CSV files.
    """
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return

    best_records = []
    
    # Iterate over each file in the specified directory
    for filename in os.listdir(directory_path):
        if filename.endswith('_results.csv'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if 'test_mcc' column exists
                if 'test_mcc' in df.columns:
                    # Find the row with the maximum 'test_mcc'
                    best_row = df.loc[df['test_mcc'].idxmax()]
                    # Add the source filename to the record
                    best_row['source_file'] = filename
                    best_records.append(best_row)
                else:
                    print(f"--- Skipping {filename}: 'test_mcc' column not found. ---")

            except Exception as e:
                print(f"--- Could not process {filename}: {e} ---\n")

    # Check if any records were found
    if not best_records:
        print("No '_results.csv' files with 'test_mcc' found to process.")
        return

    # Create a DataFrame from the list of best records
    summary_df = pd.DataFrame(best_records)
    
    # Reorder columns to have 'source_file' first for better readability
    cols = ['source_file'] + [col for col in summary_df.columns if col != 'source_file']
    summary_df = summary_df[cols]

    # Sort the summary table by 'test_mcc' in descending order
    summary_df = summary_df.sort_values(by='test_mcc', ascending=False)

    print("--- Summary of Best MCC Records ---")
    print(summary_df.to_string()) # Use to_string() to ensure all columns are displayed

def count_smell_count(directory_path='/model/data/R-SE/actionable'):
    """
    Analyzes all CSV files in a given directory to count 'actionable' values
    grouped by the 'Smell' column.

    Args:
        directory_path (str): The path to the directory containing CSV files.
    """
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return

    # Iterate over each file in the specified directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                # 1. Open the CSV file
                df = pd.read_csv(file_path)

                # Check if required columns exist
                if 'Smell' in df.columns and 'actionable' in df.columns:
                    print(f"--- Analysis for: {filename} ---")
                    
                    # 2. Group by 'Smell' and count 'actionable' values (0 and 1)
                    # pd.crosstab is a convenient way to create a frequency table
                    analysis_result = pd.crosstab(df['Smell'], df['actionable'])
                    # Add a 'Total' column which is the sum of counts for each smell
                    analysis_result['Total'] = analysis_result.sum(axis=1)
                    print(analysis_result)
                    print("\n" + "="*40 + "\n")
                else:
                    print(f"--- Skipping {filename}: Missing 'Smell' or 'actionable' column ---\n")

            except Exception as e:
                print(f"--- Could not process {filename}: {e} ---\n")

def merge_smell_data(source_dir='/model/lxj/LLM4ACS/RQ2/results1', lookup_dir='/model/lxj/actionableSmell'):
    """
    Merges data from two sets of CSV files based on a unique identifier.

    Args:
        source_dir (str): Directory containing source CSVs (ending in '_pre.csv').
        lookup_dir (str): Directory containing lookup CSVs (e.g., 'ProjectName_Split.csv').
    """
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found at '{source_dir}'")
        return
    if not os.path.isdir(lookup_dir):
        print(f"Error: Lookup directory not found at '{lookup_dir}'")
        return

    # 1. Iterate through files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('_pre.csv'):
            source_file_path = os.path.join(source_dir, filename)
            
            try:
                # 2. Extract project name and construct lookup file path
                parts = filename.split('_')
                if len(parts) > 2:
                    project_name = parts[2]
                    lookup_filename = f"{project_name}_Split.csv"
                    lookup_file_path = os.path.join(lookup_dir, lookup_filename)

                    if not os.path.exists(lookup_file_path):
                        print(f"--- Skipping {filename}: Lookup file not found at {lookup_file_path} ---\n")
                        continue

                    # 1. Open CSV file A & 3. Open CSV file B
                    df_A = pd.read_csv(source_file_path)
                    df_B = pd.read_csv(lookup_file_path)

                    # Ensure 'unique_id' exists in both dataframes
                    if 'unique_id' not in df_A.columns or 'unique_id' not in df_B.columns:
                        print(f"--- Skipping {filename}: 'unique_id' column missing in one of the files ---\n")
                        continue
                    
                    # 4. Merge data from B into A based on 'unique_id'
                    # Keep all columns from A, and add all from B that are not in A
                    merged_df = pd.merge(df_A, df_B, on='unique_id', how='left', suffixes=('', '_from_B'))
                    
                    # Overwrite the original file A with the merged data
                    merged_df.to_csv(source_file_path, index=False)
                    print(f"--- Successfully merged data into {filename} ---")

                else:
                    print(f"--- Skipping {filename}: Could not determine project name from filename. ---")

            except Exception as e:
                print(f"--- Could not process {filename}: {e} ---\n")

                import matplotlib.pyplot as plt

def plot_metrics_violin(directory_path='/model/lxj/LLM4ACS/RQ2/results1'):
    """
    Generates and saves violin plots for specified metrics from CSV files.

    For each CSV file ending in '_results.csv' in the given directory, this function
    creates a violin plot for a set of performance metrics. Files are grouped by
    the first two parts of their names (e.g., 'proj_model'), and each group
    is assigned a unique color palette. The plot is saved as a high-resolution
    SVG image.

    Args:
        directory_path (str): The path to the directory containing the CSV files.
    """
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return

    # Define the metrics to be plotted
    metrics_to_plot = ['test_accuracy', 'test_precision',  'test_recall', 'test_f1', 'test_mcc', 'test_auc']

    # Set font sizes for better readability
    title_fontsize = 52
    label_fontsize = 52
    tick_fontsize = 52

    # --- Grouping and Color Palette Logic ---
    group_palettes = {}
    # Define a list of distinct color palettes for different groups
    available_palettes = ['hls', 'Paired',  'husl', 'vlag']
    palette_index = 0

    # Iterate over each file in the specified directory
    for filename in os.listdir(directory_path):
        if filename.endswith('_results.csv'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                # Determine the group key from the filename (e.g., "project_model")
                parts = filename.split('_')
                if len(parts) >= 2:
                    group_key = f"{parts[0]}_{parts[1]}"
                else:
                    group_key = parts[0] # Fallback for simple filenames

                # Assign a new palette if this is a new group
                if group_key not in group_palettes:
                    group_palettes[group_key] = available_palettes[palette_index % len(available_palettes)]
                    palette_index += 1
                
                current_palette = group_palettes[group_key]

                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if all required metric columns exist
                if not all(metric in df.columns for metric in metrics_to_plot):
                    print(f"--- Skipping {filename}: One or more metric columns are missing. ---")
                    continue

                # Reshape the DataFrame from wide to long format for plotting
                df_melted = df[metrics_to_plot].melt(var_name='Metric', value_name='Value')
                
                # Remove 'test_' prefix from the 'Metric' column for cleaner labels
                df_melted['Metric'] = df_melted['Metric'].str.replace('test_', '')

                # Create the violin plot using the group's assigned palette
                plt.figure(figsize=(14, 10))
                sns.violinplot(x='Metric', y='Value', data=df_melted, palette=current_palette)
                
                # Set the y-axis limits to be between 0 and 1
                plt.ylim(0, 1)
                
                plt.title(f'{filename.replace("_results.csv", "")}', fontsize=title_fontsize)
                plt.xlabel('')
                plt.ylabel('')
                plt.xticks(rotation=45, fontsize=tick_fontsize) # Rotate labels and set font size
                plt.yticks(fontsize=tick_fontsize) # Set y-tick font size
                plt.tight_layout() # Adjust plot to ensure everything fits

                # Define the output image path, changing extension to .svg
                output_filename = filename.replace('.csv', '.svg')
                output_path = os.path.join(directory_path, output_filename)

                # Save the plot as an SVG file
                plt.savefig(output_path, format='svg')
                plt.close() # Close the figure to free up memory

                print(f"--- Successfully generated plot for {filename} and saved to {output_filename} ---")

            except Exception as e:
                print(f"--- Could not process {filename}: {e} ---\n")

def analyze_smell_performance_by_group(directory_path='/model/lxj/LLM4ACS/RQ2/results1'):
    """
    Groups CSV files by name, merges them, calculates classification metrics
    (including TP, FP, TN, FN) for each 'Smell' type, and performs pairwise
    hypothesis testing to compare their detection performance.

    Args:
        directory_path (str): The path to the directory containing the CSV files.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return

    # Dictionary to hold dataframes for each group before merging
    grouped_dfs = {}

    # 1. Group and read CSV files
    for filename in os.listdir(directory_path):
        if filename.endswith('_pre.csv'):
            file_path = os.path.join(directory_path, filename)
            
            # Determine the group key from the filename (e.g., "project_model")
            parts = filename.split('_')
            if len(parts) >= 2:
                group_key = f"{parts[0]}_{parts[1]}"
            else:
                continue # Skip files that don't match the expected format

            try:
                df = pd.read_csv(file_path)
                # Check for required columns
                if not all(col in df.columns for col in ['Smell', 'true_label', 'predicted_label']):
                    print(f"--- Skipping {filename}: Missing required columns (Smell, true_label, predicted_label). ---")
                    continue
                
                if group_key not in grouped_dfs:
                    grouped_dfs[group_key] = []
                grouped_dfs[group_key].append(df)

            except Exception as e:
                print(f"--- Could not process {filename}: {e} ---\n")

    if not grouped_dfs:
        print("No suitable '_pre.csv' files found to process.")
        return

    # 2. Merge, analyze, and report for each group
    for group_name, dfs_to_merge in grouped_dfs.items():
        print(f"\n{'='*20} Analysis for Group: {group_name} {'='*20}")
        
        # Merge all dataframes in the current group
        merged_df = pd.concat(dfs_to_merge, ignore_index=True)
        
        # Get unique smell types in this group
        smell_types = sorted(merged_df['Smell'].unique())
        
        results = []
        smell_data_for_test = {}
        for smell in smell_types:
            # Filter data for the specific smell
            smell_df = merged_df[merged_df['Smell'] == smell]
            
            y_true = smell_df['true_label']
            y_pred = smell_df['predicted_label']
            
            # Calculate confusion matrix components
            # confusion_matrix returns [[TN, FP], [FN, TP]]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            
            results.append({
                'Smell': smell,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'MCC': mcc,
                'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
            })
            
            # Store data for hypothesis testing
            smell_data_for_test[smell] = {'y_true': y_true, 'y_pred': y_pred}
            smell_data_for_test['God Class'] = {}
        # Create a DataFrame from the results for pretty printing
        if results:
            results_df = pd.DataFrame(results)
            # Reorder columns for better readability
            column_order = ['Smell', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'TP', 'FP', 'TN', 'FN']
            results_df = results_df[column_order]
            print("--- Performance Metrics by Smell Type ---")
            print(results_df.to_string(index=False))
        else:
            print("No smells found to analyze for this group.")

        # 3. Perform pairwise hypothesis testing
        if len(smell_types) > 1:
            print("\n--- Pairwise Performance Comparison (Chi-squared test) ---")
            print("H0: The proportion of correct predictions is independent of the smell type (i.e., no performance difference).")
            print("A low p-value (< 0.05) suggests a statistically significant difference in performance.\n")
            smell_types.append('God Class')
            test_results = []
            # Iterate through all unique pairs of smells
            for i in range(len(smell_types)):
                for j in range(i + 1, len(smell_types)):
                    smell1_name = smell_types[i]
                    smell2_name = smell_types[j]
                    
                    data1 = smell_data_for_test[smell1_name]
                    data2 = smell_data_for_test[smell2_name]
                    
                    # Number of correct/incorrect predictions for each smell
                    correct1 = (data1['y_true'] == data1['y_pred']).sum()
                    incorrect1 = len(data1['y_true']) - correct1
                    
                    if smell2_name == 'God Class':
                        correct2 = 1000
                        incorrect2 = 368
                    else:
                        correct2 = (data2['y_true'] == data2['y_pred']).sum()
                        incorrect2 = len(data2['y_true']) - correct2
                    
                    # Create a 2x2 contingency table
                    #       Correct | Incorrect
                    # --------------------------
                    # Smell1|  c1     |   i1
                    # Smell2|  c2     |   i2
                    contingency_table = [[correct1, incorrect1], [correct2, incorrect2]]
                    
                    try:
                        chi2, p_value, _, _ = chi2_contingency(contingency_table)
                        significant = 'Yes' if p_value < 0.05 else 'No'
                        test_results.append({
                            'Comparison': f"{smell1_name} vs {smell2_name}",
                            'Chi2-statistic': chi2,
                            'p-value': p_value,
                            'Significant (p<0.05)': significant
                        })
                    except ValueError as e:
                        # This can happen if a row/column in the table sums to zero
                        test_results.append({
                            'Comparison': f"{smell1_name} vs {smell2_name}",
                            'Chi2-statistic': 'N/A',
                            'p-value': 'N/A',
                            'Significant (p<0.05)': f"Error: {e}"
                        })

            if test_results:
                test_results_df = pd.DataFrame(test_results)
                print(test_results_df.to_string(index=False))

        print("=" * (42 + len(group_name)))

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def simple_cellvaluetype_subgraph(file_path, save_path=None, dpi=300):
    """简化版本，避免边标签错误"""
    
    G = nx.read_graphml(file_path)
    
    # 找到目标方法
    target_method = None
    for node in G.nodes():
        if 'cellValueType' in node and 'ColumnMetadata' in node:
            target_method = node
            break
    
    if not target_method:
        print("未找到目标方法")
        return None
    
    print(f"找到目标方法: {target_method}")
    
    # 获取相关节点
    connected_nodes = set([target_method])
    connected_nodes.update(G.neighbors(target_method))
    
    # 添加指向目标的节点
    for node in G.nodes():
        if G.has_edge(node, target_method):
            connected_nodes.add(node)
    
    print(f"找到 {len(connected_nodes)} 个相关节点")
    
    # 创建子图
    subG = G.subgraph(connected_nodes).copy()
    
    # 绘制
    plt.figure(figsize=(20, 16), dpi=dpi)
    pos = nx.spring_layout(subG, k=4, iterations=100, seed=42)
    
    # 节点样式和标签
    node_colors = []
    node_sizes = []
    labels = {}
    
    for node in subG.nodes():
        node_data = subG.nodes[node]
        node_type = node_data.get('type', 'unknown')
        label = node_data.get('label', node.split('.')[-1])
        
        if node == target_method:
            node_colors.append('#FF6B35')  # 橙红色
            node_sizes.append(6000)
            labels[node] = f"{label}\n[TARGET METHOD]"
        elif node_type == 'class':
            node_colors.append('#E74C3C')  # 红色
            node_sizes.append(4000)
            # 添加类的关键属性
            attrs = []
            if 'cbo' in node_data:
                attrs.append(f"CBO:{node_data['cbo']}")
            if 'wmc' in node_data:
                attrs.append(f"WMC:{node_data['wmc']}")
            if 'loc' in node_data:
                attrs.append(f"LOC:{node_data['loc']}")
            attr_str = ', '.join(attrs[:3])  # 显示前3个属性
            labels[node] = f"{label}\n({attr_str})" if attrs else label
            
        elif node_type == 'method':
            node_colors.append('#3498DB')  # 蓝色
            node_sizes.append(3000)
            # 添加方法的关键属性
            attrs = []
            if 'loc' in node_data:
                attrs.append(f"LOC:{node_data['loc']}")
            if 'wmc' in node_data:
                attrs.append(f"WMC:{node_data['wmc']}")
            if 'parametersqty' in node_data:
                attrs.append(f"Params:{node_data['parametersqty']}")
            attr_str = ', '.join(attrs[:3])
            labels[node] = f"{label}\n({attr_str})" if attrs else label
        else:
            node_colors.append('#2ECC71')  # 绿色
            node_sizes.append(2000)
            labels[node] = label
    
    # 绘制边（根据关系类型设置不同样式）
    edge_colors = []
    edge_widths = []
    for edge in subG.edges():
        edge_data = subG.edges[edge]
        relation = edge_data.get('relation', 'unknown')
        
        if relation == 'calls':
            edge_colors.append('#E67E22')  # 橙色
            edge_widths.append(3.0)
        elif relation == 'belongs_to':
            edge_colors.append('#9B59B6')  # 紫色
            edge_widths.append(2.0)
        else:
            edge_colors.append('#95A5A6')  # 灰色
            edge_widths.append(1.5)
    
    # 绘制图形
    nx.draw_networkx_edges(subG, pos, 
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.7, 
                          arrows=True, 
                          arrowsize=25,
                          arrowstyle='->',
                          connectionstyle="arc3,rad=0.1")
    
    nx.draw_networkx_nodes(subG, pos, 
                          node_color=node_colors, 
                          node_size=node_sizes, 
                          alpha=0.9, 
                          linewidths=2, 
                          edgecolors='black')
    
    nx.draw_networkx_labels(subG, pos, labels, 
                           font_size=10, 
                           font_weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='white', 
                                   edgecolor='none', 
                                   alpha=0.8))
    
    plt.title(f"cellValueType Method Dependencies with Node Attributes\nNodes: {len(connected_nodes)}, Edges: {subG.number_of_edges()}", 
             fontsize=16, fontweight='bold', pad=20)
    
    # 图例
    legend_elements = [
        patches.Patch(color='#FF6B35', label='Target Method (cellValueType)'),
        patches.Patch(color='#E74C3C', label='Class (with CBO, WMC, LOC)'),
        patches.Patch(color='#3498DB', label='Method (with LOC, WMC, Params)'),
        patches.Patch(color='#2ECC71', label='Other'),
        patches.Patch(color='#E67E22', label='Calls Relationship'),
        patches.Patch(color='#9B59B6', label='Belongs To Relationship'),
        patches.Patch(color='#95A5A6', label='Other Relationship')
    ]
    plt.legend(handles=legend_elements, 
              loc='upper left', 
              bbox_to_anchor=(0.02, 0.98),
              fontsize=10)
    
    
    plt.axis('off')
    plt.tight_layout()
    
    # 保存
    if save_path is None:
        save_path = f"cellValueType_subgraph.png"
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.2)
    
    print(f"\n图片已保存: {save_path}")
    print(f"分辨率: {dpi} DPI")
    
    plt.show()
    
    return save_path, subG

if __name__ == '__main__':
    # analyze_smell_performance_by_group()
    # 使用函数
    simple_cellvaluetype_subgraph('/model/data/R-SE/tools/graphgen/cassandra/cassandra_4399.graphml','/model/data/R-SE/sourceProject/actionableCS/cassandra/cassandra-5.0-alpha1/src/java/org/apache/cassandra/index/sai/utils/cellValueType_subgraph_with_attributes.png', dpi=300)

