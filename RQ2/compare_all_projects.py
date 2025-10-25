"""
比较所有项目和融合方法的实验结果

本脚本用于汇总分析不同项目和不同融合方法的实验结果，
生成综合报告和可视化图表。

用法:
python compare_all_projects.py --results_dir <results_directory>
"""

import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def load_all_comparisons(results_dir):
    """加载所有项目的比较结果"""
    comparison_dir = os.path.join(results_dir, 'comparison')
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir, exist_ok=True)
        
    all_comparisons = {}
    
    # 查找所有项目的比较结果文件
    for file in os.listdir(comparison_dir):
        if file.endswith('_comparison.csv'):
            project = file.split('_comparison.csv')[0]
            file_path = os.path.join(comparison_dir, file)
            df = pd.read_csv(file_path)
            all_comparisons[project] = df
            
    return all_comparisons


def create_overall_comparison(all_comparisons):
    """创建所有项目和方法的综合比较表"""
    rows = []
    
    for project, df in all_comparisons.items():
        for _, row in df.iterrows():
            data = {
                'Project': project,
                'Fusion Method': row['Fusion Method'],
                'Accuracy': row['accuracy'],
                'Precision': row['precision'],
                'Recall': row['recall'],
                'F1': row['f1'],
                'MCC': row['mcc']
            }
            
            # 检查是否有AUC和AP指标
            # if 'auc' in row and not pd.isna(row['auc']):
            #     data['AUC'] = row['auc']
            # if 'ap' in row and not pd.isna(row['ap']):
            #     data['AP'] = row['ap']
                
            rows.append(data)
            
    return pd.DataFrame(rows)


def calculate_method_rankings(overall_df):
    """计算每种方法在各个项目上的排名"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    # 检查是否有AUC和AP指标
    # if 'AUC' in overall_df.columns:
    #     metrics.append('AUC')
    # if 'AP' in overall_df.columns:
    #     metrics.append('AP')
    
    rankings = pd.DataFrame()
    
    # 对每个项目，计算每个融合方法在每个指标上的排名
    for project in overall_df['Project'].unique():
        project_df = overall_df[overall_df['Project'] == project]
        
        for metric in metrics:
            # 计算该指标的排名（越高越好）
            project_df[f'{metric}_rank'] = project_df[metric].rank(ascending=False)
            
        if rankings.empty:
            rankings = project_df
        else:
            rankings = pd.concat([rankings, project_df])
    
    return rankings


def calculate_average_performance(rankings):
    """计算每种融合方法在所有项目上的平均表现"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    rank_metrics = [f'{m}_rank' for m in metrics]
    
    # 检查是否有AUC和AP指标
    # if 'AUC' in rankings.columns:
    #     metrics.append('AUC')
    #     rank_metrics.append('AUC_rank')
    # if 'AP' in rankings.columns:
    #     metrics.append('AP')
    #     rank_metrics.append('AP_rank')
    
    avg_performance = rankings.groupby('Fusion Method')[metrics].mean().reset_index()
    avg_ranks = rankings.groupby('Fusion Method')[rank_metrics].mean().reset_index()
    
    # 计算平均排名
    avg_ranks['Average Rank'] = avg_ranks[rank_metrics].mean(axis=1)
    
    # 合并平均表现和排名
    result = pd.merge(avg_performance, avg_ranks[['Fusion Method', 'Average Rank']], on='Fusion Method')
    
    # 按平均排名排序
    return result.sort_values('Average Rank')


def plot_average_performance(avg_performance, output_file=None):
    """绘制各融合方法的平均表现对比图"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
    
    # 检查是否有AUC和AP指标
    # if 'AUC' in avg_performance.columns:
    #     metrics.append('AUC')
    # if 'AP' in avg_performance.columns:
    #     metrics.append('AP')
    
    # 准备绘图数据
    plot_data = []
    for _, row in avg_performance.iterrows():
        for metric in metrics:
            plot_data.append({
                'Fusion Method': row['Fusion Method'],
                'Metric': metric,
                'Value': row[metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Metric', y='Value', hue='Fusion Method', data=plot_df)
    plt.title('Average Performance Across All Projects')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Fusion Method')
    
    if output_file:
        plt.savefig(output_file, dpi=800, bbox_inches='tight')
    else:
        plt.show()


def plot_heatmap(avg_performance, output_file=None):
    """绘制热力图，展示不同融合方法在各指标上的表现"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
    
    # 检查是否有AUC和AP指标
    # if 'AUC' in avg_performance.columns:
    #     metrics.append('AUC')
    # if 'AP' in avg_performance.columns:
    #     metrics.append('AP')
    
    # 准备热力图数据
    heatmap_data = avg_performance.set_index('Fusion Method')[metrics]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title('Performance Heatmap Across Metrics')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=800, bbox_inches='tight')
    else:
        plt.show()


def plot_project_performance(overall_df, metric='F1', output_file=None):
    """绘制每个项目上不同方法的表现比较"""
    plt.figure(figsize=(16, 10))
    
    ax = sns.barplot(x='Project', y=metric, hue='Fusion Method', data=overall_df)
    plt.title(f'{metric} Score Comparison Across Projects')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Fusion Method')
    
    # 旋转x轴标签，避免重叠
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=800, bbox_inches='tight')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='比较所有项目的实验结果')
    parser.add_argument('--results_dir', type=str, default='./results0', help='结果目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.join(args.results_dir, 'overall')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载所有项目的比较结果
    print("Loading comparison results for all projects...")
    all_comparisons = load_all_comparisons(args.results_dir)
    
    if not all_comparisons:
        print("No comparison results found. Please run comparisons for individual projects first.")
        return
    
    print(f"Found comparison results for {len(all_comparisons)} projects.")
    
    # 创建综合比较表
    overall_df = create_overall_comparison(all_comparisons)
    
    # 计算方法排名
    rankings = calculate_method_rankings(overall_df)
    
    # 计算平均表现
    avg_performance = calculate_average_performance(rankings)
    
    # 保存结果
    overall_file = os.path.join(output_dir, "overall_comparison.csv")
    overall_df.to_csv(overall_file, index=False)
    print(f"Overall comparison saved to: {overall_file}")
    
    rankings_file = os.path.join(output_dir, "method_rankings.csv")
    rankings.to_csv(rankings_file, index=False)
    print(f"Method rankings saved to: {rankings_file}")
    
    avg_performance_file = os.path.join(output_dir, "average_performance.csv")
    avg_performance.to_csv(avg_performance_file, index=False)
    print(f"Average performance saved to: {avg_performance_file}")
    
    # 打印平均表现
    print("\n===== Average Performance Across All Projects =====")
    print(avg_performance.to_string())
    
    # 绘制平均表现对比图
    avg_perf_chart = os.path.join(output_dir, "average_performance_chart.png")
    plot_average_performance(avg_performance, avg_perf_chart)
    print(f"Average performance chart saved to: {avg_perf_chart}")
    
    # 绘制热力图
    heatmap_file = os.path.join(output_dir, "performance_heatmap.png")
    plot_heatmap(avg_performance, heatmap_file)
    print(f"Performance heatmap saved to: {heatmap_file}")
    
    # 绘制项目F1对比图
    f1_comparison = os.path.join(output_dir, "f1_project_comparison.png")
    plot_project_performance(overall_df, 'F1', f1_comparison)
    print(f"F1 project comparison chart saved to: {f1_comparison}")
    
    # 绘制项目MCC对比图
    mcc_comparison = os.path.join(output_dir, "mcc_project_comparison.png")
    plot_project_performance(overall_df, 'MCC', mcc_comparison)
    print(f"MCC project comparison chart saved to: {mcc_comparison}")
    
    print("\nOverall comparison completed.")


if __name__ == "__main__":
    main()
