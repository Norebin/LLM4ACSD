"""
比较不同融合方法的实验结果

本脚本用于比较不同注意力融合方法在代码气味可操作性分类任务上的表现。
它读取各个融合方法的实验结果，并生成比较报告。

用法:
python compare_fusion_methods.py --project <project_name> --results_dir <results_directory>
"""

import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir, project):
    """加载所有融合方法的实验结果"""
    fusion_types = ['self_attention', 'cross_attention', 'progress']
    results = {}
    
    for fusion_type in fusion_types:
        result_file = os.path.join(results_dir, f"lora_phi4_{project}_{fusion_type}_results.csv")
        if os.path.exists(result_file):
            df = pd.read_csv(result_file)
            results[fusion_type] = df
        else:
            print(f"Warning: Results file not found for {fusion_type}: {result_file}")
    
    return results


def get_best_epoch_results(results_dict):
    """获取每种方法在测试集上F1得分最高的epoch的结果"""
    best_results = {}
    
    for fusion_type, df in results_dict.items():
        best_epoch = df.loc[df['test_mcc'].idxmax()]
        best_results[fusion_type] = {
            'epoch': int(best_epoch['epoch']),
            'accuracy': best_epoch['test_accuracy'],
            'precision': best_epoch['test_precision'],
            'recall': best_epoch['test_recall'],
            'f1': best_epoch['test_f1'],
            'mcc': best_epoch['test_mcc'],
            'auc': best_epoch.get('test_auc', None),
            'ap': best_epoch.get('test_ap', None)
        }
    
    return best_results


def create_comparison_table(best_results):
    """创建比较表格"""
    table_data = []
    
    for fusion_type, metrics in best_results.items():
        row = {'Fusion Method': fusion_type}
        row.update(metrics)
        table_data.append(row)
    
    return pd.DataFrame(table_data)


def plot_learning_curves(results_dict, metric='test_f1', output_file=None):
    """绘制学习曲线"""
    plt.figure(figsize=(12, 8))
    
    for fusion_type, df in results_dict.items():
        plt.plot(df['epoch'], df[metric], marker='o', label=fusion_type)
    
    plt.title(f'Learning Curves ({metric})', fontsize=26)
    plt.xlabel('Epoch', fontsize=26)
    plt.ylabel(metric, fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=24, loc='lower right')
    
    if output_file:
        plt.savefig(output_file, format='svg', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_metrics_comparison(comparison_df, output_file=None):
    """绘制不同融合方法的评估指标对比"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # 如果存在AUC和AP指标，也包括它们
    # if 'auc' in comparison_df.columns and not comparison_df['auc'].isnull().all():
    #     metrics.extend(['auc', 'ap'])
    
    # 转换数据格式用于绘图
    plot_data = []
    for _, row in comparison_df.iterrows():
        for metric in metrics:
            plot_data.append({
                'Fusion Method': row['Fusion Method'],
                'Metric': metric,
                'Value': row[metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(20, 16))
    sns.barplot(x='Metric', y='Value', hue='Fusion Method', data=plot_df)
    plt.title('Performance Metrics Comparison', fontsize=30)
    plt.xlabel('Metric', fontsize=30)
    plt.ylabel('Value', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Fusion Method', fontsize=26)
    
    if output_file:
        plt.savefig(output_file, format='svg', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='比较不同融合方法的实验结果')
    parser.add_argument('--project', type=str, default='all_projects', help='项目名称')
    parser.add_argument('--results_dir', type=str, default='/model/lxj/LLM4ACS/RQ2/results0', help='结果目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.join(args.results_dir, 'comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载结果
    print(f"Loading results for project: {args.project}")
    results_dict = load_results(args.results_dir, args.project)
    
    if not results_dict:
        print("No results found. Please ensure experiments have been run.")
        return
    
    # 获取每种方法的最佳结果
    best_results = get_best_epoch_results(results_dict)
    
    # 创建比较表格
    comparison_df = create_comparison_table(best_results)
    
    # 保存比较表格
    comparison_file = os.path.join(output_dir, f"{args.project}_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Comparison table saved to: {comparison_file}")
    
    # 打印比较表格
    print("\n===== Performance Comparison =====")
    print(comparison_df.to_string())
    
    # 绘制F1学习曲线
    f1_curve_file = os.path.join(output_dir, f"{args.project}_f1_learning_curves.svg")
    plot_learning_curves(results_dict, 'test_f1', f1_curve_file)
    print(f"F1 learning curves saved to: {f1_curve_file}")
    
    # 绘制准确率学习曲线
    acc_curve_file = os.path.join(output_dir, f"{args.project}_accuracy_learning_curves.svg")
    plot_learning_curves(results_dict, 'test_accuracy', acc_curve_file)
    print(f"Accuracy learning curves saved to: {acc_curve_file}")
    
    # 绘制指标对比图
    metrics_comparison_file = os.path.join(output_dir, f"{args.project}_metrics_comparison.svg")
    plot_metrics_comparison(comparison_df, metrics_comparison_file)
    print(f"Metrics comparison chart saved to: {metrics_comparison_file}")
    
    print("\nComparison completed.")


if __name__ == "__main__":
    main()
