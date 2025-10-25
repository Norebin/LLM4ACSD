import matplotlib.pyplot as plt
import numpy as np
def GNN_plot():
    # Data from the table
    networks = ['GCN', 'GIN', 'GAT', 'GraphSAGE']
    accuracy = [0.6567, 0.6963, 0.6500, 0.7301]
    precision = [0.7286, 0.6780, 0.6700, 0.7725]
    recall = [0.4527, 0.7031, 0.5495, 0.6195]
    f1_score = [0.5584, 0.6903, 0.6042, 0.6876]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each metric as a line
    plt.plot(networks, accuracy, marker='o', linestyle='-', label='Accuracy')
    plt.plot(networks, precision, marker='s', linestyle='--', label='Precision')
    plt.plot(networks, recall, marker='^', linestyle='-.', label='Recall')
    plt.plot(networks, f1_score, marker='d', linestyle=':', label='F1-score')

    # Add title and labels
    # plt.title('Comparison of Graph Network Model Performance')
    plt.xlabel('Graph Network Model')
    plt.ylabel('Score')

    # Show legend
    plt.legend()

    # Display the specific value on each data point
    for i, network in enumerate(networks):
        plt.text(network, accuracy[i], f'{accuracy[i]:.4f}', ha='center', va='bottom')
        plt.text(network, precision[i], f'{precision[i]:.4f}', ha='center', va='bottom')
        plt.text(network, recall[i], f'{recall[i]:.4f}', ha='center', va='bottom')
        plt.text(network, f1_score[i], f'{f1_score[i]:.4f}', ha='center', va='bottom')

    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()
    # You can also use other formats like .pdf or .eps
    plt.savefig('/model/data/R-SE/pygraph_network_performance.svg', format='svg')

def pair_t():
    import pandas as pd
    from scipy import stats
    import itertools
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 1. 准备数据
    # 我们将你的数据整理成一个字典，然后转换成 Pandas DataFrame
    data = {
        'Model': ['DeepSeek', 'CodeLlama', 'Qwen', 'Phi4'],
        'Complex Method': [0.7764, 0.8091, 0.8145, 0.8090],
        'Feature Envy': [0.8129, 0.7722, 0.7801, 0.7799],
        'God Class': [0.7666, 0.7517, 0.7494, 0.7676],
        'Long Method': [0.7777, 0.7629, 0.7556, 0.7760],
        'Long Parameter List': [0.7667, 0.7697, 0.7846, 0.7651]
    }

    df = pd.DataFrame(data)
    df = df.set_index('Model')

    print("--- 原始 F1-Score 数据 ---")
    print(df)
    print("\n" + "="*40 + "\n")

    # 2. 确定所有需要比较的异味对
    smell_types = df.columns
    smell_pairs = list(itertools.combinations(smell_types, 2))

    # 3. 执行配对t检验并存储结果
    results = []
    for pair in smell_pairs:
        smell1, smell2 = pair
        
        # 获取两种异味对应的F1-score列表
        f1_scores1 = df[smell1]
        f1_scores2 = df[smell2]
        
        # 执行配对t检验
        t_stat, p_value = stats.ttest_rel(f1_scores1, f1_scores2)
        
        results.append({
            'Smell 1': smell1,
            'Smell 2': smell2,
            'T-statistic': t_stat,
            'P-value': p_value
        })

        # 将结果转换为 DataFrame 以方便查看
        results_df = pd.DataFrame(results)

        # 4. 解读和展示结果
        # 添加一列来判断差异是否显著 (p < 0.05)
        results_df['Significant Difference (p < 0.05)'] = results_df['P-value'] < 0.05

        print("--- 配对 t-检验结果 ---")
        print(results_df)

        # 5. (推荐) 使用热力图可视化 p-value 矩阵
        # 创建一个空的 DataFrame 用于存储 p-value 矩阵
        p_value_matrix = pd.DataFrame(index=smell_types, columns=smell_types, dtype=float)

        # 填充矩阵
        for index, row in results_df.iterrows():
            p_value_matrix.loc[row['Smell 1'], row['Smell 2']] = row['P-value']
            p_value_matrix.loc[row['Smell 2'], row['Smell 1']] = row['P-value'] # 镜像填充

        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(p_value_matrix, annot=True, cmap='viridis_r', fmt=".4f", linewidths=.5)
        plt.title('P-value Matrix from Paired T-tests on F1-scores', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        plt.savefig('/model/data/R-SE/pari_t.svg', format='svg')

if __name__ == '__main__':
    pair_t()