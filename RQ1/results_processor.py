import os
import pandas as pd
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='数据平衡实验结果处理工具')
    parser.add_argument('--results_dir', type=str, default='results', help='保存结果的目录')
    args = parser.parse_args()
    
    # 确保结果目录存在
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 读取终端输出的结果并组织成DataFrame
    print("请将实验结果文本粘贴在此处，完成后输入 'END' 并回车:")
    
    lines = []
    while True:
        line = input()
        if line == 'END':
            break
        lines.append(line)
    
    # 解析结果
    results = []
    current_project = None
    current_balance_method = None
    
    for line in lines:
        line = line.strip()
        
        # 检测项目名
        if line.startswith("项目:"):
            current_project = line.split("项目:")[1].strip()
            continue
            
        # 检测平衡方法和指标
        if line and not line.startswith("-") and not line.startswith("平衡方法"):
            parts = line.split()
            if len(parts) >= 5:  # 平衡方法 准确率 精确率 召回率 F1
                balance_method = parts[0].strip()
                try:
                    accuracy = float(parts[1].strip())
                    precision = float(parts[2].strip())
                    recall = float(parts[3].strip())
                    f1 = float(parts[4].strip())
                    
                    results.append({
                        'project': current_project,
                        'balance_method': balance_method,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    })
                except ValueError:
                    pass  # 忽略无法解析的行
    
    if not results:
        print("未能解析任何结果。请确保格式正确。")
        return
        
    # 创建DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存为CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = os.path.join(args.results_dir, f"balance_experiment_results_{timestamp}.csv")
    # results_df.to_csv(csv_file, index=False)
    print(f"结果已保存至: {csv_file}")
    
    # 输出汇总分析
    print("\n=== 结果汇总分析 ===")
    
    # 按平衡方法计算平均指标
    balance_methods_summary = results_df.groupby('balance_method').mean(numeric_only=True)
    print("\n平衡方法的平均表现:")
    print(balance_methods_summary[['accuracy', 'precision', 'recall', 'f1']])
    
    # 获取每个项目上表现最好的平衡方法
    best_f1_methods = results_df.loc[results_df.groupby('project')['f1'].idxmax()]
    print("\n每个项目上F1得分最高的平衡方法:")
    for _, row in best_f1_methods.iterrows():
        print(f"项目: {row['project']}, 最佳方法: {row['balance_method']}, F1: {row['f1']:.4f}")
    
    # 找出整体表现最好的平衡方法
    best_method = balance_methods_summary['f1'].idxmax()
    print(f"\n整体表现最好的平衡方法: {best_method}, 平均F1: {balance_methods_summary.loc[best_method, 'f1']:.4f}")
    
if __name__ == '__main__':
    main()
