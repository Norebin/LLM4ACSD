import argparse
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import os
import numpy as np
from tqdm import tqdm

# 导入我们自己写的模块
from data_processing import CodeSmellDataset, NODE_FEATURE_KEYS, BalancedCodeSmellDataset, ImbalanceHandler
from models import ModelFactory, get_model_info
from sklearn.metrics import matthews_corrcoef, roc_auc_score

os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)


def train(model, loader, optimizer, criterion, device):
    """模型训练函数"""
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training", leave=False):
        if data is None: continue # 跳过加载失败的数据
        data = data.to(device)
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            print("输入特征有NaN或Inf！")
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, return_predictions=False):
    """模型评估函数"""
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = [] # 用于AUC计算
    for data in tqdm(loader, desc="Evaluating", leave=False):
        if data is None: continue
        data = data.to(device)
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            print("输入特征有NaN或Inf！")
        out = model(data)
        preds = out.argmax(dim=1)
        scores = torch.softmax(out, dim=1)[:, 1] # 获取正类的概率
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(data.y.cpu().tolist())
        all_scores.extend(scores.cpu().tolist())
    
    if len(all_labels) == 0:
        print("警告: 没有有效的预测结果！")
        metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'mcc': 0, 'auc': 0}
        if return_predictions:
            return metrics, [], []
        return metrics
        
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵:")
    print(cm)
    
    # 计算各种指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    # 计算AUC，需要检查标签是否只有一个类别
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_scores)
    else:
        auc = 0.5 # 如果只有一个类别，AUC没有意义，设为0.5

    metrics = {
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1,
        'mcc': mcc,
        'auc': auc
    }
    
    # 如果需要，一并返回预测结果和真实标签
    if return_predictions:
        return metrics, all_preds, all_labels
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='GNN for Code Smell Actionability Prediction')
    parser.add_argument('--model_type', type=str, default='GIN', choices=ModelFactory.get_available_models(), help='Type of GNN model')
    parser.add_argument('--csv_dir', type=str, default='/model/lxj/actionableSmell', help='Directory containing CSV files')
    parser.add_argument('--graph_root', type=str, default='/model/data/R-SE/tools/graphgen', help='Root directory of GraphML files')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--project', type=str, default=None, help='指定要训练的项目名称，如"cassandra"，不指定则训练所有项目')
    parser.add_argument('--mode', type=str, default='all', choices=['single', 'all'], help='训练模式：单个项目(single)或所有项目合并(all)')
    parser.add_argument('--balance_method', type=str, default='none', 
                      choices=ImbalanceHandler.get_available_methods(), 
                      help='数据平衡方法，用于处理类别不平衡')
    parser.add_argument('--compare_balance', action='store_true', 
                      help='比较不同的数据平衡方法，会依次尝试所有方法')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子，用于数据平衡和模型初始化')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 获取可用的项目名称
    project_names = []
    for f in os.listdir(args.csv_dir):
        if f.endswith('_Split.csv'):
            project_name = f.replace('_Split.csv', '')
            project_names.append(project_name)
    project_names.sort()
    print(f"可用项目: {project_names}")

    # 根据模式选择训练方式
    if args.mode == 'all' and args.project is None:
        # 合并所有项目数据进行训练
        print("模式: 使用所有项目合并数据进行训练")
        run_training(None, args, device)
    elif args.mode == 'single' and args.project is not None:
        # 对指定项目进行训练
        print(f"模式: 使用单个项目 {args.project} 进行训练")
        run_training(args.project, args, device)
    elif args.mode == 'single' and args.project is None:
        # 对每个项目分别进行训练
        print("模式: 对每个项目分别进行训练")
        
        # 如果要比较不同的平衡方法
        if args.compare_balance:
            print("比较不同的数据平衡方法")
            all_balance_methods = ImbalanceHandler.get_available_methods()
            balance_results = {}
            
            for project in project_names:
                print(f"\n{'='*50}")
                print(f"训练项目: {project}")
                print(f"{'='*50}")
                
                project_results = {}
                
                for balance_method in all_balance_methods:
                    print(f"\n{'*'*30}")
                    print(f"平衡方法: {balance_method}")
                    print(f"{'*'*30}")
                    
                    try:
                        metrics = run_training(project, args, device, balance_method)
                        project_results[balance_method] = metrics
                    except Exception as e:
                        print(f"项目 {project} 使用平衡方法 {balance_method} 训练失败: {e}")
                
                balance_results[project] = project_results
                
            # 打印所有项目和平衡方法的结果汇总
            print("\n\n最终结果汇总:")
            
            for project, method_results in balance_results.items():
                print(f"\n项目: {project}")
                print("平衡方法".ljust(15), "准确率".ljust(10), "精确率".ljust(10), "召回率".ljust(10), "F1分数".ljust(10))
                print("-" * 60)
                
                for method, metrics in method_results.items():
                    print(f"{method.ljust(15)} "
                          f"{metrics['accuracy']:.4f}".ljust(10),
                          f"{metrics['precision']:.4f}".ljust(10),
                          f"{metrics['recall']:.4f}".ljust(10),
                          f"{metrics['f1']:.4f}".ljust(10),
                          f"{metrics['auc']:.4f}".ljust(10),
                          f"{metrics['mcc']:.4f}".ljust(10),
                          )
            
            # 计算每个平衡方法的平均指标
            print("\n各平衡方法在所有项目上的平均表现:")
            print("平衡方法".ljust(15), "准确率".ljust(10), "精确率".ljust(10), "召回率".ljust(10), "F1分数".ljust(10))
            print("-" * 60)
            
            for method in all_balance_methods:
                # 收集所有项目在此方法下的指标
                method_metrics = []
                for project, project_results in balance_results.items():
                    if method in project_results:
                        method_metrics.append(project_results[method])
                
                if not method_metrics:
                    continue
                    
                # 计算平均值
                avg_accuracy = np.mean([m['accuracy'] for m in method_metrics])
                avg_precision = np.mean([m['precision'] for m in method_metrics])
                avg_recall = np.mean([m['recall'] for m in method_metrics])
                avg_f1 = np.mean([m['f1'] for m in method_metrics])
                avg_auc = np.mean([m['auc'] for m in method_metrics])
                avg_mcc = np.mean([m['mcc'] for m in method_metrics])
                
                print(f"{method.ljust(15)} "
                      f"{avg_accuracy:.4f}".ljust(10),
                      f"{avg_precision:.4f}".ljust(10),
                      f"{avg_recall:.4f}".ljust(10),
                      f"{avg_f1:.4f}".ljust(10),
                      f"{avg_auc:.4f}".ljust(10),
                      f"{avg_mcc:.4f}".ljust(10))
                
        else:
            # 不比较平衡方法，只使用指定的一种
            results = {}
            for project in project_names:
                print(f"\n{'='*50}")
                print(f"训练项目: {project}")
                print(f"{'='*50}")
                try:
                    metrics = run_training(project, args, device)
                    results[project] = metrics
                except Exception as e:
                    print(f"项目 {project} 训练失败: {e}")
            
            # 打印所有项目的结果汇总
            print("\n\n最终结果汇总:")
            print("项目名称".ljust(15), "准确率".ljust(10), "精确率".ljust(10), "召回率".ljust(10), "F1分数".ljust(10))
            print("-" * 60)
            for project, metrics in results.items():
                print(f"{project.ljust(15)} "
                      f"{metrics['accuracy']:.4f}".ljust(10),
                      f"{metrics['precision']:.4f}".ljust(10),
                      f"{metrics['recall']:.4f}".ljust(10),
                      f"{metrics['f1']:.4f}".ljust(10),
                      f"{metrics['auc']:.4f}".ljust(10),
                      f"{metrics['mcc']:.4f}".ljust(10))

    else:
        # 先对所有项目分别进行训练，然后对合并的数据进行训练
        print("模式: 先训练所有单个项目，然后训练合并数据")
        
        # 如果要比较不同的平衡方法
        if args.compare_balance:
            print("比较不同的数据平衡方法")
            all_balance_methods = ImbalanceHandler.get_available_methods()
            balance_results = {}
            
            # 处理每个项目
            for project in project_names:
                print(f"\n{'='*50}")
                print(f"训练项目: {project}")
                print(f"{'='*50}")
                
                project_results = {}
                
                for balance_method in all_balance_methods:
                    print(f"\n{'*'*30}")
                    print(f"平衡方法: {balance_method}")
                    print(f"{'*'*30}")
                    
                    try:
                        metrics = run_training(project, args, device, balance_method)
                        project_results[balance_method] = metrics
                    except Exception as e:
                        print(f"项目 {project} 使用平衡方法 {balance_method} 训练失败: {e}")
                
                balance_results[project] = project_results
            
            # 处理合并数据
            print(f"\n{'='*50}")
            print(f"训练所有项目合并数据")
            print(f"{'='*50}")
            
            combined_results = {}
            for balance_method in all_balance_methods:
                print(f"\n{'*'*30}")
                print(f"平衡方法: {balance_method}")
                print(f"{'*'*30}")
                
                try:
                    metrics = run_training(None, args, device, balance_method)
                    combined_results[balance_method] = metrics
                except Exception as e:
                    print(f"合并数据使用平衡方法 {balance_method} 训练失败: {e}")
            
            balance_results["all_combined"] = combined_results
            
            # 打印所有结果汇总
            print("\n\n最终结果汇总:")
            
            for project, method_results in balance_results.items():
                print(f"\n项目: {project}")
                print("平衡方法".ljust(15), "准确率".ljust(10), "精确率".ljust(10), "召回率".ljust(10), "F1分数".ljust(10))
                print("-" * 60)
                
                for method, metrics in method_results.items():
                    print(f"{method.ljust(15)} "
                          f"{metrics['accuracy']:.4f}".ljust(10),
                          f"{metrics['precision']:.4f}".ljust(10),
                          f"{metrics['recall']:.4f}".ljust(10),
                          f"{metrics['f1']:.4f}".ljust(10),
                          f"{metrics['auc']:.4f}".ljust(10),
                          f"{metrics['mcc']:.4f}".ljust(10))
            
            # 计算每个平衡方法的平均指标(不包括合并数据)
            print("\n各平衡方法在单个项目上的平均表现:")
            print("平衡方法".ljust(15), "准确率".ljust(10), "精确率".ljust(10), "召回率".ljust(10), "F1分数".ljust(10))
            print("-" * 60)
            
            for method in all_balance_methods:
                # 收集所有项目在此方法下的指标(排除合并数据)
                method_metrics = []
                for project, project_results in balance_results.items():
                    if project != "all_combined" and method in project_results:
                        method_metrics.append(project_results[method])
                
                if not method_metrics:
                    continue
                    
                # 计算平均值
                avg_accuracy = np.mean([m['accuracy'] for m in method_metrics])
                avg_precision = np.mean([m['precision'] for m in method_metrics])
                avg_recall = np.mean([m['recall'] for m in method_metrics])
                avg_f1 = np.mean([m['f1'] for m in method_metrics])
                
                print(f"{method.ljust(15)} "
                      f"{avg_accuracy:.4f}".ljust(10),
                      f"{avg_precision:.4f}".ljust(10),
                      f"{avg_recall:.4f}".ljust(10),
                      f"{avg_f1:.4f}".ljust(10))
            
        else:
            # 不比较平衡方法，只使用指定的一种
            results = {}
            for project in project_names:
                print(f"\n{'='*50}")
                print(f"训练项目: {project}")
                print(f"{'='*50}")
                try:
                    metrics = run_training(project, args, device)
                    results[project] = metrics
                except Exception as e:
                    print(f"项目 {project} 训练失败: {e}")
            
            print(f"\n{'='*50}")
            print(f"训练所有项目合并数据")
            print(f"{'='*50}")
            combined_metrics = run_training(None, args, device)
            results["all_combined"] = combined_metrics
            
            # 打印所有项目的结果汇总
            print("\n\n最终结果汇总:")
            print("项目名称".ljust(15), "准确率".ljust(10), "精确率".ljust(10), "召回率".ljust(10), "F1分数".ljust(10))
            print("-" * 60)
            for project, metrics in results.items():
                print(f"{project.ljust(15)} "
                      f"{metrics['accuracy']:.4f}".ljust(10),
                      f"{metrics['precision']:.4f}".ljust(10),
                      f"{metrics['recall']:.4f}".ljust(10),
                      f"{metrics['f1']:.4f}".ljust(10),
                      f"{metrics['auc']:.4f}".ljust(10),
                      f"{metrics['mcc']:.4f}".ljust(10))

def run_training(project, args, device, balance_method=None):
    """运行一个项目的训练和评估"""
    # 使用指定的平衡方法，如果为None则使用args中的方法
    if balance_method is None:
        balance_method = args.balance_method
        
    # 1. 加载原始数据集
    print("加载训练数据...")
    train_dataset_raw = CodeSmellDataset(csv_dir=args.csv_dir, graph_root_dir=args.graph_root, split='train', project=project)
    print("加载测试数据...")
    test_dataset = CodeSmellDataset(csv_dir=args.csv_dir, graph_root_dir=args.graph_root, split='test', project=project)
    
    # 显示原始训练集的标签分布
    train_labels = train_dataset_raw.smell_data['actionable'].value_counts().sort_index()
    print(f"原始训练集标签统计: {train_labels}")
    print(f"测试集标签统计: {test_dataset.smell_data['actionable'].value_counts().sort_index()}")
    
    # 应用数据平衡方法
    if balance_method != 'none':
        print(f"正在应用数据平衡方法: {balance_method}")
        train_dataset = BalancedCodeSmellDataset(train_dataset_raw, balance_method=balance_method, random_state=args.random_seed)
    else:
        train_dataset = train_dataset_raw
        
    # 使用PyG的DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, follow_batch=['x', 'y'], num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, follow_batch=['x', 'y'], num_workers=4)
    
    # 2. 初始化模型
    input_dim = len(NODE_FEATURE_KEYS)
    output_dim = 2  # 二元分类：可操作 vs 不可操作

    hidden_dims = [args.hidden_dim] * args.num_layers

    # 获取模型默认参数并更新
    model_info = get_model_info(args.model_type)
    model_params = model_info.get('default_params', {})
    
    model_params.update({
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'output_dim': output_dim,
        'dropout': args.dropout
    })

    model = ModelFactory.create_model(
        model_type=args.model_type,
        model_params=model_params
    ).to(device)
    
    print(f"模型: {args.model_type.upper()} 输入特征: {input_dim}")
    print(f"模型设备: {next(model.parameters()).device}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # 3. 训练和评估循环
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epoch", total=args.epochs):
        loss = train(model, train_loader, optimizer, criterion, device)
        
        if epoch % 10 == 0: # 每10个epoch评估一次
            train_metrics = evaluate(model, train_loader, device)
            test_metrics = evaluate(model, test_loader, device)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            print(f'  训练 -> 准确率: {train_metrics["accuracy"]:.4f}, 精确率: {train_metrics["precision"]:.4f}, 召回率: {train_metrics["recall"]:.4f}, F1: {train_metrics["f1"]:.4f}, AUC: {train_metrics["auc"]:.4f}, MCC: {train_metrics["mcc"]:.4f}')
            print(f'  测试 -> 准确率: {test_metrics["accuracy"]:.4f}, 精确率: {test_metrics["precision"]:.4f}, 召回率: {test_metrics["recall"]:.4f}, F1: {test_metrics["f1"]:.4f}, AUC: {train_metrics["auc"]:.4f}, MCC: {train_metrics["mcc"]:.4f}')

    print("训练完成.")
    final_test_metrics = evaluate(model, test_loader, device)
    print("\n--- 最终测试指标 ---")
    for key, value in final_test_metrics.items():
        print(f"{key}: {value:.4f}")
        
    return final_test_metrics
        

if __name__ == '__main__':
    main()