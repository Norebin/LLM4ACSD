import datetime
import json
import pandas as pd
import numpy as np
import torch
import random
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

def set_seed(seed=42):
    """
    设置随机种子以确保实验的可重复性
    
    参数:
        seed: 随机种子值，默认为42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)

def open_smell_file(file_path):
    """
    打开smell数据文件并返回DataFrame
    
    参数:
        file_path: JSON数据文件路径
        
    返回:
        包含smell数据的DataFrame，且包含'index'列
    """
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 确保DataFrame有index列，如果没有则添加
        if 'index' not in df.columns:
            df['index'] = range(len(df))
            
        return df
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return pd.DataFrame() 

def peek(loader):
    """
    查看数据加载器(DataLoader)中的第一个批次数据
    
    参数:
        loader: PyTorch数据加载器
        
    返回:
        数据加载器中的第一个批次数据
    """
    # 获取迭代器
    data_iter = iter(loader)
    
    # 获取第一个批次
    first_batch = next(data_iter)
    
    # 返回第一个批次
    return first_batch 

# utils命名空间
class utils:
    @staticmethod
    def predict(model, data_loader, labels, verbose=True):
        """
        使用模型对数据进行预测并计算准确率
        
        参数:
            model: 训练好的模型
            data_loader: 包含测试数据的数据加载器
            labels: 真实标签
            verbose: 是否打印预测进度，默认为True
        
        返回:
            predictions: 模型的预测结果
            accuracy: 预测准确率
        """
        device = next(model.parameters()).device
        model.eval()
        
        predictions = []
        correct = 0
        total = 0
        all_true_labels = []
        all_pred_probs = []
        
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                if verbose and (i + 1) % 10 == 0:
                    print(f"预测进度: {i+1}/{len(data_loader)}")
                
                # 确保数据的维度正确
                if isinstance(data, list) and len(data) == 2:
                    # 如果data_loader返回(数据,标签)对
                    inputs = data[0]
                else:
                    # 如果data_loader只返回数据
                    inputs = data
                
                # 将输入数据展平
                if inputs.ndim > 2:
                    inputs = torch.flatten(inputs, start_dim=1)
                
                # 将数据移动到正确的设备上
                inputs = inputs.to(device)
                
                # 获取模型输出
                probs = model(inputs).cpu().numpy()
                
                all_pred_probs.extend(probs)
                # 获取预测结果
                predicted = (probs > 0.5).astype(int)
                predicted = predicted.squeeze()  # 或者 predicted = predicted.ravel()
                predictions.extend(predicted)
                
                if isinstance(labels, torch.Tensor):
                    true_labels = torch.argmax(labels[total:total+len(predicted)], dim=1).cpu().numpy()
                else:
                    true_labels = np.argmax(labels[total:total+len(predicted)], axis=1)
                all_true_labels.extend(true_labels)
                # print(f"predicted shape: {predicted.shape}, true_labels shape: {true_labels.shape}")
                # 计算当前批次的正确预测数
                batch_correct = (predicted == true_labels).sum()
                correct += batch_correct
                total += len(predicted)
        
        # 计算总体准确率
        accuracy = correct / total if total > 0 else 0
        all_true_labels = np.array(all_true_labels)
        predictions = np.array(predictions)
        all_pred_probs = np.array(all_pred_probs)
        
        # 计算其他指标
        precision = precision_score(all_true_labels, predictions, zero_division=0)
        recall = recall_score(all_true_labels, predictions, zero_division=0)
        f1 = f1_score(all_true_labels, predictions, zero_division=0)
        try:
            auc = roc_auc_score(all_true_labels, all_pred_probs)
        except Exception:
            auc = 0.0
        mcc = matthews_corrcoef(all_true_labels, predictions)
        
        if verbose:
            print(f"预测完成！准确率: {accuracy:.4f} 精确率: {precision:.4f} 召回率: {recall:.4f} F1: {f1:.4f} AUC: {auc:.4f} MCC: {mcc:.4f}")
        
        return predictions, accuracy, precision, recall, f1, auc, mcc
    @staticmethod
    def get_now(format_str="%Y_%m_%d_%H_%M_%S"):
        now = datetime.datetime.now()
        return now.strftime(format_str)