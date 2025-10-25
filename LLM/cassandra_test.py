import json
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_curve, roc_auc_score, confusion_matrix, precision_recall_fscore_support)
import numpy as np

def calculate_metrics(y_true, y_pred, y_pred_prob=None):
    """
    y_true: 真实标签 (0或1)  y_pred: 预测标签 (0或1)  y_pred_prob: 预测为正类的概率，用于绘制ROC曲线 (0到1之间的浮点数)
    返回各种性能指标的字典
    """
    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", labels=[0, 1])
   
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() 
    # 其他指标
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity,
        'F1 Score': f1,
        'Confusion Matrix': cm
    } 
    # 如果提供了概率值，计算ROC相关指标
    if y_pred_prob is not None:
        auc = roc_auc_score(y_true, y_pred_prob)
        metrics['ROC AUC'] = auc
    return metrics

def print_metrics(metrics):
    print("二分类评估指标:")
    print("-" * 40)
    for metric_name, value in metrics.items():
        if metric_name != 'Confusion Matrix':
            print(f"{metric_name}: {value:.4f}")
    cm = metrics['Confusion Matrix']
    print("\n混淆矩阵:")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

if __name__ == '__main__':
    # Simulated JSONL content as a list of dictionaries
    with open('/model/data/Research/output/cassandra_deepseek_freeze.jsonl', 'r', encoding='utf-8') as f:
        jsonl_data = [json.loads(line) for line in f]

    # Extract predictions and labels
    y_true = []
    y_pred = []
    for line in jsonl_data:
        if line["predict"] != '':
            predict = line["predict"].strip()
            if '0' in predict:
                y_pred.append(0)
            elif '1' in predict:
                y_pred.append(1)
            else:
                continue
            y_true.append(int(line["label"].strip()))

    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics)

    # 计算各类别的precision, recall, f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])
    # 计算整体的roc
    roc = roc_auc_score(y_true, y_pred)
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Print results
    print(f"\nROC:{roc}")
    print("\nClass-wise Metrics:")
    print(f"Class 0: Precision={precision[0]:.4f}, Recall={recall[0]:.4f}, F1-Score={f1[0]:.4f}")
    print(f"Class 1: Precision={precision[1]:.4f}, Recall={recall[1]:.4f}, F1-Score={f1[1]:.4f}")
    print("\nConfusion Matrix:")
    print(cm)