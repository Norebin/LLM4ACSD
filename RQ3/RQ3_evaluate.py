# 评估大模型在数据集上预测的指标（Precision、accuracy、recall、F1、MCC）
# 模型预测结果位于./RQ3文件夹下，每个jsonl文件按照模型-微调方式-测试项目命名，内容形如{"prompt": "system\nYou are a helpful assistant.\nuser\nGiven a snippet of 'Complex Method' smell code, determine whether it requires mandatory refactoring. Output 1 if the code smell must be refactored, or 0 if refactoring is optional or not necessary. Respond with only a single digit: 1 or 0, and provide no explanation.\nCode:public static void combineExcelContentProperty(ExcelContentProperty combineExcelContentProperty, ExcelContentProperty excelContentProperty) {\n    if (excelContentProperty == null) {\n        return;\n    }\n    if (excelContentProperty.getField() != null) {\n        combineExcelContentProperty.setField(excelContentProperty.getField());\n    }\n    if (excelContentProperty.getConverter() != null) {\n        combineExcelContentProperty.setConverter(excelContentProperty.getConverter());\n    }\n    if (excelContentProperty.getDateTimeFormatProperty() != null) {\n        combineExcelContentProperty.setDateTimeFormatProperty(excelContentProperty.getDateTimeFormatProperty());\n    }\n    if (excelContentProperty.getNumberFormatProperty() != null) {\n        combineExcelContentProperty.setNumberFormatProperty(excelContentProperty.getNumberFormatProperty());\n    }\n    if (excelContentProperty.getContentStyleProperty() != null) {\n        combineExcelContentProperty.setContentStyleProperty(excelContentProperty.getContentStyleProperty());\n    }\n    if (excelContentProperty.getContentFontProperty() != null) {\n        combineExcelContentProperty.setContentFontProperty(excelContentProperty.getContentFontProperty());\n    }\n}\nassistant\n", "predict": "1", "label": "1\n"}
# 指标的计算需要依据prediction和label，但需要注意的是，模型的预测并不总是能遵守我们的指令（即按照0、1输出预测）所以需要考虑如何处理
# 将指标计算结果存在表中

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# 定义结果目录和输出文件
RESULTS_DIR = Path("./RQ3")
OUTPUT_CSV = Path("./RQ3_evaluation_results.csv")

def parse_value(value: str) -> Optional[int]:
    """
    从一个字符串中解析出第一个出现的 '0' 或 '1'。
    处理模型预测和标签可能出现的格式不一致问题。
    """
    if not isinstance(value, str):
        return None
    
    # 使用正则表达式查找第一个 '0' 或 '1'
    match = re.search(r'(0|1)', value)
    if match:
        return int(match.group(0))
    
    return None # 如果找不到，则返回None

def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """
    根据真实标签和预测标签计算一系列分类指标。
    处理了标签可能不完整的情况（例如，预测中只有一类）。
    """
    if not y_true or not y_pred:
        return {
            "Precision": 0, "Recall": 0, "Accuracy": 0, 
            "F1_Score": 0, "MCC": 0, "TP": 0, "FN": 0, 
            "FP": 0, "TN": 0, "Parsed_Count": 0, "Total_Count": 0
        }

    # zero_division参数可以在某个类没有被预测时避免警告，将该指标设为0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    metrics = {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
    }
    return metrics

def evaluate_file(file_path: Path) -> Dict[str, Any]:
    """
    读取单个jsonl文件，解析内容，并计算评估指标。
    """
    y_true, y_pred = [], []
    total_lines = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            try:
                data = json.loads(line)
                label = parse_value(data.get("label"))
                prediction = parse_value(data.get("predict"))

                if label is not None and prediction is not None:
                    y_true.append(label)
                    y_pred.append(prediction)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"警告：处理文件 {file_path.name} 的某一行时出错: {e}")

    file_metrics = calculate_metrics(y_true, y_pred)
    file_metrics["Parsed_Count"] = len(y_true)
    file_metrics["Total_Count"] = total_lines
    
    return file_metrics

def parse_filename(filename: str) -> Tuple[str, str]:
    """从复杂的文件名中解析出模型名称和数据集名称。"""
    base_name = filename.removesuffix(".jsonl")
    
    # 优先处理更具体、更长的分隔符
    if "_all_datasets" in base_name:
        model_name = base_name.removesuffix("_all_datasets")
        dataset_name = "all_datasets"
    elif "_test_" in base_name:
        # 使用rsplit确保从右侧分割，处理模型名称中可能包含'_test_'的情况
        model_name, test_part = base_name.rsplit("_test_", 1)
        dataset_name = f"test_{test_part}"
    else:
        # 如果上述规则都不匹配，则将整个文件名视为模型名
        model_name = base_name
        dataset_name = "unknown"
        
    return model_name, dataset_name

def main():
    """
    主函数：遍历结果目录，处理所有jsonl文件，
    计算指标并将结果汇总到CSV文件中。
    """
    if not RESULTS_DIR.is_dir():
        print(f"错误：结果目录 '{RESULTS_DIR}' 不存在。")
        return

    all_results = []
    
    files_to_process = sorted([f for f in RESULTS_DIR.iterdir() if f.name.endswith(".jsonl")])

    for file_path in files_to_process:
        print(f"正在处理: {file_path.name}...")
        
        model_name, dataset_name = parse_filename(file_path.name)
        
        metrics = evaluate_file(file_path)
        
        result_row = {
            "Model": model_name,
            "Dataset": dataset_name,
            **metrics
        }
        all_results.append(result_row)

    if not all_results:
        print("未找到任何结果文件进行评估。")
        return

    # 创建DataFrame并进行格式化
    df = pd.DataFrame(all_results)
    
    # 将数值类型指标格式化为浮点数，保留4位小数
    float_cols = ["Precision", "Recall", "Accuracy", "F1_Score", "MCC"]
    for col in float_cols:
        df[col] = df[col].apply(lambda x: f"{x:.4f}")

    # 重新排列列的顺序，使其更具可读性
    ordered_cols = [
        "Model", "Dataset", "Accuracy", "Precision", "Recall", 
        "F1_Score", "MCC", "TP", "FN", "FP", "TN", 
        "Parsed_Count", "Total_Count"
    ]
    df = df[ordered_cols]

    # 保存到CSV文件
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print("\n评估完成!")
    print("=" * 50)
    print(df.to_string())
    print("=" * 50)
    print(f"\n结果已保存到: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
