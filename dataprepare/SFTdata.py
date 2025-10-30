import os
import pandas as pd
import json
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def train():
    # 转化为 Alpaca 格式
    alpaca_data_a = []
    # 读取 CSV 文件
    df_a = pd.read_csv("/model/data/Research/PEFTdata/testset_CM.csv")
    for _, row in df_a.iterrows():
        entry = {
            "instruction": "Given the following code snippet, identify if it contains a code smell (respond with 1 for code smell present, 0 for clean code).",
            "input": row["text"],
            "output": str(row["label"])
        }
        alpaca_data_a.append(entry)
    df_b = pd.read_csv("/model/data/Research/PEFTdata/trainset_CM.csv")
    for _, row in df_b.iterrows():
        entry = {
            "instruction": "Given the following code snippet, identify if it contains a code smell (respond with 1 for code smell present, 0 for clean code).",
            "input": row["text"],
            "output": str(row["label"])
        }
        alpaca_data_a.append(entry)
    df_c = pd.read_csv("/model/data/Research/PEFTdata/validset_CM.csv")
    for _, row in df_c.iterrows():
        entry = {
            "instruction": "Given the following code snippet, identify if it contains a code smell (respond with 1 for code smell present, 0 for clean code).",
            "input": row["text"],
            "output": str(row["label"])
        }
        alpaca_data_a.append(entry)
    df_d = pd.read_csv("/model/data/Research/PEFTdata/MLCQ.csv")
    for _, row in df_d.iterrows():
        entry = {
            "instruction": "Analyze the following code snippet with a known code smell, and specify its smell type and severity level (severity options: none, minor, major, critical).",
            "input": row["code"],
            "output": f"Smell type: {row['smell']}, Severity: {row['severity']}"
        }
        alpaca_data_a.append(entry)
    # 保存为 JSON 文件
    with open("SFT_code_smell_alpaca.json", "w", encoding="utf-8") as f:
        json.dump(alpaca_data_a, f, ensure_ascii=False, indent=2)

def valid():
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/model/LiangXJ/Model/CodeLlama")
    token_lengths = []
    alpaca_data_a = []
    # 读取 CSV 文件
    df_a = pd.read_csv("/model/data/Research/output/cassandra_action_code.csv")
    for _, row in df_a.iterrows():
        if row['code'] == 'not find code' or row['code'] == 'not find file':
            continue
        else:
            # 计算token长度
            tokens = tokenizer.encode(row['code'])
            token_lengths.append(len(tokens))
            entry = {
                "instruction": "Given a code smell description, determine whether it requires mandatory refactoring. Output 1 if the code smell must be refactored, or 0 if refactoring is optional or not necessary. Respond with only a single digit: 1 or 0, and provide no explanation.",
                "input": f"Code Smell: {row['code']}",
                "output": str(row["actionable"])
            }
            alpaca_data_a.append(entry)
    
    # 统计token长度分布
    print(f"有效数据：{len(alpaca_data_a)}条")
    print(f"Token length statistics:")
    print(f"Mean: {np.mean(token_lengths):.2f}")
    print(f"Median: {np.median(token_lengths):.2f}")
    print(f"Max: {np.max(token_lengths)}")
    print(f"Min: {np.min(token_lengths)}")
    print(f"95th percentile: {np.percentile(token_lengths, 95):.2f}")
    
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50)
    plt.title('Distribution of Code Token Lengths')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.savefig('token_distribution.png')
    plt.close()

    with open("/model/lxj/LLaMA-Factory/data/codesmell/Valid_action_smell_alpaca.json", "w", encoding="utf-8") as f:
        json.dump(alpaca_data_a, f, ensure_ascii=False, indent=2)

def create_json_dataset(df):
    """
    将DataFrame转换为指定格式的JSON列表。
    """
    json_data = []
    for _, row in df.iterrows():
        json_data.append({
            "instruction": f"Given a snippet of '{row['Smell']}' smell code, determine whether it requires mandatory refactoring. Output 1 if the code smell must be refactored, or 0 if refactoring is optional or not necessary. Respond with only a single digit: 1 or 0, and provide no explanation.",             
            "input": f"Code:{str(row['code'])}", # 确保code是字符串
            "output": str(row['actionable'])
        })
    return json_data

def split_data_and_save_json(project, train_ratio=0.7):
    csv_file_path = '/model/data/Research/actionable/'+project+'_action_code.csv'
    try:
        df = pd.read_csv(csv_file_path)
        df['actionable'] = pd.to_numeric(df['actionable'], errors='coerce').fillna(0).astype(int) # 转为数字，无效值填充为0
        df['checker'] = df['checker'].astype(str).fillna('') # 将check列转为字符串，NaN转为空字符串
        df['code'] = df['code'].astype(str).fillna('')   # 将code列转为字符串，NaN转为空字符串
        df = df[df['code'] != 'not find code']
    except FileNotFoundError:
        print(f"错误：CSV文件 '{csv_file_path}' 未找到。")
        return
    except Exception as e:
        print(f"读取CSV文件时发生错误：{e}")
        return
    # 确保重要列存在
    required_columns = ['Smell', 'actionable', 'code', 'checker']
    if not all(col in df.columns for col in required_columns):
        print(f"错误：CSV文件必须包含以下列: {', '.join(required_columns)}")
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"缺失的列: {', '.join(missing_cols)}")
        return
    # 唯一索引生成
    df['original_index_for_id'] = df.index
    df['unique_id'] = project + '_' + df['original_index_for_id'].astype(str)
    df.drop(columns=['original_index_for_id'], inplace=True) # 清理辅助列
    df['dataset_split'] = '' # 初始化标记列
    # 数据预处理
    train_indices = []
    test_indices = []
    # 按 'Smell' 和 'actionable' 分组
    grouped = df.groupby(['Smell', 'actionable'], observed=True) # observed=True 推荐用于新版pandas
    for _, group_df in grouped:
        with_comma_df = group_df[group_df['checker'].str.contains(',', na=False)]
        without_comma_df = group_df[~group_df['checker'].str.contains(',', na=False)]
        n_total_group = len(group_df)
        n_train_group_target = int(n_total_group * train_ratio)      
        current_group_train_indices_local = [] # 本组内的训练集索引       
        # 从group_df的原始索引中获取，这些索引是相对于最开始加载的df的
        group_original_indices = list(group_df.index) 
        # 1. 优先从含逗号的样本中抽取训练数据
        # 从 with_comma_df 中抽取，其索引也是原始 df 的索引
        comma_indices_for_sampling = with_comma_df.index.tolist()
        # 打乱含逗号样本的索引
        import random
        random.seed(42)
        random.shuffle(comma_indices_for_sampling)
        can_take_from_with_comma = min(len(comma_indices_for_sampling), n_train_group_target)
        selected_comma_train_indices = comma_indices_for_sampling[:can_take_from_with_comma]
        current_group_train_indices_local.extend(selected_comma_train_indices)   
        # 2. 如果训练样本还不够，从不含逗号的样本中补齐
        remaining_train_needed = n_train_group_target - len(current_group_train_indices_local)
        if remaining_train_needed > 0:
            # 不含逗号的样本索引（且未被选中的）
            # 从 group_original_indices 中排除已选的 selected_comma_train_indices
            # 并且这些索引必须属于 without_comma_df
            available_without_comma_indices = [
                idx for idx in group_original_indices 
                if idx not in selected_comma_train_indices and idx in without_comma_df.index
            ]
            random.shuffle(available_without_comma_indices) # 打乱
            can_take_from_without_comma = min(len(available_without_comma_indices), remaining_train_needed)
            selected_no_comma_train_indices = available_without_comma_indices[:can_take_from_without_comma]
            current_group_train_indices_local.extend(selected_no_comma_train_indices)
        train_indices.extend(current_group_train_indices_local)
        # 本组的测试样本索引 = 本组所有样本索引 - 本组训练样本索引
        current_group_test_indices_local = [idx for idx in group_original_indices if idx not in current_group_train_indices_local]
        test_indices.extend(current_group_test_indices_local)
    # 去重，以防万一（理论上不应有重复）
    train_indices = sorted(list(set(train_indices)))
    test_indices = sorted(list(set(test_indices)))
    # 在原始df上标记划分结果
    # 确保这些索引在df的范围内
    valid_train_indices = [idx for idx in train_indices if idx in df.index]
    valid_test_indices = [idx for idx in test_indices if idx in df.index]
    df.loc[valid_train_indices, 'dataset_split'] = 'train'
    df.loc[valid_test_indices, 'dataset_split'] = 'test'
    # 根据标记好的索引，从 *原始* df（现在包含 unique_id 和 dataset_split）中获取训练集和测试集
    # 这样 train_df 和 test_df 也会包含这些新列，但 create_json_dataset 不会使用它们
    train_df = df.loc[valid_train_indices]
    test_df = df.loc[valid_test_indices]
    # 打乱最终的训练集和测试集（主要为了JSON输出时的随机性，如果需要的话）
    train_df_shuffled = shuffle(train_df, random_state=42).reset_index(drop=True)
    test_df_shuffled = shuffle(test_df, random_state=42).reset_index(drop=True)
    print(f"--- Project: {project} ---")
    print(f"原始数据量: {len(df)}")
    print(f"标记为训练集数量: {len(valid_train_indices)} (实际获取: {len(train_df)})")
    print(f"标记为测试集数量: {len(valid_test_indices)} (实际获取: {len(test_df)})")
    # 检查是否有样本未被分配 (理论上不应有，除非train_ratio导致某些组无法分配)
    unassigned_count = df['dataset_split'].eq('').sum()
    if unassigned_count > 0:
        print(f"警告: 有 {unassigned_count} 个样本未被分配到训练集或测试集。")
    if not train_df_shuffled.empty:
        train_with_comma_count = train_df_shuffled['checker'].str.contains(',', na=False).sum()
        print(f"训练集中 'checker' 列含逗号的样本数: {train_with_comma_count} (占比: {train_with_comma_count/len(train_df_shuffled):.2%})") 
    instruction_text = f"项目 {project}: 分析给定代码片段，识别代码异味（Smell），判断其是否可操作（actionable），并列出相关检查点（checker）。"
    train_json_data = create_json_dataset(train_df_shuffled)
    test_json_data = create_json_dataset(test_df_shuffled)
    try:
        # 保存修改后的CSV文件 (包含 unique_id 和 dataset_split)
        # 将 unique_id 和 dataset_split 列放到前面方便查看
        cols = ['unique_id', 'dataset_split'] + [col for col in df.columns if col not in ['unique_id', 'dataset_split']]
        df_to_save = df[cols]
        output_csv_path = "/model/lxj/actionableSmell/" +project+"_Split.csv"
        train_json_path = "/model/lxj/LLaMA-Factory/data/codesmell/train_" +project+ ".json"
        test_json_path = "/model/lxj/LLaMA-Factory/data/codesmell/test_" +project+ ".json"
        df_to_save.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"带标记的CSV已保存到: {output_csv_path}")
        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(train_json_data, f, ensure_ascii=False, indent=4)
        print(f"训练集JSON已保存到: {train_json_path}")
        with open(test_json_path, 'w', encoding='utf-8') as f:
            json.dump(test_json_data, f, ensure_ascii=False, indent=4)
        print(f"测试集JSON已保存到: {test_json_path}")
        print("-" * 30)
    except IOError as e:
        print(f"保存文件时发生IO错误：{e}")
    except Exception as e:
        print(f"处理数据或保存文件时发生未知错误：{e}")
        
if __name__ == "__main__":
    projects = ['dbeaver','stirling','jsoup','cayenne','pig','struts','mockito','dubbo','cassandra','jedis','easyexcel']
    # for project in projects:
    #     split_data_and_save_json(project)
    for project in projects:
        json_path = os.path.join("/model/lxj/LLaMA-Factory/data/codesmell/", f'train_{project}.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 假设data是一个列表或字典，遍历并修改output字段
        def convert_output_to_str(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == 'output' and isinstance(v, (int, float)):
                        obj[k] = str(v)
                    else:
                        convert_output_to_str(v)
            elif isinstance(obj, list):
                for item in obj:
                    convert_output_to_str(item)
        convert_output_to_str(data)
        # 保存修改后的json文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)