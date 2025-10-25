import json
import os
import glob
import pandas as pd


def read_labels(path):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    all_actionable_data = []
    for csv_file in csv_files:
        print(f"处理文件: {csv_file}")
        df = pd.read_csv(csv_file)
        test_df = df[df['dataset_split'] == 'test']
        if not test_df.empty and 'actionable' in test_df.columns:
            # 提取actionable列数据并添加到结果列表
            actionable_data = test_df['actionable'].tolist()
            all_actionable_data.extend(actionable_data)
    all_actionable_data = pd.read_csv(path, sep=';').iloc[:, 0].values
    print(f"总共收集了{len(all_actionable_data)}条actionable数据")
    label_series = pd.Series(all_actionable_data)
    # if path.endswith('.csv'):
    #     smells = pd.read_csv(path)
    #     label_series = smells["smellKey"]
    # else:
    #     assert path.endswith('.json')
    #     labels_file = open(path, 'r').readlines()
    #     smells = []
    #     for i in labels_file:
    #         test_line = json.loads(i)
    #         smells.append(test_line["smellKey"])
    #     label_series = pd.Series(smells)

    return label_series


def read_functions(path, to_json=False):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    # 创建一个空的DataFrame用于拼接
    all_test_data = pd.DataFrame()
    
    for csv_file in csv_files:
        print(f"处理文件: {csv_file}")
        df = pd.read_csv(csv_file)
        test_df = df[df['dataset_split'] == 'test']
        if not test_df.empty and 'code' in test_df.columns:
            # 直接拼接DataFrame
            all_test_data = pd.concat([all_test_data, test_df[['code']]])
    
    print(f"总共收集了{len(all_test_data)}条code数据")
    
    if not to_json:
        # 需要返回list时才转换为list
        return all_test_data['code'].tolist()
    else:
        # 直接使用DataFrame的to_json方法
        return all_test_data.to_json(orient="records", lines=True)
    # if path.endswith('.csv'):
    #     smells = pd.read_csv(path)
    #     functions = smells[["function"]]
    #     if not to_json:  # return list
    #         return functions["function"].tolist()
    #     else:  # return df
    #         return functions.to_json(orient="records", lines=True)

    # else:
    #     assert path.endswith('.json')
    #     labels_file = open(path, 'r').readlines()
    #     functions = []
    #     for i in labels_file:
    #         test_line = json.loads(i)
    #         functions.append(test_line["function"])

    #     return functions
if __name__ == "__main__":
    csv_files = glob.glob(os.path.join("/model/lxj/actionableSmell", "*.csv"))
    all_actionable_data = []
    for csv_file in csv_files:
        print(f"处理文件: {csv_file}")
        df = pd.read_csv(csv_file)
        test_df = df[df['dataset_split'] == 'test']
        if not test_df.empty and 'actionable' in test_df.columns:
            # 提取actionable列数据并添加到结果列表
            actionable_data = test_df['Smell'].tolist()
            all_actionable_data.extend(actionable_data)
    print(f"总共收集了{len(all_actionable_data)}条actionable数据")
    
    # 创建smell到数字的映射
    unique_smells = list(set(all_actionable_data))
    smell_to_num = {smell: idx for idx, smell in enumerate(unique_smells)}
    
    # 将smell转换为对应的数字
    numeric_labels = [smell_to_num[smell] for smell in all_actionable_data]
    
    # 创建DataFrame并保存
    labels_df = pd.DataFrame({
        'label': numeric_labels
    })
    
    output_path = "/model/lxj/Baseline/TripletLoss/data/smells.csv"
    labels_df.to_csv(output_path, index=False, header=True)
    print(f"标签数据已保存到: {output_path}")
    