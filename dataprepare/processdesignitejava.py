import os

import pandas as pd

def feature_envy(file):
    # 读取CSV文件
    # 假设文件路径为 'path/to/your/DesignSmells.csv'，请替换为实际路径
    column_names = ['Project Name','Package Name','Type Name','Design Smell','Cause of the Smell']  # 调整列数
    df = pd.read_csv(file, names=column_names, header=None, on_bad_lines='skip')
    # 筛选出Design Smell为Feature Envy的行
    feature_envy_df = df[df['Design Smell'] == 'Feature Envy'].copy()
    # 定义一个函数来提取because后的单词
    def extract_method_name(text):
        if pd.isna(text):
            return None
        try:
            # 分割字符串并找到because后的第一个单词
            parts = text.split('because')
            if len(parts) > 1:
                words = parts[1].strip().split()
                if words:  # 检查是否有单词
                    return words[0]
        except:
            return None
        return None
    # 创建新列Method Name
    feature_envy_df['Method Name'] = feature_envy_df['Cause of the Smell'].apply(extract_method_name)
    # 找到Type Name列的位置
    type_name_idx = feature_envy_df.columns.get_loc('Type Name')
    # 获取所有列名
    cols = feature_envy_df.columns.tolist()
    # 将Method Name插入到Type Name后面
    new_cols = cols[:type_name_idx + 1] + ['Method Name'] + cols[type_name_idx + 1:-1]
    # 重新整理数据框
    feature_envy_df = feature_envy_df[new_cols]
    feature_envy_df.rename(columns={'Design Smell': 'Smell'}, inplace=True)
    return feature_envy_df
    # 如果需要保存结果到新文件
    # feature_envy_df.to_csv('filtered_design_smells.csv', index=False)

def implement_smell(file):
    column_names = ['Project Name', 'Package Name', 'Type Name', 'Method Name', 'Implementation Smell', 'Cause of the Smell', 'Method start line no']
    df = pd.read_csv(file, names=column_names, header=None, on_bad_lines='skip')
    df = df.rename(columns={'Implementation Smell' : 'Smell'})
    # 筛选出Design Smell为Feature Envy的行
    target_smells = ['Complex Method', 'Long Method', 'Long Parameter List']
    target_df = df[df['Smell'].isin(target_smells)].copy()
    return target_df

if __name__ == '__main__':
    dir = '/model/data/Research/tools/designitejava/cayenne'
    for version in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, version)):
            path = os.path.join(dir, version)
            part1 = feature_envy(path + '/DesignSmells.csv')
            part2 = implement_smell(path + '/ImplementationSmells.csv')
            common_columns = ['Project Name', 'Package Name', 'Type Name', 'Method Name', 'Smell', 'Cause of the Smell']
            combined_df = pd.concat([part1[common_columns], part2[common_columns]], ignore_index=True)
            combined_df.to_csv(path + f'/Smell.csv', index=False)
            print("成功合并并保存为 Smell.csv")
            print("\n合并后的数据预览：")
            print(combined_df.head())