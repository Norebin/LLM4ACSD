import os
import pandas as pd

# 配置项：可以修改这些变量来指定要提取的列名
METHOD_CSV_TARGET_COLUMN = 'method'  # 从 method.csv 中提取的列名
CLASS_CSV_TARGET_COLUMN = 'type'  # 从 class.csv 中提取的列名
OUTPUT_COLUMN_NAME = 'Extracted_Data'  # 添加到 Smell.csv 中的新列名

def process_smell_files(root_path, metric_path):
    # 遍历根路径下的所有子目录
    for dirpath, _, filenames in os.walk(root_path):
        if 'Smell.csv' in filenames:
            smell_file = os.path.join(dirpath, 'Smell.csv')
            print(f"处理文件: {smell_file}")
            # 读取 Smell.csv
            smell_df = pd.read_csv(smell_file)
            # 添加新列用于存储提取的数据
            smell_df[OUTPUT_COLUMN_NAME] = None
            # 遍历 Smell.csv 的每一行
            for index, row in smell_df.iterrows():
                project_name = row['Project Name']
                package_name = row['Package Name']
                type_name = row['Type Name']
                method_name = row['Method Name'] if pd.notna(row['Method Name']) else ''
                # 拼接 metric_path
                CK_path = os.path.join(metric_path, project_name)
                # 拼接类名
                class_name = f"{package_name}.{type_name}"
                # 根据 Method Name 是否为空选择文件
                if method_name:
                    target_file = os.path.join(CK_path, 'method.csv')
                    target_column = METHOD_CSV_TARGET_COLUMN
                    match_columns = {'class': class_name, 'method': method_name}
                else:
                    target_file = os.path.join(CK_path, 'class.csv')
                    target_column = CLASS_CSV_TARGET_COLUMN
                    match_columns = {'class': class_name}
                # 检查目标文件是否存在
                if os.path.exists(target_file):
                    # 读取目标 CSV 文件
                    target_df = pd.read_csv(target_file)
                    # 查找匹配行
                    match_condition = True
                    for col, value in match_columns.items():
                        if col == 'method':
                            # 移除 method 列中的 /0 或类似后缀，只匹配方法名部分
                            match_condition &= (target_df[col].str.split('/').str[0] == value)
                        else:
                            # 其他列（如 class）保持严格匹配
                            match_condition &= (target_df[col] == value)
                    matched_rows = target_df[match_condition]
                    # 如果找到匹配行，提取目标列的值
                    if not matched_rows.empty:
                        extracted_value = matched_rows.iloc[0][target_column]
                        smell_df.at[index, OUTPUT_COLUMN_NAME] = extracted_value
                    else:
                        print(f"未在 {target_file} 中找到匹配行: {match_columns}")
                else:
                    print(f"文件不存在: {target_file}")

            # 保存更新后的 Smell.csv
            smell_df.to_csv(smell_file, index=False)
            print(f"已更新文件: {smell_file}")


if __name__ == "__main__":
    # 指定根路径（请替换为实际路径）
    root_path = "/model/lxj/actionableCS/designitejava/cassandra"
    metric_path = "/model/lxj/actionableCS/CK/cassandra"
    process_smell_files(root_path, metric_path)