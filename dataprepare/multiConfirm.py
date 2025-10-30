import os
import pandas as pd
import re
import csv
from pathlib import Path
import uuid
#designitejava--Feature Envy, Long Method, Complex Method, Long Parameter List
#Decor--LongMethod
#pmd--Complex Method, God Class
#checkstyle--complex class, complex method, FeatureEnvy, ParametersPerMethod

# 基础路径
BASE_PATH = "/model/data/Research/tools"
OUTPUT_PATH = "/model/data/Research/temp/"
TOOLS = ["Decor", "designitejava", "pmd"]

# 确保输出目录存在
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

def extract_project_version1(file_path, tool):
    """从文件路径中提取项目名和版本号"""
    file_name = Path(file_path).name
    parent_dirs = Path(file_path).parts

    if tool == "Checkstyle":
        # 路径示例: Checkstyle/cassandra/cassandra-cassandra-0.3.0-final/cassandra-cassandra-0.3.0-final_smell.csv
        match = re.search(r"(.*?)-(.*?)-([\d\.\-a-z]+)_smell\.csv", file_name)
        if match:
            project = match.group(1)  # cassandra
            version = match.group(3)  # 0.3.0-final
            return project, version
    elif tool == "Decor":
        # 路径示例: Decor/cassandra/cassandra-0.3.0-final.csv
        match = re.search(r"(.*?)-([\d\.\-a-z]+)\.csv", file_name)
        if match:
            project = match.group(1)
            version = match.group(2)
            return project, re.search(r"(?:jedis-){1,2}([\d.]+(?:-[a-zA-Z]+\d*)?)", file_path).group(1)
    elif tool == "designitejava":
        # 路径示例: designitejava/cassandra/0.3.0-final/Smell.csv
        for part in parent_dirs[::-1]:
            if re.match(r"(.*?)-([\d\.\-a-z]+)", part):
                version = '-'.join(file_path.split('/')[-2].split('-')[1:])
                project = parent_dirs[parent_dirs.index(part) - 1].capitalize()
                return project, re.search(r"(?:jedis-){1,2}([\d.]+(?:-[a-zA-Z]+\d*)?)", file_path).group(1)
    elif tool == "pmd":
        # 路径示例: pmd/cassandra/output/cassandra-cassandra-0.3.0-final.csv
        match = re.search(r"(.*?)-([\d\.\-a-z]+)\.csv", file_name)
        if match:
            project = match.group(1)
            version = match.group(2)
            return project, re.search(r"(?:jedis-){1,2}([\d.]+(?:-[a-zA-Z]+\d*)?)", file_path).group(1)
    return None, None
def extract_project_version(file_path, tool):
    """从文件路径中提取项目名和版本号"""
    file_name = Path(file_path).stem
    parent_dirs = Path(file_path).parts
    if tool == "designitejava":
        file_name = file_path.split('/')[-2]
        project = file_name.split('-')[0]
        version = '-'.join(file_name.split('-')[2:]) if len(file_name.split('-'))>=4 else '-'.join(file_name.split('-')[1:])
        return re.search(r'\d.*', file_name).group(0)
    elif tool == "pmd":
        # 路径示例: /model/data/Research/tools/pmd/struts/output/struts-STRUTS_2_2_0.csv
        project = file_name.split('-')[0]
        version = '-'.join(file_name.split('-')[2:]) if len(file_name.split('-'))>=4 else '-'.join(file_name.split('-')[1:])
        return re.search(r'\d.*', file_name).group(0)
    elif tool == "Decor":
        # 路径示例: /model/data/Research/tools/pmd/struts/output/struts-STRUTS_2_2_0.csv
        project = file_name.split('-')[0]
        version = '-'.join(file_name.split('-')[1:])
        return re.search(r'\d.*', file_name).group(0)
    
def normalize_csv_package_name(package_name_csv_val):
    """Handles (Default), NaN, None, or empty strings from CSV for package name."""
    if pd.isna(package_name_csv_val) or package_name_csv_val in ["(Default)", ""]:
        return ""
    if any(c.isupper() for c in package_name_csv_val):
        return re.sub(r'\.[A-Z_].*$', '', package_name_csv_val)
    return str(package_name_csv_val)

def load_csv_files(tool, project):
    """加载指定工具的所有 CSV 文件"""
    tool_path = os.path.join(BASE_PATH, tool)
    data = []
    large_class_count = 0
    x = 'Smell.csv' if tool=='designitejava' else '.csv'
    cassandra_path = os.path.join(tool_path, project)
    for root, _, files in os.walk(cassandra_path):
        for file in files:
            if file.endswith(x):
                file_path = os.path.join(root, file)
                version = extract_project_version(file_path, tool)
                print(file_path)
                if project and version:
                    try:
                        df = pd.read_csv(file_path)
                        # 添加 checker 和版本字段
                        df["checker"] = str(tool)
                        df["Project"] = str(project)
                        df["Version"] = str(version)
                        df["Smell"] = df["Smell"].replace("Large Class", "God Class")
                        df["Smell"] = df["Smell"].replace("LongMethod", "Long Method")
                        df["Smell"] = df["Smell"].replace("LongParameterList", "Long Parameter List")
                        df["Smell"] = df["Smell"].replace("FeatureEnvy", "Feature Envy")
                        df['Package Name'] = df['Package Name'].apply(normalize_csv_package_name)
                        large_class_count += len(df[df["Smell"] == "Large Class"])
                        data.append(df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

def merge_tool_results():
    """合并四个工具的结果"""
    all_data = []
    project_dir = '/model/data/Research/sourceProject/actionableCS'
    projects = [d for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d))]
    for project in projects:  
        for tool in TOOLS:
            df = load_csv_files(tool, project)
            if not df.empty:
                all_data.append(df)
        if not all_data:
            print("No data found.")
            return
        # 合并所有数据
        merged_df = pd.concat(all_data, ignore_index=True)
        merged_df["Method Name"] = merged_df["Method Name"].fillna("")
        large_class_records = merged_df[merged_df["Smell"] == "Large Class"]
        print(f"Total Large Class records in merged result: {len(large_class_records)}")
        # 按关键字段分组，合并 checker
        group_cols = ["Project", "Version", "Package Name", "Type Name", "Method Name", "Smell"]
        merged_df = merged_df.groupby(group_cols, as_index=False).agg({
            "checker": lambda x: ",".join(sorted(set(x)))
        })
        merged_df["Version"] = merged_df["Version"].astype(str)
        
        print(merged_df["Version"].dtype)
        
        # 保存按项目和版本的汇总表
        for (project, version), group in merged_df.groupby(["Project", "Version"]):
            output_dir = os.path.join(OUTPUT_PATH, project)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_file = os.path.join(output_dir, f"{project}-{version}.csv")
            group['Version'] = group['Version'].astype(str)
            print(group["Version"].dtype)
            group.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
            print(f"Saved summary to {output_file}")

if __name__ == "__main__":
    merge_tool_results()

# find /model/data/Research/tools/Decor/jedis/ -type f -name "jedis-jedis-*.csv" | while read -r file; do
#     newname=$(echo "$file" | sed 's/jedis-jedis-/jedis-/')
#     mv "$file" "$newname"
# done
# find /model/data/Research/tools/designitejava/struts/ -type d -name "struts-STRUTS_*" | while read -r dir; do
#     newname=$(echo "$dir" | sed 's/struts-STRUTS_/struts-/')
#     mv "$dir" "$newname"
# done