import os
import re
import subprocess

import pandas as pd

def find_java_file(directory, project_version, java_path):
    """
    在目录及其子目录中递归查找目标Java文件。
    :param directory: 项目根目录
    :param java_path: 目标Java文件的相对路径（如 org/apache/hadoop/hbase/DroppedSnapshotException.java）
    :return: 如果找到文件返回True，否则返回False
    """
    # 将路径分隔符统一为当前系统的分隔符
    java_path = java_path.replace('/', os.sep)
    target_dir, target_filename = os.path.split(java_path)
    target_dir_parts = target_dir.split(os.sep)
    # 遍历目录及其子目录
    for root, dirs, files in os.walk(os.path.join(directory, project_version)):
        root_parts = root.split(os.sep)
        # 如果当前路径包含 target_dir 的所有部分，则匹配
        if root_parts[-len(target_dir_parts):] == target_dir_parts:
            # 检查该目录下是否包含目标文件
            if target_filename in files:
                return os.path.join(root, java_path)
    return False
# Function to read the CSV file and filter rows based on class-name containing a specific string
def filter_and_check_classnames(csv_file, filter_string, txt_file):
    # Step 1: 读取CSV文件
    df = pd.read_excel(csv_file)

    # Step 2: 筛选出class-name列包含指定字符串的行
    filtered_df = df[df['class-name'].str.contains(filter_string, na=False)]

    # Step 3: 逐条操作 class-name 列
    with open(txt_file, 'r') as file:
        existing_files = file.read().splitlines()  # 读取所有已存在的文件路径

    # Step 4: 遍历筛选后的 DataFrame
    for _, row in filtered_df.iterrows():
        class_name = row['class-name']
        # 提取 class-name 列中按 split 后的最后一个词
        class_file = class_name.split('.')[-1] + '.java'

        # Step 5: 查找是否存在该 .java 文件
        if class_file not in existing_files:
            print(class_name)  # 如果不存在则输出 class-name

def check_java_exsit(dir_path="/model/LiangXJ/ACSDetect/Projectsrc", target_path="/model/lxj/ACSDetector/DataSet/"):
    for sub_dir in os.listdir(dir_path):
        if sub_dir != 'hbase_new': continue
        failed = 0
        temp = os.path.join(dir_path, sub_dir)
        target_dir=os.path.join(target_path, sub_dir, 'types', 'pureJava')
        for filename in os.listdir(target_dir):
            if filename.endswith(".java"):
                match = re.match(r'^(.*?-\d+\.\d+\.\d+)', filename)
                project_version = match.group(1)
                java_path = filename.split('-', 3)[-1].split('-')
                java_path = str(java_path[0]).replace('.','/') + '/' + str(java_path[1])
                if not find_java_file(os.path.join(dir_path,sub_dir),project_version,java_path):
                    failed += 1
                    print(filename)
        # result_a = subprocess.run(['find', target_dir, '-name', '*.java'], stdout=subprocess.PIPE, text=True)
        # files_a = result_a.stdout.strip().split('\n')
        # for files in files_a:
        #     files_a_names = files.split('-')[-1]
        #     version = files.split('-')[2]
        #     current_dir = os.path.join(dir_path, sub_dir, 'hbase-rel-'+version)
        #     file = files_a_names
        #     result = subprocess.run(['find', current_dir, '-type', 'f', '-name', file], stdout=subprocess.PIPE, text=True)
        #     if result.stdout == '':
        #         print(files)
        #         failed += 1
        print(f"无法找到的java文件：{failed}")
if __name__ == '__main__':
    # csv_file = '/model/LiangXJ/developCodeSmell.xlsx'
    # filter_string = 'mahout'  # 需要筛选的字符串
    # txt_file = '/model/LiangXJ/CodeSmellProject/apache-mahout-distribution-0.10.2/output.txt'  # 存放已存在的 Java 文件列表的 TXT 文件
    # filter_and_check_classnames(csv_file, filter_string, txt_file)
    check_java_exsit()
