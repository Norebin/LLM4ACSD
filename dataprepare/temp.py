import os
import pandas as pd


def convert_xlsx_to_csv(directory):
    # 遍历指定目录下的所有文件夹
    for root, dirs, files in os.walk(directory):
        # 检查是否是output文件夹
        if 'output' in dirs:
            output_path = os.path.join(root, 'output')
            # 遍历output文件夹中的所有文件
            for filename in os.listdir(output_path):
                # 检查是否是xlsx文件
                if filename.endswith('.xlsx'):
                    # 构造完整的文件路径
                    xlsx_path = os.path.join(output_path, filename)
                    # 读取xlsx文件
                    try:
                        df = pd.read_excel(xlsx_path)
                        # 构造csv文件名（去掉.xlsx后缀，添加.csv）
                        csv_filename = filename.replace('.xlsx', '.csv')
                        csv_path = os.path.join(output_path, csv_filename)
                        # 转换为csv文件
                        df.to_csv(csv_path, index=False)
                        print(f"已转换: {xlsx_path} -> {csv_path}")
                    except Exception as e:
                        print(f"转换文件 {xlsx_path} 时出错: {str(e)}")

def pmd_process(directory):
    # 遍历指定目录下的所有文件夹
    for root, dirs, files in os.walk(directory):
        # 检查是否是output文件夹
        if 'output' in dirs:
            output_path = os.path.join(root, 'output')
            # 遍历output文件夹中的所有文件
            for filename in os.listdir(output_path):
                # 检查是否是xlsx文件
                if filename.endswith('.csv'):
                    # 构造完整的文件路径
                    csv_path = os.path.join(output_path, filename)
                    # 读取CSV文件
                    df = pd.read_csv(csv_path)
                    # 处理method列
                    if 'Method Name' in df.columns:
                        df['Method Name'] = df['Method Name'].apply(lambda x: x.split('(')[0] if pd.notna(x) else x)
                    # 保存更改后的文件
                    df.to_csv(csv_path, index=False)
                    print(f"已处理并保存: {csv_path}")

def decor_process(directory):
    # 遍历指定目录下的所有文件夹
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path)
                # 删除Smell列为LongParameterList的行
                df = df[df['Smell'] != 'LongParameterList']
                # 处理Package Name列
                df['Package Name'] = df['Package Name'].apply(lambda x: '.'.join(x.split('.')[:-1]) if pd.notna(x) else x)
                # 保存更改后的文件
                df.to_csv(csv_path, index=False)
                print(f"已处理并保存: {csv_path}")

def convert_column_name(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                # 重命名列
                df.rename(columns={
                    '项目及版本号': 'Project Name',
                    '包名': 'Package Name',
                    '类名': 'Type Name',
                    '方法名': 'Method Name',
                    '异味类型': 'Smell',
                    '异味源代码': 'code',
                    '代码描述': 'comment',
                    '代码指标': 'actionable'
                }, inplace=True)
                # 保存更改后的文件
                df.to_csv(file_path, index=False)
                print(f"已更新列名并保存: {file_path}")
if __name__ == "__main__":
    # directory = "/model/data/Research/tools/Decor"
    # print(f"开始转换 {directory} 下的xlsx文件...")
    # decor_process(directory)
    # print("转换完成！")
    root_dir = "/model/data/Research/tools/Checkstyle/excel_output/"
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            # 查找当前文件夹中的xlsx文件
            for file in os.listdir(folder_path):
                project_path = os.path.join(folder_path, file)
                for x in os.listdir(project_path):
                    if x.endswith('.xlsx'):
                        file_path = os.path.join(project_path, x)
                        
                        # 读取Excel文件
                        try:
                            df = pd.read_excel(file_path)
                            df['File Path'] = df['File Path'].astype(str)
                            # 处理"File Path"列（如果存在）
                            if 'File Path' in df.columns:
                                # 按'/'分割一次，取第二部分
                                df['File Path'] = df['File Path'].apply(lambda x: x.split('/', 1)[1] if '/' in x else x)
                                
                                # 保存修改后的文件（可以另存为新文件或覆盖原文件）
                                output_path = os.path.join(folder_path, f"{os.path.splitext(x)[0]+'.csv'}")
                                df.to_csv(output_path, index=False)
                                print(f"处理完成: {output_path}")
                            else:
                                print(f"文件 {file} 中没有'File Path'列")
                                
                        except Exception as e:
                            print(f"处理文件 {file} 时出错: {str(e)}")