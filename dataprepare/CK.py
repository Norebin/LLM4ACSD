import os
from pathlib import Path
import subprocess


def process_folders(input_base_path, output_base_path, jar_path):
    # 确保输入基础路径存在
    if not os.path.exists(input_base_path):
        print(f"输入路径 {input_base_path} 不存在")
        return
    # 创建输出基础路径（如果不存在）
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    # 遍历输入路径下的所有文件夹
    for folder_name in os.listdir(input_base_path):
        input_path = os.path.join(input_base_path, folder_name)
        # 只处理目录
        if os.path.isdir(input_path):
            # 提取版本号部分（假设文件夹名格式类似cassandra-cassandra-0.3.0-final）
            # 我们取最后一个 '-' 后面的部分作为版本号
            # version = folder_name.split('-',maxsplit=2)[-1]
            # output_folder = f"{'-'.join(version)}"
            output_path = os.path.join(output_base_path, folder_name)
            # 创建输出子文件夹（如果不存在）
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            else:
                continue
            # 构建并执行java命令
            cmd = [
                "java",
                "-jar",
                jar_path,
                input_path,
                'false',
                '0',
                'false',
                output_path + '/'
            ]

            print(f"正在处理: {folder_name}")
            print(f"执行命令: {' '.join(cmd)}")
            try:
                # 执行命令
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"成功处理 {folder_name}")
                print(f"输出: {result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"处理 {folder_name} 时出错:")
                print(f"错误信息: {e.stderr}")
            except Exception as e:
                print(f"未知错误处理 {folder_name}: {str(e)}")


if __name__ == "__main__":
    # 配置路径
    projects = [project.name for project in Path('/model/data/Research/sourceProject/actionableCS').iterdir() if project.is_dir()]
    for project in projects:
        if project != "dbeaver":continue
        input_base_path = "/model/data/Research/sourceProject/actionableCS/" + project # 输入基础路径
        output_base_path = "/model/data/Research/tools/CK/" + project# 输出基础路径
        jar_path = "/model/LiangXJ/ck-0.7.0-jar-with-dependencies.jar"  # ck.jar 的路径
        # 执行处理
        process_folders(input_base_path, output_base_path, jar_path)