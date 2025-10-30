import os
import re
import tarfile
import urllib
import zipfile
import subprocess

def download_target_project(dir_path="/model/lxj/ACSDetector/DataSet"):
    # 提取数字的正则表达式
    number_pattern = re.compile(r"\d")
    # 检查目录是否存在
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"The directory {dir_path} does not exist.")
    # 存储结果的字典
    results = {}
    # 遍历上级目录及其所有子目录
    for root, dirs, files in os.walk(dir_path):
        for sub_dir in dirs:
            temp = os.path.join(root, sub_dir, 'types/pureJava')
            for _root, _dirs, _files in os.walk(temp):
                for file_name in _files:
                    # 按 "-" 分割文件名
                    parts = file_name.split('-')
                    for parts in [parts[1], parts[2], parts[3]]:
                        if number_pattern.search(parts):
                            # 初始化子目录的结果集合
                            if sub_dir not in results:
                                results[sub_dir] = set()
                            # 将提取到的部分添加到集合中
                            results[sub_dir].add(parts)
                            break
    # 打印结果
    # for sub_dir, extracted_numbers in results.items():
    #     print(f"Sub-directory: 准备下载{sub_dir}")
    #     print(f"Extracted numbers: {extracted_numbers}")
    base_url = "https://archive.apache.org/dist/ant/source/apache-ant-{version}-src.tar.gz"
    base_url = "https://s3.amazonaws.com/jruby.org/downloads/{version}/jruby-src-{version}.tar.gz"
    base_url = "https://github.com/apache/storm/archive/refs/tags/v{version}.tar.gz"
    # base_url = "https://archive.apache.org/dist/kafka/{version}/kafka-{version}-src.tgz"
    for project, versions in results.items():
        if project != 'storm': continue
        for version in versions:
            print(f"开始下载：{version}")
            download_url = base_url.format(version=version)
            version_dir = os.path.join('/model/LiangXJ/ACSDetect/Projectsrc' , project)
            os.makedirs(os.path.join(version_dir, version), exist_ok=True)
            save_path = os.path.join(version_dir, f"{version}.tar.gz")
            try:
                print(f"Downloading {download_url} to {save_path}...")
                urllib.request.urlretrieve(download_url, save_path)
                print(f"Downloaded {download_url} successfully.")
            except Exception as e:
                print(f"Failed to download {download_url}: {e}")
    return results

def unzip_src(dir_path='/model/LiangXJ/ACSDetect/Projectsrc/'):
    for root, dirs, files in os.walk(dir_path):
        for sub_dir in dirs:
            # if sub_dir != 'jruby': continue
            sub_dir_path = os.path.join(root, sub_dir)
            for version in os.listdir(sub_dir_path):
                file_name = os.path.join(sub_dir_path, version)
                # for file_name in os.listdir(version_path):
                # file_path = os.path.join(version_path, file_name)
                if file_name.endswith('.zip'):
                    subprocess.run(['unzip', '-o', file_name, '-d', sub_dir_path], check=True)
                    # subprocess.run(['unzip', '-o', '-j', file_name, '-d', sub_dir_path], check=True)
                    print(f"解压文件: {file_name} 到 {sub_dir_path}")
                elif file_name.endswith('.tar.gz') or file_name.endswith('.tgz'):
                    subprocess.run(['tar', '-xzvf', file_name, '-C', sub_dir_path], check=True)
                    # subprocess.run(['tar', '-xzvf', file_name, '--strip-components=1', '-C', sub_dir_path], check=True)
                    print(f"解压文件: {file_name} 到 {sub_dir_path}")

if __name__ == '__main__':
    # download_target_project()
    unzip_src()