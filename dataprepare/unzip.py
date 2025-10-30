import os
import tarfile

def extract_tar_gz(directory):
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在")
        return
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.tar.gz'):
            filepath = os.path.join(directory, filename)
            try:
                # 打开tar.gz文件
                with tarfile.open(filepath, 'r:gz') as tar:
                    # 解压到当前目录（可以自定义输出路径）
                    tar.extractall(path=directory)
                    print(f"成功解压: {filename}")
            except Exception as e:
                print(f"解压 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    # 指定要解压的目录（可以改为你的路径）
    target_directory = "/model/data/Research/sourceProject/actionableCS/struts"
    extract_tar_gz(target_directory)