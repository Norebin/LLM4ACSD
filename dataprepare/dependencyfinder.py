import os
import subprocess

if __name__ == "__main__":
    # 用户输入要遍历的根目录
    root_dir = input("请输入要遍历的根目录路径：")
    # DependencyExtractor脚本所在目录
    extractor_dir = "/model/LiangXJ/DependencyFinder-1.4.3/bin"
    extractor_script = os.path.join(extractor_dir, "DependencyExtractor")


    # 遍历根目录下所有子目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 只处理第一层子目录
        for dirname in dirnames:
            last_dir = os.path.basename(dirpath)
            output_dir = os.path.join("/model/data/Research/tools/DependencyFinder", last_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, dirname + ".xml")
            sub_dir = os.path.join(dirpath, dirname)
            print(f"正在处理: {sub_dir}")
            # 构造命令
            cmd = [
                "bash", extractor_script,
                "-minimize",
                "-xml",
                "-out", output_file,
                sub_dir
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"处理{sub_dir}时出错: {e}")
        # 只遍历第一层子目录
        break

