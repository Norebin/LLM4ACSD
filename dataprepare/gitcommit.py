import git
import pandas as pd
import os
from pathlib import Path
import time

# 配置
# REPO_PATH = "/model/lxj/actionableGit/cassandra"  # 替换为你的 Git 仓库本地路径
# CSV_DIR = "/model/lxj/actionableCS/designitejava/cassandra"  # 替换为 CSV 文件所在目录
# VERSIONS = get_sorted_versions('/model/lxj/actionableCS/designitejava/cassandra')# 替换为你的版本标签列表
# OUTPUT_FILE = "smell_evolution_results.txt"

# 初始化 Git 仓库
# repo = git.Repo(REPO_PATH)
def get_file_path(repo, repo_path, version, package_name, class_name, cache=None):
    """
    在特定版本的项目目录中查找类文件路径，使用缓存避免重复遍历。
    参数:
        repo: GitPython 的 Repo 对象
        version: 版本标签（如 "v1.0"）
        package_name: 包名（如 "com.example"）
        class_name: 类名（如 "Foo"）
        cache: 版本文件路径缓存
    返回: 文件路径（如 "src/com/example/Foo.java"）或 None
    """
    if cache is None:
        cache = {}
    cache_key = (version, package_name, class_name)
    if cache_key in cache:
        return cache[cache_key]
    # 切换到指定版本
    repo.git.checkout(version)
    root_dir = Path(repo_path)
    package_parts = package_name.split(".")
    expected_file = f"{class_name}.java"  # 假设是 Java 项目，修改为其他语言后缀如 .py 如果需要
    # 遍历目录，查找匹配的文件
    for dirpath, _, filenames in os.walk(root_dir):
        if expected_file in filenames:
            relative_path = Path(dirpath) / expected_file
            dir_parts = str(relative_path.parent.relative_to(root_dir)).split(os.sep)
            if all(part in dir_parts for part in package_parts):
                cache[cache_key] = str(relative_path.relative_to(root_dir))
                return cache[cache_key]
    cache[cache_key] = None
    return None


def analyze_commit_diff(repo, base_version, start_version, end_version, file_path, method_name, smell_type):
    """
    分析提交差异，判断异味是否被解决。
    返回: (状态, 相关提交信息)
    """
    try:
        diff_output = repo.git.diff(f"{start_version}..{end_version}", file_path)
        commit_log = repo.git.log(f"{base_version}..{start_version}", "--oneline", "--", file_path)
        if not diff_output:
            return False, commit_log
        # 简单规则判断（根据异味类型）
        # actual_changed = False
        # lines = diff_output.splitlines()
        # for line in lines:
        #     if method_name in line:
        #         actual_changed = True
        #         break
        return True, commit_log
    except git.exc.GitCommandError:
        return True, ''

def analyze_smell_evolution():
    """
    分析异味在所有后续版本中的演变。
    """
    start_time = time.time()
    # 加载所有版本的 CSV 数据
    csv_data = {}
    for version in VERSIONS:
        csv_file = os.path.join(CSV_DIR, version, "Smell.csv")
        if os.path.exists(csv_file):
            csv_data[version] = pd.read_csv(csv_file)
        else:
            print(f"警告: {csv_file} 不存在，跳过版本 {version}")
    # 文件路径缓存
    path_cache = {}
    # 存储结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as output:
        output.write("异味演变分析结果\n")
        output.write("=" * 50 + "\n")
        # 对每个版本的异味进行追踪
        for i, start_version in enumerate(VERSIONS[:-1]):  # 最后一个版本无需分析后续
            if start_version not in csv_data:
                continue
            df_start = csv_data[start_version]
            output.write(f"\n从 {start_version} 开始的异味追踪\n")
            output.write("-" * 50 + "\n")
            # 遍历该版本的异味
            for _, row in df_start.iterrows():
                smell_key = f"{row['Package Name']}.{row['Type Name']}.{row['Method Name']}.{row['Smell']}"
                found_in_later = False
                resolved_version = None
                commits = None
                # 检查后续所有版本
                for later_version in VERSIONS[i + 1:]:
                    if later_version not in csv_data:
                        continue
                    df_later = csv_data[later_version]
                    # 检查异味是否仍然存在
                    if any(
                            (df_later["Package Name"] == row["Package Name"]) &
                            (df_later["Type Name"] == row["Type Name"]) &
                            (df_later["Method Name"] == row["Method Name"]) &
                            (df_later["Smell"] == row["Smell"])
                    ):
                        found_in_later = True
                        break
                    elif resolved_version is None:
                        # 记录首次消失的版本
                        resolved_version = later_version
                # 根据结果分类
                if not found_in_later and resolved_version:
                    # 异味在后续版本消失，验证提交记录
                    file_path = get_file_path(repo, "cassandra-"+start_version, row["Package Name"], row["Type Name"], path_cache)
                    output.write(f"异味 {smell_key} 在 {start_version} 出现，于 {resolved_version} 消失\n")
                    if file_path:
                        status, commits = analyze_commit_diff(
                            repo, "cassandra-"+start_version, "cassandra-"+resolved_version, file_path, row["Method Name"], row["Smell"]
                        )
                        output.write(f"状态: {status}\n")
                        output.write(f"相关提交记录:\n{commits}\n")
                    else:
                        output.write("状态: 无法定位文件\n")
                elif found_in_later:
                    output.write(f"异味 {smell_key} 在 {start_version} 出现，在后续版本中仍存在\n")
                else:
                    output.write(f"异味 {smell_key} 在 {start_version} 出现，后续版本无数据可追踪\n")
                output.write("\n")
        # 清理：恢复到默认分支
        repo.git.checkout("main")
    end_time = time.time()
    print(f"分析完成，耗时 {end_time - start_time:.2f} 秒，结果已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    analyze_smell_evolution()