import csv
import os
import re
import git
import subprocess
import pandas as pd
from functools import lru_cache
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
# from pkg_resources import parse_version
from test import parse_version
from packaging.version import Version
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor
from gitcommit import get_file_path, analyze_commit_diff

# 动态读取版本号并排序
def get_sorted_versions(path):
    # 读取目录下所有文件夹，去掉扩展名
    items = os.listdir(path)
    # versions = [item for item in items if os.path.isdir(os.path.join(path, item))]
    versions = [Path(f).stem for f in items]
    # 使用 parse_version 进行语义排序
    sorted_versions = sorted(versions, key=parse_version)
    return sorted_versions

# 加载所有版本的异味数据
def load_smell_data(versions):
    smell_data = {}
    for version in versions:
        filename = os.path.join(INPUT_DIR, version, 'Smell.csv')
        smell_data[version] = []
        with open(filename, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                smell_id = (
                    row["Package Name"],
                    row["Type Name"],
                    row["Method Name"],
                    row["Smell"]
                )
                smell_data[version].append({
                    "id": smell_id,
                    "row": row  # 保存完整行数据
                })
    return smell_data


# 追踪异味的消失版本
def track_smell_disappearance(smell_data, versions, repo_path, output_dir):
    result_by_version = defaultdict(list)
    action = 0
    nonaction = 0
    # 文件路径缓存
    path_cache = {}
    repo = git.Repo(repo_path)
    for i, base_version in enumerate(tqdm(versions[:-1])):
        # 获取基准版本的异味
        base_smells = smell_data[base_version]
        subsequent_versions = versions[i+1:]  # 包含当前版本及后续版本
        for smell in base_smells:
            smell_id = smell["id"]
            original_row = smell["row"]
            disappeared_version = None
            last_detected_version = base_version
            # 从当前版本开始检查后续版本
            for version in subsequent_versions:  # 从下一个版本开始
                version_smells = {s["id"] for s in smell_data[version]}
                if smell_id not in version_smells:
                    disappeared_version = version
                    break
                else:
                    last_detected_version = version
            # 创建结果行
            result_row = original_row.copy()
            local_file_path = get_file_path(repo, repo_path, base_version.removeprefix('dbeaver-'), smell_id[0], smell_id[1], path_cache)
            status = False
            if local_file_path and disappeared_version:
                # "cassandra-" + base_version, "cassandra-" + disappeared_version
                status, commits = analyze_commit_diff(
                    repo, versions[0].removeprefix('dbeaver-'), base_version.removeprefix('dbeaver-'), disappeared_version.removeprefix('dbeaver-'), local_file_path,
                    smell_id[2], smell_id[3]
                )
                result_row["commit times"] = commits.count('\n') + 1
            elif local_file_path and last_detected_version:
                # "cassandra-" + base_version, "cassandra-" + last_detected_version
                status, commits = analyze_commit_diff(
                    repo, versions[0].removeprefix('dbeaver-'), base_version.removeprefix('dbeaver-'), base_version.removeprefix('dbeaver-'), local_file_path,
                    smell_id[2], smell_id[3]
                )
                result_row["commit times"] = commits.count('\n') + 1
            if disappeared_version and status:
                result_row["actionable"] = 1
                result_row["gap version"] = abs(versions.index(disappeared_version) - versions.index(base_version))
                result_row["Disappeared Version"] = disappeared_version
                action = action + 1
            else:
                result_row["actionable"] = 0
                result_row["gap version"] = -1
                result_row["Disappeared Version"] = "None"
                nonaction = nonaction + 1
            result_by_version[base_version].append(result_row)
        save_results(base_version, result_by_version[base_version], output_dir)
    print(f"可操作异味数量：{action}, 不可操作异味数量：{nonaction}")
    return result_by_version


# 保存结果到 CSV
def save_results(version, result_by_version, output_dir):
    output_filename = os.path.join(output_dir, f"{version}_actionable.csv")
    fieldnames = ["Project Name", "Package Name", "Type Name", "Method Name","Smell",
                  "Cause of the Smell", "Disappeared Version", "gap version", "commit times", "actionable", ]

    with open(output_filename, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_by_version)
    print(f"Saved results for {version} to {output_filename}")


# 主函数
def main(input_dir, output_dir):
    # 版本排序
    VERSIONS = get_sorted_versions(input_dir)
    # 加载数据
    smell_data = load_smell_data(VERSIONS)
    # 追踪异味消失情况
    repo = "/model/data/Research/actionableGit/dbeaver"
    result_by_version = track_smell_disappearance(smell_data, VERSIONS, repo, output_dir)
    # 保存结果
    # save_results(result_by_version, output_dir)

def get_version_range(df, versions):
    results = []
    versions = [re.search(r'\d.*', version).group(0) for version in versions]
    for _, group in df.groupby('SmellID'):
        # 按版本号排序
        group = group.sort_values(by='Version', key=lambda x: x.apply(parse_version))
        # 获取起始和终结版本
        begin_version = re.search(r'\d.*', group['Version'].iloc[0]).group(0)
        end_version = re.search(r'\d.*', group['Version'].iloc[-1]).group(0)
        # 保留一条记录，并添加 begin 和 end 列
        row = group.iloc[0].copy()
        row['begin'] = begin_version
        row['end'] = end_version
        row['disappear'] = versions[versions.index(end_version) + 1] if end_version != versions[-1] else end_version
        row['gap'] = versions.index(end_version) - versions.index(begin_version) + 1
        results.append(row)
    return pd.DataFrame(results)

@lru_cache(maxsize=128)
def get_file_list(repo_path, type_name, tag):
    command = f'git -C {repo_path} ls-tree -r {tag} --name-only'
    result = subprocess.run(
        command,
        capture_output=True, text=True, check=True, shell=True
    )
    return result.stdout.splitlines()

def check_git_changes(notmatch, package_name, file_path, type_name, method, begin, end, disappear, repo_path):
    # if os.path.exists(repo_path):
    #     print(f"更新仓库: {repo_path}")
    #     subprocess.run(['git', '-C', repo_path, 'fetch'], check=True)
    # else:
    #     print(f"克隆仓库: {repo_path}")
    #     subprocess.run(['git', 'clone', repo_url, repo_path], check=True)
    # subprocess.run(['git', '-C', repo_path, 'checkout', f'tags/{begin}'], check=True)
    # 拼接代码路径
    try:
        # result = subprocess.run(
        #     ['git', '-C', repo_path, 'ls-files'],
        #     capture_output=True, text=True, check=True
        # )
        files = get_file_list(repo_path, type_name, f'tags/{begin}')
        # files = result.stdout.splitlines()
        # package_path = package.replace('.', '/')
        matches = []
        target_file = f"{type_name}.java"
        if file_path.startswith("Ambiguous"):
            # package_name = re.search(r"package\s*'([^']*)'", file_path).group(1)
            # package_name = re.sub(r'\.[A-Z_].*$', '', package_name).replace('.','/')
            matches = [
                path for path in files
                if target_file in path
            ]
        elif file_path.startswith("File Not Found"):
            matches = [
                path for path in files
                if target_file in path
            ]
        else:
            try:
                matches.append(file_path)
            except Exception as e:
                print(file_path, begin)
        if not matches:
            notmatch += 1
            return "", 0, "", 0
        result = subprocess.run(
            ['git', '-C', repo_path, 'diff', f'tags/{begin}', f'tags/{disappear}', '--', matches[0]],
            capture_output=True, text=True, errors='ignore', check=True
        )
        commit = subprocess.run(
            ['git', '-C', repo_path, 'log', '--follow', f'tags/{begin}..tags/{disappear}', '--', matches[0]],
            capture_output=True, text=True, check=True
        )
        commit_count = subprocess.run(
            ['git', '-C', repo_path, 'rev-list', '--count', f'tags/{begin}..tags/{disappear}', '--', matches[0]],
            capture_output=True, text=True, check=True
        )
        commit_count = int(commit_count.stdout.strip())
        if end == disappear or pd.isna(disappear):
            action = 0
        else:
            action = 1 if bool(result.stdout.strip()) else 0
        return result.stdout.strip(), action, commit.stdout.strip(), commit_count
    except subprocess.CalledProcessError as e:
        print(f"git operation fail: {e}")
        return "", 0, "", 0

def process_row(row, repo_path):
    if row["begin"] == row["end"]:
        actionable_list.append(0)
    try:
        diff, actionable, commit_history, commit_count = check_git_changes(
            row['Package Name'],
            row['Type Name'],
            row['Method Name'],
            row['begin'],
            row['end'],
            repo_path
        )
        return actionable, commit_history, commit_count, diff
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        return 0, "", 0, ""  # 默认值

def find_file_path(package: str, type_name: str) -> Optional[str]:
    package_path = package.replace('.', '/')
    try:
        cmd = f"git ls-files '**/{type_name}.java'"
        files = subprocess.check_output(cmd, shell=True).decode('utf-8').strip().split('\n')
        matching_files = [f for f in files if package_path in f]
        if matching_files:
            return matching_files[0]
        elif files:
            # If no exact package match but files with the class name exist
            return files[0]
        return None
    except subprocess.CalledProcessError:
        return None
def find_matching_version(version_str, version_list):
    if not pd.isna(version_str):
        for item in version_list:
            if item.endswith(version_str):  # 检查输入是否是列表元素的子串
                return item
    return ""  # 如果没有找到匹配项
def process_single_row(args):
    row, repo_path = args
    tags = subprocess.check_output(
            ["git", "-C", repo_path, "tag", "-l", "--sort=-v:refname"],
            stderr=subprocess.PIPE,
            universal_newlines=True
        ).splitlines()
    
    notmatch = 0  # 每个进程独立的计数器
    dif, actionable, commit_his, commit_cnt = check_git_changes(
        notmatch, row['Package Name'],
        row['File Path'], row['Type Name'], row['Method Name'], 
        find_matching_version(row['begin'], tags), find_matching_version(row['end'], tags), 
        find_matching_version(row['disappear'], tags),
        repo_path
    )
    return {
        'diff': dif,
        'actionable': actionable,
        'commit_history': commit_his,
        'commit_count': commit_cnt
    }

def temp(project):
    smell_versions = pd.read_csv('/model/data/Research/temp_merged_output/'+project+'.csv').fillna("")

    repo_path = "/model/data/Research/sourceProject/actionableGit/" + project
    
    # 准备并行处理的参数
    process_args = [(row, repo_path) for _, row in smell_versions.iterrows()]
    
    DEBUG_MODE = False  # 调试时设为True，运行时设为False

    if DEBUG_MODE:
        # 单线程顺序执行（便于调试）
        results = []
        for args in tqdm(process_args, desc="Processing rows sequentially"):
            try:
                results.append(process_single_row(args))
            except Exception as e:
                row, path = args
                pd.set_option('display.max_colwidth', None)  # 不限制列宽
                pd.set_option('display.max_rows', None)     # 显示所有行
                pd.set_option('display.max_columns', None)  # 显示所有列
                print(f"Error processing {row}: {str(e)}")
                raise
    else:
        # 并行执行
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(
                executor.map(process_single_row, process_args),
                total=len(process_args),
                desc="Processing rows in parallel"
            ))
    
    # 将结果分配到相应的列
    smell_versions['actionable'] = [r['actionable'] for r in results]
    smell_versions['commit_history'] = [r['commit_history'] for r in results]
    smell_versions['commit_count'] = [r['commit_count'] for r in results]
    smell_versions['diff'] = [r['diff'] for r in results]
    
    smell_versions.to_csv("/model/data/Research/temp_merged_output/"+project+'_action.csv', index=False, encoding='utf-8')

def clean_path(full_path: str, base_dir: str = '/model/data/Research/sourceProject/actionableCS/') -> str:
    """清理路径，去除base_dir和project/version部分"""
    full_path = Path(full_path)
    base_path = Path(base_dir)
    try:
        relative_path = full_path.relative_to(base_path)
        return str(Path(*relative_path.parts[2:]))
    except ValueError:
        return str(full_path)
    
if __name__ == "__main__":
    # 输入文件夹,输出文件夹
    def test(project):
        INPUT_DIR = "/model/data/Research/temp_merged_output/" + project
        OUTPUT_DIR = "/model/data/Research/actionable/" + project
        # os.makedirs(OUTPUT_DIR, exist_ok=True)
        # main(INPUT_DIR, OUTPUT_DIR)

        versions = get_sorted_versions(INPUT_DIR)
        source_path = "/model/data/Research/sourceProject/actionableCS/" + project
        repo_path = "/model/data/Research/sourceProject/actionableGit/" + project
        all_data = []
        for version in versions:
            filename = os.path.join(INPUT_DIR, version + '.csv')
            df = pd.read_csv(filename)
            df = df.fillna("")
            all_data.append(df)
        merged_df = pd.concat(all_data, ignore_index=True)
        merged_df['Version'] = merged_df['Version'].astype(str)
        merged_df['Smell'] = merged_df['Smell'].astype(str)
        merged_df["SmellID"] = (merged_df["File Path"] + "." +
                                merged_df["Type Name"] + "." + merged_df["Method Name"] + "." +
                                merged_df["Smell"])
        smell_versions = get_version_range(merged_df, versions)
        smell_versions.to_csv('/model/data/Research/temp_merged_output/'+project+'.csv', index=False, encoding='utf-8')
    projects = [project.name for project in Path('/model/data/Research/temp_merged_output').iterdir() if project.is_dir()]
    for project in projects:
        if os.path.exists('/model/data/Research/temp_merged_output/'+str(project)+'.csv'):
            continue
        test(project)
    for project in projects:
        if os.path.exists('/model/data/Research/temp_merged_output/'+str(project)+'_action.csv'):
            continue
        temp(project)
