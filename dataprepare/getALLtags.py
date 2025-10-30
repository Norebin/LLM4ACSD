import re
from collections import defaultdict
import subprocess

import requests
import os
import sys
import zipfile
import argparse
from tqdm import tqdm
from test import unique_versions
from judgeActionable import get_sorted_versions

def get_repo_tags(owner, repo, token='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'):
    """获取仓库的所有标签"""
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    url = f"https://api.github.com/repos/{owner}/{repo}/tags"
    tags = []
    page = 1
    while True:
        response = requests.get(f"{url}?page={page}&per_page=200", headers=headers)
        if response.status_code != 200:
            print(f"Error fetching tags: {response.status_code} - {response.text}")
            sys.exit(1)
        page_tags = response.json()
        if not page_tags:
            break
        tags.extend(page_tags)
        page += 1
    return tags

def filter_versions(versions):
    # 分离正式版和预发布版，并解析版本号
    def parse_version(version):
        # 提取版本号部分，例如 "cassandra-5.0.3" -> "5.0.3"
        match = re.match(r"v-(\d+\.\d+(?:\.\d+)?(?:-.+)?)", version)
        if not match:
            return None
        version_str = version.split('-')[-1]
        # 分割版本号和预发布后缀
        if '-' in version_str:
            base_version, prerelease = version_str.split('-', 1)
        elif '.' in version_str and version_str.count('.') == 1 and any(c.isalpha() for c in version_str):
            # 处理如 "3.2M1" 这样的情况，假设最后一个点后是预发布标记
            base_version, prerelease = version_str.rsplit('.', 1)
        else:
            base_version, prerelease = version_str, None
        # 分割主要、次要、补丁版本
        parts = base_version.split('.')
        major = int(parts[0])
        minor = int(parts[1])
        patch = int(parts[2]) if len(parts) > 2 else 0
        return {
            "name": version,
            "major": major,
            "minor": minor,
            "patch": patch,
            "prerelease": prerelease  # None 表示正式版
        }
    def parse_version2(version):
        version = version.replace('cayenne-parent-', '')
        match = re.match(r'^(\d+)(?:\.(\d+))?(?:\.(\d+))?([A-Za-z]+\d*)?$', version)
        if not match:
            raise ValueError(f"Cannot parse version: {version}")
        main_version = [
            int(match.group(1)),  # 主版本号
            int(match.group(2)) if match.group(2) else 0,  # 次版本号
            int(match.group(3)) if match.group(3) and match.group(3).isdigit() else 0  # 补丁版本号
        ]
        # 处理里程碑、Beta、RC等特殊版本
        special_version = 0
        special_type = 0
        if match.group(4):
            special_mark = match.group(4)
            if special_mark.startswith('M'):
                special_type = 1  # 里程碑版本
                special_version = int(special_mark[1:])
            elif special_mark.startswith('B'):
                special_type = 2  # Beta版本
                special_version = int(special_mark[1:])
            elif special_mark.startswith('RC'):
                special_type = 3  # 候选版本
                special_version = int(special_mark[2:])
            elif special_mark == 'final':
                special_type = 4
        return (main_version + [special_type, special_version])
    # 存储正式版和预发布版
    formal_versions = defaultdict(list)
    prerelease_versions = defaultdict(list)
    for version in versions:
        parsed = parse_version2(version)
        if parsed:
            key = (parsed["major"], parsed["minor"])  # 以 major.minor 作为分组依据
            if parsed["prerelease"]:
                prerelease_versions[key].append(parsed)
            else:
                formal_versions[key].append(parsed)
    # 筛选正式版：取最小补丁版本
    filtered_formal = []
    for key, group in formal_versions.items():
        # 按 patch 从小到大排序，取第一个
        min_version = min(group, key=lambda x: x["patch"])
        filtered_formal.append(min_version["name"])
    # 筛选预发布版：定义预发布优先级，取最早的

    def prerelease_priority(prerelease):
        if not prerelease:
            return (float('inf'),)  # 正式版优先级最高（但这里不涉及）
        prefix_order = {'alpha': 0, 'beta': 1, 'rc': 2}  # alpha < beta < rc
        match = re.match(r"([a-z]+)(\d+)?", prerelease)
        if match:
            prefix, num = match.groups()
            num = int(num) if num else 0
            return (prefix_order.get(prefix, -1), num)
        return (-1, 0)  # 默认最低优先级

    filtered_prerelease = []
    for key, group in prerelease_versions.items():
        # 按预发布优先级排序，取最早的
        min_version = min(group, key=lambda x: prerelease_priority(x["prerelease"]))
        filtered_prerelease.append(min_version["name"])
    # 合并结果并排序（可选）
    result = sorted(filtered_formal + filtered_prerelease)
    return result

def download_tag(owner, repo, down_tags, output_dir=None, token=None):
    """下载特定标签版本的代码"""
    if not output_dir:
        output_dir = os.getcwd()
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 下载ZIP文件 https://api.github.com/repos/{owner}/{repo}/zipball/refs/tags/version
    # 下载tar文件 https://api.github.com/repos/{owner}/{repo}/tarball/refs/tags/version
    for tag_name in down_tags:
        # zip_url = f"https://github.com/{owner}/{repo}/archive/refs/tags/v{tag_name}.tar.gz"
        zip_url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/refs/tags/{tag_name}"
        zip_file = os.path.join(output_dir, f"{tag_name}.tar.gz")
        if os.path.exists(zip_file):
            continue
        print(f"Downloading {tag_name} from {zip_url}...")
        headers = {}
        if token:
            headers["Authorization"] = f"token {token}"
        response = requests.get(zip_url, headers=headers, stream=True)
        if response.status_code != 200:
            print(f"Error downloading tag: {response.status_code} - {response.text}")
            return False
        # 使用tqdm显示下载进度
        total_size = int(response.headers.get('content-length', 0))
        with open(zip_file, 'wb') as f, tqdm(
                desc=tag_name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    return True

def download_tag_assets(owner, repo, down_tags, output_dir=None, token=None):
    headers = {'Accept': 'application/vnd.github.v3+json',
               'Authorization':f'token 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'
               }
    if token:
        headers['Authorization'] = f'token {token}'
    if output_dir is None:
        output_dir = '.'
    os.makedirs(output_dir, exist_ok=True)
    urls = []
    for tag in down_tags:
        # 获取release信息
        url = f'https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}'
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print(f'获取 {tag} release 信息失败: {resp.status_code}')
            continue
        release = resp.json()
        assets = release.get('assets', [])
        for asset in assets:
            name = asset['name']
            if name.endswith('linux.gtk.x86_64.tar.gz'):
                download_url = asset['browser_download_url']
                file_path = os.path.join(output_dir, name)
                print(f'正在下载: {name}')
                urls.append(download_url)
                # cmd = [
                #     'wget',
                #     '-P', output_dir,
                    
                # ]
                # subprocess.run(cmd)
    return True

def list_tags(tags, count=10):
    """列出标签"""
    print(f"Found {len(tags)} tags:")
    for i, tag in enumerate(tags[:count]):
        print(f"{i + 1}. {tag['name']}")
    if len(tags) > count:
        print(f"... and {len(tags) - count} more")


def main(args):
    try:
        owner, repo = args.repo.split("/")
    except ValueError:
        print("Error: Repository must be in the format 'owner/repo' (e.g., 'elastic/elasticsearch')")
        sys.exit(1)

    # 获取所有标签
    # tags = get_repo_tags(owner, repo, args.token)
    # if args.list or args.list_all:
    #     if args.list_all:
    #         list_tags(tags, len(tags))
    #     else:
    #         list_tags(tags)
    #     return
    if args.tag:
        # 下载特定标签
        download_tag_assets(owner, repo, args.tag, args.output, args.token)
    else:
        # 列出最近的10个标签并提示用户选择
        list_tags(tags)
        try:
            choice = int(input("\nEnter the number of the tag to download (or 0 to quit): "))
            if 1 <= choice <= len(tags):
                download_tag(owner, repo, tags[choice - 1]["name"], args.output, args.token)
            elif choice != 0:
                print("Invalid choice")
        except ValueError:
            print("Invalid input")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download specific GitHub repository tags")
    parser.add_argument("--repo", help="Repository in format 'owner/repo' (e.g., 'elastic/elasticsearch')")
    parser.add_argument("--tag", help="Specific tag to download")
    parser.add_argument("--list", action="store_true", help="List available tags")
    parser.add_argument("--list-all", action="store_true", help="List all available tags")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--token", help="GitHub Personal Access Token for private repos or higher rate limits")
    args = parser.parse_args()
    # 显示某个项目所有的版本号
    versions = get_repo_tags('dbeaver', 'dbeaver',)
    versions = [item["name"] for item in versions]
    # versions.remove('struts2-parent-2.3.14.1')
    # versions.remove('struts2-parent-2.3.1.2')
    # versions.remove('struts2-parent-2.0.10')
    # versions = get_sorted_versions("/model/data/Research/designitejava/cassandra")
    # versions = [x for x in versions if '+' not in x]
    # print(versions)
    # filtered_version = [v for v in versions if v.get("name", "").startswith("v")]
    # args.tag = filter_versions([item["name"] for item in versions])
    args.tag = unique_versions(versions)
    # args.tag = ["cassandra-" + x for x in versions]
    args.repo = 'dbeaver/dbeaver'
    args.output = '/model/data/Research/sourceProject/actionableCS/dbeaver/jar'
    main(args)
