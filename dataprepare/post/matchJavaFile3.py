import pandas as pd
from pathlib import Path
import os
import re # 用于正则表达式解析包声明

# --- 配置路径 ---
DESIGNITE_OUTPUT_BASE_DIR = Path("/model/data/Research/tools/designitejava/")
SOURCE_PROJECT_BASE_DIR = Path("/model/data/Research/sourceProject/actionableCS/")
SMELL_CSV_FILENAME = "Smell.csv"
FILE_PATH_COLUMN_NAME = "File Path"
JAVA_FILE_NOT_FOUND_MSG = "File Not Found"

# 正则表达式用于从 Java 文件中提取包声明
# 匹配 'package com.example.mypackage;' 这样的行
# 它会处理 package 关键字前后的空格，以及分号前的空格
PACKAGE_REGEX = re.compile(r"^\s*package\s+([a-zA-Z_][\w\.]*)\s*;")

def get_package_from_java_content(java_file_path: Path):
    """
    读取Java文件的内容，从中提取包声明。
    返回包名字符串，如果未找到包声明则返回空字符串"" (表示默认包)，
    如果读取或解析文件出错则返回 None。
    """
    try:
        # 尝试使用几种常用编码打开文件
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252']
        package_name = "" # 默认为空字符串 (默认包)
        
        file_content_read = False
        for encoding in encodings_to_try:
            try:
                with open(java_file_path, 'r', encoding=encoding) as f:
                    # 为了效率，通常包声明在文件的前面几行
                    # 可以限制读取的行数，但为了准确性，这里读取整个文件
                    # 不过，一旦找到就停止
                    for line in f:
                        match = PACKAGE_REGEX.match(line)
                        if match:
                            package_name = match.group(1)
                            file_content_read = True
                            break # 找到包声明，停止读取当前文件
                    if file_content_read: # 如果此编码成功读取并找到包
                        break # 停止尝试其他编码
            except UnicodeDecodeError:
                continue # 如果当前编码失败，尝试列表中的下一个
            except Exception as e:
                # print(f"警告: 读取文件 {java_file_path} 时使用编码 {encoding} 发生错误: {e}")
                # 如果发生其他类型的读取错误，也尝试下一个编码
                continue
        
        # 如果尝试所有编码后都未能读取文件内容（例如文件权限问题），
        # 此时 file_content_read 仍为 False，应该返回 None 表示无法确定
        # 但如果文件能被读取，只是没有 package 语句，则 package_name 会是 "" (默认包)
        # 这里的逻辑是：如果至少有一种编码能读完文件（即使没找到package语句），就算成功读取。
        # 实际上，上面的循环会在成功读取并找到包后break，或尝试完所有编码。
        # 如果所有编码都失败导致无法读取任何内容，那么可能应该返回 None。
        # 简单起见，如果循环结束而 file_content_read 为 False，但没有因IO错误抛出，
        # 就意味着文件被读取了（至少部分），但没有找到 package 语句。

        return package_name # 返回找到的包名或默认包""

    except Exception as e: # 捕获打开文件之前的错误，如路径问题
        print(f"严重警告: 处理文件 {java_file_path} 的包信息时发生无法恢复的错误: {e}")
        return None
def build_java_file_cache1(source_project_version_path: Path):
    """
    为指定项目版本的源文件构建一个快速查找缓存。
    键: (package_name, type_name_without_extension)
    值: absolute_file_path
    """
    cache = {}
    if not source_project_version_path.is_dir():
        print(f"警告: 源项目版本目录不存在: {source_project_version_path}")
        return cache

    for java_file in source_project_version_path.rglob("*.java"):
        try:
            # 获取相对于版本根目录的路径
            relative_path = java_file.relative_to(source_project_version_path)
            
            # 从文件名获取类型名 (不含 .java)
            type_name = java_file.stem
            
            # 从目录结构推断包名
            # relative_path.parent 是父目录, parts是各级目录组成的元组
            # 例如: org/example/MyClass.java -> parent = org/example, parts = ('org', 'example')
            if relative_path.parent == Path('.'): # 文件在根目录, 即默认包
                package_name = ""
            else:
                parent_parts = list(relative_path.parent.parts)
                org_index = parent_parts.index('org')
                relevant_parts = parent_parts[org_index:]
                package_name = '.'.join(relevant_parts)
            cache[(package_name, type_name)] = str(java_file.resolve())
        except Exception as e:
            print(f"处理文件 {java_file} 时出错: {e}")
    return cache

def build_java_file_cache(source_project_version_path: Path):
    """
    为指定项目版本的源文件构建一个快速查找缓存。
    键: (package_name_from_declaration, type_name_from_filename)
    值: absolute_file_path
    """
    cache = {}
    if not source_project_version_path.is_dir():
        print(f"警告: 源项目版本目录不存在: {source_project_version_path}")
        return cache

    java_files_scanned = 0
    java_files_cached = 0
    for java_file in source_project_version_path.rglob("*.java"):
        java_files_scanned += 1
        type_name = java_file.stem # 文件名作为类型名 (不含 .java)
        
        # 从Java文件内容中提取包名
        declared_package_name = get_package_from_java_content(java_file)
        
        if declared_package_name is not None:
            # 使用从文件内容中声明的包名
            cache[(declared_package_name, type_name)] = str(java_file.resolve())
            java_files_cached += 1
        else:
            # 如果 get_package_from_java_content 返回 None，表示无法处理该文件
            print(f"注意: 文件 {java_file.name} 因无法确定包名将不会被缓存。")
            
    print(f"扫描了 {java_files_scanned} 个 .java 文件。成功缓存了 {java_files_cached} 个文件的路径信息。")
    return cache

def process_smell_file(smell_csv_path: Path, java_file_cache: dict):
    """
    读取Smell.csv文件，匹配Java文件路径，并添加新列。
    """
    # if not smell_csv_path.exists():
    #     print(f"Smell.csv 文件未找到: {smell_csv_path}")
    #     return

    try:
        # 尝试显式指定UTF-8，如果失败则让pandas自动检测
        try:
            df = pd.read_csv(smell_csv_path, dtype=str) # 读取所有列为字符串以避免类型问题
        except UnicodeDecodeError:
            print(f"警告: 读取 {smell_csv_path.name} 时UTF-8解码失败，尝试自动检测编码...")
            df = pd.read_csv(smell_csv_path, dtype=str, encoding_errors='replace')
        except Exception as e:
            print(f"读取CSV文件失败 {smell_csv_path}: {e}")
            return
            
    except Exception as e: # 其他pd.read_csv 可能的错误
        print(f"读取CSV文件时发生未知错误 {smell_csv_path}: {e}")
        return

    # 确保必要的列存在，并且填充空值，避免后续 .iterrows() 出错
    required_cols = ['Package Name', 'Type Name']
    for col in required_cols:
        if col not in df.columns:
            print(f"错误: {smell_csv_path.name} 缺少必要的列: '{col}'. 可用列: {df.columns.tolist()}. 跳过此文件.")
            return
    # 将NaN值替换为空字符串，以便后续处理
    df.fillna("", inplace=True)

    failed = 0
    file_paths = []
    for index, row in df.iterrows():
        package_name_csv = row['Package Name']
        type_name_csv = row['Type Name'] 

        # 处理CSV中的特殊包名 "(Default)" 或空包名
        if package_name_csv == "(Default)" or package_name_csv == "":
            package_name_lookup = ""
        else:
            package_name_lookup = package_name_csv

        # 获取外部类名用于文件查找 (e.g., Outer$Inner -> Outer)
        outer_type_name_lookup = type_name_csv.split('$')[0]

        # 在缓存中查找
        found_path = java_file_cache.get((package_name_lookup, outer_type_name_lookup), JAVA_FILE_NOT_FOUND_MSG)
        if found_path == JAVA_FILE_NOT_FOUND_MSG:
            failed += 1
        file_paths.append(found_path)

    print(f"失败{failed}")
    if FILE_PATH_COLUMN_NAME in df.columns:
         print(f"警告: 列 '{FILE_PATH_COLUMN_NAME}' 已存在于 {smell_csv_path}。将覆盖此列。")
    df[FILE_PATH_COLUMN_NAME] = file_paths

    try:
        df.to_csv(smell_csv_path, index=False, encoding='utf-8')
        print(f"已更新: {smell_csv_path.name} (位于 {smell_csv_path.parent})")
    except Exception as e:
        print(f"写入CSV文件失败 {smell_csv_path}: {e}")


def main():
    if not DESIGNITE_OUTPUT_BASE_DIR.is_dir():
        print(f"错误: DesigniteJava 输出目录不存在: {DESIGNITE_OUTPUT_BASE_DIR}")
        return
    if not SOURCE_PROJECT_BASE_DIR.is_dir():
        print(f"错误: 源项目根目录不存在: {SOURCE_PROJECT_BASE_DIR}")
        return

    # 1. 遍历DesigniteJava输出目录中的项目
    for project_dir in DESIGNITE_OUTPUT_BASE_DIR.iterdir():
        if str(project_dir) != '/model/data/Research/tools/designitejava/struts':
            continue
        if project_dir.is_dir():
            project_name = project_dir.name
            
            # 2. 遍历项目下的版本文件夹
            for version_dir in project_dir.iterdir():
                if version_dir.is_dir():
                    version_name = version_dir.name
                    print(f"\n--- 正在处理项目: {project_name}, 版本: {version_name} ---")

                    smell_csv_file_path = version_dir / SMELL_CSV_FILENAME
                    source_project_version_path = SOURCE_PROJECT_BASE_DIR / project_name / version_name

                    if not smell_csv_file_path.exists():
                        print(f"{SMELL_CSV_FILENAME} 在 {version_dir.name} (项目 {project_name}) 中未找到，跳过。")
                        continue
                    
                    java_file_cache = {} # 初始化为空字典
                    if not source_project_version_path.is_dir():
                        print(f"对应的源项目版本目录 {source_project_version_path.name} (项目 {project_name}) 不存在。该版本下的文件路径将无法找到。")
                    else:
                        # 3. 为当前项目版本构建Java文件缓存 (读取文件内容)
                        print(f"正在为 {source_project_version_path.name} (项目 {project_name}) 构建Java文件缓存 (将读取.java文件内容)...")
                        java_file_cache = build_java_file_cache(source_project_version_path)
                        
                        # 检查缓存是否为空以及是否有Java文件实际存在
                        if not java_file_cache:
                            # 检查目录下是否真的没有java文件，还是都无法解析
                            has_java_files = any(source_project_version_path.rglob("*.java"))
                            if has_java_files:
                                print(f"警告: 在 {source_project_version_path.name} 中找到Java文件，但缓存为空。可能所有文件都无法确定包名或读取失败。")
                            else:
                                print(f"信息: 未在 {source_project_version_path.name} 中找到任何Java文件。")


                    # 4. 处理Smell.csv文件
                    process_smell_file(smell_csv_file_path, java_file_cache)
    
    print("\n--- 所有处理完成 ---")

if __name__ == "__main__":
    # main()
    project_name = 'struts'
    for root, _, files in os.walk('/model/data/Research/tools/Decor/'+project_name):
        for file in files:
            if file.endswith('.csv'):
                smell_csv_file_path = os.path.join(root, file)
                source_project_version_path = SOURCE_PROJECT_BASE_DIR / project_name / file.removesuffix(".csv")

                if not os.path.exists(smell_csv_file_path):
                    print(f"{SMELL_CSV_FILENAME} 在 {version_dir.name} (项目 {project_name}) 中未找到，跳过。")
                    continue
                
                java_file_cache = {} # 初始化为空字典
                if not source_project_version_path.is_dir():
                    print(f"对应的源项目版本目录 {source_project_version_path.name} (项目 {project_name}) 不存在。该版本下的文件路径将无法找到。")
                else:
                    # 3. 为当前项目版本构建Java文件缓存 (读取文件内容)
                    print(f"正在为 {source_project_version_path.name} (项目 {project_name}) 构建Java文件缓存 (将读取.java文件内容)...")
                    java_file_cache = build_java_file_cache(source_project_version_path)
                    
                    # 检查缓存是否为空以及是否有Java文件实际存在
                    if not java_file_cache:
                        # 检查目录下是否真的没有java文件，还是都无法解析
                        has_java_files = any(source_project_version_path.rglob("*.java"))
                        if has_java_files:
                            print(f"警告: 在 {source_project_version_path.name} 中找到Java文件，但缓存为空。可能所有文件都无法确定包名或读取失败。")
                        else:
                            print(f"信息: 未在 {source_project_version_path.name} 中找到任何Java文件。")


                # 4. 处理Smell.csv文件
                process_smell_file(smell_csv_file_path, java_file_cache)