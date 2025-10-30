import os
import csv

def find_java_file_in_version(source_version_root_path, package_name, type_name):
    """
    在给定的源文件版本根目录中查找对应的Java文件。

    Args:
        source_version_root_path (str): 对应项目版本的源文件根目录。
                                         例如 /model/data/Research/sourceProject/actionableCS/dbeaver/dbeaver-3.6.0
        package_name (str): Java包名 (例如 "org.jkiss.dbeaver.core")。
        type_name (str): Java类型名 (例如 "DBeaverCore" 或 "OuterClass$InnerClass")。

    Returns:
        str or None: 如果找到，返回Java文件的绝对路径，否则返回None。
    """
    if not os.path.isdir(source_version_root_path):
        # print(f"  Debug: Source version root path does not exist: {source_version_root_path}")
        return None

    # 处理内部类：文件名通常是外部类名
    # 例如 "OuterClass$InnerClass" 应该查找 "OuterClass.java"
    outer_type_name = type_name.split('$')[0]
    target_java_filename = f"{outer_type_name}.java"

    # 将包名转换为相对路径格式
    # 例如 "org.jkiss.dbeaver.core" -> "org/jkiss/dbeaver/core"
    if package_name: # 如果包名非空
        package_as_path = package_name.replace('.', os.sep)
        # 期望的文件路径后缀，例如 "org/jkiss/dbeaver/core/DBeaverCore.java"
        expected_path_suffix = os.path.join(package_as_path, target_java_filename)
    else: # 默认包 (包名为空)
        package_as_path = "" # 确保 os.path.join 能正确处理
        expected_path_suffix = target_java_filename


    # print(f"  Debug: Searching for suffix '{expected_path_suffix}' in '{source_version_root_path}'")
    for root, _, files in os.walk(source_version_root_path):
        if target_java_filename in files:
            potential_file_path = os.path.join(root, target_java_filename)
            # print(f"  Debug: Found candidate file: {potential_file_path}")

            # 为了确保匹配的是正确的包路径下的文件，检查路径后缀
            # 标准化路径分隔符以便可靠比较
            normalized_potential_path = potential_file_path.replace(os.sep, '/')
            normalized_expected_suffix = expected_path_suffix.replace(os.sep, '/')

            # 检查 potential_file_path 是否以 expected_path_suffix 结尾
            # 如果 package_name 为空, normalized_expected_suffix 就是 "FileName.java"
            # 我们需要确保它位于某个源根目录下，而不是 случайно 在子包中找到同名文件
            # (例如, 避免匹配 a/b/FileName.java 当期望的是 FileName.java 或 src/FileName.java)
            # 通过检查 '.../' + normalized_expected_suffix 确保它不是一个更长包路径的一部分
            # 或者，可以检查 (root_path + separator + expected_suffix)
            
            # 一个更可靠的检查是，从 potential_path 中移除 source_version_root_path 部分（以及可能的 src/main/java 等）
            # 然后看剩下的部分是否等于 expected_path_suffix
            # 例如:
            # potential_path = /source_root/src/main/java/com/example/MyClass.java
            # expected_suffix = com/example/MyClass.java
            # -> (src/main/java/ + com/example/MyClass.java).endswith(com/example/MyClass.java)

            # 更简单：确保路径末尾完全匹配预期的包和文件名结构
            if normalized_potential_path.endswith('/' + normalized_expected_suffix) or \
               (not package_name and normalized_potential_path.endswith('/' + target_java_filename)): # 处理默认包的情况，可能直接在src下
                # 确保不是匹配到错误的更深层级
                # 例如，如果 expected_path_suffix 是 "MyClass.java" (默认包)
                # 那么 /project/src/MyClass.java 是一个匹配
                # 而 /project/src/com/example/MyClass.java 不是 (除非 type_name 也是 com.example.MyClass)
                #
                # 假设我们总是从某个 "source root" (如 src/main/java, src, 或 version_root_path 本身)开始
                #  查找。如果 `potential_file_path` 减去 `source_version_root_path` 后，再去掉常见的
                #  source root (如 src/main/java, src/java, src) 等于 `expected_path_suffix`，则是精确匹配。
                #
                # 对于 `os.walk`，`root` 已经是实际包含文件的目录。
                # 所以 `os.path.join(root, target_java_filename)` 是完整路径。
                # 检查 `root` 是否以 `package_as_path` 结尾（或其一部分，如果包名本身就是根）
                normalized_root = root.replace(os.sep, '/')
                normalized_package_as_path = package_as_path.replace(os.sep, '/')
                if normalized_root.endswith(normalized_package_as_path) or \
                   (not package_name and (normalized_root.endswith("/src") or normalized_root == source_version_root_path.replace(os.sep, '/'))): # 默认包可能在src下或直接在根下
                    # print(f"    Debug: Path suffix match for {potential_file_path}")
                    return potential_file_path
                # else:
                    # print(f"    Debug: Path suffix mismatch. Root: {normalized_root}, Expected package path: {normalized_package_as_path}")


    # print(f"  Debug: File not found with suffix '{expected_path_suffix}'")
    return None


# --- 主程序 ---
designite_base_dir = "/model/data/Research/tools/designitejava/"
source_projects_base_dir = "/model/data/Research/sourceProject/actionableCS/"

# 检查基础目录是否存在
if not os.path.isdir(designite_base_dir):
    print(f"错误：DesigniteJava 目录不存在: {designite_base_dir}")
    exit(1)
if not os.path.isdir(source_projects_base_dir):
    print(f"错误：源项目目录不存在: {source_projects_base_dir}")
    exit(1)

# 遍历 DesigniteJava 下的每个项目
for project_name in os.listdir(designite_base_dir):
    if project_name != 'dbeaver':
        continue
    project_path_designite = os.path.join(designite_base_dir, project_name)
    project_path_source = os.path.join(source_projects_base_dir, project_name)

    if not os.path.isdir(project_path_designite):
        continue # 跳过非目录文件

    print(f"处理项目: {project_name}")

    if not os.path.isdir(project_path_source):
        print(f"  警告: 未找到对应的源项目目录: {project_path_source}")
        continue
    failed = 0 
    # 遍历项目下的每个版本文件夹
    for version_folder_name in os.listdir(project_path_designite):
        version_path_designite = os.path.join(project_path_designite, version_folder_name)
        version_path_source = os.path.join(project_path_source, version_folder_name) # 源项目中对应的版本目录

        if not os.path.isdir(version_path_designite):
            continue

        print(f"  处理版本: {version_folder_name}")

        smell_csv_file = os.path.join(version_path_designite, "Smell.csv")
        if not os.path.exists(smell_csv_file):
            print(f"    警告: Smell.csv 未在以下路径找到: {smell_csv_file}")
            continue

        if not os.path.isdir(version_path_source):
            print(f"    警告: 未找到此版本的源文件目录: {version_path_source}")
            continue

        try:
            with open(smell_csv_file, mode='r', encoding='utf-8', newline='') as csvfile:
                # 注意：Designite输出的CSV可能是用制表符分隔的，或者有特殊的引用字符处理
                # 先尝试标准逗号分隔，如果不行，可能需要指定 dialect 或 delimiter
                # reader = csv.DictReader(csvfile)
                # Designite Java 的CSV输出通常是逗号分隔，但有时列名可能有额外的空格
                # 我们先读取头部并清理
                header_line = csvfile.readline()
                headers = [h.strip() for h in header_line.strip().split(',')]
                
                # 检查必要的列是否存在
                required_cols = ["Package Name", "Type Name", "Method Name"] # Project Name 也在，但我们直接从文件夹获取
                if not all(col in headers for col in ["Package Name", "Type Name"]): # Method Name is optional for file finding
                    print(f"    错误: CSV文件 {smell_csv_file} 缺少必要的列 (Package Name, Type Name). 实际列: {headers}")
                    continue

                reader = csv.DictReader(csvfile, fieldnames=headers) # 使用清理过的头部重新创建reader

                for i, row in enumerate(reader):
                    # csv.DictReader 会使用清理过的 headers 作为键
                    package_name = row.get("Package Name", "").strip()
                    type_name = row.get("Type Name", "").strip()
                    method_name = row.get("Method Name", "").strip() # 方法名暂时记录，用于后续操作

                    if not type_name: # Type Name 是定位文件的关键
                        # print(f"      跳过第 {i+2} 行，因为 Type Name 为空: {row}")
                        continue
                    
                    # 如果 package_name 是 "<default>" 或类似 Designite 的输出，将其视为空包名
                    if package_name.lower() == "<default>" or package_name == "-":
                        package_name = ""

                    # print(f"    查找: 包='{package_name}', 类='{type_name}', 方法='{method_name}'")
                
                    found_java_file = find_java_file_in_version(version_path_source, package_name, type_name)

                    if found_java_file:
                        print(f"      匹配成功: {package_name}::{type_name}::{method_name} -> {found_java_file}")
                        # 在这里，您可以添加进一步处理 found_java_file 的代码
                        # 例如，打开文件，搜索 method_name 等
                    else:
                        failed += 1
                        print(f"      匹配失败: {package_name}::{type_name} 在 {version_path_source} 中未找到对应Java文件。")
        except FileNotFoundError:
            print(f"    错误: Smell.csv 文件未找到于: {smell_csv_file}")
        except Exception as e:
            print(f"    处理 Smell.csv 文件时发生错误 {smell_csv_file}: {e}")
print(f"失败{failed}条记录")
print("\n处理完成。")