import os
import csv
from concurrent.futures import ThreadPoolExecutor
import time # 用于计时

# --- 1. Java 文件索引构建模块 ---
def build_java_file_index(source_version_root_path):
    """
    为给定的源文件版本根目录构建Java文件索引。

    Args:
        source_version_root_path (str): 项目版本的源文件根目录。

    Returns:
        dict: 一个字典，键是元组 (package_name, outer_type_name)，值是Java文件的绝对路径。
    """
    java_file_index = {}
    if not os.path.isdir(source_version_root_path):
        return java_file_index

    # 常见的Java源文件相对根目录的模式
    # 顺序很重要：从更具体/深层嵌套的到更通用的。
    # "" 表示包结构直接从 source_version_root_path 开始。
    relative_src_roots_patterns = [
        os.path.join("src", "main", "java"),
        os.path.join("src", "test", "java"), # 也索引测试文件，如果smell可能出现在那里
        os.path.join("src", "main", "kotlin"), # 如果也考虑Kotlin并且结构相同
        "src", # 通用源文件夹
        ""  # 包结构直接在版本根目录下开始
    ]
    
    # 将相对源根目录转换为绝对路径以便于比较
    # 并确保它们存在，只使用实际存在的源根目录进行索引
    valid_absolute_src_roots = []
    for rel_pattern in relative_src_roots_patterns:
        abs_pattern_path = os.path.normpath(os.path.join(source_version_root_path, rel_pattern))
        if os.path.isdir(abs_pattern_path): # 只考虑实际存在的目录作为源根
            valid_absolute_src_roots.append(abs_pattern_path)
    
    # 如果没有任何有效的源根目录模式匹配（例如，空的 version_path_source），则直接返回空索引
    if not valid_absolute_src_roots:
        # print(f"    [Indexer] No valid source root patterns found in {source_version_root_path}")
        return java_file_index

    for dir_path_abs, _, file_names in os.walk(source_version_root_path, topdown=True):
        norm_current_dir_abs = os.path.normpath(dir_path_abs)
        
        # 尝试确定当前目录是基于哪个有效的源根目录
        current_base_src_root = None
        package_sub_path_str = None

        for abs_src_root in valid_absolute_src_roots:
            if norm_current_dir_abs.startswith(abs_src_root):
                current_base_src_root = abs_src_root
                if norm_current_dir_abs == current_base_src_root:
                    package_sub_path_str = ""
                else:
                    package_sub_path_str = os.path.relpath(norm_current_dir_abs, current_base_src_root)
                
                if package_sub_path_str == ".": # os.path.relpath 可能返回 "."
                    package_sub_path_str = ""
                break # 找到了最匹配的源根目录

        if current_base_src_root is None:
            # 如果当前目录不在任何已知的源根目录下，则跳过（例如，可能是 .git, target, build 等目录）
            # 通过修改 os.walk 的 topdown=True 和 dirs[:] 来剪枝可以更主动地避免遍历这些目录
            # dirs[:] = [d for d in dirs if d not in ['.git', 'target', 'build', 'node_modules']] # 示例剪枝
            continue

        package_name_str = package_sub_path_str.replace(os.sep, '.') if package_sub_path_str else ""

        for file_name in file_names:
            if not file_name.endswith(".java"): # 或 .kt
                continue

            full_file_path = os.path.normpath(os.path.join(norm_current_dir_abs, file_name))
            outer_type_name = file_name[:-5]  # 移除 ".java"
            
            key = (package_name_str, outer_type_name)
            if key not in java_file_index: # 保留找到的第一个 (如果源根目录顺序正确，通常来自更具体的)
                java_file_index[key] = full_file_path
            # else:
            #     # 如果需要处理重复情况，例如，一个类在多个源根中定义
            #     print(f"    [Indexer] 警告: 索引键重复 ('{package_name_str}', '{outer_type_name}')")
            #     print(f"                 旧路径: {java_file_index[key]}")
            #     print(f"                 新路径: {full_file_path}")


    return java_file_index

# --- 2. 单个版本处理模块 (用于并发) ---
def process_version_task(project_name, version_folder_name, project_designite_base_path, project_source_base_path):
    """
    处理单个项目版本的任务：构建索引、读取CSV、匹配记录。
    """
    thread_prefix = f"({project_name}/{version_folder_name})"
    # print(f"  {thread_prefix} 开始处理...")

    version_designite_path = os.path.join(project_designite_base_path, version_folder_name)
    version_source_path = os.path.join(project_source_base_path, version_folder_name)

    # 1. 构建Java文件索引
    java_files_idx = {}
    if os.path.isdir(version_source_path):
        # print(f"    {thread_prefix} 正在为 {version_source_path} 创建Java文件索引...")
        java_files_idx = build_java_file_index(version_source_path)
        # if not java_files_idx:
        #     print(f"    {thread_prefix} 警告: 在 {version_source_path} 中未能索引到任何Java文件，或目录结构不符合预期。")
        # else:
        #     print(f"    {thread_prefix} 索引创建完成，共找到 {len(java_files_idx)} 个Java文件条目。")
    else:
        print(f"    {thread_prefix} 警告: 源文件目录不存在: {version_source_path}")


    # 2. 处理 Smell.csv
    smell_csv_file = os.path.join(version_designite_path, "Smell.csv")
    if not os.path.exists(smell_csv_file):
        print(f"    {thread_prefix} 警告: Smell.csv 未在以下路径找到: {smell_csv_file}")
        return f"{thread_prefix}: Smell.csv not found."

    matches_found = 0
    rows_processed = 0
    results_log = [] # 用于收集此线程的详细结果，如果需要的话

    try:
        with open(smell_csv_file, mode='r', encoding='utf-8', newline='') as csvfile:
            # 读取并清理头部
            header_line = csvfile.readline()
            if not header_line: # 空文件
                print(f"    {thread_prefix} 警告: Smell.csv 为空: {smell_csv_file}")
                return f"{thread_prefix}: Smell.csv is empty."
                
            headers = [h.strip() for h in header_line.strip().split(',')]
            
            required_cols = ["Package Name", "Type Name"]
            if not all(col in headers for col in required_cols):
                msg = f"    {thread_prefix} 错误: CSV文件 {smell_csv_file} 缺少必要的列 (Package Name, Type Name). 实际列: {headers}"
                print(msg)
                return msg

            reader = csv.DictReader(csvfile, fieldnames=headers)
            
            for i, row in enumerate(reader):
                rows_processed +=1
                package_name = row.get("Package Name", "").strip()
                type_name_csv = row.get("Type Name", "").strip() # 可能包含 $ 用于内部类
                method_name = row.get("Method Name", "").strip()

                if not type_name_csv:
                    # results_log.append(f"      跳过第 {i+2} 行 (从1开始计数，包括头部)，因为 Type Name 为空: {row}")
                    continue
                
                # 标准化包名，处理 Designite 可能输出的 "<default>"
                if package_name.lower() == "<default>" or package_name == "-":
                    package_name = ""

                outer_type_name = type_name_csv.split('$')[0]
                lookup_key = (package_name, outer_type_name)
                
                found_java_file = java_files_idx.get(lookup_key)

                if found_java_file:
                    matches_found += 1
                    # 详细日志可以收集起来，而不是直接打印，以避免并发打印混乱
                    # results_log.append(f"      匹配成功: {package_name}::{type_name_csv}::{method_name} -> {found_java_file}")
                # else:
                    # results_log.append(f"      匹配失败: {package_name}::{type_name_csv} (查找键: {lookup_key}) 在索引中未找到。")
        
        summary_msg = f"    {thread_prefix} 完成: CSV行数={rows_processed}, 匹配数={matches_found}, 索引条目数={len(java_files_idx)}"
        print(summary_msg)
        # for log_entry in results_log: # 如果需要打印详细日志
        #     print(log_entry)
        return summary_msg

    except FileNotFoundError: # Should be caught by os.path.exists earlier
        msg = f"    {thread_prefix} 错误: Smell.csv 文件未找到于: {smell_csv_file}"
        print(msg)
        return msg
    except Exception as e:
        msg = f"    {thread_prefix} 处理 Smell.csv 文件时发生错误 {smell_csv_file}: {e}"
        print(msg)
        return msg # 返回错误信息

# --- 3. 主程序 ---
if __name__ == "__main__":
    start_time = time.time()

    designite_base_dir = "/model/data/Research/tools/designitejava/" # 请替换为实际路径
    source_projects_base_dir = "/model/data/Research/sourceProject/actionableCS/" # 请替换为实际路径

    if not os.path.isdir(designite_base_dir):
        print(f"错误：DesigniteJava 目录不存在: {designite_base_dir}")
        exit(1)
    if not os.path.isdir(source_projects_base_dir):
        print(f"错误：源项目目录不存在: {source_projects_base_dir}")
        exit(1)

    tasks_to_submit = []

    # 收集所有待处理的任务
    for project_name in os.listdir(designite_base_dir):
        project_designite_path = os.path.join(designite_base_dir, project_name)
        project_source_path = os.path.join(source_projects_base_dir, project_name)

        if not os.path.isdir(project_designite_path):
            continue

        print(f"准备项目: {project_name}")

        if not os.path.isdir(project_source_path):
            print(f"  警告: 项目 {project_name} 未找到对应的源项目目录: {project_source_path}")
            # 即使源项目目录不存在，也继续处理其版本，process_version_task会处理空索引
            
        for version_folder_name in os.listdir(project_designite_path):
            version_designite_full_path = os.path.join(project_designite_path, version_folder_name)
            if os.path.isdir(version_designite_full_path): # 确保版本是一个目录
                # (project_name, version_folder_name, project_designite_path, project_source_path)
                tasks_to_submit.append(
                    (project_name, version_folder_name, project_designite_path, project_source_path)
                )
            else:
                print(f"  跳过非目录条目: {version_designite_full_path}")
    
    print(f"\n共收集到 {len(tasks_to_submit)} 个版本任务待处理。\n")
    
    # 使用 ThreadPoolExecutor 并行处理任务
    # 根据CPU核心数调整 MAX_WORKERS，对于I/O密集型任务，可以设置比CPU核心数稍多一些
    # 对于主要是本地文件I/O，os.cpu_count() * 2 或 * 4 可能是一个好的起点
    MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4) # 限制最大线程数，例如32，并确保至少有几个线程
    
    if not tasks_to_submit:
        print("没有找到可处理的版本任务。")
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交任务
            future_to_task_info = {
                executor.submit(process_version_task, task_project, task_version, task_proj_designite_base, task_proj_source_base): 
                f"{task_project}/{task_version}" 
                for task_project, task_version, task_proj_designite_base, task_proj_source_base in tasks_to_submit
            }

            for future in future_to_task_info: # concurrent.futures.as_completed(future_to_task_info) # 如果想按完成顺序处理
                task_info = future_to_task_info[future]
                try:
                    result = future.result() # 等待任务完成并获取结果 (或异常)
                    # print(f"任务 {task_info} 的结果: {result}") # process_version_task 内部已打印总结
                except Exception as exc:
                    print(f"任务 {task_info} 执行时产生异常: {exc}")

    end_time = time.time()
    print(f"\n所有任务处理完成。总耗时: {end_time - start_time:.2f} 秒。")