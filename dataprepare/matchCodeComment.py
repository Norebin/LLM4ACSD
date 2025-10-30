import os
import pandas as pd
from jpype import JClass
import jpype
import jpype.imports
from jpype.types import *
import re
from tqdm import tqdm


def start_jvm():
    """启动 JVM 并加载 JavaParser"""
    import jpype
    import jpype.imports
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=["/model/lxj/cfexplainer/code smell/javaparser-core-3.26.2.jar"])
    ParserConfiguration_class = JClass("com.github.javaparser.ParserConfiguration")
    LanguageLevel_class = JClass("com.github.javaparser.ParserConfiguration$LanguageLevel")
    JavaParser_class = JClass("com.github.javaparser.JavaParser")
    ClassOrInterfaceDeclaration = JClass("com.github.javaparser.ast.body.ClassOrInterfaceDeclaration")
    ConstructorDeclaration = JClass("com.github.javaparser.ast.body.ConstructorDeclaration")
    MethodDeclaration = JClass("com.github.javaparser.ast.body.MethodDeclaration")
    # 创建 ParserConfiguration 实例
    config = ParserConfiguration_class()
    try:
        selected_level = getattr(LanguageLevel_class, 'BLEEDING_EDGE')
    except AttributeError:
        print(f"警告: 无效的 Java 版本. 回退到 BLEEDING_EDGE.")
        selected_level = LanguageLevel_class.BLEEDING_EDGE # 或者一个你认为安全的默认值
    config.setLanguageLevel(selected_level)
    configured_parser = JavaParser_class(config)
    return (configured_parser,
            ClassOrInterfaceDeclaration,
            ConstructorDeclaration,
            MethodDeclaration)


import subprocess

def find_java_file(project_path, package_name, type_name):
    """使用 Ubuntu 命令查找匹配的 .java 文件"""
    file = f"{type_name}.java"
    command = f"find {project_path} -name {file}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()
    matching_files = output.decode("utf-8").split("\n")[:-1]  # 去掉最后一个空元素
    if not matching_files:
        return None
    return matching_files


def extract_code_and_comment(java_file_path, method_name, type_name):
    """提取类、方法或构造函数的代码和注释"""
    StaticJavaParser, ClassOrInterfaceDeclaration, ConstructorDeclaration, MethodDeclaration= start_jvm()
    try:
        with open(java_file_path, "r", encoding="utf-8") as file:
            java_code = file.read()
        # 解析 Java 文件
        compilation_unit = StaticJavaParser.parse(java_code).getResult().get() 
        if method_name:  # 提取方法或构造函数
            if method_name == type_name:  # 认为是构造函数
                longest_constructor = None
                max_length = -1
                for constructor in compilation_unit.findAll(ConstructorDeclaration):
                    if constructor.getNameAsString() == method_name:
                        constructor_code = str(constructor)
                        if len(constructor_code) > max_length:
                            max_length = len(constructor_code)
                            longest_constructor = constructor
                if longest_constructor:
                    raw_code = str(longest_constructor)
                    # 移除所有注释
                    clean_code = re.sub(r'(?s)/\*.*?\*/|//.*?$|/\*\*.*?\*/', '', raw_code,
                                        flags=re.MULTILINE | re.DOTALL)
                    clean_code = re.sub(r'\n\s*\n', '\n', clean_code).strip()
                    comment_opt = longest_constructor.getComment()
                    comment = comment_opt.get().getContent() if comment_opt.isPresent() and comment_opt.get().isJavadocComment() else ""
                    return clean_code, comment.strip()
            else:  # 普通方法
                longest_method = None
                max_length = -1
                for method in compilation_unit.findAll(MethodDeclaration):
                    if method.getNameAsString() == method_name:
                        method_code = str(method)
                        if len(method_code) > max_length:
                            max_length = len(method_code)
                            longest_method = method
                if longest_method:
                    raw_code = str(longest_method)
                    # 移除所有注释
                    clean_code = re.sub(r'(?s)/\*.*?\*/|//.*?$|/\*\*.*?\*/', '', raw_code,
                                        flags=re.MULTILINE | re.DOTALL)
                    clean_code = re.sub(r'\n\s*\n', '\n', clean_code).strip()
                    comment_opt = longest_method.getComment()
                    comment = comment_opt.get().getContent() if comment_opt.isPresent() and comment_opt.get().isJavadocComment() else ""
                    return clean_code, comment.strip()
        else:  # 提取类
            for class_decl in compilation_unit.findAll(ClassOrInterfaceDeclaration):
                if class_decl.getNameAsString() == type_name:
                    raw_code = str(class_decl)
                    # 移除所有注释
                    clean_code = re.sub(r'(?s)/\*.*?\*/|//.*?$|/\*\*.*?\*/', '', raw_code,
                                        flags=re.MULTILINE | re.DOTALL)
                    clean_code = re.sub(r'\n\s*\n', '\n', clean_code).strip()
                    comment_opt = class_decl.getComment()
                    comment = comment_opt.get().getContent() if comment_opt.isPresent() and comment_opt.get().isJavadocComment() else ""
                    return clean_code, comment.strip()
        return "not find code", "not find code"
    except Exception as e:
        print(f"Error processing {java_file_path}: {str(e)}")
        return "not find code", "not find code"

def process_csv_files(base_dir, output_dir):
    """处理所有 CSV 文件"""
    # for project_folder in os.listdir(base_dir):
        # project_path = os.path.join(base_dir, project_folder)
        # if not os.path.isdir(project_path):
        #     continue
    for csv_file in os.listdir(base_dir):
        if (not csv_file.endswith(".csv")) or os.path.exists(os.path.join(output_dir, csv_file)):
            continue
        # if csv_file != "4.0.0_actionable.csv":
        #     continue
        csv_path = os.path.join(base_dir, csv_file)
        df = pd.read_csv(csv_path)
        # 添加新列
        df["code"] = ""
        df["comment"] = ""
        # 处理每一行
        for index, row in df.iterrows():
            project_name = row["Project Name"]
            full_package_name = row["Package Name"]
            # 如果没有找到大写字母，使用原始的类名
            package_name = full_package_name
            type_name = row["Type Name"]
            type_name = row["Type Name"].split('$')[0] if '$' in row["Type Name"] else row["Type Name"]
            method_name = row["Method Name"] if pd.notna(row["Method Name"]) else ""
            # 拼接项目路径
            full_project_path = os.path.join("/model/lxj/actionableCS/cassandra", project_name)
            java_file_path = find_java_file(full_project_path, package_name, type_name)
            if java_file_path:
                for x in java_file_path:
                    code, comment = extract_code_and_comment(java_file_path, method_name, type_name)
                    if code == 'not find code':
                        print(f'没找到这段代码{package_name}_{type_name}_{method_name}')
                        continue
                df.at[index, "code"] = code
                df.at[index, "comment"] = comment
            else:
                print(f'没找到这个java文件{package_name}_{type_name}_{method_name}')
                df.at[index, "code"] = "not find file"
                df.at[index, "comment"] = "not find file"
        # 覆盖原 CSV 文件
        df.to_csv(os.path.join(output_dir, csv_file), index=False)
        print(f"Processed {csv_path}")

def process_action(input, output):
    not_find_file = 0
    not_find_code = 0 
    df = pd.read_csv(input)
    df = df.fillna("")
    df["code"] = ""
    df["comment"] = ""
    project_name = df['Project'][0]
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        version = row['Version']
        full_package_name = row["Package Name"]
        # 查找最后一个点号之前的大写字母位置
        parts = full_package_name.split('.')
        if any(c.isupper() for c in full_package_name):
            for i in range(0, len(parts)-1, 1):
                if any(c.isupper() for c in parts[i]):
                    package_name = '.'.join(parts[:i])
                    type_name = parts[i]
                    break
        else:
            # 如果没有找到大写字母，使用原始的类名
            package_name = full_package_name
            type_name = row["Type Name"].split('$')[0] if '$' in row["Type Name"] else row["Type Name"]
        method_name = row["Method Name"] if pd.notna(row["Method Name"]) else ""
        source_code_path = os.path.join('/model/data/Research/sourceProject/actionableCS', project_name, project_name+'-'+version)
        java_file_path = []
        if row["File Path"].startswith("File Not Found") or row["File Path"].startswith("Ambiguous"):
            java_file_path.append(find_java_file(source_code_path, package_name, type_name))
        else:
            java_file_path.append(os.path.join('/model/data/Research/sourceProject/actionableCS',row["Project"],row["Project"]+'-'+row["Version"],row["File Path"]))
        if java_file_path:
            for x in java_file_path:
                code, comment = extract_code_and_comment(x, method_name, type_name)
                df.at[index, "code"] = code
                df.at[index, "comment"] = comment
                if code != 'not find code':
                    break
                else:
                    print(f'没找到这段代码{package_name}_{type_name}_{method_name}')
                    not_find_code += 1
        else:
            print(f'没找到这个java文件{package_name}_{type_name}_{method_name}')
            not_find_file += 1
            df.at[index, "code"] = "not find file"
            df.at[index, "comment"] = "not find file"
    # 覆盖原 CSV 文件
    df.to_csv(output, index=False)
    print(f"Processed {input}, 有{not_find_file}个文件没有找到,有{not_find_code}个代码没有找到")
def test_find_code_function(java_file_path, method_name, type_name):
    code, comment = extract_code_and_comment(java_file_path, method_name, type_name)
    print(code)
    print(comment)
if __name__ == "__main__":
    # 基础路径
    projects = [
    name for name in os.listdir('/model/data/Research/temp') 
    if os.path.isdir(os.path.join('/model/data/Research/temp', name))]
    for project in projects:
        if os.path.exists('/model/data/Research/actionable/'+str(project)+'_action_code.csv'):
            continue
        base_dir = '/model/data/Research/temp_merged_output/' + project + '_action.csv'
        output_dir = "/model/data/Research/actionable/" + project + "_action_code.csv"
        process_action(base_dir, output_dir)
    # test_find_code_function('/model/data/Research/sourceProject/actionableCS/stirling/stirling-0.42.0/src/main/java/stirling/software/SPDF/UI/impl/LoadingWindow.java','LoadingWindow','LoadingWindow')