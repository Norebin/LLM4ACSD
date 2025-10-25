import javalang
import re
from itertools import chain
import jpype
import jpype.imports
from jpype.types import *

code = '''
class StringUtil {
    //返回的第一个内容为字母n的个数，第二个内容为字母o的个数
    public static int [] count(String str) {
        int countData [] = new int [2] ;
        char [] data = str.toCharArray() ;    //将字符串变为字符数组
        for (int x = 0 ; x < data.length ; x ++) {
           if (data[x] == 'n' || data[x] == 'N') {
              countData[0] ++ ;
           }
           if (data[x] == 'o' || data[x] == 'O') {
              countData[1] ++ ;
           }
         }
         return countData ;
    }
}
'''
JAR_PATH = "/model/lxj/cfexplainer/code smell/javaparser-core-3.26.2.jar"
# jpype.startJVM(classpath=[JAR_PATH])
code_lines = code.splitlines()
drop_comment = re.sub(r'(?s)/\*.*?\*/|//.*?$|/\*\*.*?\*/', '', code, flags=re.MULTILINE | re.DOTALL)
clean_code = re.sub(r'\n\s*\n', '\n', drop_comment).strip()
def extract_method_from_java_file(java_file_path, method_name, param_types):
    # 启动 JVM 并加载 Javaparser
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=["/model/lxj/cfexplainer/code smell/javaparser-core-3.26.2.jar"])
    try:
        from com.github.javaparser import StaticJavaParser
        from com.github.javaparser.ast.body import MethodDeclaration
        # 读取并解析 Java 文件
        with open(java_file_path, "r", encoding="utf-8") as file:
            java_code = file.read()
        java_code = re.sub(r'(?s)/\*.*?\*/|//.*?$|/\*\*.*?\*/', '', java_code, flags=re.MULTILINE | re.DOTALL)
        java_code = re.sub(r'\n\s*\n', '\n', java_code).strip()
        # 解析 Java 文件为 CompilationUnit
        compilation_unit = StaticJavaParser.parse(java_code)

        # 遍历所有方法，找到目标方法
        for method in compilation_unit.findAll(MethodDeclaration):
            if method.getNameAsString() == method_name:
                method_param_types = [
                    param.getType().asString() for param in method.getParameters()
                ]
                if len(method_param_types) == len(param_types):
                    return str(method)
        return  " "
    except:
        print(f'no {method_name} with {param_types} in {java_file_path}')
        return " "

def extract_method_code(file_path, target_method_name):
    """
    使用 javalang 从 Java 文件中提取指定方法的代码。
    """
    # with open(file_path, "r") as f:
    #     code = f.read()

    tree = javalang.parse.parse(code)
    method_code = None
    node_types = (javalang.tree.ClassDeclaration, javalang.tree.InterfaceDeclaration)
    for _, node in chain(
        tree.filter(javalang.tree.ClassDeclaration),
        tree.filter(javalang.tree.InterfaceDeclaration)
    ):
        # 遍历类中的所有方法
        for method in node.methods:
            parameters = [(param.type.name, param.name) for param in method.parameters]
            start_line = method.position.line - 1
            if method.body:
                end_line = method.body[-1].position[0]
            else:
                end_line = start_line + 1
            method_code = code_lines[start_line:end_line + 1]
            break
    for _, node in tree:
        if isinstance(node, javalang.tree.MethodDeclaration):
            if node.name == target_method_name:
                # 提取方法代码
                method_code = code[node.position[0] - 1:node.position[0] - 1 + node.body.position[1]]
                break

    if method_code:
        return method_code.strip()
    else:
        raise ValueError(f"Method '{target_method_name}' not found in the file.")

if __name__ == "__main__":
    java_file_path = "/model/LiangXJ/CodeSmellProject/pig/src/org/apache/pig/PigServer.java"  # 替换为实际 Java 文件路径
    method_name = "printHistory"  # 替换为目标方法名称
    extracted_method = extract_method_from_java_file(java_file_path, method_name)
    print(extracted_method)
    # extract_method_code('', 'serializedSize')
