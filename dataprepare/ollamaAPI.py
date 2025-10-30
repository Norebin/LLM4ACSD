import requests
import json
import pandas as pd
import sys
from typing import Optional

# Ollama API 调用模块
class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        """初始化 Ollama 客户端，默认地址为本地默认端口"""
        self.base_url = base_url
    def generate_completion(self, prompt: str, model: str = "deepseek-r1:70b", stream: bool = False) -> Optional[str]:
        """调用 Ollama API 生成文本补全"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # 如果请求失败，抛出异常
            result = response.json()
            return result["response"]
        except requests.RequestException as e:
            print(f"API 调用失败: {e}", file=sys.stderr)
            return None


# 主程序：处理 CSV 文件并生成代码注释
def generate_code_comments(input_csv: str, output_csv: str, model: str = "deepseek-r1:70b"):
    """读取 CSV 文件，生成代码注释并保存结果"""
    # 初始化 Ollama 客户端
    ollama = OllamaClient()
    # 读取 CSV 文件
    try:
        df = pd.read_csv(input_csv)
        if "code" not in df.columns:
            raise ValueError("CSV 文件中缺少 'code' 列")
    except Exception as e:
        print(f"读取 CSV 文件失败: {e}", file=sys.stderr)
        return
    # 用于存储生成的注释
    comments = []
    # 遍历每一行代码，生成注释
    for index, row in df.iterrows():
        code = row["code"]
        # 构造 prompt，要求模型生成注释
        prompt = f"Please generate concise comments for the following code, if it is a method, it is a method comment, if it is a class, it is a class comment:\n\n{code}"
        comment = ollama.generate_completion(prompt, model=model)

        if comment:
            comments.append(comment)
            print(f"已为第 {index + 1} 行代码生成注释: {comment}")
        else:
            comments.append("生成注释失败")
            print(f"第 {index + 1} 行代码生成注释失败")

    # 将结果添加到 DataFrame
    df["comment"] = comments

    # 保存到新的 CSV 文件
    try:
        df.to_csv(output_csv, index=False)
        print(f"结果已保存到 {output_csv}")
    except Exception as e:
        print(f"保存文件失败: {e}", file=sys.stderr)


# 示例使用
if __name__ == "__main__":
    # 示例 CSV 文件路径
    input_file = "/model/data/Research/actionableCS/cassandra/0.3.0-final_actionable.csv"
    output_file = "/model/data/Research/test.csv"

    # 假设 input_code.csv 的内容如下：
    # code
    # "def add(a, b): return a + b"
    # "for i in range(10): print(i)"

    # 运行程序
    generate_code_comments(input_file, output_file, model="deepseek-r1:70b")