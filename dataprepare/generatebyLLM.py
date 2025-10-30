import os
import pandas as pd
from typing import Optional


# 假设使用的大模型接口（这里以简单的占位函数表示）
def generate_code_intent(code: str, model_name: str = "grok") -> Optional[str]:
    """
    调用大模型生成代码意图描述
    :param code: 输入的代码字符串
    :param model_name: 指定模型名称，例如 "grok", "qwen", "llama"
    :return: 生成的意图描述，或 None（如果失败）
    """
    try:
        if model_name == "grok":
            # 假设使用 xAI 的 Grok 模型（这里用伪代码表示）
            from xai_api import GrokAPI  # 虚构的 API 调用
            grok = GrokAPI()
            response = grok.generate(f"Describe the intent of this code accurately and concisely: {code}")
            return response
        elif model_name == "qwen":
            # 假设使用 Qwen 模型（可以通过 Hugging Face 或 API 调用）
            from transformers import pipeline
            generator = pipeline("text-generation", model="qwen-7b")  # 示例模型
            response = generator(f"Describe the intent of this code accurately and concisely: {code}", max_length=100)
            return response[0]["generated_text"]
        elif model_name == "llama":
            # 假设使用 LLaMA 模型（本地或 API）
            from llama_api import LlamaAPI  # 虚构的 API 调用
            llama = LlamaAPI()
            response = llama.generate(f"Describe the intent of this code accurately and concisely: {code}")
            return response
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    except Exception as e:
        print(f"Error generating intent with {model_name}: {e}")
        return None


def process_csv_files(directory: str, model_name: str = "grok"):
    """
    处理目录下的所有 CSV 文件，生成代码意图描述
    :param directory: CSV 文件所在目录
    :param model_name: 指定使用的大模型
    """
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {file_path}")
            # 读取 CSV 文件
            try:
                df = pd.read_csv(file_path)
                # 检查是否包含 "comment" 和 "code" 列
                if "comment" not in df.columns or "code" not in df.columns:
                    print(f"Skipping {filename}: Missing 'comment' or 'code' column")
                    continue
                # 处理每一行
                for index, row in df.iterrows():
                    if pd.isna(row["comment"]) or row["comment"].strip() == "":
                        code = row["code"]
                        print(f"Found empty comment at row {index}, code: {code}")

                        # 调用大模型生成意图描述
                        intent = generate_code_intent(code, model_name)
                        if intent:
                            print(f"Generated intent: {intent}")
                            # 将生成的意图写回 DataFrame
                            df.at[index, "comment"] = intent
                        else:
                            print(f"Failed to generate intent for code: {code}")
                # 保存修改后的 CSV 文件
                # output_file = os.path.join(directory, f"updated_{filename}")
                df.to_csv(file_path, index=False)
                print(f"Saved updated file: {output_file}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    # 示例用法
    csv_directory = "/model/lxj/actionableCS/designitejava/cassandra"  # 替换为你的 CSV 文件目录
    selected_model = "grok"  # 可替换为 "qwen", "llama" 等
    process_csv_files(csv_directory, model_name=selected_model)