import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_dataset_with_llm_embeddings(
    json_path: str,
    output_path: str,
    model_name: str = "/model/LiangXJ/Model/Qwen/Qwen2.5-Coder-14B-Instruct",
    layers_to_extract: List[int] = [12, 24, 47],  # 浅层、中层、深层
    max_length: int = 4096,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    """
    从 JSON 文件中读取代码和标签，使用 LLM 提取中间层表示（均值池化和最后一个 token），并保存为数据集。
    
    参数：
        json_path (str): 输入 JSON 文件路径，格式为 [{"input": code, "output": label}, ...]
        output_path (str): 输出文件路径，保存为 .npz 格式
        model_name (str): LLM 模型名称，默认为 Qwen2-7B
        layers_to_extract (List[int]): 要提取的层索引（如 [12, 24, 47] 表示第 12、24、47 层）
        max_length (int): 输入序列最大长度
        batch_size (int): 批量处理大小
        device (str): 计算设备（cuda 或 cpu）
    
    输出：
        保存 .npz 文件，包含：
        - mean_pooled_{layer_idx}: 每层的均值池化表示
        - last_token_{layer_idx}: 每层的最后一个 token 表示
        - labels: 标签数组
    """
    # 1. 加载模型和分词器
    logger.info(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",  # "auto" 需要 accelerate 库
        trust_remote_code=True
    )
    model.eval()
    
    # 2. 读取 JSON 数据
    logger.info(f"Reading JSON file from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    codes = [item["input"] for item in data]
    labels = [item["output"] for item in data]
    num_samples = len(codes)
    logger.info(f"Loaded {num_samples} samples.")
    
    # 3. 初始化存储
    hidden_size = model.config.hidden_size  # 5120 for Qwen2-7B
    embeddings = {
        f"mean_pooled_{layer_idx}": np.zeros((num_samples, hidden_size), dtype=np.float32)
        for layer_idx in layers_to_extract
    }
    embeddings.update({
        f"last_token_{layer_idx}": np.zeros((num_samples, hidden_size), dtype=np.float32)
        for layer_idx in layers_to_extract
    })
    labels_array = np.array(labels)
    
    # 4. 批量处理数据
    for i in range(0, num_samples, batch_size):
        batch_codes = codes[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}/{(num_samples + batch_size - 1) // batch_size}")
        
        # 分词
        inputs = tokenizer(
            batch_codes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        # 前向传播
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # 元组，包含所有层的隐藏状态
        
        # 处理每层的表示
        for layer_idx in layers_to_extract:
            layer_hidden = hidden_states[layer_idx]  # (batch_size, sequence_length, hidden_size)
            
            # 均值池化
            attention_mask = inputs.attention_mask
            masked_hidden = layer_hidden * attention_mask.unsqueeze(-1)  # 屏蔽填充 token
            sum_hidden = masked_hidden.sum(dim=1)  # 按序列维度求和
            valid_lengths = attention_mask.sum(dim=1).unsqueeze(-1)  # 有效 token 数
            mean_pooled = sum_hidden / valid_lengths.clamp(min=1)  # 避免除以 0
            
            # 最后一个 token 的表示
            last_token_idx = attention_mask.sum(dim=1) - 1  # 最后一个有效 token 的索引
            last_token_hidden = layer_hidden[
                torch.arange(layer_hidden.size(0)), last_token_idx, :
            ]
            
            # 保存到数组
            batch_end = min(i + batch_size, num_samples)
            embeddings[f"mean_pooled_{layer_idx}"][i:batch_end] = mean_pooled.cpu().numpy()
            embeddings[f"last_token_{layer_idx}"][i:batch_end] = last_token_hidden.cpu().numpy()
    
    # 5. 保存数据集
    logger.info(f"Saving dataset to {output_path}...")
    np.savez(output_path, **embeddings, labels=labels_array)
    logger.info("Dataset saved successfully.")
    
    # 清理显存
    del model
    torch.cuda.empty_cache() if device == "cuda" else None

# 示例用法
if __name__ == "__main__":   
    # 运行函数
    build_dataset_with_llm_embeddings(
        csv_path="/model/lxj/actionableSmell/stirling_Split.json",
        output_path="code_embeddings.npz",
        layers_to_extract=[12, 24, 47],  # 浅层、中层、深层
        max_length=4096,
        batch_size=1
    )