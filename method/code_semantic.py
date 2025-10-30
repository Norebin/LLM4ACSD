import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
import os
from tqdm import tqdm
import json  
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 配置参数 LLaMA-Factory/src/llamafactory/extras/constants.py
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/bdww/anaconda3/envs/lxj_LLM/lib/
MODEL_NAME = {"origin":"/model/LiangXJ/Model/Qwen/Qwen2.5-Coder-14B-Instruct",
              "dora":"/model/LiangXJ/Model/PEFT/Qwen2.5-Coder-14B-Instruct/dora_all_merge",
              "lora":"/model/LiangXJ/Model/PEFT/Qwen2.5-Coder-14B-Instruct/lora_all_merge"}
OUTPUT_DIR = "semantic_features"
BATCH_SIZE = 1  # 可以根据您的GPU内存调整
LAYERS_TO_EXTRACT = {
    "shallow": [10],       # 浅层9,10,11         8,9,10          9,10,11         10,11,12
    "middle": [20],        # 中层19,20,21      18,19,20          19,20,21       22,23,24
    "deep": [39]  # 深层 phi4 38,39,40     deepseek 29,30,31   codellama 37,38,39  qwen45,46,47
}

def read_csv_data(csv_path, flag):
    """读取CSV文件并筛选测试集"""
    print(f"读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    # 检查所需列是否存在
    required_cols = ['unique_id', 'dataset_split', 'code', 'actionable']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV文件中缺少'{col}'列")
    # 筛选测试集数据
    test_df = df[df['dataset_split'] == flag].copy()
    print(f"总数据条目: {len(df)}, 测试集条目: {len(test_df)}")
    return test_df, df

def setup_model_and_tokenizer(model_name):
    """设置模型和分词器"""
    print(f"加载模型和分词器: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token='[PAD]'
    # 设置output_hidden_states=True以获取所有层的隐状态
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict_in_generate=True,
        output_hidden_states=True,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
     # attn_implementation="sdpa"
    # 如果有GPU则使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # 设置为评估模式
    return model, tokenizer, device

def extract_features_batch(codes, model, tokenizer, device):
    """批量提取代码特征"""
    # 对代码进行分词，截断过长的序列
    inputs = tokenizer(codes, padding=False, truncation=True, max_length=8192, return_tensors="pt").to(device)
    # 获取隐状态，不需要计算梯度
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # 获取所有层的隐状态 (layers+1, batch, seq_len, hidden_dim)
    # hidden_states[0]是词嵌入，hidden_states[1]开始是第1层transformer的输出
    hidden_states = outputs.hidden_states
    features = {}
    # 1. 提取不同层次的平均池化特征
    for layer_name, layer_indices in LAYERS_TO_EXTRACT.items():
        # 选择指定层的隐状态并在这些层上进行平均
        layer_features = torch.stack([hidden_states[i] for i in layer_indices], dim=0).mean(dim=0)
        # 为每个样本创建平均池化特征 (忽略padding token)
        pooled_features = []
        for i in range(layer_features.size(0)):
            # 获取该样本的实际长度（非padding部分）
            seq_len = torch.sum(inputs['attention_mask'][i])
            # 只对非padding的token进行平均池化
            sample_features = layer_features[i, :seq_len].mean(dim=0)
            pooled_features.append(sample_features.cpu().float().numpy())
        features[f"{layer_name}_avg_pooling"] = np.array(pooled_features)
    # 2. 提取最后一个token的向量表示
    last_layer = hidden_states[-1]  # 最后一层的隐状态
    last_token_features = []  
    for i in range(last_layer.size(0)):
        # 获取最后一个非padding token的位置
        seq_len = torch.sum(inputs['attention_mask'][i]) - 1  # -1是因为索引从0开始
        # 提取最后一个token的特征
        last_token_feature = last_layer[i, seq_len]
        last_token_features.append(last_token_feature.cpu().float().numpy())
    features["last_token"] = np.array(last_token_features)
    return features

def process_and_extract_features(test_df, model, tokenizer, device, project):
    """处理所有测试数据并提取特征"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 准备保存数据的结构
    feature_types = ["shallow_avg_pooling", "middle_avg_pooling", "deep_avg_pooling", "last_token"]
    # 获取隐藏层的维度大小
    sample_code = test_df['code'].iloc[0]
    sample_inputs = tokenizer(sample_code, return_tensors="pt").to(device)
    with torch.no_grad():
        sample_outputs = model(**sample_inputs, output_hidden_states=True)
    hidden_dim = sample_outputs.hidden_states[0].size(-1)
    
    # 获取数据集大小
    n_samples = len(test_df)
    
    # 初始化存储特征的字典
    features_dict = {feature_type: [] for feature_type in feature_types}
    unique_ids = []
    
    # 按批次处理数据
    for i in tqdm(range(0, len(test_df), BATCH_SIZE), desc="处理代码批次"):
        batch_df = test_df.iloc[i:i+BATCH_SIZE]
        batch_codes = batch_df['code'].tolist()
        batch_ids = batch_df['unique_id'].tolist()
        
        # 提取特征
        batch_features = extract_features_batch(batch_codes, model, tokenizer, device)
        
        # 收集批次的特征
        for feature_type in feature_types:
            features_dict[feature_type].append(torch.tensor(batch_features[feature_type], dtype=torch.float32))
        # 收集ID
        unique_ids.extend(batch_ids)
    
    # 合并所有批次的特征
    for feature_type in feature_types:
        features_dict[feature_type] = torch.cat(features_dict[feature_type], dim=0)
    # 获取标签
    labels = torch.tensor(test_df['actionable'].values, dtype=torch.int8)
    
    # 保存所有特征和标签为单个PyTorch文件
    torch_path = project
    print(f"将特征保存为PyTorch格式: {torch_path}")
    
    torch.save({
        'metadata': {
            'model_name': MODEL_NAME,
            'feature_types': feature_types,
            'unique_ids': unique_ids,
            'layers_extracted': LAYERS_TO_EXTRACT
        },
        'features': features_dict,
        'labels': labels,
    }, torch_path)
    
    print("特征提取完成!")
    return {
        "feature_types": feature_types,
        "n_samples": n_samples,
        "hidden_dim": hidden_dim,
        "torch_path": torch_path
    }

def main(csv_path):
    """主函数"""
    # 1. 读取数据
    test_df, full_df = read_csv_data(csv_path)
    # 3. 处理并提取特征
    results = process_and_extract_features(test_df, model, tokenizer, device)
    # 4. 返回结果摘要
    return {
        "总数据量": len(full_df),
        "测试集数量": len(test_df),
        "提取特征数量": results["n_samples"],
        "特征种类": results["feature_types"],
        "隐状态维度": results["hidden_dim"],
        "PyTorch输出文件": results["torch_path"]
    }

if __name__ == "__main__":
    for key,model_name in MODEL_NAME.items():
        model, tokenizer, device = setup_model_and_tokenizer(model_name)
        # 指定CSV文件路径
        projects = ['dbeaver','stirling','jsoup','cayenne','pig','mockito','dubbo','cassandra','jedis','easyexcel']
        try:
            for project in projects:
                for flag in ['train', 'test']:
                    pt_path = os.path.join(OUTPUT_DIR, f"{key}_qwen_code_semantic_{project}_{flag}.pt")
                    if os.path.exists(pt_path):
                        continue
                    csv_path = "/model/lxj/actionableSmell/"+project+"_Split.csv"  # 请替换为您的CSV文件路径
                    test_df, full_df = read_csv_data(csv_path, flag)
                    results = process_and_extract_features(test_df, model, tokenizer, device, pt_path)
                    print("\n结果摘要:")
                    for k, v in results.items():
                        print(f"{k}: {v}")
        finally:
            # 显式释放GPU内存
            del model
            torch.cuda.empty_cache()
    # 加载特征和标签
    def load_pt():
        data = torch.load('semantic_features/code_semantic_features.pt')
        # 获取元数据
        model_name = data['metadata']['model_name']
        unique_ids = data['metadata']['unique_ids']
        # 获取特征
        features = data['features']  # 字典，包含所有特征类型
        shallow_features = features['shallow_avg_pooling']  # 浅层特征，形状: [n_samples, hidden_dim]
        middle_features = features['middle_avg_pooling']    # 中层特征
        deep_features = features['deep_avg_pooling']        # 深层特征
        last_token_features = features['last_token']        # 最后token特征
        # 获取标签
        labels = data['labels']  # 形状: [n_samples]