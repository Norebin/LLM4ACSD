import numpy as np
from tqdm import tqdm

import logic.embeds as embeds
import pathlib
import code_utils as cu


def build_embeddings(data_path):
    # codebert_file_name = f"{pathlib.Path(data_path).stem}_codebert_pooler_output.npy"
    # embeds.build_codebert(data_path, codebert_file_name)

    # graphcodebert_pooler_file_name = f"{pathlib.Path(data_path).stem}_graphcodebert_pooler_output.npy"
    # embeds.build_graphcodebert(data_path, graphcodebert_pooler_file_name)

    graphcodebert_hidden_state_file_name = f"{pathlib.Path(data_path).stem}_graphcodebert_hidden_state.npy"
    embeds.build_graphcodebert(data_path, graphcodebert_hidden_state_file_name)

    # bert_file_name = f"{pathlib.Path(data_path).stem}_bert_nli_mean_token_pooler_output.npy"
    # embeds.build_bert_nli_mean(data_path, bert_file_name)


def separate_train_test_embeddings(embeds, train_path, test_path):
    train_df = cu.open_smell_file(train_path)

    test_df = cu.open_smell_file(test_path)
    for embed in tqdm(embeds):
        embed_path = pathlib.Path(embed)
        embed_content = np.load(embed, "r")

        train_file_name = f"{str(embed_path.parent)}/{pathlib.Path(embed).stem}_train.npy"
        test_file_name = f"{str(embed_path.parent)}/{pathlib.Path(embed).stem}_test.npy"

        with open(train_file_name, "wb") as train_file:
            np.save(train_file, embed_content[train_df["index"]])

        with open(test_file_name, "wb") as test_file:
            np.save(test_file, embed_content[test_df["index"]])


def separate_train_test_embeddings1(embeds, train_ratio=0.6, random_seed=42):
    np.random.seed(random_seed)
    for embed in tqdm(embeds):
        embed_path = pathlib.Path(embed)
        embed_content = np.load(embed, "r")
        
        # 获取样本总数
        total_samples = embed_content.shape[0]
        # 生成随机索引
        indices = np.random.permutation(total_samples)
        # 计算训练集大小
        train_size = int(total_samples * train_ratio)
        # 划分训练集和测试集索引
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        # 保存文件路径
        train_file_name = f"{str(embed_path.parent)}/{pathlib.Path(embed).stem}_train.npy"
        test_file_name = f"{str(embed_path.parent)}/{pathlib.Path(embed).stem}_test.npy"
        
        # 保存训练集
        with open(train_file_name, "wb") as train_file:
            np.save(train_file, embed_content[train_indices])
        
        # 保存测试集
        with open(test_file_name, "wb") as test_file:
            np.save(test_file, embed_content[test_indices])
        
        print(f"已保存: {train_file_name} (样本数: {len(train_indices)})")
        print(f"已保存: {test_file_name} (样本数: {len(test_indices)})")


if __name__ == '__main__':
    # build_embeddings('/model/lxj/actionableSmell')
    # 使用原函数
    # separate_train_test_embeddings("""/home/user/PycharmProjects/Model_Scratch/data/9000_smells_bert_nli_mean_token_pooler_output.npy
    # /home/user/PycharmProjects/Model_Scratch/data/9000_smells_codebert_pooler_output.npy
    # /home/user/PycharmProjects/Model_Scratch/data/9000_smells_graphcodebert_hidden_state.npy
    # /home/user/PycharmProjects/Model_Scratch/data/9000_smells_graphcodebert_pooler_output.npy""".split("\n"),
    #                                "data/raw/9000_smells_train.json",
    #                                "data/raw/9000_smells_test.json"
    #                                )
    
    # 使用新函数，不需要json文件
    separate_train_test_embeddings1("""/model/lxj/Baseline/TripletLoss/data/actionableSmell_bert_nli_mean_token_pooler_output.npy
/model/lxj/Baseline/TripletLoss/data/actionableSmell_codebert_pooler_output.npy
/model/lxj/Baseline/TripletLoss/data/actionableSmell_graphcodebert_hidden_state.npy
/model/lxj/Baseline/TripletLoss/data/actionableSmell_graphcodebert_pooler_output.npy""".split("\n"))