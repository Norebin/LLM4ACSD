import os
import pandas as pd
import numpy as np
import networkx as nx
import subprocess
import javalang
from itertools import chain
from typing import List, Optional
import os.path as osp
from torch_geometric.data import Dataset, Data
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
from tqdm import tqdm
import re
from javaparse import extract_method_from_java_file
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SmellDataset(Dataset):
    def __init__(self, xlsx_file=None,
                 graph_dir='/model/LiangXJ/graph2metric',
                 code_dir='/model/LiangXJ/CodeSmellProject',
                 transform=None, pre_transform=None):
        """
        初始化数据集
        Args:
            xlsx_file: 包含class-name, smell, label的Excel文件路径
            graph_dir: 存放graphml文件的目录
            code_dir: 源代码所在目录
        """
        self.xlsx_file = xlsx_file
        self.graph_dir = graph_dir
        self.code_dir = code_dir

        # 初始化code-llama模型
        if xlsx_file != None:
            self.data_df = pd.read_excel(xlsx_file)
            self.tokenizer = AutoTokenizer.from_pretrained("/model/LiangXJ/Model/CodeLlama")
            self.model = AutoModelForCausalLM.from_pretrained("/model/LiangXJ/Model/CodeLlama", device_map="auto")

        super(SmellDataset, self).__init__(transform, pre_transform)
        self.data_list = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self): #处理后的文件存放路径
        return '/model/LiangXJ/processed_graph_data'

    @property #处理后的数据保存名
    def processed_file_names(self):
        return 'data.pt'

    def len(self):
        return len(self.data_list)

    def get(self, idx: int):
        return self.data_list[idx]

    def itempath(java_file):
        return os.path.join('/model/LiangXJ/CodeSmellProject', java_file)
        # 获取当前样本信息
        row = self.data_df.iloc[idx]
        class_name = row['class-name']
        label = row['label']

        # 处理图数据
        data = self._process_graph(class_name, label)

        # 保存处理后的数据
        torch.save(data, processed_path)
        return data

    def process(self):
        """处理单个图数据"""
        data_list= []
        project_mapping = {
            'mahout': 'mahout',
            'jena': 'jena',
            'jackrabbit': 'jackrabbit',
            'cayenne': 'cayenne',
            'cdt': 'cdt',
            'lucene': 'solr',
            'solr': 'solr',
            'snowball': 'solr',
            'pig': 'pig',
            'tools': 'pig',
            'cassandra': 'cassandra',
            'cxf': 'cxf'
        }
        for index, row in tqdm(self.data_df.iterrows(), total=len(self.data_df), desc="正在处理表中数据"):
            label_mapping = {'NON-SEVERE': 0, 'MEDIUM': 1, 'SEVERE': 2}
            label = label_mapping.get(row['label'])
            class_name = row['class-name']
            smell_type = row['smell']
            if len(class_name.split('.')) > 2:
                project_name = project_mapping[class_name.split('.')[2]]
            else:
                project_name = 'jena'
            graph_path = os.path.join(self.graph_dir, f"{class_name}.graphml")
            if os.path.exists(graph_path):
                G = nx.read_graphml(graph_path)
            else:
                continue
            node_features = [] # 存储节点特征
            node_mapping = {}  # 用于重新映射节点索引
            keyword = class_name.split('.')[-1]
            nodes_with_keyword = {n for n, attrs in G.nodes(data=True) if keyword in attrs.get("label", "")}
            filtered_sccs = [
                scc for scc in nx.strongly_connected_components(G) if scc & nodes_with_keyword
            ]
            G_max = G.subgraph(max(filtered_sccs, key=len) if filtered_sccs else max(nx.strongly_connected_components(G), key=len))
            # G_max = G.subgraph(max(nx.strongly_connected_components(G), key=len))
            for idx, (node, data) in tqdm(enumerate(G_max.nodes(data=True)), total=len(G_max.nodes()), desc="正在处理图中节点"):
                node_mapping[node] = idx
                node_type = data.get('node_type')
                code_metric = data.get('code_metric')
                if isinstance(code_metric, str):
                    code_metric = [float(x) for x in code_metric.split(',')]

                if node_type == 2:  # 方法节点
                    # 解析方法信息并获取代码
                    method_code = self._extract_method_code(project_name, data.get('label'))
                    # 获取代码语义特征
                    code_features = self._get_semantic_features(method_code).to('cpu')
                    print(code_features)
                    # 组合特征
                    features = torch.cat([torch.tensor(code_metric), code_features.flatten()])
                else:  # 类型0或1的节点
                    features = torch.tensor(code_metric)

                node_features.append(features)

            # 处理边
            edge_index = []
            for u, v in G_max.edges():
                edge_index.append([node_mapping[u], node_mapping[v]])

            # 创建PyG数据对象
            # max_length = max(tensor.size(0) for tensor in node_features)
            # padded_tensors = torch.nn.utils.rnn.pad_sequence(node_features, batch_first=True)
            # padded_tensors = [padded_tensors[i] for i in range(padded_tensors.size(0))]
            padded_tensors = [
                torch.nn.functional.pad(tensor, (0, 4145 - tensor.size(0)), mode='constant', value=0)
                for tensor in node_features
            ]
            padded_tensor_stack = torch.stack(padded_tensors)
            first_49_features = padded_tensor_stack[:, :49]  # 提取前 49 个特征
            # 使用 Min-Max Scaling 归一化到 [0, 1] 范围
            min_values = first_49_features.min(dim=0, keepdim=True)[0]
            max_values = first_49_features.max(dim=0, keepdim=True)[0]
            normalized_features = (first_49_features - min_values) / (max_values - min_values + 1e-8)
            # 更新原始张量中的前 49 个特征为归一化后的值
            padded_tensor_stack[:, :49] = normalized_features
            data = Data(
                x=padded_tensor_stack,
                edge_index=torch.tensor(edge_index).t().contiguous(),
                y=torch.tensor([label])
            )
            data.__setitem__("class_name", class_name)
            data.__setitem__("smell_type", smell_type)
            data_list.append(data)
        print('Saving...')
        torch.save(data_list, self.processed_paths[0])
    def _find_method_end(self, code_lines, start_line: int):
        brace_count = 0
        found_first_brace = False
        for i in range(start_line, len(code_lines)):
            line = code_lines[i]
            # 计算花括号的数量
            brace_count += line.count('{')
            brace_count -= line.count('}')
            if '{' in line:
                found_first_brace = True
            if found_first_brace and brace_count == 0:
                return i + 1
        return start_line + 1

    def _extract_method_code(self,project_name, method_label):
        """从方法标签中提取方法代码"""
        # 解析方法标签获取类名和方法名
        match = re.match(r'([\w.]+\.(\w+))\.([\w]+)\(([\w.,\s]+)\)', method_label)
        if not match:
            return ""
        dir_name = match.group(1).split('.')[-2]
        class_name = match.group(2)
        method_name = match.group(3)
        # params = len(match.group(4).split(',')) if match.group(4) else 0
        params = match.group(4).replace(' ', '').split(',')

        # 构建源文件路径
        class_path = class_name + '.java'
        project_path = os.path.join(self.code_dir, project_name)
        try:
            java_file = subprocess.check_output(['find', project_path, '-type', 'f', '-name', class_path], text=True)
            # java_file = subprocess.check_output(['find', project_path, '-type', 'f', '-name', class_path, '-path', f'*/{dir_name}/*'], text=True)
        except subprocess.CalledProcessError as e:
            return " "
        # 读取并解析源文件
        method_code = " "
        for file in java_file.splitlines():
            method_code = extract_method_from_java_file(file, method_name, params)
        # with open(java_file.strip(), 'r', encoding='utf-8') as f:
        #     code = f.read()
        #     drop_comment = re.sub(r'(?s)/\*.*?\*/|//.*?$|/\*\*.*?\*/', '', code, flags=re.MULTILINE | re.DOTALL)
        #     code_content = re.sub(r'\n\s*\n', '\n', drop_comment).strip()
        #     code_lines = code_content.splitlines()
        #     try:
        #         tree = javalang.parse.parse(code_content)
        #         for _, class_decl in chain(
        #     tree.filter(javalang.tree.ClassDeclaration),
        #             tree.filter(javalang.tree.InterfaceDeclaration)
        #         ):
        #             # 遍历类中的所有方法
        #             for method in class_decl.methods:
        #                 parameters = [(param.type.name, param.name) for param in method.parameters]
        #                 if method.name == method_name and len(parameters) == params:
        #                     start_line = method.position.line - 1  # javalang的行号从1开始
        #                     if method.body:
        #                         end_line = method.body[-1].position.line + 1
        #                     else:
        #                         end_line = start_line + 1
        #                     method_code = '\n'.join(code_lines[start_line:end_line])
        #                     print(method_code)
        #                     return method_code
        #     except javalang.tokenizer.LexerError as lexer_error:
        #         print(code_content)
        return method_code

    def _get_semantic_features(self, code):
        """使用code-llama获取代码的语义特征"""
        inputs = self.tokenizer(code, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        return torch.mean(hidden_states, dim=1)

if __name__ == '__main__':
    MODEL_CLASSES = {
        'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
        'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
        'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
        't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer)
    }

    model_type = "roberta"
    model_name_or_path = "/model/LiangXJ/Model/microsoft/graphcodebert-base"
    tokenizer_name = "/model/LiangXJ/Model/microsoft/graphcodebert-base"
    # train, val, test
    # partition = sys.argv[1]
    partition = 'train'

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

    # config = config_class.from_pretrained(model_name_or_path)
    # tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
    # language_model = model_class.from_pretrained(model_name_or_path, from_tf=bool('.ckpt' in model_name_or_path),
    #                                              config=config)

    dataset = SmellDataset(xlsx_file='/model/LiangXJ/developCodeSmell.xlsx', code_dir='/model/LiangXJ/CodeSmellProject')
    print(dataset)
    print(dataset.data_list[0])
    # print(dataset.data_list[0].x)
    # print(dataset.data_list[0].edge_index)
    # print(dataset.data_list[0]._SAMPLE)
