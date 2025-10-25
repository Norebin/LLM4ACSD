from re import I, X
import os
import tqdm
import json
import torch
from torch import LongTensor, as_tensor
from torch.utils.data import Dataset
import numpy as np

'''
class JsonDataLoader(Dataset):
    def __init__(self, jsonFolderPath, vocabdict, metricdict):
        self.vocabdict = vocabdict
        self.metricdict = metricdict

        x_list = []
        edge_index_list = []
        edge_attr_list = []
        metrics_list = []

        num_list = []

        for root ,dirs, files in os.walk(jsonFolderPath):
            for file in files:
                
                data = json.load(open(jsonFolderPath+'/'+file))
                
                x = self.get_x(data['nodes'])
                edge_index, edge_attr = self.get_edge_info(data['edges'])
                metrics = self.get_metrics(data['metrics'])
                
                x_list.append(x)
                edge_index_list.append(edge_index)
                edge_attr_list.append(edge_attr)
                metrics_list.append(metrics)
                for num in metrics:
                    num_list.append(int(num))
                
        print(len(num_list)," ",len(set(num_list)))
        self.x_list = x_list
        self.edge_index_list = edge_index_list
        self.edge_attr_list = edge_attr_list
        self.metrics_list = metrics_list

    def get_x(self, nodes):
        x = []
        for i in range(len(nodes)):
            #x.append(nodes[str(i)])
            if nodes[str(i)] not in self.vocabdict:
                x.append(self.vocabdict['__Empty_Token__'])
                #print('keyErro')
            else:
                x.append(self.vocabdict[nodes[str(i)]])
        return x
    
    def get_metrics(self, metrics):
        metric_indix = []
        for metric in metrics:
            metric_indix.append(self.metricdict[metric])
        return metric_indix

    def get_edge_info(self, edges):
        edge_src = []
        edge_tgt = []
        edge_attr_num = []
        for edge in edges:
            edge_src.append(int(edge.split("->")[0]))
            edge_tgt.append(int(edge.split("->")[1]))
            edge_attr_num.append(edges[edge])
        edge_index = [edge_src,edge_tgt]
        #edge_attr = self.num2one_hot(edge_attr_num)  #one-hot
        edge_attr = edge_attr_num
        return edge_index, edge_attr

    def num2one_hot(self, _edge_attr):
        #one-hot
        _edge_attr = np.array(_edge_attr)
        edge_attr = np.zeros((_edge_attr.shape[0],13))
        edge_attr[np.arange(_edge_attr.shape[0]),_edge_attr] = 1
        return edge_attr.tolist()
        
    def __getitem__(self, index):
        x = LongTensor(as_tensor(self.x_list[index]))
        edge_index = as_tensor(self.edge_index_list[index]).to('cuda')
        edge_attr = LongTensor(as_tensor(self.edge_attr_list[index]))
        metrics = LongTensor(as_tensor(self.metrics_list[index]))
        
        return x, edge_index, edge_attr, metrics

    def __len__(self):
        return len(self.x_list)

#data = JsonDataLoader("/home/yqx/Desktop/TD-data-process/outData/ant-rel-1.10.12/rawjson", vocabdict, metricdict)
#print(data[0])
'''

def get_x(nodes, vocabdict):
    x = []
    for i in range(len(nodes)):
        #x.append(nodes[str(i)])
        if nodes[str(i)] not in vocabdict:
            x.append(vocabdict['__Empty_Token__'])
            #print('keyErro')
        else:
            x.append(vocabdict[nodes[str(i)]])
    return x

def get_token_list(nodelist, vocabdict):
    token_list = []
    for token in nodelist:
        if token not in vocabdict:
            token_list.append(vocabdict['__Empty_Token__'])
        else:
            token_list.append(vocabdict[token])
    return token_list

def num2one_hot(_edge_attr):
    #one-hot
    _edge_attr = np.array(_edge_attr)
    edge_attr = np.zeros((_edge_attr.shape[0],13))
    edge_attr[np.arange(_edge_attr.shape[0]),_edge_attr] = 1
    return edge_attr.tolist()

def get_metrics(metrics, metricdict):
    metric_indix = []
    for metric in metrics:
        metric_indix.append(metricdict[metric])
    return metric_indix

def get_edge_info(edges):
    edge_src = []
    edge_tgt = []
    edge_attr_num = []
    for edge in edges:
        edge_src.append(int(edge.split("->")[0]))
        edge_tgt.append(int(edge.split("->")[1]))
        edge_attr_num.append(edges[edge])
    edge_index = [edge_src,edge_tgt]
    edge_attr = num2one_hot(edge_attr_num)  #one-hot
    #edge_attr = edge_attr_num
    return edge_index, edge_attr

def getAllData2Dict(jsonFolderPathList, vocabdict, metricdict):
    allJsonDicts = {}
    for jsonFolderPath in jsonFolderPathList:
        for root ,dirs, files in os.walk(jsonFolderPath):
            for file in files:
                filedict = {}

                codepath = (root+'/'+file).replace('rawjson','pureJava')[:-5]+'.java'
                # if os.path.getsize(codepath) > 1024*110:
                #     print("文件大于110kb")
                #     continue

                data = json.load(open(root+'/'+file))
                
                x = get_x(data['nodes'], vocabdict)
                edge_index, edge_attr = get_edge_info(data['edges'])
                metrics = get_metrics(data['metrics'], metricdict)

                token_list = get_token_list(data["nodelist"], vocabdict)
                if len(token_list)>5000:
                    continue
                src_metrics = data["metrics"]
                
                x = LongTensor(as_tensor(x)).to('cuda')
                edge_index = as_tensor(edge_index).to('cuda')
                edge_attr = as_tensor(edge_attr).to('cuda')
                metrics = LongTensor(as_tensor([[metric] for metric in metrics])).to('cuda')

                token_list = LongTensor(as_tensor(token_list)).to('cuda')
                src_metrics = as_tensor(src_metrics).to('cuda')

                filedict['x'] = x
                filedict['edge_index'] = edge_index
                filedict['edge_attr'] = edge_attr
                filedict['metrics'] = metrics
                filedict['token_list'] = token_list
                filedict['src_metrics'] = src_metrics
                allJsonDicts[file] = filedict
            
    return allJsonDicts

def getTwoDicts(jsonFolderPathList):
    vocabdict = {}
    vocabindex = 0
    vocabdict['__Empty_Token__'] = vocabindex
    vocabindex += 1

    metricdict = {}
    metric_num = []
    for jsonFolderPath in jsonFolderPathList:
        for root ,dirs, files in os.walk(jsonFolderPath):
            for file in files:
                data = json.load(open(jsonFolderPath+'/'+file))
                metric_num = metric_num + data['metrics']

                for token in data['nodelist']:
                    if token in vocabdict.keys():
                        continue
                    else:
                        vocabdict[token] = vocabindex
                        vocabindex += 1
        metric_num = list(set(metric_num))
        metric_num.sort()
        #print(metric_num, len(metric_num))
        for i,v in enumerate(metric_num):
            metricdict[v] = i
        #print("vocabdict",len(vocabdict))
        #print("metricdict",len(metricdict))
    return vocabdict, metricdict
'''
jsonFolderPath = "/home/yqx/Desktop/TD-data-process/outData/ant-rel-1.10.12/rawjson"
vocabdict, metricdict = getTwoDicts(jsonFolderPath)
data = getAllData2Dict(jsonFolderPath, vocabdict, metricdict)

print(data['ant-rel-1.10.12__(default package)__A.json'])
'''







