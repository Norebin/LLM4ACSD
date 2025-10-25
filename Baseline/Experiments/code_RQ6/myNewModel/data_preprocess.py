import os
import numpy as np
import json
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from astTools import *
from tqdm import tqdm
#from func_timeout import func_set_timeout, FunctionTimedOut

def getCodeGraphDataByPath(codePath):
    pureAST = code2AST(codePath) #得到AST需要的数据，递归各节点遍历出一棵树 tree
    newtree, nodelist = getNodeList(pureAST)
    ifcount,whilecount,forcount,blockcount,docount,switchcount,alltokens,vocabdict = astStaticCollection(pureAST)
    h,x,vocabdict,edge_index,edge_attr = getFA_AST(newtree, vocabdict)
    return h,x,vocabdict,edge_index,edge_attr

def num2one_hot(_edge_attr):
    #one-hot
    _edge_attr = np.array(_edge_attr)
    edge_attr = np.zeros((_edge_attr.shape[0],13))
    edge_attr[np.arange(_edge_attr.shape[0]),_edge_attr] = 1
    return edge_attr.tolist()

def getCodeGraphByDataItem(codaPath, vocabDict, vocabIndex):
    h,_,_,edge_index,edge_attr = getCodeGraphDataByPath(codaPath)
    #print(vocabdict1)
    # 创建词列表的同时，构造词典，用词序构造 h1_index
    h_index = []
    for v in h:
        if v not in vocabDict: 
            vocabDict[v] = vocabIndex
            h_index.append(vocabIndex)
            vocabIndex += 1
        else:
            h_index.append(vocabDict[v])
    #edge_attr = num2one_hot(edge_attr)
    return vocabDict, vocabIndex, h_index, edge_index, edge_attr

def getVocabDictByASTs(sourceCodePath, codeGraphSavePath, vocabDictSavePath):
    #获取词表，保存代码图对应的json文件
    vocabDict = {} # 创建词列表的同时，构造词典
    vocabIndex = 0  # 词为键，词序为值，以便于用词序构造 h1_index、h2_index
    # dataDict 保存路径
    allCodeGraphDict = {}
    astparseexceptions = 0

    if not os.path.exists(codeGraphSavePath):
        # 遍历文件夹下的所有文件
        for root, dirs, files in os.walk(sourceCodePath):
            for file in tqdm(files):
                # 判断文件是否以".java"结尾
                if file.endswith(".java"):
                    # 打印文件路径
                    codaPath = os.path.join(root, file)
                    fileName = os.path.join(file).split('.')[0]
                    #print(codaPath)
                    try:
                        vocabDict, vocabIndex, h_index, edge_index, edge_attr = getCodeGraphByDataItem(codaPath, vocabDict, vocabIndex)
                        codeGraph = {}
                        codeGraph["h_index"] = h_index
                        codeGraph["edge_index"] = edge_index
                        codeGraph["edge_attr"] = edge_attr
                        allCodeGraphDict[fileName] = codeGraph
                    except:
                        astparseexceptions+=1
                        print(codaPath)
        print(astparseexceptions) #73
        # 保存 vocabDict 文件 到本地
        vocabDictFile = open(vocabDictSavePath, "w")
        json.dump(vocabDict,vocabDictFile)
        vocabDictFile.close()

        # 保存到 allCodeGraphDict 文件
        codeGraphFile = open(codeGraphSavePath, "w")
        json.dump(allCodeGraphDict,codeGraphFile)
        codeGraphFile.close()
    else:
        print("allCodeGraphDict 已存在，读取中...")
        with open(codeGraphSavePath, 'r') as codeGraphFile:
            allCodeGraphData = codeGraphFile.read()
        allCodeGraphDict = json.loads(allCodeGraphData)

        smellCount = {}
        for key in allCodeGraphDict.keys():
            #smellName = '__'.join(key.split('__')[-4:])
            smellName = '__'.join(key.split('__')[1:-6])
            if 'blob__none__1' == smellName or 'data_class__none__1' == smellName:
                print(key)
            val = 1
            if smellName not in smellCount.keys():
                smellCount[smellName] = val
            else:
                smellCount[smellName] += 1
        #print('smellCount',smellCount,len(smellCount))
        # 最终的数据集版本所对应的数据条目：
        # 'long_method__minor': 376, 'feature_envy__minor': 244,    'data_class__none': 2731,     'blob__major': 287, 
        # 'long_method__none': 2254, 'feature_envy__none': 2512,    'data_class__minor': 472,     'blob__minor': 493, 
        # 'long_method__major': 217, 'feature_envy__major': 118,    'data_class__major': 380,     'blob__none': 2830, 
        #'long_method__critical': 64,'feature_envy__critical': 21,  'data_class__critical': 140,  'blob__critical': 114, 

        print("\n词典本地加载中！")
        # read file
        with open(vocabDictSavePath, 'r') as vocabDictFile:
            vocabData = vocabDictFile.read()
        vocabDict = json.loads(vocabData)
        
    print("vocabDict",len(vocabDict)) 
    print("allCodeGraphDict",len(allCodeGraphDict))
    # vocabDict 79084
    # allCodeGraphDict 13255
    return vocabDict, allCodeGraphDict

def getTrainAndTestSetBySeedFold(label_list, fold_num, fold_idx): # e.g. 分为5折（1，2，3，4，5）
        fold_size = len(label_list)//fold_num
        #print("data_list",len(label_list))
        print("fold_size",fold_size)
        train_label_list = []
        test_label_list = []
        for index, value in enumerate(label_list):
            if index >=  (fold_idx - 1) * fold_size and index < fold_idx * fold_size:
                test_label_list.append(value)
            else:
                train_label_list.append(value)
        return train_label_list, test_label_list

def showRitiaOfPosNeg(train_label_list, test_label_list):
    train_pos = 0
    train_neg = 0
    test_pos = 0
    test_neg = 0
    for item in train_label_list:
        label = 1 if int(item.split()[2]) > 1 else 0
        if label == 1:
            train_pos+=1
        else:
            train_neg+=1

    for item in test_label_list:
        label = 1 if int(item.split()[2]) > 1 else 0
        if label == 1:
            test_pos+=1
        else:
            test_neg+=1
    print('showRitiaOfPosNeg:', train_pos/train_neg, test_pos/test_neg)

def getMetricDict(allMetricDictPath):
    metricdict = {}
    metric_num = []
    data = json.load(open(allMetricDictPath))
    # 遍历字典的所有值
    for value in data.values():
        metric_num = metric_num + value
    metric_num = list(set(metric_num))
    metric_num.sort()
    #print(metric_num, len(metric_num))
    for i,v in enumerate(metric_num):
        v = round(v, 2)
        metricdict[v] = i
    # print("metricdict",metricdict,len(metricdict))
    # exit()
    print('metricDict:', len(metricdict))
    return metricdict

def getMetricIndex(metricdict, metrics):
    metric_index = []
    metrics = metrics.cpu().tolist()
    for metric in metrics:
        metric = round(metric, 2)
        try:
            metric_index.append([metricdict[metric]])
        except:
            #print('miss found metric:',metric)
            minor_num = 10000
            minor_key = str(0)
            for key in metricdict:
                if abs(float(key)-float(metric))<minor_num:
                    minor_num = abs(float(key)-float(metric))
                    minor_key = key
            #print('minor_key:',minor_key)
            metric_index.append([metricdict[minor_key]])
    return torch.as_tensor(metric_index).to(device)


if __name__ == '__main__':
    datasetPath = "/home/yqx/Documents/myMLCQdataset/myMLCQdataset/"
    sourceCodePath = datasetPath + 'sourceCode/'
    allMetricDictPath = datasetPath + "allDataDict.json"

    # exampleCode = sourceCodePath + "class__blob__critical__0__10551__5427e72d02bd7f1904da05cdf033359690d2dd00__StreamsDatum__30__167.java"
    # h,x,vocabdict,edge_index,edge_attr = getCodeGraphDataByPath(exampleCode)
    #print(h,x,vocabdict,edge_index,edge_attr)

    
    codeGraphSavePath = datasetPath + "allCodeGraphDict.json"
    vocabDictSavePath = datasetPath + "vocabDict.json"
    vocabDict, allCodeGraphDict = getVocabDictByASTs(sourceCodePath, codeGraphSavePath, vocabDictSavePath)

    #getMetricDict(allMetricDictPath)
   