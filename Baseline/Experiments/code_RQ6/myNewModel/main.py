from unittest import result
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from mymodel import *
from data_preprocess import *
from data_loader import getTwoDicts, getAllData2Dict
from focalloss import FocalLoss
torch.cuda.init()


parser = argparse.ArgumentParser()
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--dim_feedforward', type=int, default=512)
parser.add_argument('--dropout', type=int, default=0.1)
parser.add_argument('--hidden_dropout_prob', type=int, default=0.1)
parser.add_argument('--attention_probs_dropout_prob', type=int, default=0.1)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_attention_heads', type=int, default=8)
parser.add_argument('--alpha', type=int, default=0.2)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

criterion = FocalLoss().to(device)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def getBatchList(allCodeGraphDict, line_list):
    batchlist = []
    for line in line_list:
        try:
            info = line.rstrip('\n').split()
            #print('info',info)
            codeName = info[0]+'.java'
            metrics = torch.as_tensor(allMetricDict[codeName]).to(device)
            label = 1 if int(info[2]) > 1 else 0
            data = allCodeGraphDict[info[0]]
            h_index = torch.LongTensor(data['h_index']).to(device)
            edge_index = torch.LongTensor(data['edge_index']).to(device)
            edge_attr = torch.as_tensor(num2one_hot(data['edge_attr'])).to(device)
            a_sample = [h_index, edge_index, edge_attr, metrics, label]
            batchlist.append(a_sample)
        except:
            pass
    return batchlist

def getBatch(line_list, batch_size, batch_index, device):
    start_line = batch_size*batch_index
    end_line = start_line+batch_size
    dataList = getBatchList(allCodeGraphDict,line_list[start_line:end_line])
    return dataList

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list
def test(testlist, model_index, allCodeGraphDict, batch_size, curModel):
    with torch.no_grad():
        
        #model.load_state_dict(torch.load('./model/epoch'+str(model_index)+'.pkl'))
        model.eval()

        notFound = 0
        testCount = 0
        y_preds = []
        y_trues = []
        y_labels = []
        #y_probs = []
        y_scores = []
        batches = split_batch(testlist, batch_size)
        Test_data_batches = trange(len(batches), leave=True, desc = "Test")
        for i in Test_data_batches:
            try:
                line_info = batches[i][0].rstrip('\n').split()
                codeName = line_info[0]+".java"
                metrics = torch.as_tensor(allMetricDict[codeName]).to(device)
                label = 1 if int(line_info[2]) > 1 else 0
                data = allCodeGraphDict[line_info[0]]
                x = torch.LongTensor(torch.as_tensor(data['h_index'])).to(device)
                edge_index = torch.LongTensor(torch.as_tensor(data['edge_index'])).to(device)
                edge_attr = torch.as_tensor(num2one_hot(data['edge_attr'])).to(device)
                testCount += 1

                #predict
                if curModel == myDNN:
                    output = model(metrics)
                elif curModel == myBiLSTM:
                    metrics_index = getMetricIndex(metricDict, metrics)
                    output = model(x, metrics_index)
                elif curModel == FullModel:
                    metrics_index = getMetricIndex(metricDict, metrics)
                    output = model(x, edge_index, edge_attr, metrics_index)

                _, predicted = torch.max(output.data, 1)
                #y_probs+=F.softmax(output, dim=1).cpu().numpy().tolist()
                testCount += 1
                y_trues += [label]
                y_preds += predicted.tolist()
                y_label = [0,1] if label == 1 else [1,0]
                y_labels += y_label
                y_scores += output.tolist()[0]
            except:
                notFound += 1
        

        # macro
        macro_p = precision_score(y_trues, y_preds, average='macro')
        macro_r = recall_score(y_trues, y_preds, average='macro')
        macro_f1 = f1_score(y_trues, y_preds, average='macro')
        print('macro p, r, f1:', macro_p, macro_r, macro_f1)
        macro_p = float(format(macro_p, '.4f'))
        macro_r = float(format(macro_r, '.4f'))
        macro_f1 = float(format(macro_f1, '.4f'))
        
        confusion = confusion_matrix(y_trues, y_preds)
        # 获取混淆矩阵的各个元素
        true_positives = confusion[1, 1]  # 真正例
        false_positives = confusion[0, 1]  # 假正例
        false_negatives = confusion[1, 0]  # 假负例
        true_negatives = confusion[0, 0]  # 真负例
        print('TP, FP, FN, TN: ', true_positives, false_positives, false_negatives, true_negatives)
        # 计算精确度（Precision）
        precision = true_positives / (true_positives + false_positives)
        # 计算召回率（Recall）
        recall = true_positives / (true_positives + false_negatives)
        # 计算 F1 分数
        f1 = 2 * (precision * recall) / (precision + recall)
        # 计算虚警率（False Positive Rate）
        false_positive_rate = false_positives / (false_positives + true_negatives)
        # 计算AUC
        #auc = roc_auc_score(y_trues, np.array(y_probs)[:,1])
        auc = roc_auc_score(y_labels, y_scores)
        #print('auc',auc)
        # print("Precision:", precision)
        # print("Recall:", recall)
        # print("F1 Score:", f1)
        # print("False Positive Rate:", false_positive_rate)
        
        p = float(format(precision, '.4f'))
        r = float(format(recall, '.4f'))
        f1 = float(format(f1, '.4f'))
        fpr = float(format(false_positive_rate, '.4f'))
        auc = float(format(auc, '.4f'))
        print('p, r, f1, fpr, auc: ', p, r, f1, fpr, auc)

        return true_positives, false_positives, false_negatives, true_negatives, macro_p, macro_r, macro_f1, p, r, f1, fpr, auc



def Undersampling(trainlist):
    #print('trainlist',len(trainlist))
    random.shuffle(trainlist)
    pos = 0
    neg = 0
    rate = 1
    posSamples = []
    negSamples = []
    selectSamples = []
    for sample in trainlist:
        if int(sample.split(' ')[2]) < 2:
            neg+=1
            negSamples.append(sample)
        else:
            pos+=1
            posSamples.append(sample)
    #print('sample ratio(pos:neg): ',pos,':',neg)
    if pos>=neg:
        sampleNum = int(neg*rate)
        selectSamples = negSamples[:sampleNum] + posSamples[:sampleNum]
    elif neg>pos:
        sampleNum = int(pos*rate)
        selectSamples = negSamples[:sampleNum] + posSamples[:sampleNum]
    random.shuffle(selectSamples)
    pos = 0
    neg = 0
    for item in selectSamples:
        if int(item.split(' ')[2]) < 2:
            neg+=1
        else:
            pos+=1
    #print('after sampling(pos:neg): ',pos,':',neg)
    return selectSamples

def train(trainlist, testlist, curModel):
    optimizer = optim.Adam(model.parameters(), lr=t_lr, weight_decay=args.weight_decay)
    model.train()
    #print("loaded ", './saveModel/epoch'+str(start_train_model_index)+'.pkl')
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # print("模型总参：", get_parameter_number(model))
    # print("nhead ", args.nhead," batch_size ", t_batch_size)
    # print("dropout = ",args.dropout)

    #trainlist = trainlist[:int(len(trainlist)*0.2)]
    epochs = trange(t_epoch, leave=True, desc = "Epoch")
    iterations = 0
    for epoch in epochs:
        #print(epoch)
        totalloss=0.0
        main_index=0.0
        #batches = create_batches(trainDataList)
        #for index, batch in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
        count = 0
        right = 0
        acc = 0
        
        train_data = Undersampling(trainlist)
        #random.shuffle(trainlist)

        for batch_index in range(int(len(train_data)/t_batch_size)):
            batch = getBatch(train_data, t_batch_size, batch_index, device)
            optimizer.zero_grad()
            
            batchloss= 0
            for data in batch:
                model.train()
                x, edge_index, edge_attr, metrics, label = data
                

                #print("data",data)
                #print(type(label),label)
                label=torch.Tensor([[0,1]]).to(device) if label==1 else torch.Tensor([[1,0]]).to(device)
                #print("label ",label.device," ",label)
                
                if curModel == myDNN:
                    output = model(metrics)
                elif curModel == myBiLSTM:
                    metrics_index = getMetricIndex(metricDict, metrics)
                    output = model(x, metrics_index)
                elif curModel == FullModel:
                    metrics_index = getMetricIndex(metricDict, metrics)
                    output = model(x, edge_index, edge_attr, metrics_index)
                #print("output",output)
                #print("label",label)

                batchloss = batchloss + criterion(output, label)

                count += 1
                right += torch.sum(torch.eq(torch.argmax(output, dim=1), torch.argmax(label, dim=1)))
            #print("batchloss",batchloss)
            acc = right*1.0/count
            batchloss.backward(retain_graph = True)
            optimizer.step()
            loss = batchloss.item()
            totalloss += loss
            main_index = main_index + len(batch)
            loss = totalloss/main_index
            epochs.set_description("Epoch (Loss=%g) (Acc = %g)" % (round(loss,5) , acc))
            iterations += 1
        '''
        if (epoch)%10==0:
            torch.save(model.state_dict(), saveModelsPath+'/'+modelName+'epoch'+str(epoch)+'.pkl')
        '''  
        if (epoch)%20==0 and epoch >10:
            true_positives, false_positives, false_negatives, true_negatives, macro_p, macro_r, macro_f1, p, r, f1, fpr, auc = test(testlist, epoch, allCodeGraphDict, 1, curModel)
            test_p_r_f1 = open(test_result_path, 'a')
            test_p_r_f1.write(str(curModel.__name__)+ str(epoch) +'\n')
            test_p_r_f1.write('fusion matrix(TP, FP, FN, TN): '+str(true_positives) +" "+ str(false_positives) +" "+ str(false_negatives) +" "+ str(true_negatives)+"\n")
            test_p_r_f1.write('macro: '+str(macro_p) +" "+ str(macro_r) +" "+ str(macro_f1)+"\n")
            test_p_r_f1.write('binary: '+str(p) +" "+ str(r) +" "+ str(f1) +" "+ str(fpr) +" "+ str(auc)+"\n")
            test_p_r_f1.close()
    true_positives, false_positives, false_negatives, true_negatives, macro_p, macro_r, macro_f1, p, r, f1, fpr, auc = test(testlist, epoch, allCodeGraphDict, 1, curModel)
    
    return true_positives, false_positives, false_negatives, true_negatives, macro_p, macro_r, macro_f1, p, r, f1, fpr, auc

if __name__ == '__main__':
    # hyperparameters settings by Taguchi DoE
    t_hidden = 128
    t_lr = 0.001
    t_epoch = 100
    t_batch_size = 64
    t_gcn_layers = 3
    t_transformer_layers = 3
    print('Taguchi:', t_hidden, t_lr, t_epoch, t_batch_size, t_gcn_layers, t_transformer_layers)


    datasetPath = "/home/yqx/Documents/myMLCQdataset/myMLCQdataset/"
    sourceCodePath = datasetPath + 'sourceCode/'
    codeGraphSavePath = datasetPath + "allCodeGraphDict.json"
    allMetricDictPath = datasetPath + "allDataDict.json"
    vocabDictSavePath = datasetPath + "vocabDict.json"

    with open(allMetricDictPath, 'r') as f:
        allMetricDictData = f.read()
    allMetricDict = json.loads(allMetricDictData)
    vocabDict, allCodeGraphDict = getVocabDictByASTs(sourceCodePath, codeGraphSavePath, vocabDictSavePath)
    metricDict = getMetricDict(allMetricDictPath)
    
    vocablen = len(vocabDict) + 10000
    metriclen = len(metricDict) + 10000

    # 最终的数据集版本所对应的数据条目：
        # 'long_method__minor': 376, 'feature_envy__minor': 244,    'data_class__none': 2731,     'blob__major': 287, 
        # 'long_method__none': 2254, 'feature_envy__none': 2512,    'data_class__minor': 472,     'blob__minor': 493, 
        # 'long_method__major': 217, 'feature_envy__major': 118,    'data_class__major': 380,     'blob__none': 2830, 
        #'long_method__critical': 64,'feature_envy__critical': 21,  'data_class__critical': 140,  'blob__critical': 114,
    
    overall_label_path = datasetPath + "label.txt"


    smellType = ['data_class','blob','feature_envy','long_method']
    modelList = [myDNN, myBiLSTM, FullModel]

    for smell in smellType:
        
        if smell == 'data_class' or smell == 'blob':
            metric_list_len = 36
        else:
            metric_list_len = 24

        for curModel in modelList:
            print(curModel)
            test_result_path =  curModel.__name__ +'__'+ smell + '_result.txt'
            
            smell_label_path = datasetPath + 'labels_' + smell + '.txt'
            with open(smell_label_path) as f:
                label_list = f.readlines()
            print("\n -----------------------DataInfo------------------------")
            fold_num = 5
            seed = 666
            print(smell)
            print("seed =",seed)
            print("fold_num =",fold_num)
            random.seed(seed)
            random.shuffle(label_list)

            result_list = []

            for fold_idx in range(1,6):
                print(' fold_idx:',fold_idx)
                #数据集划分，按项目名称划分，进行5折/3折交叉验证
                train_label_list, test_label_list = getTrainAndTestSetBySeedFold(label_list, fold_num, fold_idx)
                #print("train_label_list",len(train_label_list))
                #print("test_label_list",len(test_label_list))
                showRitiaOfPosNeg(train_label_list, test_label_list)

                if curModel == myDNN:
                    model = myDNN(t_hidden, metric_list_len).to(device)
                elif curModel == myBiLSTM:
                    model = myBiLSTM(vocablen, metriclen, t_hidden, args.dropout, args.alpha).to(device)
                elif curModel == FullModel:
                    model = FullModel(vocablen, metriclen, t_hidden, t_hidden, metric_list_len, args.nhead, t_gcn_layers, t_transformer_layers, args.dim_feedforward, args.dropout, args.alpha).to(device)

                true_positives, false_positives, false_negatives, true_negatives, macro_p, macro_r, macro_f1, p, r, f1, fpr, auc = train(train_label_list, test_label_list, curModel)
                result_list.append([true_positives, false_positives, false_negatives, true_negatives, macro_p, macro_r, macro_f1, p, r, f1, fpr, auc])
                test_p_r_f1 = open(test_result_path, 'a')
                test_p_r_f1.write(str(curModel.__name__)+'\n')
                test_p_r_f1.write('fusion matrix(TP, FP, FN, TN): '+str(true_positives) +" "+ str(false_positives) +" "+ str(false_negatives) +" "+ str(true_negatives)+"\n")
                test_p_r_f1.write('macro: '+str(macro_p) +" "+ str(macro_r) +" "+ str(macro_f1)+"\n")
                test_p_r_f1.write('binary: '+str(p) +" "+ str(r) +" "+ str(f1) +" "+ str(fpr) +" "+ str(auc)+"\n")
                test_p_r_f1.close()

            # save mean result
            np.set_printoptions(suppress=True)
            array = np.array(result_list)
            column_means = np.mean(array, axis=0)
            test_p_r_f1 = open(test_result_path, 'a')
            test_p_r_f1.write(str(column_means))
            test_p_r_f1.write("\n")
            test_p_r_f1.close()
    

            
            
            