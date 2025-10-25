from unittest import result
import torch
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
from data_loader import getTwoDicts, getAllData2Dict
from focalloss import FocalLoss

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

datasetPath = '/home/yqx/Downloads/comment_data/comment-data/DesigniteJava-master/myData/AAAA_data_collection_in_versions/DataSet/'

def get_train_test_list(projectList):
    trainlist = []
    testlist = []
    for projectname in projectList:
        train_list_path = datasetPath+projectname+"/types/trainlabels.txt"
        trainlist += open(train_list_path).readlines()
    
        test_list_path = datasetPath+projectname+"/types/testlabels.txt"
        testlist += open(test_list_path).readlines()
    
    new_trainlist = []
    new_testlist = []
    for line in trainlist:
        #print(line.split('    '))

        if line.split('    ')[1] == '0\n':
            new_trainlist.append(line.split('    ')[0]+'    '+'0')
        else:
            new_trainlist.append(line.split('    ')[0]+'    '+'1')
    for line in testlist:
        #print(line.split('    '))
        if line.split('    ')[1] == '0\n':
            new_testlist.append(line.split('    ')[0]+'    '+'0')
        else:
            new_testlist.append(line.split('    ')[0]+'    '+'1')
    #print(new_trainlist)
    return new_trainlist, new_testlist
    
def init_data_and_model(projectList, curModel):
    jsonFolderPathList = []
    for projectname in projectList:
        jsonFolderPathList.append(datasetPath+ projectname +'/types/rawjson')
    trainlist, testlist = get_train_test_list(projectList)
    print('trainlist:testlist',len(trainlist),len(testlist))
    vocabdict, metricdict = getTwoDicts(jsonFolderPathList)
    
    vocablen, metriclen = len(vocabdict), len(metricdict)
    print('vocablen, metriclen',vocablen, metriclen)
    allJsonDict = getAllData2Dict(jsonFolderPathList, vocabdict, metricdict)
    print('allJsonDict',len(allJsonDict))
    if curModel == myDNN:
        model = myDNN(t_hidden).to(device)
    elif curModel == myBiLSTM:
        model = myBiLSTM(vocablen, metriclen, t_hidden, args.dropout, args.alpha).to(device)
    elif curModel == FullModel:
        model = FullModel(vocablen, metriclen, t_hidden, t_hidden, args.nhead, t_gcn_layers, t_transformer_layers, args.dim_feedforward, args.dropout, args.alpha).to(device)
    return trainlist, testlist, allJsonDict, model

criterion = FocalLoss().to(device)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def getBatchList(allJsonDict,line_list):
    batchlist = []
    for line in line_list:
        try:
            jsonName = line.split('    ')[0][:-5]+'.json'
            label = int(line.split('    ')[1])
            data = allJsonDict[jsonName]
            x = data['x']
            edge_index = data['edge_index']
            edge_attr = data['edge_attr']
            metrics = data['metrics']
            token_list = data['token_list']
            src_metrics = data['src_metrics']
            
            a_sample = [x, edge_index, edge_attr, metrics, token_list, src_metrics, label]
            
            batchlist.append(a_sample)
        except:
            #print(jsonName)
            continue
    #print('batchlist',len(batchlist))
    return batchlist


def getBatch(line_list, batch_size, batch_index, device):
    start_line = batch_size*batch_index
    end_line = start_line+batch_size
    dataList = getBatchList(allJsonDict,line_list[start_line:end_line])
    return dataList

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

def test(testlist, model_index, allJsonDict, batch_size, curModel):
    with torch.no_grad():
        
        #model.load_state_dict(torch.load('./model/epoch'+str(model_index)+'.pkl'))
        model.eval()

        notFound = 0
        testCount = 0
        y_preds = []
        y_trues = []
        y_probs = []
        batches = split_batch(testlist, batch_size)
        Test_data_batches = trange(len(batches), leave=True, desc = "Test")
        for i in Test_data_batches:
            #lable
            line_info = batches[i][0].split('    ')
            jsonName = line_info[0][:-5]+'.json'
            label = int(line_info[1])
            #data
            try:
                data = allJsonDict[jsonName]
                x = data['x']
                edge_index = data['edge_index']
                edge_attr = data['edge_attr']
                metrics = data['metrics']
                token_list = data['token_list']
                src_metrics = data['src_metrics']
                testCount += 1
            except:
                notFound += 1
            #predict
            if curModel == myDNN:
                output = model(src_metrics)
            elif curModel == myBiLSTM:
                output = model(token_list, metrics)
            elif curModel == FullModel:
                output = model(x, edge_index, edge_attr, metrics)
            _, predicted = torch.max(output.data, 1)
            y_probs+=F.softmax(output, dim=1).cpu().numpy().tolist()
            
            y_trues += [label]
            y_preds += predicted.tolist()

        
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
        auc = roc_auc_score(y_trues, np.array(y_probs)[:,1])
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
        if sample.split('    ')[1] == '0':
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
        if item.split('    ')[1] == '0':
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
                x, edge_index, edge_attr, metrics, token_list, src_metrics, label = data
                #print("data",data)
                #print(type(label),label)
                label=torch.Tensor([[0,1]]).to(device) if label==1 else torch.Tensor([[1,0]]).to(device)
                #print("label ",label.device," ",label)
                
                if curModel == myDNN:
                    output = model(src_metrics)
                elif curModel == myBiLSTM:
                    output = model(token_list, metrics)
                elif curModel == FullModel:
                    output = model(x, edge_index, edge_attr, metrics)
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
        if (epoch)%10==0:
            true_positives, false_positives, false_negatives, true_negatives, macro_p, macro_r, macro_f1, p, r, f1, fpr, auc = test(testlist, epoch, allJsonDict, 1, curModel)
            test_p_r_f1 = open(test_result_path, 'a')
            test_p_r_f1.write(str(curModel.__name__)+ str(epoch) +'\n')
            test_p_r_f1.write('fusion matrix(TP, FP, FN, TN): '+str(true_positives) +" "+ str(false_positives) +" "+ str(false_negatives) +" "+ str(true_negatives)+"\n")
            test_p_r_f1.write('macro: '+str(macro_p) +" "+ str(macro_r) +" "+ str(macro_f1)+"\n")
            test_p_r_f1.write('binary: '+str(p) +" "+ str(r) +" "+ str(f1) +" "+ str(fpr) +" "+ str(auc)+"\n")
            test_p_r_f1.close()
    true_positives, false_positives, false_negatives, true_negatives, macro_p, macro_r, macro_f1, p, r, f1, fpr, auc = test(testlist, epoch, allJsonDict, 1, curModel)
    
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


    projectList = ['Ant','jruby','kafka','mockito','storm','tomcat']
    modelList = [myDNN, myBiLSTM, FullModel][2:]

    for curModel in modelList:
        print(curModel)
        test_result_path = 'result.txt'
        times = 3
        result_list = []
        trainlist, testlist, allJsonDict, model = init_data_and_model(projectList, curModel)

        for i in range(times):
            true_positives, false_positives, false_negatives, true_negatives, macro_p, macro_r, macro_f1, p, r, f1, fpr, auc = train(trainlist, testlist, curModel)
            result_list.append([true_positives, false_positives, false_negatives, true_negatives, macro_p, macro_r, macro_f1, p, r, f1, fpr, auc])
            test_p_r_f1 = open(test_result_path, 'a')
            test_p_r_f1.write(str(curModel.__name__)+'\n')
            test_p_r_f1.write('fusion matrix(TP, FP, FN, TN): '+str(true_positives) +" "+ str(false_positives) +" "+ str(false_negatives) +" "+ str(true_negatives)+"\n")
            test_p_r_f1.write('macro: '+str(macro_p) +" "+ str(macro_r) +" "+ str(macro_f1)+"\n")
            test_p_r_f1.write('binary: '+str(p) +" "+ str(r) +" "+ str(f1) +" "+ str(fpr) +" "+ str(auc)+"\n")
            test_p_r_f1.close()
        # save mean result
        array = np.array(result_list)
        column_means = np.mean(array, axis=0)
        test_p_r_f1 = open(test_result_path, 'a')
        test_p_r_f1.write(str(column_means))
        test_p_r_f1.write("\n")
        test_p_r_f1.close()