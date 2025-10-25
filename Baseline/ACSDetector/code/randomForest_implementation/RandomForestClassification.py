# -*- coding: utf-8 -*-
import os
import json
from nbformat import write
import pandas as pd
import numpy as np
import random
import math
import collections
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
datasetPath = '/home/yqx/Downloads/comment_data/comment-data/DesigniteJava-master/myData/AAAA_data_collection_in_versions/DataSet/'


if __name__ == '__main__':
    # 获取 类似的数据
    '''
    Alcohol,Malic acid,Ash,Alcalinity of ash,Magnesium,Total phenols,Flavanoids,Nonflavanoid phenols,Proanthocyanins,Color intensity,Hue,OD280/OD315 of diluted wines,Proline,label
    14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065,1

    NOF	NOPF	NOM	NOPM	LOC	WMC	NC	DIT	LCOM	FANIN	FANOUT label
    0	0	2	2	7	2	0	0	-1	0	0                        1
    1	0	1	1	6	1	0	0	0	0	0                        2
    1	0	1	1	7	1	0	0	0	0	0                        2
    '''
    
    def Undersampling(trainlist):
        print('trainlist',len(trainlist))
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
        print('sample ratio(pos:neg): ',pos,':',neg)
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
        print('after sampling(pos:neg): ',pos,':',neg)
        return selectSamples
    def getTrainOrTestCSV(datalist, csv_txt, jsonFilePathList):
        label_dict = {}
        for line in datalist:
            name = line.split('    ')[0][:-5]+'.json'
            label = line.split('    ')[1]
            #print('label',label)
            label_dict[name] = label
        csv_data = []
        csv_data.append('LOC,CC,PC,label\n')
        
        for file in jsonFilePathList:
            try:
                data = json.load(open(file))
                src_metrics = data["metrics"]
                file_name = file.split('/')[-1]
                label = 1 if label_dict[file_name] == '1' else 0
                csv_data.append(','.join(map(lambda x:str(x), src_metrics))+','+str(label)+'\n')
            except:
                print('not found')
        data_txt = open(csv_txt,'w')
        data_txt.writelines(csv_data)
        data_txt.close()

    def get_train_test_list(projectList):
        trainlist = []
        testlist = []
        for projectname in projectList:
            train_list_path = datasetPath+projectname+"/methods/trainlabels.txt"
            trainlist += open(train_list_path).readlines()
        
            test_list_path = datasetPath+projectname+"/methods/testlabels.txt"
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

    
    projectList = ['Ant','jruby','kafka','mockito','storm','tomcat']
    trainlist,testlist = get_train_test_list(projectList)
    times = 3
    result_list = []
    test_result_path = 'result.txt'
    for i in range(times):
        trainlist = Undersampling(trainlist)
        jsonFilePathList_train = []
        jsonFilePathList_test = []
        for item in trainlist:
            projectname = item.split('-')[0]
            if projectname == 'ant':
                projectname = 'Ant'
            jsonFilePath = datasetPath+projectname+"/methods/rawjson/"+item.split('    ')[0][:-5]+'.json'
            jsonFilePathList_train.append(jsonFilePath)
        
        for item in testlist:
            projectname = item.split('-')[0]
            if projectname == 'ant':
                projectname = 'Ant'
            jsonFilePath = datasetPath+projectname+"/methods/rawjson/"+item.split('    ')[0][:-5]+'.json'
            jsonFilePathList_test.append(jsonFilePath)
        
        
        #print("trainlist",trainlist)
        getTrainOrTestCSV(trainlist, 'train_csv.txt', jsonFilePathList_train)
        getTrainOrTestCSV(testlist, 'test_csv.txt', jsonFilePathList_test)

        ##################################################################################################
        train_data = pd.read_csv("train_csv.txt")
        #train_data = train_data[train_data['label'].isin([1, 2])].sample(frac=1, random_state=66).reset_index(drop=True)

        test_data = pd.read_csv("test_csv.txt")
        #test_data = test_data[test_data['label'].isin([1, 2])].sample(frac=1, random_state=66).reset_index(drop=True)

        # 随机森林参数
        NUM_TREES = 128
        MAX_DEPTH = 16
        # 训练随机森林
        clf = RandomForestClassifier(n_estimators=NUM_TREES, max_depth=MAX_DEPTH)

        feature_list = ['LOC','CC','PC']
        clf.fit(train_data.loc[:, feature_list], train_data.loc[:, 'label'])

        
        # 预测
        y_pred = clf.predict(test_data.loc[:, feature_list])
        y_test = test_data.loc[:, 'label']
        
        # 计算auc
        y_scores = clf.predict_proba(test_data.loc[:, feature_list])[:, 1]  # 二分类
        auc = roc_auc_score(y_test, y_scores)

        # macro
        macro_p = precision_score(y_test, y_pred, average='macro')
        macro_r = recall_score(y_test, y_pred, average='macro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print('macro p, r, f1:', macro_p, macro_r, macro_f1)
        macro_p = float(format(macro_p, '.4f'))
        macro_r = float(format(macro_r, '.4f'))
        macro_f1 = float(format(macro_f1, '.4f'))
        
        confusion = confusion_matrix(y_test, y_pred)
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
        p = float(format(precision, '.4f'))
        r = float(format(recall, '.4f'))
        f1 = float(format(f1, '.4f'))
        fpr = float(format(false_positive_rate, '.4f'))
        auc = float(format(auc, '.4f'))
        print('p, r, f1, fpr, auc: ', p, r, f1, fpr, auc)
        result_list.append([true_positives, false_positives, false_negatives, true_negatives, macro_p, macro_r, macro_f1, p, r, f1, fpr, auc])
        test_p_r_f1 = open(test_result_path, 'a')
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
