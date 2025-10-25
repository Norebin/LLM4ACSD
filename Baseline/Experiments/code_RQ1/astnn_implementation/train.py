import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import time
from tqdm import tqdm, trange
from focalloss import FocalLoss
from preprocess_data import preprocess_data
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from model import ASTNN
import torch.nn.functional as F
import torch.optim as optim

EPOCH = 100
custom_learning_rate = 0.001


def get_batch(dataset, i, batch_size):
    return dataset.iloc[i: i + batch_size]

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

def init_data_and_model(projectList):

    train_labels, test_labels = get_train_test_list(projectList)
    
    typefilepathList = []
    for projectname in projectList:
        typefilepathList.append(datasetPath + projectname +'/methods/pureJavaMethod')

    w2v, programs = preprocess_data(train_labels, test_labels, typefilepathList)
    
    print('Reading data...')
    #w2v = Word2Vec.load('./data/w2v_128').wv
    embeddings = torch.tensor(np.vstack([w2v.wv.vectors, [0] * 128]))

    #programs = pd.read_pickle('./data/programs.pkl')

    #print("programs",programs)
    #train_labels = open(train_labels).readlines()
    #test_labels = open(test_labels).readlines()
    train_code_name = []
    test_code_name = []
    for line in train_labels:
        train_code_name.append(line.split('    ')[0])
    for line in test_labels:
        test_code_name.append(line.split('    ')[0])

    train_set_indexTree = []
    train_set_label = []
    test_set_indexTree = []
    test_set_label = []
    for i,name in enumerate(programs['name']):
        if programs['name'][i] in train_code_name and programs['index_tree'][i] != []:
            train_set_indexTree.append(programs['index_tree'][i])
            train_set_label.append(programs['label'][i])
        elif programs['name'][i] in test_code_name and programs['index_tree'][i] != []:
            test_set_indexTree.append(programs['index_tree'][i])
            test_set_label.append(programs['label'][i])

    training_data = {"index_tree":train_set_indexTree,"label":train_set_label}
    training_set = pd.DataFrame(training_data, columns = ['index_tree','label'])

    test_data = {"index_tree":test_set_indexTree,"label":test_set_label}
    test_set = pd.DataFrame(test_data, columns = ['index_tree','label'])
    '''
    training_set.to_pickle('./data/train_programs.pkl')
    test_set.to_pickle('./data/test_programs.pkl')
    training_set=pd.read_pickle('./data/train_programs.pkl')
    test_set=pd.read_pickle('./data/test_programs.pkl')
    '''
    validation_set = test_set

    net = ASTNN(output_dim=2,
            embedding_dim=128, num_embeddings=len(w2v.wv.vectors) + 1, embeddings=embeddings,
            batch_size=BATCH_SIZE)
    net.cuda()

    #criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss().cuda()
    #optimizer = torch.optim.Adamax(net.parameters())
    optimizer = optim.Adamax(net.parameters(), lr=custom_learning_rate)
    return net,criterion,optimizer,training_set,validation_set,test_set

def train(dataset, backward=True):
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    batchloss = 0
    while i < len(dataset):
        data = get_batch(dataset, i, BATCH_SIZE)
        input, label = data['index_tree'], torch.Tensor([[0,1] if label==1 else [1,0] for label in data['label']]).cuda()
        tlabel = [label for label in data['label']]
        i += BATCH_SIZE
        
        #print(" label",label.shape)
        net.zero_grad()
        net.batch_size = len(input)
        
        output = net(input)

        _, predicted = torch.max(output.data, 1)

        loss = criterion(output, label)
        batchloss += loss
        if backward and i%64==0:   #  BATCH_SIZE = 64
            #print(i)
            batchloss.backward()
            #loss.backward(retain_graph = True)
            optimizer.step()
            batchloss = 0

        # calc acc
        #pred = output.data.argmax(1)
        #correct = pred.eq(label).sum().item()
        correct = torch.sum(torch.eq(torch.argmax(output, dim=1), torch.argmax(label, dim=1)))
        total_acc += correct
        total += len(input)
        total_loss += loss.item() * len(input)
     
    return total_loss/total, total_acc/total

def test(dataset, backward=False):
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    y_preds = []
    y_trues = []
    y_probs = []
    batchloss = 0
    while i < len(dataset):
        data = get_batch(dataset, i, BATCH_SIZE)
        input, label = data['index_tree'], torch.Tensor([[0,1] if label==1 else [1,0] for label in data['label']]).cuda()
        tlabel = [label for label in data['label']]
        i += BATCH_SIZE
        
        #print(" label",label.shape)
        net.zero_grad()
        net.batch_size = len(input)
        
        output = net(input)

        _, predicted = torch.max(output.data, 1)

        loss = criterion(output, label)
        batchloss += loss
        if backward and i%64==0:   #  BATCH_SIZE = 64
            #print(i)
            batchloss.backward()
            #loss.backward(retain_graph = True)
            optimizer.step()
            batchloss = 0

        # calc acc
        #pred = output.data.argmax(1)
        #correct = pred.eq(label).sum().item()
        correct = torch.sum(torch.eq(torch.argmax(output, dim=1), torch.argmax(label, dim=1)))
        total_acc += correct
        total += len(input)
        total_loss += loss.item() * len(input)
        #print("tlabel",tlabel)
        #print("predicted.tolist()",predicted.tolist())
        #exit()
        y_trues += tlabel
        y_preds += predicted.tolist()
        y_probs += F.softmax(output, dim=1).cpu().detach().numpy().tolist()

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
    print('loss ', total_loss/total)
    print('acc ', total_acc/total)
        

    return total_loss/total, total_acc/total, true_positives, false_positives, false_negatives, true_negatives, macro_p, macro_r, macro_f1, p, r, f1, fpr, auc


def Undersampling(training_set):
    #print('training_set',len(training_set['label']))
    pos = 0
    neg = 0
    rate = 1
    posSamples = []
    negSamples = []
    selectSamples = []
    for i,label in enumerate(training_set['label']):
        #print('label',label)
        if label == 1:
            pos+=1
            posSamples.append([training_set['index_tree'][i],label])
        else:
            neg+=1
            negSamples.append([training_set['index_tree'][i],label])
    #print('sample ratio(pos:neg): ',pos,':',neg)
    if pos>=neg:
        sampleNum = int(neg*rate)
        selectSamples = negSamples[:sampleNum] + posSamples[:sampleNum]
        #selectSamples = negSamples + posSamples[:neg]
        #print('after sampling(pos:neg): ',sampleNum,':',sampleNum)
    elif neg>pos:
        sampleNum = int(pos*rate)
        selectSamples = negSamples[:sampleNum] + posSamples[:sampleNum]
        #selectSamples = negSamples[:pos] + posSamples
        #print('after sampling(pos:neg): ',sampleNum,':',sampleNum)
    random.shuffle(selectSamples)
    index_tree = []
    label = []
    for item in selectSamples:
        index_tree.append(item[0])
        label.append(item[1])
    data = {'index_tree':index_tree, 'label':label}
    Sampled_set = pd.DataFrame(data,columns = ['index_tree','label'])
    return Sampled_set

    
if __name__ == '__main__':
    
    projectList = ['Ant','jruby','kafka','mockito','storm','tomcat']
    datasetPath = '/home/yqx/Downloads/comment_data/comment-data/DesigniteJava-master/myData/AAAA_data_collection_in_versions/DataSet/'
    epochs = trange(EPOCH, leave=True, desc = "Epoch")
    #三次求平均值
    BATCH_SIZE = 1
    p_sum = r_sum = f1_sum = 0
    test_result_path = 'result.txt'
    result_list = []
    times = 3
    for i in range(times):
        net,criterion,optimizer,training_set,validation_set,test_set = init_data_and_model(projectList)
        print('Start Training...')
        
        for epoch in epochs:
            start_time = time.time()
            #print(len(training_set),training_set)
            training_data = Undersampling(training_set)

            loss, acc = train(training_data, epoch)
            epochs.set_description("Epoch (Loss=%g) (Acc = %g)" % (round(loss,5) , acc))
            end_time = time.time()
            
            #torch.save(net.state_dict(), './data/params_epoch[%d].pkl' % (epoch + 1))

        _, _, true_positives, false_positives, false_negatives, true_negatives, macro_p, macro_r, macro_f1, p, r, f1, fpr, auc = test(test_set, backward=False)
        result_list.append([true_positives, false_positives, false_negatives, true_negatives, macro_p, macro_r, macro_f1, p, r, f1, fpr, auc])
        test_p_r_f1 = open(test_result_path, 'a')
        test_p_r_f1.write('result: '+str(true_positives)+' '+str(false_positives)+' '+str(false_negatives)+' '+str(true_negatives)+' '+str(macro_p)+' '+str(macro_r)+' '+str(macro_f1)+' '+str(p)+' '+str(r)+' '+str(f1)+' '+str()+' '+str(fpr)+' '+str(auc)+"\n")
        test_p_r_f1.close()
        
    # save mean result
    np.set_printoptions(suppress=True)
    array = np.array(result_list)
    column_means = np.mean(array, axis=0)
    test_p_r_f1 = open(test_result_path, 'a')
    test_p_r_f1.write('mean_result '+str(column_means))
    test_p_r_f1.write("\n")
    test_p_r_f1.close()