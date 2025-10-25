import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization, LSTM, Conv1D
from keras.models import Sequential
from dataclasses import dataclass
import tensorflow_datasets as tfds 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler
import os
from   datetime import date
from Pretraining import *
from keras import callbacks
# import tensorflow_addons as tfa

import pandas as pd
import numpy as np
from statistics import mean

from datetime import datetime
import time
import glob
import re
from pprint import pprint
import matplotlib.pyplot as plt

@dataclass
class Config:
    # preprocessing params
    MAX_LEN = 256      
    VOCAB_SIZE = 30000  
    # model params
    EMBED_DIM = 128
    FF_DIM = 2048  
    NUM_LAYERS = 6
    LR = 0.001   
    DROPOUT = 0.3
    BATCH_SIZE = 128   
    # transformer 
    NUM_HEAD = 8 
    # parallelism
    BUFFER_SIZE = tf.data.experimental.AUTOTUNE
    CODE_VERSION = 'Not defined'
    def save(self, fname):
        dict1 = {'MAX_LEN' : self.MAX_LEN, 
                'VOCAB_SIZE' : self.VOCAB_SIZE,
                'EMBED_DIM' : self.EMBED_DIM,
                'BATCH_SIZE' : self.BATCH_SIZE,
                'LR' : self.LR,
                'DROPOUT' : self.DROPOUT,
                'FF_DIM' : self.FF_DIM,
                'NUM_LAYERS' : self.NUM_LAYERS,
                'CODE_VERSION' : self.CODE_VERSION,
                }
        file1 = open(fname, "w") 
        str1 = repr(dict1)
        file1.write("config = " + str1 + "\n")
        file1.close()

def get_vectorize_layer1( vocab_size, max_seq, vec_log='tv_layer.pkl'):

    try: 
        from_disk = pickle.load(open(vec_log, "rb"))
        vectorize_layer = TextVectorization.from_config(from_disk['config'])
        vectorize_layer.set_weights(from_disk['weights'])
        print('vectorize layer already existed')

    except:
        print('vectorize layer not found')

        vectorize_layer = TextVectorization(
            max_tokens=vocab_size,
            output_mode="int",
            standardize=None,
            output_sequence_length=max_seq,
        )
        text_ds = tf.data.Dataset.from_tensor_slices(texts).prefetch(config.BUFFER_SIZE)
        vectorize_layer.adapt(text_ds)

        # Insert mask token in vocabulary
        vocab = vectorize_layer.get_vocabulary()
        vocab = vocab[2 : vocab_size ]
        vectorize_layer.set_vocabulary(vocab)


    return vectorize_layer

def encode(texts):
    texts = tf.expand_dims(texts, -1)  
    encoded_texts = tf.squeeze(vectorize_layer(texts))
    return encoded_texts

def data_balancing(x, y):
    ROS = SMOTE(sampling_strategy=1, k_neighbors =3)
    X_bal, y_bal = ROS.fit_resample(x, y)

    return X_bal, y_bal

def split_data(data, training_ratio, _valid=False):
    #train_size=0.8
    X = data.drop(columns = ['label']).copy()
    y = data['label']

    X_train, X_rem, y_train, y_rem = train_test_split(X,y,stratify = y, train_size=training_ratio)

    if  _valid == False:
        return X_train, X_rem, y_train, y_rem
    
    else:
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)
        return  X_train, X_valid, X_test, y_train, y_valid, y_test

def load_labeled_dataset(fname, training_ratio=0.7):
     # 创建一个空的DataFrame来存储所有处理后的数据
    all_data = pd.DataFrame()
    # 要保留的列名
    columns_to_keep = [
        'unique_id', 'Project', 'Version', 'Package Name', 
        'Type Name', 'Method Name', 'Smell', 'File Path', 'code', 'actionable'
    ]
    # 遍历目录下的所有文件
    for filename in os.listdir(fpath):
        if filename.endswith('.csv'):
            file_path = os.path.join(fpath, filename)
            try:
                df = pd.read_csv(file_path)
                # 筛选dataset_split为train的数据
                if 'dataset_split' in df.columns:
                    df = df[df['dataset_split'] == 'train']
                # 只保留指定的列
                # 检查哪些列存在
                existing_columns = [col for col in columns_to_keep if col in df.columns]
                df = df[existing_columns]
                # 将处理后的数据添加到总DataFrame中
                all_data = pd.concat([all_data, df], ignore_index=True)
                print(f"成功处理文件: {filename}")             
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    all_data.rename(columns = {'actionable':'label'}, inplace = True)
    all_data = pre_processing(all_data)

    # dataset = pd.read_csv (fname, index_col =None, header =0) 
    # all_data = dataset[['Code', 'Smelly']]
    # all_data = all_data.rename(columns={"Code": "code", "Smelly": "label"})

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    X_train,  X_test, y_train, y_test= split_data(all_data, training_ratio= training_ratio)
    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
    return train_df, test_df

def get_finetuning_dataset(train_df, test_df):
    # Prepare datasets for training

    x_train = encode(train_df.code.values)
    y_train = train_df.label.values

    # for testing
    x_test = encode(test_df.code.values)
    y_test = test_df.label.values

    return x_train, y_train, x_test, y_test #train_ds, test_ds

def data_preprocessing(x_train, y_train, x_test, y_test, balance = False):
    if balance:
        #print(Counter(y_train))
        x_train, y_train = data_balancing(x_train, y_train)
        print(Counter(y_train))
        x_test, y_test = data_balancing( x_test, y_test)
        print(Counter(y_test))

    return x_train, y_train, x_test, y_test

def create_classifier_model(pretrained_model):
    inputs = layers.Input((config.MAX_LEN,), dtype=tf.int64)
    sequence_output = pretrained_model(inputs)
    pooled_output = layers.GlobalMaxPooling1D()(sequence_output)
    outputs = layers.Dense(1, activation="sigmoid")(pooled_output)
    classifer_model = keras.Model(inputs, outputs, name="classification")
    optimizer = keras.optimizers.Adam()
    classifer_model.compile(
        optimizer=optimizer, loss=loss, metrics=metrics
    )
    return classifer_model

def fine_tune(pretrained_model, x_train, y_train, x_test, y_test):
    print('================== Start training ==================\n')
    # Freeze it
    pretrained_model.trainable = False
    classifer_model = create_classifier_model(pretrained_model)
    #classifer_model.summary()

    # Train the classifier with frozen  stage (feature-based)
    start_time = time.time()
    classifer_model.fit(x_train, y_train, epochs=epochs, validation_data=[x_test, y_test ], verbose=0)
    train_time_1 = round((time.time() - start_time))
    history_1 = classifer_model.evaluate(x_test, y_test)

    # Unfreeze the model for fine-tuning
    pretrained_model.trainable = True
    classifer_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    start_time = time.time()
    classifer_model.fit(x_train, y_train, epochs=epochs, validation_data=[x_test, y_test], verbose=0)
    train_time_2 = round((time.time() - start_time))
    
    start_time = time.time()
    history_2 = classifer_model.evaluate(x_test, y_test)
    EVAL_TIME = round((time.time() - start_time))
    print('EVAL_TIME = ', EVAL_TIME)

    return history_1, history_2, train_time_1, train_time_2, classifer_model, #EVAL_TIME

def fine_tune_feature_based(pretrained_model, x_train, y_train, x_test, y_test):
    print('================== Start training ==================\n')
    # Freeze it
    pretrained_model.trainable = False
    classifer_model = create_classifier_model(pretrained_model)
    #classifer_model.summary()

    # Train the classifier with frozen  stage
    start_time = time.time()
    classifer_model.fit(x_train, y_train, epochs=epochs, validation_data=[x_test, y_test ], verbose=0)
    train_time_1 = round((time.time() - start_time))
    start_time = time.time()
    history_1 = classifer_model.evaluate(x_test, y_test)
    eval_time_1 = round((time.time() - start_time))


    return history_1,  eval_time_1, classifer_model

def getToday ():
    return str(date.today())

def getTime ():
    t = time.localtime()
    current_time = time.strftime('%H-%M-%S', t)
    return current_time

def Save_Excel(result, fname, mode='w'):
    if not os.path.isfile(fname) : result.to_csv(fname, header='column_names')
    else : 
        result.to_csv(fname,  mode='a', header=False)

def mcc(tp,fp,tn,fn):
    try:   
        return (tp*tn - fp*fn)/((tp+fp)*(tp+fn)*(fn+tn)*(fp+tn))**0.5
    except:
        return 0

def get_result(history, time,  proj, m_name, eval_proj=None, n_samples =-1):
    
    if (n_samples == -1) and (eval_proj==None):
        results = pd.DataFrame(columns = ['Dataset', 'Model', 'Time (Sec)', 'Accuracy', 'Precision', 'Recall',
                                            'F1-score', 'AUC','MCC','TP', 'TN', 'FP',  'FN'])
        pres = np.round(history[2] * 100,2)
        rec = np.round(history[3] * 100,2)

        new_row = pd.DataFrame({'Dataset'    : [proj[0:-4]],
                            'Model' : [m_name],
                            'Time (Sec)'       : [time],
                            'Accuracy'   : [np.round(history[1] * 100,2)],
                            'Precision'  : [pres],
                            'Recall'     : [rec], 
                            'F1-score'   : [np.round( 2* ((pres*rec)/(pres+rec)),2)],
                            'AUC'        : [np.round(history[4] * 100,2)],
                            'MCC'        : [np.round(mcc(history[5],history[7],history[6],history[8]) * 100,2)],
                            'TP'         : [history[5]],
                            'TN'         : [history[6]],
                            'FP'         : [history[7]],
                            'FN'         : [history[8]]})
        results = pd.concat([results, new_row], ignore_index=True)
    elif n_samples != -1:
        results = pd.DataFrame  (columns = ['Dataset', 'Model', 'Size', 'Time (Sec)', 'Accuracy', 'Precision', 'Recall',
                                            'F1-score', 'AUC','MCC','TP', 'TN', 'FP',  'FN'])
        pres = np.round(history[2] * 100,2)
        rec = np.round(history[3] * 100,2)

        new_row = pd.DataFrame({'Dataset'    : [proj[0:-4]],
                            'Model' : [m_name],
                            'Size'  : [n_samples],
                            'Time (Sec)'       : [time],
                            'Accuracy'   : [np.round(history[1] * 100,2)],
                            'Precision'  : [pres],
                            'Recall'     : [rec], 
                            'F1-score'   : [np.round( 2* ((pres*rec)/(pres+rec)),2)],
                            'AUC'        : [np.round(history[4] * 100,2)],
                            'MCC'        : [np.round(mcc(history[5],history[7],history[6],history[8]) * 100,2)],
                            'TP'         : [history[5]],
                            'TN'         : [history[6]],
                            'FP'         : [history[7]],
                            'FN'         : [history[8]],
                            })
        results = pd.concat([results, new_row], ignore_index=True)
    elif eval_proj != None:
        results = pd.DataFrame(columns = ['Training dataset', 'Evaluation dataset', 'Model', 'Time (Sec)', 'Accuracy', 'Precision', 'Recall',
                                            'F1-score', 'AUC','MCC','TP', 'TN', 'FP',  'FN'])
        pres = np.round(history[2] * 100,2)
        rec = np.round(history[3] * 100,2)

        new_row = pd.DataFrame({'Training dataset'    : [proj[0:-4]],
                            'Evaluation dataset' : [eval_proj[0:-4]],
                            'Model'  : [m_name],
                            'Time (Sec)'       : [time],
                            'Accuracy'   : [np.round(history[1] * 100,2)],
                            'Precision'  : [pres],
                            'Recall'     : [rec], 
                            'F1-score'   : [np.round( 2* ((pres*rec)/(pres+rec)),2)],
                            'AUC'        : [np.round(history[4] * 100,2)],
                            'MCC'        : [np.round(mcc(history[5],history[7],history[6],history[8]) * 100,2)],
                            'TP'         : [history[5]],
                            'TN'         : [history[6]],
                            'FP'         : [history[7]],
                            'FN'         : [history[8]],
                            })
        results = pd.concat([results, new_row], ignore_index=True)
    return results


def cross_proj_evaluation(model, m_name, file_path, train_proj, eval_projects):
    results = pd.DataFrame()
    for proj in eval_projects:
        fname = file_path+ proj
        print('-------------- Evaluation on ---------------\n', fname)

        try:
            train_df, _ = load_labeled_dataset(fname, training_ratio=0.99)
            x_train, y_train, _, _ =  get_finetuning_dataset(train_df, train_df)
            _, _, x_test, y_test =  data_preprocessing(x_train, y_train, x_train, y_train, balance=True)

        except  Exception as e: 
            print(e)
            print('Data has less than 6 positive labels: ', fname)
            continue
        start_time = time.time()
        history = model.evaluate(x_test, y_test)
        train_time = round((time.time() - start_time))
        
        fine_tuning_results =  get_result(history, train_time,  train_proj , m_name, eval_proj=proj)
        results = results.append(fine_tuning_results)
    return results

def get_evaluation_projects(code_smell):
    eval_projects = pd.read_csv ('cross-projects.csv', index_col =None, header =0)
    eval_projects = pd.unique(eval_projects[code_smell])
    eval_projects =  [x+'.csv' for x in eval_projects if str(x) != 'nan']
    return eval_projects

if __name__ == "__main__":
    config = Config()
    vec_layer_logs = 'Code/vec_layer_logs/tv_layer.pkl'
    dataset = load_data_from_file('/model/lxj/actionableSmell','train')
    keywords = get_keywords('Java-keywords.txt')
    all_data = pre_processing(dataset)
    X_train, X_valid, X_test ,_,_,_ = split_data(all_data, training_ratio= 0.8, _valid=True)
    # # Get vectorize layer
    vectorize_layer = get_vectorize_layer(keywords+
        X_train.code.values.tolist(),
        config.VOCAB_SIZE,
        config.MAX_LEN, 
        vec_log= vec_layer_logs,
        special_tokens= ['[mask]'],
    )
    # for loops cs>>models>>proj
    not_completed = []

    CODE_SMELLS = ['Data class', 'God class', 'Feature envy', 'Long method']

    DATA_PATH = 'Data/Fine-tuning/Code smells/'
    SAVE_PATH = '/model/lxj/CoRT/Code/Results/' + getToday() + '_' + getTime()

    # Set training parameters 
    optimizer = keras.optimizers.Adam()
    loss="binary_crossentropy"
    metrics=['accuracy', tf.metrics.Precision(), tf.metrics.Recall(),  tf.metrics.AUC(),
            tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), 
            tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]
    epochs=5
    MODEL_NAME = {'ANN':'logs/hparam_tuning/DL 16-11-2022/run-18/dl_model.h5',
              'CNN':'logs/hparam_tuning/CNN 06-12-2022/run-20/cnn_model.h5', 
              'LSTM': 'logs/hparam_tuning/LSTM 21-11-2022/run-19/lstm_model.h5',
              'Transformer':'/model/lxj/CoRT/Code/logs/20250606-055753_Transformer/transformer_model.h5',
              }

    #CODE_SMELLS = ['Long method']

    m_name = 'Transformer' #item[0]
    m_best = MODEL_NAME[m_name] #item[1]
    print(m_name, m_best)

    # Load pretrained model
    model_path = m_best #"transformer_model.h5"
    loaded_model = get_model(m_name)
    loaded_model.load_weights(model_path)
    pretrained_model = tf.keras.Model(loaded_model.input, loaded_model.get_layer(index=len(loaded_model.layers)-1).output)

    # for code_smell in CODE_SMELLS:
    #     PROJECTS = os.listdir(DATA_PATH + code_smell +'/') 

    #     for proj in PROJECTS:

    #         fname = DATA_PATH + code_smell +'/' + proj
    #         print(fname)

    try:

        test_data = load_data_from_file('/model/lxj/actionableSmell', 'test')
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        X_train,  X_test, y_train, y_test= split_data(test_data, training_ratio= 0.7)
        train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
        x_train, y_train, x_test, y_test =  get_finetuning_dataset(train_df, test_df)
        x_train, y_train, x_test, y_test =  data_preprocessing(x_train, y_train, x_test, y_test, balance=True)
    except:
        print('Data has less than 6 positive labels: ')
        # continue
    
    try:
        # fine tuning     
        history_1, history_2, train_time_1, train_time_2, _ = fine_tune(pretrained_model, x_train, y_train, x_test, y_test)
        print(history_2)
        # Save results
        fine_tuning_results =  get_result(history_2, train_time_2,  'allproject', m_name)
        feature_based_results = get_result(history_1, train_time_1,  'allproject', m_name)

        Save_Excel(fine_tuning_results,  SAVE_PATH + '_FineTuningResults_SMOTE_'+m_name+'.csv')
        Save_Excel(feature_based_results,  SAVE_PATH + '_FeatureBasedResults_SMOTE_'+m_name+'.csv')
        # Save_Excel (fine_tuning_results,  SAVE_PATH + '_' + code_smell+ '_FineTuningResults_SMOTE_'+m_name+'.csv')
        # Save_Excel (feature_based_results,  SAVE_PATH +'_' + code_smell+ '_FeatureBasedResults_SMOTE_'+m_name+'.csv')
    except Exception as e: 
        print(e)
        # not_completed.append((code_smell, proj))
    print(not_completed)