import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization, LSTM, Conv1D
from keras.models import Sequential
from dataclasses import dataclass
import tensorflow_datasets as tfds 
from sklearn.model_selection import train_test_split
import pickle

from keras import callbacks

# HP tuning 
from tensorboard.plugins.hparams import api as hp
import optuna

import pandas as pd
import numpy as np
from statistics import mean

from datetime import datetime
import time
import glob
import re
from pprint import pprint

# visualization 
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
print("tf version:",tf.__version__)
print("keras version:", keras.__version__)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)

# ## Set-up Hyperparameters
@dataclass
class Config:
    # preprocessing params
    MAX_LEN = 256       #512 in bert
    VOCAB_SIZE = 30000  # 30000 in bert     
    # mask  15% in bert
    KW_MASK_RATE = 0.25
    RAND_MASK_RATE = 0.10
    # model params
    EMBED_DIM = 128
    FF_DIM = 2048  # number of units 768 in bert
    NUM_LAYERS = 6 # 3, 6, 12 in bert
    LR = 0.001   # 0.0001 in bert
    DROPOUT = 0.3 # 0.1 in bert
    BATCH_SIZE = 128     #256 in bert
    # transformer 
    NUM_HEAD = 8 # 8,16 in atten, 12 in bert
    # parallelism
    BUFFER_SIZE = tf.data.experimental.AUTOTUNE
    CODE_VERSION = 'Not defined'
    def save(self, fname):
      dict1 = {'MAX_LEN' : self.MAX_LEN, 
               'VOCAB_SIZE' : self.VOCAB_SIZE,
               'EMBED_DIM' : self.EMBED_DIM,
               'BATCH_SIZE' : self.BATCH_SIZE,
               'KW_MASK_RATE' : self.KW_MASK_RATE,
               'RAND_MASK_RATE' : self.RAND_MASK_RATE,
               'LR' : self.LR,
               'DROPOUT' : self.DROPOUT,
               'FF_DIM' : self.FF_DIM,
               'NUM_LAYERS' : self.NUM_LAYERS,
               'NUM_HEAD' : self.NUM_HEAD,
               'CODE_VERSION' : self.CODE_VERSION,   
               }
      file1 = open(fname, "w") 
      str1 = repr(dict1)
      file1.write("config = " + str1 + "\n")
      file1.close()
config = Config()

# ## Load the data



def load_data_from_file(fpath, flag):
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
                    df = df[df['dataset_split'] == flag]
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
    return  all_data 


def get_keywords(fpath):
  keywords_fname = '/model/lxj/CoRT/Java-keyword.txt'
  keywords = [line.rstrip() for line in open(keywords_fname)]
  return keywords


# ## Preprocessing
# ### Filtering
def data_filter(data):
  fdata = pd.DataFrame([x for a, x in data.iterrows() if len(x[0].split(' ')) > 50]).reset_index(drop =True)
  return fdata


# ### Truncation
def truncateText(text ,max_len):
    cursor = 0  
    end = max_len
    lst = []
    text = text.split(' ')
    while (end < len(text)):
        substr = text[cursor : end]
        lst.append(' '.join(substr))
        cursor =  end        
        end = (cursor+max_len) if (cursor+max_len<len(text)) else len(text)        
    if (end-cursor) > (max_len/2): 
        lst.append(' '.join(text[cursor : end]))

    return lst


def truncation(data):
    fdata = pd.DataFrame()
    listdict = []

    for a, text in data.iterrows():
        t_text = truncateText(text[0], config.MAX_LEN) 
        listdict.append(pd.DataFrame(data = {'code':t_text, 'label': text[1]}))
        fdata = pd.concat(list(listdict), axis=0, ignore_index =True) 

    return fdata


# ### Tokenization


def get_vectorize_layer(texts, vocab_size, max_seq, vec_log='Code/vec_layer_logsvec_layer_logs/tv_layer.pkl', special_tokens=["[mask]"]):
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
        
        try:
            # 直接使用文本列表而不是数据集
            vectorize_layer.adapt(texts)
            # 创建数据集并添加错误处理
            # text_ds = tf.data.Dataset.from_tensor_slices(texts)
            # text_ds = text_ds.prefetch(tf.data.AUTOTUNE)
            # vectorize_layer.adapt(text_ds)
            # 处理词表
            vocab = vectorize_layer.get_vocabulary()
            vocab = vocab[2 : vocab_size - len(special_tokens)] + special_tokens
            vectorize_layer.set_vocabulary(vocab)
            
            # 保存
            layer_configs = vectorize_layer.get_config()
            layer_weights = vectorize_layer.get_weights()
            
            # pickle.dump({
            #     'config': layer_configs,
            #     'weights': layer_weights
            # }, open(vec_log, "wb"))
            
            print(f"词表大小: {len(vocab)}")
            # print(f"保存文件大小: {os.path.getsize(vec_log)} bytes")
            
        except Exception as e:
            print(f"处理数据时出错: {str(e)}")
            raise

    return vectorize_layer



def encode(texts):
    texts = tf.expand_dims(texts, -1)  
    encoded_texts = tf.squeeze(vectorize_layer(texts))
    return encoded_texts


# ## Proxytask
def get_keywords_pos(text, keywords_dic):
    keywors_pos_list = [] 

    for i in range(text.shape[0]):
        temp = []
        # get the position of keywords
        for a in range(0, len(keywords)):
            if keywords[a] in keywords_dic:  # 确保关键词在字典中
                pos = np.flatnonzero(text[i] == keywords_dic[keywords[a]])
                if len(pos) > 0:  # 只添加非空数组
                    temp.append(pos)

        # 合并所有位置
        if temp:  # 如果有找到任何位置
            keywors_pos_list.append(np.concatenate(temp))
        else:
            keywors_pos_list.append(np.array([]))  # 如果没有找到任何位置，添加空数组

    return keywors_pos_list

def get_masked_input_and_labels(encoded_txt):
    encoded_texts = []
    for row in encoded_txt:
        encoded_texts.append(row.tolist())

    encoded_texts = np.array(encoded_texts, int)
    texts_masked = np.copy(encoded_texts)
    label_list = np.copy(encoded_texts)

    # Num of instances
    _m = encoded_texts.shape[0]

    # Get keywords positions list 
    keywors_pos_list = get_keywords_pos(encoded_texts, keywords_dic)
    
    # 修改这部分代码
    inp_mask = []
    for i in range(_m):
        # 确保keywors_pos_list[i]是numpy数组
        if isinstance(keywors_pos_list[i], list):
            keywors_pos_list[i] = np.array(keywors_pos_list[i])
        
        # 创建随机掩码
        if len(keywors_pos_list[i]) > 0:  # 确保有位置可以掩码
            rand = (np.random.rand(len(keywors_pos_list[i])) < config.KW_MASK_RATE)
            inp_mask.append(rand)
        else:
            inp_mask.append(np.array([]))

    selection = []
    for i in range(_m):
        if len(inp_mask[i]) > 0:  # 确保有掩码
            selection.append(keywors_pos_list[i][inp_mask[i]])
        else:
            selection.append(np.array([]))

    # Apply the mask by replacing the selected tokens with MASK_TOKEN_ID
    for i in range(_m):
        if len(selection[i]) > 0:  # 确保有选择的位置
            texts_masked[i, selection[i]] = MASK_TOKEN_ID

    # Randomly mask 10% of the text
    inp_mask_2 = (np.random.rand(*encoded_texts.shape) < config.RAND_MASK_RATE)
    # Do not mask special tokens 
    inp_mask_2[encoded_texts <= 2] = False
    # Apply the mask
    texts_masked[inp_mask_2] = MASK_TOKEN_ID 
    
    # Make keywords mask to true
    inp_mask_2[texts_masked == MASK_TOKEN_ID] = True
    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    # Set labels for masked tokens
    labels[inp_mask_2] = texts_masked[inp_mask_2]

    texts_masked = np.array(texts_masked, int)
    labels = np.array(labels, int)

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    return texts_masked, label_list, sample_weights

# ## Prepare datasets 
# ### Pre-training DS


def get_pretraining_dataset(X_train, X_valid, X_test): # training %

  trainig_dataset_path = 'Code/Preprocessed datasets/pre-training/training/'
  validation_dataset_path = 'Code/Preprocessed datasets/pre-training/validation/'
  testing_dataset_path = 'Code/Preprocessed datasets/pre-training/testing/'

  try:
    mlm_ds = tf.data.experimental.load(trainig_dataset_path)
    valid_ds = tf.data.experimental.load(validation_dataset_path)
    test_ds = tf.data.experimental.load(testing_dataset_path)
    print('preprocessed datasets already existed')

  except:
    print('preprocessed datasets not found')

    # Prepare data for pre-trained model 
    x_all_raw = tf.data.Dataset.from_tensor_slices(
                tf.cast(X_train.code.values, tf.string)) 

    x_all_code = x_all_raw.map(encode, 
                  num_parallel_calls=config.BUFFER_SIZE)

    x_valid_raw = tf.data.Dataset.from_tensor_slices(
                tf.cast(X_valid.code.values, tf.string)) 

    x_valid_code = x_valid_raw.map(encode, 
                  num_parallel_calls=config.BUFFER_SIZE)
    
    x_test_raw = tf.data.Dataset.from_tensor_slices(
                tf.cast(X_test.code.values, tf.string)) 

    x_test_code = x_test_raw.map(encode, 
                  num_parallel_calls=config.BUFFER_SIZE)
    
    # Applying the Proxytask on training ds
    a = tfds.as_numpy(x_all_code)
    x_masked_train, y_masked_labels, sample_weights = get_masked_input_and_labels(a)

    mlm_ds = tf.data.Dataset.from_tensor_slices(
      ( x_masked_train, y_masked_labels, sample_weights)
      )

    mlm_ds = mlm_ds.shuffle(1000).batch(config.BATCH_SIZE).prefetch(config.BUFFER_SIZE)

    # Applying the Proxytask on validation and testing datasets
    x_masked_val, y_masked_labels_val, sample_weights_val = get_masked_input_and_labels(tfds.as_numpy(x_valid_code))

    valid_ds = tf.data.Dataset.from_tensor_slices(
      (x_masked_val, y_masked_labels_val, sample_weights_val)
      )

    valid_ds = valid_ds.shuffle(1000).batch(config.BATCH_SIZE).prefetch(config.BUFFER_SIZE)
    
    x_masked_test, y_masked_labels_test, sample_weights_test = get_masked_input_and_labels(tfds.as_numpy(x_test_code))

    test_ds = tf.data.Dataset.from_tensor_slices(
      (x_masked_test, y_masked_labels_test, sample_weights_test)
      )
    
    test_ds = test_ds.shuffle(1000).batch(config.BATCH_SIZE).prefetch(config.BUFFER_SIZE)
    
    # Save datasets
    # tf.data.experimental.save(mlm_ds, trainig_dataset_path)
    # tf.data.experimental.save(valid_ds, validation_dataset_path)
    # tf.data.experimental.save(test_ds, testing_dataset_path)
  return mlm_ds, valid_ds, test_ds, #x_masked_val, y_masked_labels_val


# ## Create DL model 


class TextGenerator(keras.callbacks.Callback):
    def __init__(self, sample_tokens, top_k=5):
        self.sample_tokens = sample_tokens
        self.k = top_k

    def decode(self, tokens):
        temp = []
        for t in tokens:
           if t != 0 and t < len(id2token):
             temp.append(id2token[t])
        return " ".join(temp)

    def convert_ids_to_tokens(self, id):
        if id >= len(id2token):
          return ''
        else:
          return id2token[id]

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.sample_tokens)

        masked_index = np.where(self.sample_tokens == MASK_TOKEN_ID)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index]

        top_indices = mask_prediction[0].argsort()[-self.k :][::-1]
        values = mask_prediction[0][top_indices]

        for i in range(len(top_indices)):
            p = top_indices[i]
            v = values[i]
            tokens = np.copy(sample_tokens[0])
            tokens[masked_index[0]] = p
            #print(tokens)
            result = {
                "input_text": self.decode(sample_tokens[0].numpy()),
                "prediction": self.decode(tokens),
                "probability": v,
                "predicted mask token": self.convert_ids_to_tokens(p),
            }
            pprint(result)

def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


# ### ANN model

def create_dl_model():
    inputs = layers.Input((config.MAX_LEN,), dtype=tf.int64)
    #x_in = layers.Dense(256)(inputs) 
    embeddings = layers.Embedding(config.VOCAB_SIZE, config.EMBED_DIM, name="embedding" )(inputs)
    
    L4 = embeddings

    for i in range(config.NUM_LAYERS):
      L1 = layers.Dense(config.FF_DIM, activation="relu")(L4) 
      L3 = layers.Dropout(rate = config.DROPOUT, )(L1) 

    mlm_output = layers.Dense(config.VOCAB_SIZE, name="mlm_cls", activation="softmax")(L3)

    mlm_model = keras.Model(inputs, mlm_output, name="dl_model")  

    return mlm_model


# ### CNN model

def create_cnn_model():

    inputs = layers.Input((config.MAX_LEN,), dtype=tf.int64)
    embeddings = layers.Embedding(config.VOCAB_SIZE, config.EMBED_DIM, name="embedding" )(inputs)

    L2 = embeddings

    for i in range(config.NUM_LAYERS):
      L1 = Conv1D(filters = config.FF_DIM, kernel_size =3, padding = 'SAME')(L2)
      L2 = layers.ReLU()(L1) 
    #L3 = layers.MaxPool1D( padding = 'SAME')(L2)

    #L4 = layers.Flatten()(L3)
    mlm_output = layers.Dense(config.VOCAB_SIZE, name="mlm_cls", activation="softmax")(L2)

    mlm_model = keras.Model(inputs, mlm_output, name="cnn_model")

    #optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    #mlm_model.compile(optimizer=optimizer)

    return mlm_model


# ### LSTM model

def create_lstm_model():
    inputs = layers.Input((config.MAX_LEN,), dtype=tf.int64)
    #x_in = layers.Dense(256)(inputs) 
    embeddings = layers.Embedding(config.VOCAB_SIZE, config.EMBED_DIM, name="embedding" )(inputs)

    L2 = embeddings

    for i in range(config.NUM_LAYERS):
      L1 = LSTM(config.FF_DIM, return_sequences= True)(L2)

      L2 = layers.Dropout(rate = config.DROPOUT, )(L1) 

    mlm_output = layers.Dense(config.VOCAB_SIZE, name="mlm_cls", activation="softmax")(L2)

    mlm_model = keras.Model(inputs, mlm_output, name="lstm_model")

    #optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    #mlm_model.compile(optimizer=optimizer)
    
    return mlm_model


# ### Transformer model

def transformer_model():

    inputs = layers.Input((config.MAX_LEN,), dtype=tf.int64)
    word_embeddings = layers.Embedding(config.VOCAB_SIZE, config.EMBED_DIM, name="word_embedding")(inputs)
    
    position_embeddings = layers.Embedding(
        input_dim=config.MAX_LEN,
        output_dim=config.EMBED_DIM,
        weights=[get_pos_encoding_matrix(config.MAX_LEN, config.EMBED_DIM)],
        name="position_embedding",)(tf.range(start=0, limit=config.MAX_LEN, delta=1))
    
    embeddings = word_embeddings + position_embeddings

    encoder_output = embeddings

    for i in range(config.NUM_LAYERS):
      encoder_output = transformer_Encoder(encoder_output, encoder_output, encoder_output, i)
      decoder_output = transformer_Decoder(embeddings, embeddings, embeddings, encoder_output, i)

    mlm_output = layers.Dense(config.VOCAB_SIZE, name="mlm_cls", activation="softmax")(decoder_output)
    
    mlm_model = keras.Model(inputs, mlm_output, name="transformer_model")


    return mlm_model


def transformer_Encoder (query, key, value, i):

    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(num_heads=config.NUM_HEAD, key_dim=config.EMBED_DIM // config.NUM_HEAD,)(query, key, value)
    
    attention_output = layers.Dropout(config.DROPOUT)(attention_output)
    
    attention_output = layers.LayerNormalization(epsilon=1e-6,)(query + attention_output)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            layers.Dense(config.FF_DIM, activation="relu"),
            layers.Dense(config.EMBED_DIM),
        ], name="encoder_{}_ffn".format(i),)
    
    ffn_output = ffn(attention_output)
    
    ffn_output = layers.Dropout(config.DROPOUT)(ffn_output)
    
    sequence_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
    
    return sequence_output

def transformer_Decoder (query, key, value, enc_output, i):

    # Multi headed self-attention 1
    attention_output = layers.MultiHeadAttention(num_heads=config.NUM_HEAD, key_dim=config.EMBED_DIM // config.NUM_HEAD,)(query, key, value)
    
    attention_output = layers.Dropout(config.DROPOUT)(attention_output)
    
    attention_output = layers.LayerNormalization(epsilon=1e-6,)(query + attention_output)

    # Multi headed self-attention 2
    attention_output2 = layers.MultiHeadAttention(num_heads=config.NUM_HEAD, key_dim=config.EMBED_DIM // config.NUM_HEAD,)(attention_output, enc_output, enc_output)
    
    attention_output2 = layers.Dropout(config.DROPOUT)(attention_output2)
    
    attention_output2 = layers.LayerNormalization(epsilon=1e-6,)(attention_output + attention_output2)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            layers.Dense(config.FF_DIM, activation="relu"),
            layers.Dense(config.EMBED_DIM),
        ], name="decoder_{}_ffn".format(i),)
    
    ffn_output = ffn(attention_output2)
    
    ffn_output = layers.Dropout(config.DROPOUT)(ffn_output)
    
    sequence_output = layers.LayerNormalization(epsilon=1e-6)(attention_output2 + ffn_output)
    
    return sequence_output


# ## Pre-training

def pre_train_model(train_ds, valid_ds, epochs, model_name, callbacks, config, logs):
    tf.get_logger().setLevel('ERROR')
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    with strategy.scope():
        model = get_model(model_name)
    
    History = model.fit(train_ds, epochs=epochs, validation_data = valid_ds, callbacks=callbacks)

    eval_results = model.evaluate(valid_ds, verbose ='0', batch_size=config.BATCH_SIZE)
    loss = eval_results[0] 
    accuracy = eval_results[1] 
    
    model.save(logs + "/" + model.name + ".h5")
    config.save(logs + "/" + model.name + "_config.txt")

    return accuracy, loss, model

def get_model(model_name):
    if model_name == 'DL': model = create_dl_model()
    if model_name == 'LSTM': model = create_lstm_model()
    if model_name == 'CNN': model = create_cnn_model()
    if model_name == 'Transformer': model = transformer_model()

    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    loss =  keras.losses.SparseCategoricalCrossentropy()

    model.compile(
            optimizer= optimizer,
            loss='sparse_categorical_crossentropy',
            weighted_metrics=["sparse_categorical_accuracy"], #jit_compile=True,
    )

    return model

def parallel_truncation(data):
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    def process_chunk(chunk):
        return chunk.apply(lambda x: truncateText(x['code'], config.MAX_LEN), axis=1)
    # 将数据分成多个块
    num_chunks = min(os.cpu_count(), len(data))
    chunks = np.array_split(data, num_chunks)
    
    # 并行处理
    with ThreadPoolExecutor(max_workers=num_chunks) as executor:
        results = list(executor.map(process_chunk, chunks))
    
    # 合并结果
    return pd.concat(results, ignore_index=True)

# 5. 优化truncateText函数
def truncateText(text, max_len):
    # 使用更高效的分割方法
    words = text.split()
    if len(words) <= max_len:
        return [text]
    
    # 预分配列表大小
    result = []
    cursor = 0
    
    # 使用列表推导式
    while cursor < len(words):
        end = min(cursor + max_len, len(words))
        if end - cursor > max_len/2:
            result.append(' '.join(words[cursor:end]))
        cursor = end
    
    return result

def pre_processing(data, _filtering = False, _truncation= False):
    new_data = data[['code', 'label']].copy(deep=False)
    new_data['code'] = new_data['code'].astype(str, copy=False)
    

    if _filtering:
        new_data = data_filter(new_data)

    if _truncation: 
        new_data = parallel_truncation(new_data)

    return new_data


def get_callbacks(sample_tokens, log_dir):
    #sample_tokens = vectorize_layer(["[mask] processJarAttrs ( ) throws BuildException" ])
    #sample_tokens = vectorize_layer(["[mask] index = 1" ])
    generator_callback = TextGenerator(sample_tokens.numpy())
    
    tboard_callback = keras.callbacks.TensorBoard(log_dir = log_dir,
                                                    histogram_freq = 1,
                                                    profile_batch = '20,31')
    
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                            mode ="min", patience = 5, 
                                            restore_best_weights = True, verbose=1)
    #Hparam_callback = hp.KerasCallback(logdir, hparams)

    return [generator_callback, tboard_callback, earlystopping]


def split_data(data, training_ratio= 0.8, _valid=False):
    X = data.drop(columns = ['label']).copy()
    y = data['label']

    X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=training_ratio)

    if  _valid == False:
        return X_train, X_rem, y_train, y_rem
    
    else:
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)
        return  X_train, X_valid, X_test, y_train, y_valid, y_test


# ## ----- Main
if __name__ == "__main__":
    dataset = load_data_from_file('/model/lxj/actionableSmell')
    keywords = get_keywords('Java-keywords.txt')
    all_data = pre_processing(dataset)
    X_train, X_valid, X_test ,_,_,_ = split_data(all_data, training_ratio= 0.8, _valid=True)
    #!rm -rf ./vec_layer_logs
    # Get vec layer for tokenization
    vec_layer_logs = 'Code/vec_layer_logs/tv_layer.pkl'

    vectorize_layer = get_vectorize_layer(keywords+
        X_train.code.values.tolist(),
        config.VOCAB_SIZE,
        config.MAX_LEN, 
        vec_log= vec_layer_logs,
        special_tokens= ['[mask]'],
    )

    # Get mask token id for masked language model
    # MASK_TOKEN_ID = vectorize_layer(["[mask]"]).numpy()[0][0]
    # print('--------------', MASK_TOKEN_ID)
    # # Get dictionary of keywords
    # keywords_dic = {keyword: encode(keyword).numpy().tolist()[0] for keyword in keywords} #keywords_dic[keywords[1]]
    # # Get dictionary of tokens ids
    # id2token = dict(enumerate(vectorize_layer.get_vocabulary()))  #id2token[1]
    # token2id = {y: x for x, y in id2token.items()}  #token2id['[UNK]']
    # mlm_train_ds, mlm_valid_ds, mlm_test_ds= get_pretraining_dataset(X_train, X_valid, X_test)
    # config.CODE_VERSION = 'v7'
    # MODEL_NAME = 'Transformer' # DL, LSTM, CNN, Transformer
    # logs = "/model/lxj/CoRT/Code/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")+ '_' + MODEL_NAME
    # sample_tokens = vectorize_layer(["[mask] index = 1" ])
    # callbacks= get_callbacks(sample_tokens, logs) 
    # model = get_model(MODEL_NAME)
    # _,_, model= pre_train_model(mlm_train_ds, mlm_valid_ds, 5, MODEL_NAME, callbacks, config, logs)


# # ## Test the model 
# def decode(tokens):
#     temp = []
#     for t in tokens:
#         if t != 0 and t < len(id2token):
#             temp.append(id2token[t])
#     return " ".join(temp)

# def convert_ids_to_tokens(id):
#     if id >= len(id2token):
#         return ''
#     else:
#         return id2token[id]

# def get_predictions(prediction, k):
#   masked_index = np.where(test_tokens == MASK_TOKEN_ID)
#   masked_index = masked_index[1]
#   mask_prediction = prediction[0][masked_index]

#   top_indices = mask_prediction[0].argsort()[-k:][::-1]
#   values = mask_prediction[0][top_indices]

#   for i in range(len(top_indices)):
#       p = top_indices[i]
#       v = values[i]
#       tokens = np.copy(test_tokens[0])
#       tokens[masked_index[0]] = p
#               #print(tokens)
#       result = {
#           "input_text": decode(test_tokens[0].numpy()),
#           "prediction": decode(tokens),
#           "probability": v,
#           "predicted mask token": convert_ids_to_tokens(p),
#       }
#       pprint(result)

# test_tokens = vectorize_layer(["[mask] text = ' some text '" ])
# prediction = model.predict(test_tokens)
# top_k =3

# get_predictions(prediction, top_k)


# # ## Hyperparameter Tuning 

# # Clear any logs from previous runs
# #!rm -rf ./logs/hparam_tuning

# def check_existed_params(df, params):

#   if MODEL_NAME == 'Transformer':
#       existed_params = df.loc[(df['state'] == 'COMPLETE') & (df['params_Learning rate'] == params['Learning rate']) & 
#                           (df['params_Dropout rate'] == params['Dropout rate'])& 
#                           (df['params_Batch size'] == params['Batch size'])& 
#                           (df['params_Num. layers'] == params['Num. layers'])& 
#                           (df['params_Layers Dim'] == params['Layers Dim'])& 
#                           (df['params_Emmbedding Dim.'] == params['Emmbedding Dim.'])& 
#                           (df['params_Num. heads'] == params['Num. heads'])] 
#   else:
#       existed_params = df.loc[(df['state'] == 'COMPLETE') &  (df['params_Learning rate'] == params['Learning rate']) & 
#                           (df['params_Dropout rate'] == params['Dropout rate'])& 
#                           (df['params_Batch size'] == params['Batch size'])& 
#                           (df['params_Num. layers'] == params['Num. layers'])& 
#                           (df['params_Layers Dim'] == params['Layers Dim'])& 
#                           (df['params_Emmbedding Dim.'] == params['Emmbedding Dim.'])
#                           ] 
  
#   print('====== existed_params ', existed_params)
  
#   return not existed_params.empty


# def run(run_dir, hparams):
    
#     log_dir =  run_dir.replace('hparam_tuning/','')
#     tboard_callback = keras.callbacks.TensorBoard(log_dir = log_dir,
#                                                   histogram_freq = 1,
#                                                   profile_batch = '20,31')
    
#     with tf.summary.create_file_writer(run_dir).as_default():
#         hp.hparams(hparams)  # record the values used in this trial

#         start = time.time()
#         accuracy, loss, _ = pre_train_model(mlm_train_ds, mlm_valid_ds, 5, MODEL_NAME, tboard_callback, config, run_dir)
#         training_time = round((time.time() - start)/60)

#         tf.summary.scalar('Accuracy', accuracy, step=1)
#         tf.summary.scalar('Loss', loss, step=1)
#         tf.summary.scalar('Time (Min.)', training_time, step=1)

#         return accuracy, loss



# def objective(trial):

#     #model_name = trial.suggest_categorical('model', ['DL', 'CNN', 'LSTM', 'Transformer'])
#     lr = trial.suggest_categorical('Learning rate', [0.0001, 0.001, 0.01])
#     dropout_rate = trial.suggest_categorical('Dropout rate', [0.1, 0.2, 0.3])
#     batch_size = trial.suggest_categorical('Batch size', [64, 128, 256])
#     num_layers = trial.suggest_categorical('Num. layers', [3, 6, 12])
#     num_units = trial.suggest_categorical('Layers Dim', [512, 1024, 2048])
#     emb_dim = trial.suggest_categorical('Emmbedding Dim.', [128, 256, 512])
    
    
#     hparams = {
#                   #'model': model_name,
#                   'Learning rate': lr,
#                   'Dropout rate': dropout_rate,
#                   'Batch size': batch_size,
#                   'Num. layers': num_layers,
#                   'Layers Dim': num_units,
#                   'Emmbedding Dim.': emb_dim,

#               }

#     config.BATCH_SIZE = batch_size
#     config.NUM_LAYERS = num_layers
#     config.FF_DIM = num_units
#     config.EMBED_DIM = emb_dim
#     config.LR = lr

#     if MODEL_NAME == 'Transformer':
#       num_heads = trial.suggest_categorical('Num. heads', [8, 16])
#       hparams['Num. heads'] = num_heads
#       config.NUM_HEAD = num_heads

        
#     df = study.trials_dataframe()
#     try:
#       if check_existed_params(df, hparams): #not existed_params.empty:
#         print('================== param already existed')
#         return 100 #study.trials_dataframe().value[a['number']][1]
#     except:
#       pass

#     run_name = "run-%d" % trial.number
#     accuracy, loss = run(hp_logs + '/' + run_name, hparams)
#     trial.set_user_attr("accuracy", accuracy)


#     return loss

# config.CODE_VERSION = 'DL_model_v7_HPOpt_Parall_optuna_eval'
# MODEL_NAME = 'Transformer' # DL, LSTM, CNN, Transformer
# hp_logs = 'logs/hparam_tuning/' + MODEL_NAME+ datetime.now().strftime(" %d-%m-%Y")
# optuna_logs = "logs/optuna/" + MODEL_NAME+ datetime.now().strftime(" %d-%m-%Y")

# study_name = MODEL_NAME # Unique identifier of the study.
# storage_name = "sqlite:///{}.db".format(study_name)


# study = optuna.create_study(study_name= MODEL_NAME, 
#                             direction='minimize', 
#                             storage =storage_name, 
#                             pruner=optuna.pruners.MedianPruner(),
#                             load_if_exists =True) #maximize val accuracy or min val loss

# study.optimize(objective, n_trials=25)

# pickle.dump(study, open(optuna_logs +".pkl", "wb"))
# study.trials_dataframe().to_csv(optuna_logs + '.csv')
# trial = study.best_trial
# print('Loss: {}'.format(trial.value))
# print("Best hyperparameters: {}".format(trial.params))