#!/usr/bin/env python
# from sklearn.preprocessing import LabelBinarizer
from torch import optim
import copy
from tqdm import tqdm
import numpy as np
from torch import nn
import torch.utils.data as dt
import torch
import pandas as pd


def create_triplet(embeds_path, save_path, train_batch_size, epoch, label_path, type_path):

    device = torch.device("cuda")
    
    # 读取标签文件
    labels_df = pd.read_csv(label_path, sep=';')
    series_data = labels_df.iloc[:, 0].values  # 假设标签在第一列
    
    # 读取类型文件
    types_df = pd.read_csv(type_path)
    types_data = types_df.iloc[:, 0].values  # 假设类型在第一列
    
    loaded_ast_embeddings = np.load(embeds_path)
    loaded_ast_embeddings = loaded_ast_embeddings.reshape(-1, np.prod(loaded_ast_embeddings.shape[1:]))

    X = loaded_ast_embeddings
    y = np.array(series_data)
    t = np.array(types_data)
    
    # 创建组合标签 (label, type)
    combined_labels = np.array([(y[i], t[i]) for i in range(len(y))])

    def decode(coded):
        coded = int(coded)
        n = coded % len(X)
        p = (coded // len(X)) % len(X)
        a = (coded // len(X) // len(X)) % len(X)
        return a,p,n

    def encode(a, p, n):
        return (a * len(X) * len(X)) + (p * len(X)) + n

    X_, combined_labels_ = X, combined_labels
    data_xy = tuple([X_, combined_labels_])

    ind_list = []

    # 获取唯一的组合标签
    unique_combined_labels = np.unique(combined_labels_, axis=0)
    
    for data_class in unique_combined_labels:
        # 查找具有相同标签和类型的样本
        same_class_idx = np.where(np.all(combined_labels_ == data_class, axis=1))[0]
        # 查找具有不同标签或类型的样本
        diff_class_idx = np.where(np.any(combined_labels_ != data_class, axis=1))[0]

        # 根据数据集大小调整生成三元组的数量
        sample_size = min(int(1e5), len(same_class_idx) * len(diff_class_idx))
        
        if len(same_class_idx) >= 2 and len(diff_class_idx) > 0:
            a = np.random.choice(same_class_idx, sample_size, replace=True)
            p = np.random.choice(same_class_idx, sample_size, replace=True)
            n = np.random.choice(diff_class_idx, sample_size, replace=True)
            ind_list.append(encode(a, p, n))

    X_train = np.array(list(set(np.concatenate(ind_list))))

    class TripletDataset(dt.Dataset):
        def __init__(self, X_, triplet_indices, device):
            self.X_tensor = torch.from_numpy(X_).to(device).float()
            self.triplet_indices = triplet_indices
            self.device = device

        def __len__(self):
            return len(self.triplet_indices)

        def __getitem__(self, index):
            a, p, n = decode(self.triplet_indices[index])
            return self.X_tensor[a], self.X_tensor[p], self.X_tensor[n]

    class BaseNetwork(nn.Module):
        def __init__(self, input_size):
            super(BaseNetwork, self).__init__()
            self.linear1 = nn.Linear(input_size, 1000)
            self.bn1 = nn.BatchNorm1d(1000)
            self.linear2 = nn.Linear(1000, 500)
            self.bn2 = nn.BatchNorm1d(500)
            self.linear3 = nn.Linear(500, input_size)
            self.actRelu = nn.LeakyReLU()
            self.drop = nn.Dropout(0.3)

        def forward(self, x):
            out = self.linear1(x)
            out = self.bn1(out)
            out = self.actRelu(out)
            out = self.drop(out)
            out = self.linear2(out)
            out = self.bn2(out)
            out = self.actRelu(out)
            out = self.drop(out)
            out = self.linear3(out)
            return out

    class TripletArchitecture(nn.Module):
        def __init__(self, input_size):
            super(TripletArchitecture, self).__init__()
            self.bn = BaseNetwork(input_size)

        def forward(self, a, p, n):
            a_out = self.bn(a)
            p_out = self.bn(p)
            n_out = self.bn(n)
            return a_out, p_out, n_out

    import math
    triplet_model = TripletArchitecture(X.shape[1]).to(device)

    triplet_optim = optim.Adam(triplet_model.parameters(), lr=3e-5, betas=(0.9, 0.999), weight_decay=1e-5)
    triplet_criterion = nn.TripletMarginLoss(margin=0.3)

    triplet_dataset = TripletDataset(X, X_train, device)

    triplet_loader = dt.DataLoader(triplet_dataset, shuffle=True, batch_size=train_batch_size)

    triplet_model.train()
    triplet_model.to(device)
    best_model = None
    best_loss = 1000
    default_patience = 2
    patience = default_patience

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(triplet_optim, mode='min', factor=0.5, patience=2)

    for i in range(epoch):
        data_iter = tqdm(enumerate(triplet_loader),
                                      desc="EP_%s:%d" % ("test", i),
                                      total=len(triplet_loader),
                                      bar_format="{l_bar}{r_bar}")
        total_loss = 0

        for j, (a_, p_, n_) in data_iter:
            # NEW STUFF START
            a_.requires_grad = False
            p_.requires_grad = False
            n_.requires_grad = False
            minibatch_size = len(a_)
            pos_dists = torch.linalg.norm(a_ - p_, dim=1).to("cpu")
            neg_dists = torch.linalg.norm(a_ - n_, dim=1).to("cpu")
            
            valid_triplets_mask = torch.less(pos_dists, neg_dists)
            
            pos_mask = torch.zeros(minibatch_size, dtype=torch.bool)
            pos_mask[[torch.sort(pos_dists).indices.split(math.floor(minibatch_size * 0.3))[0]]] = True
            
            neg_mask = torch.zeros(minibatch_size, dtype=torch.bool)
            neg_mask[[torch.sort(neg_dists).indices.split(math.floor(minibatch_size * 0.3))[0]]] = True
            
            indices = torch.arange(minibatch_size)[valid_triplets_mask & pos_mask & neg_mask]
            input_size = len(indices)

            a_ = a_[indices]
            p_ = p_[indices]
            n_ = n_[indices]
            a_o, p_o, n_o = triplet_model(a_, p_, n_)
            loss = triplet_criterion(a_o, p_o, n_o)
            loss.backward()
            total_loss += loss.item()
            # data_iter.set_postfix({"Loss": total_loss / (j + 1), "Input Size": input_size})
            data_iter.set_postfix({"Loss": total_loss / (j + 1)})
            triplet_optim.step()

        total_loss /= len(triplet_loader)
        print(f'Epoch [{i + 1}/{100}], Loss: {total_loss:.4f}')
        if total_loss < best_loss - 0.0001:
            best_loss = total_loss
            best_model = copy.deepcopy(triplet_model)
            patience = default_patience
        else:
            patience -= 1
        if patience == 0:
            break

        scheduler.step(total_loss)

    # torch.save(best_model, "Test/test_best_model.pth")

    best_model.eval()
    y_tensor = torch.from_numpy(np.arange(len(y)))
    x_tensor = torch.from_numpy(X).to(device).float()
    triplet_test_loader = dt.DataLoader(dt.TensorDataset(x_tensor, y_tensor), batch_size=256)
    labels = []
    output_embeds = []
    with torch.no_grad():
        for (data_, label_) in tqdm(triplet_test_loader, total=len(triplet_test_loader)):
            labels.append(label_)
            a, _, _ = best_model(data_, data_, data_)
            output_embeds.append(a.cpu().detach().numpy())

    embeds = np.vstack(output_embeds)

    np.save(save_path, embeds)

if __name__ == "__main__":
    create_triplet('/model/lxj/Baseline/TripletLoss/data/actionableSmell_graphcodebert_pooler_output.npy',
                   '/model/lxj/Baseline/TripletLoss/data/triple/actionableSmell_graphcodebert_pooler_output.npy',
                   256,
                   20,
                   '/model/lxj/Baseline/TripletLoss/data/labels.csv',
                   '/model/lxj/Baseline/TripletLoss/data/smells.csv')
