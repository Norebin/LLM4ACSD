import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from logic.classifiers.multi_label_classifier import MultiLabelClassifier

# 设置随机种子以保证可重现性
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载数据
X = np.load('data/triple/actionableSmell_graphcodebert_pooler_output.npy')
labels_df = pd.read_csv('data/labels.csv', header=0, sep=';')
y = labels_df.values

# 确保X和y的样本数量匹配
min_samples = min(X.shape[0], y.shape[0])
X = X[:min_samples]
y = y[:min_samples]

print(f"特征向量形状: {X.shape}")
print(f"标签形状: {y.shape}")

# 转换标签为one-hot格式（由于是二分类，我们只需要一列）
y = y.reshape(-1, 1)

# 转换为张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 训练参数
num_epochs = 200
batch_size = 512
learning_rate = 0.0005
num_folds = 5

# 输入和输出维度
input_size = X.shape[1]  # 768
output_size = 1  # 二分类问题

# 初始化K折交叉验证
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# 存储每折的指标
all_accuracy = []
all_precision = []
all_recall = []
all_f1 = []
all_auc = []
all_mcc = []

fold = 1
# 进行K折交叉验证
for train_idx, test_idx in kf.split(X_tensor):
    print(f"\n开始第 {fold}/{num_folds} 折训练")
    fold += 1
    
    # 准备训练和测试数据
    X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型并移动到GPU
    model = MultiLabelClassifier(input_size, output_size).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            # 将数据移动到GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 打印每十个epoch的训练损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}')
    
    # 在测试集上评估模型
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # 将数据移动到GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            # 将预测转换为二进制值（阈值0.5）
            predicted = (outputs > 0.5).float()
            
            # 将结果移回CPU进行评估
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())
    
    # 转换为numpy数组以便计算指标
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    # 计算各种评估指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = 0
    
    # MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(y_true.flatten(), y_pred.flatten())
    
    # 存储结果
    all_accuracy.append(accuracy)
    all_precision.append(precision)
    all_recall.append(recall)
    all_f1.append(f1)
    all_auc.append(auc)
    all_mcc.append(mcc)
    
    # 打印当前折的结果
    print(f"Fold结果:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"MCC: {mcc:.4f}")

# 计算平均指标
print("\n==== K折交叉验证的平均结果 ====")
print(f"Accuracy: {np.mean(all_accuracy):.4f} ± {np.std(all_accuracy):.4f}")
print(f"Precision: {np.mean(all_precision):.4f} ± {np.std(all_precision):.4f}")
print(f"Recall: {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")
print(f"F1 Score: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
print(f"AUC: {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
print(f"MCC: {np.mean(all_mcc):.4f} ± {np.std(all_mcc):.4f}") 