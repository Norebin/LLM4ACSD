import torch
from torch import nn


class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiLabelClassifier, self).__init__()
        hidden_input_size1 = 256  # input_size // 2
        hidden_input_size2 = 128  # input_size // 4

        self.linear = nn.Linear(input_size, hidden_input_size1)
        self.linear2 = nn.Linear(hidden_input_size1, hidden_input_size2)
        self.linear3 = nn.Linear(hidden_input_size2, hidden_input_size2)
        self.linear4 = nn.Linear(hidden_input_size2, 1)
        self.drop = nn.Dropout(0.1)
        self.act = torch.nn.LeakyReLU()
        self.act2 = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.act(out)
        out = self.linear2(out)
        out = self.act(out)
        # out = self.drop(out)
        out = self.linear3(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.linear4(out)
        out = self.act2(out)
        return out

if __name__ == "__main__":
    # 简单测试模型的正向传播
    input_size = 768
    output_size = 1  # 二分类
    batch_size = 16
    
    # 创建模型实例
    model = MultiLabelClassifier(input_size, output_size)
    
    # 创建随机输入数据
    x = torch.randn(batch_size, input_size)
    
    # 运行前向传播
    output = model(x)
    
    # 打印输出形状
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出示例:\n{output[0]}")
    
    # 检查输出是否为概率值（0-1之间）
    print(f"第一个样本的输出: {output[0].item()}")
    