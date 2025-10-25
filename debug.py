import subprocess
import torch
from torch.nn.utils.rnn import pad_sequence

def test_find():
    node_features = [torch.tensor([1.0, 2.0, 3.0]),
                     torch.tensor([4.0, 5.0]),
                     torch.tensor([6.0, 7.0, 8.0, 9.0])]

    padded_tensors = pad_sequence(node_features, batch_first=True)
    padded_tensors_list = [padded_tensors[i] for i in range(padded_tensors.size(0))]
    print(padded_tensors_list)


def test_padd():
    node_features = [torch.tensor([1.0, 2.0, 3.0]),
                     torch.tensor([4.0, 5.0]),
                     torch.tensor([6.0, 7.0, 8.0, 9.0])]

    padded_features_tensor = pad_sequence(node_features, batch_first=True)

    print(padded_features_tensor)


if __name__ == '__main__':
    test_find()