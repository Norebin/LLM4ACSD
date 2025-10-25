from torch.utils.data import Dataset


class SingleTensorDataset(Dataset):
    def __init__(self, tensor) -> None:
        self.tensor = tensor

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, index):
        return self.tensor[index]
