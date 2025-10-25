import numpy as np
import torch
from torch.utils.data import Dataset


class LazyDataset(Dataset):
    def __init__(self, path, device, shape=None, indices=None) -> None:
        self.lazy_tensor: np.memmap = np.load(path, "r")
        if shape is not None:
            self.lazy_tensor = self.lazy_tensor.reshape(shape)

        self.indices = indices if indices is not None else np.arange(len(self.lazy_tensor), dtype=np.int32)
        self.device = device

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_ = self.lazy_tensor[self.indices[index]]
        return torch.tensor(data_, dtype=torch.float32, device=self.device)
