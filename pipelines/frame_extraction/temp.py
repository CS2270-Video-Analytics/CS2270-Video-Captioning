import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MemoryMappedDataset(Dataset):
    def __init__(self, file_path, dtype=np.float32, shape=(100000, 256, 256)):
        self.file_path = file_path
        # Memory-map the file, dtype is important to set for memory efficiency
        self.data = np.memmap(self.file_path, dtype=dtype, mode='r', shape=shape)
    
    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get a specific sample (this could be an image tensor, for example)
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)