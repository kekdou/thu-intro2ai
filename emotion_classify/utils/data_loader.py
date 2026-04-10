import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

class SentiDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        content = torch.LongTensor(self.x[index])
        label = torch.tensor(self.y[index], dtype=torch.long)
        return content, label
    
def get_dataloader(x_path, y_path, batch_size, shuffle=True):
    dataset = SentiDataset(x_path, y_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=0
    )
    return dataloader

