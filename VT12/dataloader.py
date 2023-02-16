import os
import pickle
from torch.utils.data import Dataset, DataLoader

import config


class LoadData(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __len__(self):
        return len(os.listdir(self.data_path))

    def __getitem__(self, idx):
        name = os.listdir(self.data_path)[idx]
        label_path = os.path.join(self.data_path, name)
        with open(label_path, 'rb') as f:
            data = pickle.load(f)
        return data['images'].to(config.device), \
            data['actor_name'].float().to(config.device), \
            data['director_name'].float().to(config.device), \
            data['description'].to(config.device), \
            data['country'][0].float().to(config.device), \
            data['label'].float().to(config.device)
