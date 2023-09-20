from torch.utils.data import Dataset
import torch
import numpy as np
import os
import random
import pandas as pd

class Dataset_validation(Dataset):

    def __init__(self, cv_id, batch_size=64, train=True):
        self.batch_size = batch_size
        FM_plus_persons_infos = pd.read_csv("prepared_data/Data_validation/persons_infos_validation_FM+.csv").iloc[:,:4]
        FM_minus_persons_infos = pd.read_csv("prepared_data/Data_validation/persons_infos_validation_FM-.csv").iloc[:,:4]
        FM_plus_data_dir = "prepared_data/Data_validation/FM+"
        FM_minus_data_dir = "prepared_data/Data_validation/FM-"
        FM_plus = [os.path.join(FM_plus_data_dir,i) for i in os.listdir(FM_plus_data_dir)]
        FM_minus = [os.path.join(FM_minus_data_dir,i) for i in os.listdir(FM_minus_data_dir)]
        FM_plus_len = len(FM_plus)
        FM_minus_len = len(FM_minus)

        self.data_paths = FM_plus + FM_minus
        self.persons_info = pd.concat((FM_plus_persons_infos,FM_minus_persons_infos))
        labels = [1]*FM_plus_len + [0]*FM_minus_len
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        data = []
        for file in (os.listdir(data_path)*6)[2:(2+self.batch_size//2)]:
            file_path = os.path.join(data_path,file)
            data.append(torch.tensor(np.load(file_path, allow_pickle=True)))
        data = torch.stack(data)
        persons_info = torch.tensor(self.persons_info.iloc[idx].values,dtype=torch.float)
        label = self.labels[idx]
        return data, persons_info, label

    def __len__(self):
        return len(self.labels)