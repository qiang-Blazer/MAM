from torch.utils.data import Dataset
import torch
import numpy as np
import os

class Dataset_FM_clips(Dataset):

    def __init__(self):
        FM_clips_data_dir = "prepared_data/labeled_FM_clips/FM_clips"
        non_FM_clips_data_dir = "prepared_data/labeled_FM_clips/non_FM_clips"
        FM_clips = [os.path.join(FM_clips_data_dir,i) for i in os.listdir(FM_clips_data_dir)]
        non_FM_clips = [os.path.join(non_FM_clips_data_dir,i) for i in os.listdir(non_FM_clips_data_dir)]
        self.data_paths = FM_clips + non_FM_clips
        labels = [1]*len(FM_clips) + [0]*len(non_FM_clips)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        data = torch.tensor(np.load(data_path, allow_pickle=True))
        label = self.labels[idx]
        return data, label

    def __len__(self):
        return len(self.labels)