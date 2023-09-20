from torch.utils.data import Dataset
import torch
import numpy as np
import os
import random
import pandas as pd

class Dataset_derivation(Dataset):

    def __init__(self, cv_id, batch_size=64, train=True):
        '''
            cv_id --- the cross validation id  (cv_id = 0/1/2/3/4)
        '''
        self.batch_size = batch_size
        FM_plus_persons_infos = pd.read_csv("prepared_data/Data_derivation/persons_infos_derivation_FM+.csv").iloc[:,:4]
        FM_minus_persons_infos = pd.read_csv("prepared_data/Data_derivation/persons_infos_derivation_FM-.csv").iloc[:,:4]
        FM_plus_data_dir = "prepared_data/Data_derivation/FM+"
        FM_minus_data_dir = "prepared_data/Data_derivation/FM-"
        FM_plus = np.array([os.path.join(FM_plus_data_dir,i) for i in os.listdir(FM_plus_data_dir)])
        FM_minus = np.array([os.path.join(FM_minus_data_dir,i) for i in os.listdir(FM_minus_data_dir)])
        FM_plus_len = len(FM_plus)
        FM_minus_len = len(FM_minus)
        FM_plus_ids = [i for i in range(FM_plus_len)]
        FM_minus_ids = [i for i in range(FM_minus_len)]
        random.shuffle(FM_plus_ids)
        random.shuffle(FM_minus_ids)

        #split for 5-fold cross validation
        CVs = [{},{},{},{},{}]
        for i in range(4):
            CVs[i]["plus"] = FM_plus_ids[i*(FM_plus_len//5) : (i+1)*(FM_plus_len//5)]
            CVs[i]["minus"] = FM_minus_ids[i*(FM_minus_len//5) : (i+1)*(FM_minus_len//5)]
        CVs[4]["plus"] = FM_plus_ids[4*(FM_plus_len//5) : ]
        CVs[4]["minus"] = FM_minus_ids[4*(FM_minus_len//5) : ]

        self.data_paths = []
        self.persons_info = None
        labels = []
        if train:
            for i in range(5):
                if i!=cv_id:
                    self.data_paths += (FM_plus[CVs[i]["plus"]].tolist() + FM_minus[CVs[i]["minus"]].tolist())
                    if self.persons_info is None:
                        self.persons_info = pd.concat((FM_plus_persons_infos.iloc[CVs[i]["plus"]], FM_minus_persons_infos.iloc[CVs[i]["minus"]]))
                    else:
                        self.persons_info = pd.concat((self.persons_info,FM_plus_persons_infos.iloc[CVs[i]["plus"]], FM_minus_persons_infos.iloc[CVs[i]["minus"]]))
                    labels += ([1]*len(FM_plus[CVs[i]["plus"]]) + [0]*len(FM_minus[CVs[i]["minus"]]))
        else:
            self.data_paths += (FM_plus[CVs[cv_id]["plus"]].tolist() + FM_minus[CVs[cv_id]["minus"]].tolist())
            self.persons_info = pd.concat((FM_plus_persons_infos.iloc[CVs[cv_id]["plus"]], FM_minus_persons_infos.iloc[CVs[cv_id]["minus"]]))
            labels += ([1]*len(FM_plus[CVs[cv_id]["plus"]]) + [0]*len(FM_minus[CVs[cv_id]["minus"]]))
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