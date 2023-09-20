import os
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataloader.Dataset_validation import Dataset_validation
from utils.utils import get_auc_and_threshold, get_roc_curve
from model.STTransformer import STTransformer



def eval(val_dataloader,batch_size=64):
    all_x_val_MI_label = np.array([])
    all_x_val_MI_pred = np.array([])
    model.eval()
    with torch.no_grad():
        for data,persons_info,labels in val_dataloader:
            features = data.reshape(batch_size,17,16,3,240)
            features = torch.cat((features,torch.zeros(features.shape)))
            clip_feature, clip_cls, x_val_MI_cls = model(features.to(device),persons_info.to(device))
            all_x_val_MI_label = np.append(all_x_val_MI_label, labels.detach().numpy())
            all_x_val_MI_pred = np.append(all_x_val_MI_pred, torch.softmax(x_val_MI_cls, 1)[:,1].detach().cpu().numpy())
    x_val_MI_auc,threshold,fpr,tpr = get_auc_and_threshold(all_x_val_MI_label, all_x_val_MI_pred)    
    x_val_MI_acc = accuracy_score(all_x_val_MI_label, all_x_val_MI_pred>threshold)

    return x_val_MI_acc, x_val_MI_auc, threshold, fpr, tpr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Mini-batch size in training')
    parser.add_argument('--joint_in_channels', type=int, default=3, help='Input channels of joint feature')
    parser.add_argument('--joint_hidden_channels', type=int, default=64, help='Hidden channels of joint feature')
    parser.add_argument('--time_window', type=int, default=10, help='The size of slide window in time dimension')
    parser.add_argument('--time_step', type=int, default=1, help='The step of slide window in time dimension')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    args = parser.parse_args()

    #load validation data
    val_dataset = Dataset_validation(cv_id=1, batch_size=args.batch_size, train=False)
    val_dataloader = DataLoader(val_dataset,batch_size=2,drop_last=True)


    #load the model
    model = STTransformer(joint_in_channels=args.joint_in_channels, joint_hidden_channels=args.joint_hidden_channels, 
                          time_window=args.time_window, time_step=args.time_step)
    model_dict = model.state_dict()
    checkpoint_dict = torch.load("runs/1/model_checkpoint_max_eval_auc.pth")
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    model.to(device)
  

    #eval
    fig,ax = plt.subplots(figsize=(10,10))
    x_val_MI_acc, x_val_MI_auc, val_threshold, fpr, tpr = eval(val_dataloader,batch_size=args.batch_size)
    get_roc_curve(ax, fpr, tpr, x_val_MI_auc, ".", metric='auc')            
    log = f'[Eval] Acc: {x_val_MI_acc:.4f} Auc: {x_val_MI_auc:.4f} Thres: {val_threshold:.4f}'
    print(log)
