import os
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score
from dataloader.Dataset_FM_clips import Dataset_FM_clips
from utils.utils import set_seed, make_runs_dir, save_args, get_auc_and_threshold, get_roc_curve
from model.STTransformer_clip import STTransformer
from loss.triplet_loss import TripletLoss


def train(FM_clip_dataloader,batch_size=64):
    all_clip_label = np.array([])
    all_clip_pred = np.array([])
    model.train()  
    for clip_data,clip_labels in FM_clip_dataloader:
        optimizer.zero_grad()
        clip_feature, clip_cls = model(clip_data.to(device))
        all_clip_label = np.append(all_clip_label, clip_labels.detach().numpy())
        all_clip_pred = np.append(all_clip_pred, torch.softmax(clip_cls, 1)[:,1].detach().cpu().numpy())

        loss = criterion_distance(clip_feature, clip_labels.to(device)) + criterion_class(clip_cls, clip_labels.to(device))
        loss.backward()
        optimizer.step()
    scheduler.step()
    clip_auc,threshold,fpr,tpr = get_auc_and_threshold(all_clip_label, all_clip_pred)    
    clip_acc = accuracy_score(all_clip_label, all_clip_pred>threshold)

    return loss, clip_acc, clip_auc, threshold, fpr, tpr



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=150, help='Number of epochs') 
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size in training')
    parser.add_argument('--joint_in_channels', type=int, default=3, help='Input channels of joint feature')
    parser.add_argument('--joint_hidden_channels', type=int, default=64, help='Hidden channels of joint feature')
    parser.add_argument('--time_window', type=int, default=10, help='The size of slide window in time dimension')
    parser.add_argument('--time_step', type=int, default=1, help='The step of slide window in time dimension')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate')
    parser.add_argument('--seed', type=int, default=0, help='The random seed')
    parser.add_argument('--tl_margin', type=float, default=0.4, help='The trilet loss margin')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    args = parser.parse_args()
    set_seed(args.seed)

    #make dir to save results
    result_dir = make_runs_dir(result_name = 'checkpoints')
    save_args(args, result_dir)


    #load FM and non-FM clips 
    FM_clip_dataset = Dataset_FM_clips()
    ##imbalanced sampling to make the number of FM clips and non-FM clips close when training 
    FM_num = sum(FM_clip_dataset.labels == 1)
    non_FM_num = sum(FM_clip_dataset.labels == 0)
    weight = np.array([1/FM_num]*FM_num + [1/non_FM_num]*non_FM_num)
    ##sample same number of clips with training data
    sampler = WeightedRandomSampler(weight,100*args.batch_size)
    FM_clip_dataloader = DataLoader(FM_clip_dataset,batch_size=args.batch_size,sampler=sampler,drop_last=True)

    #load the model
    model = STTransformer(joint_in_channels=args.joint_in_channels, joint_hidden_channels=args.joint_hidden_channels, 
                          time_window=args.time_window, time_step=args.time_step)
    model_dict = model.state_dict()
    model.to(device)
    criterion_class = torch.nn.CrossEntropyLoss()
    criterion_distance = TripletLoss(args.tl_margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.6)

    #train the model
    training_logs = ""
    max_auc = 0
    fig,ax = plt.subplots(figsize=(10,10))
    for epoch in range(1,args.epoch+1):
        loss, clip_acc, clip_auc, threshold, fpr, tpr = train(FM_clip_dataloader,batch_size=args.batch_size)
        if clip_auc >= max_auc:
            torch.save(model.state_dict(), os.path.join(result_dir,"clip_only.pth"))
            max_eval_auc = clip_auc
            get_roc_curve(ax, fpr, tpr, clip_auc, result_dir, metric='auc')            
        log = f'Epoch: {epoch:03d}, [Train] Loss: {loss:.4f} Acc: {clip_acc:.4f} Auc: {clip_auc:.4f} Thres: {threshold:.4f}'
        print(log)
        training_logs += log+"\n"  
    
    with open(os.path.join(result_dir,'logs.txt'), 'w') as f:
        f.write(training_logs)
        f.close()
