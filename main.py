import os
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CyclicLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score
from dataloader.Dataset_FM_clips import Dataset_FM_clips
from dataloader.Dataset_derivation import Dataset_derivation
from utils.utils import set_seed, make_runs_dir, save_args, get_auc_and_threshold, get_roc_curve
from model.STTransformer import STTransformer
from loss.MI_overall_loss import OverallLoss


def train(train_dataloader,FM_clip_dataloader,batch_size=64,info=False):
    all_clip_label = np.array([])
    all_clip_pred = np.array([])
    all_x_train_MI_label = np.array([])
    all_x_train_MI_pred = np.array([])
    model.train()  
    train_dataloader = iter(train_dataloader)
    FM_clip_dataloader = iter(FM_clip_dataloader)
    for i in range(len(train_dataloader)):
        optimizer.zero_grad()
        data,persons_info,labels = next(train_dataloader)
        clip_data,clip_labels = next(FM_clip_dataloader)
        features = torch.cat((data.reshape(batch_size,17,16,3,240),clip_data))
        if info:
            clip_feature, clip_cls, x_train_MI_cls = model(features.to(device),persons_info.to(device))
        else:
            clip_feature, clip_cls, x_train_MI_cls = model(features.to(device))
        all_clip_label = np.append(all_clip_label, clip_labels.detach().numpy())
        all_clip_pred = np.append(all_clip_pred, torch.softmax(clip_cls[batch_size:], 1)[:,1].detach().cpu().numpy())
        all_x_train_MI_label = np.append(all_x_train_MI_label, labels.detach().numpy())
        all_x_train_MI_pred = np.append(all_x_train_MI_pred, x_train_MI_cls[:,1].detach().cpu().numpy())

        loss = criterion(x_train_MI_cls, labels.to(device), clip_feature, clip_cls, clip_labels.to(device))
        loss.backward()
        optimizer.step()
    scheduler.step()
    clip_acc = accuracy_score(all_clip_label, all_clip_pred>0.5)
    x_train_MI_auc,threshold,fpr,tpr = get_auc_and_threshold(all_x_train_MI_label, all_x_train_MI_pred)    
    x_train_MI_acc = accuracy_score(all_x_train_MI_label, all_x_train_MI_pred>threshold)

    return loss, clip_acc, x_train_MI_acc, x_train_MI_auc, threshold


def eval(val_dataloader,batch_size=64,info=False):
    all_x_val_MI_label = np.array([])
    all_x_val_MI_pred = np.array([])
    model.eval()
    with torch.no_grad():
        for data,persons_info,labels in val_dataloader:
            features = data.reshape(batch_size,17,16,3,240)
            features = torch.cat((features,torch.zeros(features.shape)))
            if info:
                clip_feature, clip_cls, x_val_MI_cls = model(features.to(device),persons_info.to(device))
            else:
                clip_feature, clip_cls, x_val_MI_cls = model(features.to(device))
            all_x_val_MI_label = np.append(all_x_val_MI_label, labels.detach().numpy())
            all_x_val_MI_pred = np.append(all_x_val_MI_pred, x_val_MI_cls[:,1].detach().cpu().numpy())
    x_val_MI_auc,threshold,fpr,tpr = get_auc_and_threshold(all_x_val_MI_label, all_x_val_MI_pred)    
    x_val_MI_acc = accuracy_score(all_x_val_MI_label, all_x_val_MI_pred>threshold)

    return x_val_MI_acc, x_val_MI_auc, threshold, fpr, tpr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='Number of epochs') 
    parser.add_argument('--batch_size', type=int, default=32, help='Mini-batch size in training')
    parser.add_argument('--cv_id', type=int, default=0, help='The cross validation id, choose from [0,1,2,3,4]')
    parser.add_argument('--joint_in_channels', type=int, default=3, help='Input channel numbers of joint feature')
    parser.add_argument('--joint_hidden_channels', type=int, default=64, help='Hidden channel numbers of joint feature')
    parser.add_argument('--time_window', type=int, default=10, help='The size of slide window in time dimension')
    parser.add_argument('--time_step', type=int, default=1, help='The step of slide window in time dimension')
    parser.add_argument('--optimizer', type=str, default="SGD", help='Optimizer used in training')
    parser.add_argument('--lr', type=float, default=0.001, help='The starting learning rate')
    parser.add_argument('--scheduler', type=str, default="CyclicLR", help='Scheduler used in training')
    parser.add_argument('--seed', type=int, default=0, help='The random seed')
    parser.add_argument('--tl_margin', type=float, default=0.4, help='The trilet loss margin')
    parser.add_argument('--with_info', type=bool, default=True, help='If use the info characteristics')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    args = parser.parse_args()
    set_seed(args.seed)

    #make dir to save results
    result_dir = make_runs_dir(result_name = 'runs')
    save_args(args, result_dir)

    #load derivation data
    train_dataset = Dataset_derivation(cv_id=args.cv_id, batch_size=args.batch_size, train=True)
    val_dataset = Dataset_derivation(cv_id=args.cv_id, batch_size=args.batch_size, train=False)
    train_dataloader = DataLoader(train_dataset,batch_size=2,drop_last=True)
    val_dataloader = DataLoader(val_dataset,batch_size=2,drop_last=True)

    #load FM and non-FM clips 
    FM_clip_dataset = Dataset_FM_clips()
    ##imbalanced sampling to make the number of FM clips and non-FM clips close when training 
    FM_num = sum(FM_clip_dataset.labels == 1)
    non_FM_num = sum(FM_clip_dataset.labels == 0)
    weight = np.array([1/FM_num]*FM_num + [1/non_FM_num]*non_FM_num)
    ##sample same number of clips with training data
    sampler = WeightedRandomSampler(weight,len(train_dataloader)*args.batch_size)
    FM_clip_dataloader = DataLoader(FM_clip_dataset,batch_size=args.batch_size,sampler=sampler,drop_last=True)

    #load the model
    model = STTransformer(joint_in_channels=args.joint_in_channels, joint_hidden_channels=args.joint_hidden_channels, 
                          time_window=args.time_window, time_step=args.time_step)
    model_dict = model.state_dict()
    checkpoint_dict = torch.load("checkpoints/1/clip_only.pth")
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    criterion = OverallLoss(margin=args.tl_margin)
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"The argument --optimizer can not be {args.optimizer}")
    
    if args.scheduler.lower() == "steplr":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.6)
    elif args.scheduler.lower() == "exponentiallr":
        scheduler = ExponentialLR(optimizer, gamma=0.99)
    elif args.scheduler.lower() == "cycliclr":
        scheduler = CyclicLR(optimizer,base_lr=1e-5, max_lr=args.lr, step_size_up=20, mode='triangular2')
    elif args.scheduler.lower() == "cosineannealinglr":
        scheduler = CosineAnnealingLR(optimizer, T_max=20)
    elif args.scheduler.lower() == "cosineannealingwarmrestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    else:
        raise ValueError(f"The argument --scheduler can not be {args.scheduler}")

    #train the model
    training_logs = ""
    max_eval_auc = 0
    fig,ax = plt.subplots(figsize=(10,10))
    for epoch in range(1,args.epoch+1):
        loss, clip_acc, x_train_MI_acc, x_train_MI_auc, threshold = train(train_dataloader,FM_clip_dataloader,batch_size=args.batch_size,info=args.with_info)
        x_val_MI_acc, x_val_MI_auc, val_threshold, fpr, tpr = eval(val_dataloader,batch_size=args.batch_size,info=args.with_info)
        if x_val_MI_auc >= max_eval_auc:
            torch.save(model.state_dict(), os.path.join(result_dir,"model_checkpoint_max_eval_auc.pth"))
            max_eval_auc = x_val_MI_auc
            get_roc_curve(ax, fpr, tpr, x_val_MI_auc, result_dir, metric='auc')            
        log = f'Epoch: {epoch:03d}, [Train] Loss: {loss:.4f} Auc: {x_train_MI_auc:.4f} Thres: {threshold:.4f} \
[Eval] Auc: {x_val_MI_auc:.4f} Thres: {val_threshold:.4f}'
        print(log)
        training_logs += log+"\n"  
    
    with open(os.path.join(result_dir,'logs.txt'), 'w') as f:
        f.write(training_logs)
        f.close()
