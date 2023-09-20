import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc


def set_seed(seed=2022):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_runs_dir(result_name):
    run_times = len(os.listdir(result_name))+1
    result_dir = os.path.join(result_name,str(run_times))
    os.makedirs(result_dir)

    return result_dir


def save_args(args, result_dir):    
    args_dict = args.__dict__
    with open(os.path.join(result_dir,'args.txt'), 'w') as f:
        for key, value in args_dict.items():
            f.writelines(key + ' : ' + str(value) + '\n')
        f.close()


def get_auc_and_threshold(label,pred):
    fpr, tpr, thresholds = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)
    index = np.argmax(tpr-fpr)
    optimal_threshold = thresholds[index]
    
    return roc_auc,optimal_threshold,fpr,tpr


def get_roc_curve(ax, fpr, tpr, roc_auc, result_dir, metric='auc'):    
    ax.clear()
    ax.plot(fpr, tpr, color='#CC0033', lw=4, label=f'ROC curve (AUC = {roc_auc:.4f})') 
    ax.plot([0,1], [0,1], color='navy', lw=1.5, linestyle='--')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.05)
    ax.tick_params(labelsize=18)
    ax.set_xlabel('False Positive Rate',fontdict={'size':20},labelpad=12)
    ax.set_ylabel('True Positive Rate',fontdict={'size':20},labelpad=12)
    ax.set_title('ROC curve',fontdict={'size':22})
    ax.legend(loc="lower right",fontsize=20)
    plt.savefig(os.path.join(result_dir,f"roc-curve_{metric}.png"))


    