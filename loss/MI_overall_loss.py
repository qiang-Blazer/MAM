import torch 
import torch.nn as nn


class OverallLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.pdist = nn.PairwiseDistance(p=2)
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)
        self.nll_loss = nn.NLLLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        
    def forward(self, x_train_MI_cls, labels, clip_feature, clip_cls, clip_labels):
        #bag loss
        x_train_MI_CEloss = self.nll_loss(torch.log(x_train_MI_cls), labels)
        #clip feature distance loss
        n = len(clip_labels)
        dist = torch.pow(clip_feature[n:], 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(clip_feature[n:], clip_feature[n:].t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-10).sqrt()
        mask = clip_labels.expand(n, n).eq(clip_labels.expand(n, n).t())
        dist_ap = []
        dist_an = []
        for i in range(n):
            # dist_ap.append(dist[i][mask[i]].max())
            # dist_an.append(dist[i][mask[i] == 0].min())
            dist_ap.append(dist[i][mask[i]].mean())
            dist_an.append(dist[i][mask[i] == 0].mean())
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        y = torch.ones_like(dist_an)
        clip_Tripletloss = self.ranking_loss(dist_an, dist_ap, y)
        #clip loss
        clip_CEloss = self.ce_loss(clip_cls[n:], clip_labels)
        #instance loss
        instance_pred = torch.softmax(clip_cls[:n],1)
        d_mask = instance_pred.ge(0.5)
        FM_minus_center = clip_feature[n:][clip_labels==0].mean(dim=0,keepdim=True)
        FM_plus_center = clip_feature[n:][clip_labels==1].mean(dim=0,keepdim=True)
        distance = torch.cat((self.pdist(clip_feature[:n],FM_minus_center).unsqueeze(0),\
                              self.pdist(clip_feature[:n],FM_plus_center).unsqueeze(0)))
        distance = torch.softmax(distance,0)
        distance = torch.sum(torch.mul(distance,torch.transpose(d_mask,dim0=0, dim1=1)),dim=0)
        instance_loss = torch.mean(torch.mul(torch.mul(instance_pred[:,0], instance_pred[:,1]), distance))

        
        return x_train_MI_CEloss + clip_Tripletloss + clip_CEloss + instance_loss


