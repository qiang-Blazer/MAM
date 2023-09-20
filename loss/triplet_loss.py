import torch
# should pay attention to the balance of pos/neg samples in each batch
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)

    def forward(self, xs, ys):
        n = xs.size(0)

        # 计算pairwise distance   #生成结果为距离矩阵
        dist = torch.pow(xs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(xs, xs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-10).sqrt()

        # 找到hardest triplet
        mask = ys.expand(n, n).eq(ys.expand(n, n).t())
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
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss