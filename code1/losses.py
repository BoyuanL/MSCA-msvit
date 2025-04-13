import torch
from  torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
# import numpy as np
# from torch.nn.modules.distance import PairwiseDistance
# from torch.nn import Parameter
# import math
from torch.nn.modules.distance import PairwiseDistance

class ContrastiveLoss(nn.Module):
    '''
    对比损失（Contrastive Loss）的实现。对比损失用于学习一个特征空间，使得同一类别的样本在特征空间中距离更近，不同类别的样本在特征空间中距离更远。具体来说，对于每一对样本$x_1$和$x_2$，它们应该被分别映射到特征空间中的两个点$f(x_1)$和$f(x_2)$，并且当它们属于同一类别时，$f(x_1)$和$f(x_2)$应该距离更近，当它们属于不同类别时，$f(x_1)$和$f(x_2)$应该距离更远。该损失函数通过对比两个样本的相似度和它们所属的类别，来鼓励同一类别的样本相似度更高，不同类别的样本相似度更低。

该实现中，将$x_1$和$x_2$的特征向量拼接成一个大矩阵，计算该矩阵中的相似度矩阵。为了避免样本$x_i$与自己的相似度被考虑，创建对角线为1的矩阵，将其拼接成一个大矩阵，并将两个对角线上的1都变为0，得到一个掩码矩阵。然后，分别计算$x_1$和$x_2$的余弦相似度，得到对角线上的指数分子，分母是掩码矩阵中每个位置的指数和。最后，使用对数损失函数来衡量预测值与真实值之间的差异，使同一类别的样本更加接近，不同类别的样本更加远离
    '''

    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, tau=0.1):
        x1x2 = torch.cat([x1, x2], dim=0) # 将x1和x2进行拼接
        x2x1 = torch.cat([x2, x1], dim=0)

        cosine_mat = torch.cosine_similarity(torch.unsqueeze(x1x2, dim=1),  # 计算拼接后的相似度矩阵
                                             torch.unsqueeze(x1x2, dim=0), dim=2) / tau

        mask = torch.eye(x1.size(0))  # 创建对角线为1的矩阵
        mask = torch.cat([mask, mask], dim=0)    # 将两个对角线为1的矩阵拼接成一个大矩阵
        mask = 1.0 - torch.cat([mask, mask], dim=1)  # 将大矩阵水平和垂直方向上都翻转后，两个对角线上的1都变为了0
        mask = mask.cuda()
        numerators = torch.exp(torch.cosine_similarity(x1x2, x2x1, dim=1) / tau)  # 计算对角线上的指数分子
        denominators = torch.sum(torch.exp(cosine_mat) * mask, dim=1)  # 计算分母

        return -torch.mean(torch.log(numerators / denominators), dim=0)  # 返回对数损失函数的均值


class TripletLoss(nn.Module):
    '''
    TripletLoss 是一种距离度量的损失函数，用于监督学习任务中的度量学习（metric learning）问题。
    它的目的是使同一类别的样本之间的距离尽可能小，不同类别的样本之间的距离尽可能大。
    损失函数接受三个输入参数，即锚点样本 anchor，正样本 positive 和负样本 negative，
    分别表示同一类别的样本、不同类别的样本和与锚点样本同属不同类别的样本。函数中使用欧几里得距离度量了不同样本之间的距离，通过 hinge loss 的形式计算损失值。
    '''

    def __init__(self, margin=0.0):
        super(TripletLoss, self).__init__()
        self.margin = margin   # Triplet Loss 的 margin
        self.pdist = PairwiseDistance(p=2)  # 计算两个向量之间的欧几里得距离

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist.forward(anchor, positive)  # anchor和positive之间的距离
        neg_dist = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)   # 计算 Triplet Loss 中的 hinge loss
        loss = torch.mean(hinge_dist)  # 求 hinge loss 的平均值
        return loss


class TripletLoss2(nn.Module):
    def __init__(self, margin=0.0):
        super(TripletLoss2, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()

        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            # dist_ap.append(dist[i][mask[i]].max())
            # dist_an.append(dist[i][mask[i] == 0].min())
            dist_ap.append(torch.tensor([float(dist[i][mask[i]].max())]))
            dist_an.append(torch.tensor([float(dist[i][mask[i] == 0].min())]))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        # prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

