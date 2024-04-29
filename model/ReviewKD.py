import torch
from torch import nn
import torch.nn.functional as F

# period1 主要架构：ReviewKD中的ABF与HCL
"""Some scripts here are copied from ReviewKD: https://github.com/dvlab-research/ReviewKD"""

#abf：feature融合机制
class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
        )
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore


    def forward(self, x,y,shape):
        n,_,h,w = x.shape
        """x做conv1卷积，增加为y的通道数"""
        x = self.conv1(x)
        """y做插值，上采样到x的size"""
        y = F.interpolate(y ,[shape,shape],mode="nearest")
        """z是x和y两个feature直接叠加，之后做attention"""
        z = torch.cat([x, y], dim=1)
        z = self.att_conv(z)
        """x和z的第一个通道对应相乘，y和z第二个通道对应相乘。之后两者相加为A"""
        A = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        """对A做conv2卷积得到B，B作为x这层的fuse feature"""
        B = self.conv2(A)
        return B

#ReviewKD：student feature经过abf融合后的新student feature
class ReviewKD(nn.Module):
    def __init__(
        self, in_channels, out_channels, mid_channels
    ):
        super(ReviewKD, self).__init__()

        abfs = nn.ModuleList()

        for idx,in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channels[idx], out_channels[idx]))

        self.abfs = abfs

    def forward(self, student_features):

        x = student_features
        # stage 4
        out_features2 = x[2]
        # stage 3
        features = x[1]
        res_features = x[2]
        abf = self.abfs[0]
        out_features1 = abf(features, res_features,32)

        # stage 2
        features = x[0]
        res_features = out_features1
        abf = self.abfs[1]
        out_features0 = abf(features, res_features, 64)

        results = [out_features0,out_features1,out_features2]

        return results

# hcl_loss：多尺度loss
def hcl_loss(fstudent, fteacher):
    cos_loss = torch.nn.CosineSimilarity()
    loss_all = 0.0
    for item in range(len(fstudent)):
        fs = fstudent[item]
        ft = fteacher[item]
        n, c, h, w = fs.shape
        loss = torch.mean(1 - cos_loss(fs.view(fs.shape[0], -1) , ft.view(ft.shape[0], -1)))
        cnt = 1.0
        tot = 1.0
        l1 = int(h/2)
        l2 = int(h/4)
        l3 = int(h/8)
        for l in [l1,l2,l3]:
            tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
            tmpft = F.adaptive_avg_pool2d(ft, (l,l))
            cnt /= 2.0
            loss += torch.mean(1 - cos_loss(tmpfs.view(tmpfs.shape[0], -1) , tmpft.view(tmpft.shape[0], -1))) * cnt
            tot += cnt
        loss = loss / tot
        loss_all += loss
    total_loss = loss_all
    return total_loss