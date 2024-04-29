import torch
import torch.nn as nn


# period1 主要架构：RD++中的投影层MultiProjectionLayer

"""Period 1 : ProjLayer and MultiProjectionLayer """
class ProjLayer(nn.Module):
    '''
    inputs: features of encoder block
    outputs: projected features
    '''
    def __init__(self, in_c, out_c):
        super(ProjLayer, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_c, in_c//2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//2, in_c//4, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//4),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//4, in_c//2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//2, out_c, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(out_c),
                                  torch.nn.LeakyReLU(),
                                  )
    def forward(self, x):
        return self.proj(x)


class MultiProjectionLayer(nn.Module):
    def __init__(self, base = 64):
        """此处是为resnet18设计的,所以conv2,conv3,conv4的通道数为64,128,256"""
        super(MultiProjectionLayer, self).__init__()
        self.proj_a = ProjLayer(base * 1, base * 1)
        self.proj_b = ProjLayer(base * 2, base * 2)
        self.proj_c = ProjLayer(base * 4, base * 4)

    def forward(self, features, features_noise = False):
        if features_noise is not False:
            return ([self.proj_a(features_noise[0]),self.proj_b(features_noise[1]),self.proj_c(features_noise[2])], \
                  [self.proj_a(features[0]),self.proj_b(features[1]),self.proj_c(features[2])])
        else:
            return [self.proj_a(features[0]),self.proj_b(features[1]),self.proj_c(features[2])]

