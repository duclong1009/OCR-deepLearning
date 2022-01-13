import torch
from torch import nn
from torchvision import models
class VGG(nn.Module):
    def __init__(self, name, ss, ks, hidden, pretrained=True, dropout=0.5):
        super(VGG, self).__init__()
        
        if name == 'vgg11_bn':
            cnn = models.vgg11_bn(pretrained=pretrained)
        elif name == 'vgg19_bn':
            cnn = models.vgg19_bn(pretrained=pretrained)

        pool_idx = 0
        # breakpoint()
        for i, layer in enumerate(cnn.features):
            # print(layer)
            if isinstance(layer, torch.nn.MaxPool2d):        
                cnn.features[i] = torch.nn.AvgPool2d(kernel_size=tuple(ks[pool_idx]), stride=tuple(ss[pool_idx]), padding=0)
                pool_idx += 1
 
        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(512, hidden, 1)

    def forward(self, x):
        """
        Shape: 
            - x: (N, C, H, W)
            - output: (W, N, C)
        """
        # breakpoint()
        # for i in self.features:
        #     x = i(x)
        # breakpoint()
        conv = self.features(x)
        # print(conv.shape)
        # breakpoint()
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)
        # breakpoint()
#        conv = rearrange(conv, 'b d h w -> b d (w h)')
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        return conv

def vgg11_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    return VGG('vgg11_bn', ss, ks, hidden, pretrained, dropout)

def vgg19_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    return VGG('vgg19_bn', ss, ks, hidden, pretrained, dropout)

import numpy as np
