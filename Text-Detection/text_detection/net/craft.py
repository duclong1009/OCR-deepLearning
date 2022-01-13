"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""
import sys
sys.path.append("/content/drive/MyDrive/Project_Deeplearning/text_detection")

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.basenet.vgg16_bn import vgg16_bn
from net.torch_util import *
from dataloader.BaseDataset import *
import utils


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = 1
        x = self.conv(x)
        return x

class CRAFT(nn.Module):
    def __init__(self, pretrained=None, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """

        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature


# if __name__ == '__main__':
#     craft = CRAFT(pretrained="model/pretrained/vgg16_bn-6c64b313.pth")
#     craft.load_state_dict(copyStateDict(torch.load("model/pretrained/craft_mlt_25k.pth", map_location=torch.device("cpu"))))

#     image = cv2.imread('test_folder/image/im1201.jpg')
#     boxes, annos = utils.load_boxes_in_gt("test_folder/gt/gt_im1201.txt")

#     base_dataset = BaseDataset(768)
#     word_image = base_dataset.crop_image_by_bbox(image, boxes[0])[0]
#     cv2.imwrite("crop.jpg", word_image)
