"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
sys.path.append("/content/drive/MyDrive/Project_Deeplearning/text_detection")

import os
import time
import argparse
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from net.craft import CRAFT
from net.refinenet import RefineNet
from net.torch_util import copyStateDict
from gaussianMap import imgproc
import utils
import config


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default=config.pretrained_craft, type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default=config.path_test_folder, type=str, help='folder path to input images')
parser.add_argument('--refiner_model', default=config.pretrained_linkrefiner, type=str,
                    help='pretrained refiner model')

# show_time = True
# args = parser.parse_args()

def test_net(image, net, device, refine_net=None, poly=False, text_threshold=0.7, link_threshold=0.4, low_text=0.4, canvas_size=1280, mag_ratio=1.5):
    net = net.to(device)
    net.eval()

    t0 = time.time()
    # resize image
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    x = x.to(device)

    # forward pass
    print("Device1: ", device)
    with torch.no_grad():
        y, feature = net(x)

    print("Device: ", device)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()
    # score_text = y[0, :, :, 0]
    # score_link = y[0, :, :, 1]

    # refine link
    if refine_net is not None:
        refine_net = refine_net.cuda()
        refine_net.eval()
        print("Device: ", device)
        print("y", y.device, refine_net.device)
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    # if args.show_time:
        # print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
    return boxes, polys, ret_score_text

# if __name__ == '__main__':
#     # For test images in a folder
#     args = parser.parse_args()
#     image_list, _, _ = utils.get_files(args.test_folder)
#     result_folder = config.path_saved_linkrefiner_inference if args.refiner_model else config.path_saved_craft_inference
#     if not os.path.isdir(result_folder):
#         os.mkdir(result_folder)

#     # Device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # Load craft pretrained model
#     craft = CRAFT()
#     print('Loading weights from checkpoint (' + args.trained_model + ')')
#     craft.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location=device)))

#     # Load LinkRefiner
#     refine_net = None
#     if args.refiner_model != "":
#         refine_net = RefineNet()
#         print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
#         refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location=device)))
#         args.poly = True

#     t = time.time()

#     # Inference
#     for k, image_path in enumerate(image_list):
#         print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
#         image = imgproc.loadImage(image_path)

#         bboxes, polys, score_text = test_net(image, craft, device, refine_net, args.poly, args.text_threshold, args.link_threshold, args.low_text,
#                                              args.canvas_size, args.mag_ratio)

#         # save score text
#         filename, file_ext = os.path.splitext(os.path.basename(image_path))
#         mask_file = result_folder + "/res_" + filename + '_mask.jpg'
#         cv2.imwrite(mask_file, score_text)
#         utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)

#     print("elapsed time : {}s".format(time.time() - t))
