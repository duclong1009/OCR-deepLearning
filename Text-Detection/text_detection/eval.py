"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""
# -*- coding: utf-8 -*-
import os
import time
import argparse

import torch
import json
import numpy as np

import config
from net.craft import CRAFT
from net.torch_util import copyStateDict
from inference import test_net
from gaussianMap import imgproc
from metric.eval_det_iou import DetectionIoUEvaluator

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
parser.add_argument('--isTraingDataset', default=1, type=int, help='test for traing or test data')
parser.add_argument('--data_folder', default=config.word_data, type=str, help='folder path to input images')

def load_vintext_gt(dataFolder, isTraining=False):
    if isTraining:
        img_folder = os.path.join(dataFolder, "imgs")
        gt_folder = os.path.join(dataFolder, "gt")
    else:
        img_folder = os.path.join(dataFolder, "unseen_test_images")
        gt_folder = os.path.join(dataFolder, "gt_test")

    gt_names = os.listdir(gt_folder)
    img_names = os.listdir(img_folder)
    total_imgs_bboxes = []
    total_img_path = []
    for gt_name in gt_names:
        gt_path = os.path.join(gt_folder, gt_name)
        img_name = gt_name.replace("gt_", "").replace(".txt", ".jpg")
        if img_name not in img_names:
            print("Error file extension...", img_name)
            continue

        img_path = os.path.join(img_folder, img_name)
        lines = open(gt_path, encoding='utf-8').readlines()
        single_img_bboxes = []
        for line in lines:
            boxInfos = {"points": None, "text": None, "ignore": None}
            ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
            box = [int(ori_box[j]) for j in range(8)]
            word = ','.join(ori_box[8:])
            box = np.array(box, dtype=np.int32).reshape(4, 2).tolist()
            boxInfos["points"] = box
            boxInfos["text"] = word
            if word == "###":
                boxInfos["ignore"] = True
            else:
                boxInfos["ignore"] = False

            single_img_bboxes.append(boxInfos)
        total_imgs_bboxes.append(single_img_bboxes)
        total_img_path.append(img_path)
    return total_imgs_bboxes, total_img_path


# def evaluate(model, evaluator, device, data_folder, phase="Training",
#              text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=False, canvas_size=1200, mag_ratio=1.5):
#     # total_imgs_bboxes_gt: (num_image, num_box), num_box: {"points": (4, 2) - (tl, tr, br, bl), "text": "", "ignore": True/False}
#     total_imgs_bboxes_gt, total_img_path = load_vintext_gt(dataFolder=data_folder, isTraining=True if phase == "Training" else False)
#     total_img_bboxes_pre = []
#     print("phase: ", phase)
#     print("len: ", len(total_img_path))
#     for img_path in total_img_path:
#         image = imgproc.loadImage(img_path)
#         single_img_bbox = []
#         # bboxes: (num_box, 4, 2)
#         bboxes, polys, score_text = test_net(model,
#                                              image,
#                                              text_threshold,
#                                              link_threshold,
#                                              low_text,
#                                              device,
#                                              poly,
#                                              canvas_size,
#                                              mag_ratio)
#         for box in bboxes:
#             box_info = {"points": box, "text": "###", "ignore": False}
#             single_img_bbox.append(box_info)
#         total_img_bboxes_pre.append(single_img_bbox)
#     results = []
#     # total_imgs_bboxes_gt: (num_image, num_box), num_box: {"points": (4, 2) - (tl, tr, br, bl)
#     # total_img_bboxes_pre: (num_image, num_box), num_box: {"points": (4, 2) - (tl, tr, br, bl)

#     for i, (gt, pred) in enumerate(list(zip(total_imgs_bboxes_gt, total_img_bboxes_pre))):
#         draw_gt(cv2.imread(total_img_path[i]), gt, "GT_{}".format(i))
#         draw_gt(cv2.imread(total_img_path[i]), pred, "pred_{}".format(i))
#         results.append(evaluator.evaluate_image_1(gt, pred))

#     metrics = evaluator.combine_results_1(results)
#     print("Metric: ", metrics)
#     epoch = int(os.path.basename(config.pretrained_craft).split("_")[0])
#     metrics["epoch"] = epoch
#     with open("sample.json", "a") as f:
#         json.dump(metrics, f, indent=2)
#         f.write("\n")
#     return metrics

# def draw_gt(image, info, pred):
#     img = image.copy()
#     # print(info)
#     for i in range(len(info)):
#         box = info[i]['points']
#         b = np.array(box, dtype=np.int32).reshape(4, 2)
#         img = cv2.polylines(img, [b], isClosed=1, color=(0, 255, 0), thickness=2)
#         img = cv2.putText(img, str(i), (b[0][0], b[0][1]+2), fontScale=0.35, thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255))
#     cv2.imwrite("metric/"+pred+".jpg", img)

# if __name__ == '__main__':
#     args = parser.parse_args()
#     # Device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # Load pretrained craft
#     craft = CRAFT()
#     print('Loading weights from checkpoint (' + args.trained_model + ')')
#     craft.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location=device)))

#     evaluator = DetectionIoUEvaluator()
#     data_folder = args.data_folder
#     text_threshold = args.text_threshold
#     link_threshold = args.link_threshold
#     low_text = args.low_text
#     poly = args.poly
#     canvas_size = args.canvas_size
#     mag_ratio = args.mag_ratio

#     metrics = evaluate(craft, evaluator, device, data_folder, phase="Training" if args.isTraingDataset else "Test")
    
#     model_path = args.trained_model
#     lr = float(model_path.split("/")[-2].split("_")[-1])
#     epoch = int(model_path.split("/")[-1].split("_")[0])
#     loss = float(model_path.split("/")[-1].split("_")[1].replace(".pth", ""))

#     metrics.update({
#         "lr": lr,
#         "epoch": epoch,
#         "loss": loss
#     })
#     print(metrics)
#     log = "evaluation_{}.json".format("train" if args.isTraingDataset else "test")
#     with open(log, "a") as f:
#         json.dump(metrics, f)
#         f.write("\n")

def evaluate(model, evaluator, device, data_folder, phase="Training",
             text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=False, canvas_size=1200, mag_ratio=1.5):
    # total_imgs_bboxes_gt: (num_image, num_box), num_box: {"points": (4, 2) - (tl, tr, br, bl), "text": "", "ignore": True/False}
    total_imgs_bboxes_gt, total_img_path = load_vintext_gt(dataFolder=data_folder, isTraining=True if phase == "Training" else False)
    total_img_bboxes_pre = []
    print("phase: ", phase)
    print("len: ", len(total_img_path))
    for img_path in total_img_path:
        image = imgproc.loadImage(img_path)
        single_img_bbox = []
        # bboxes: (num_box, 4, 2)
        bboxes, polys, score_text = test_net(model,
                                             image,
                                             text_threshold,
                                             link_threshold,
                                             low_text,
                                             device,
                                             poly,
                                             canvas_size,
                                             mag_ratio)
        for box in bboxes:
            box_info = {"points": box, "text": "###", "ignore": False}
            single_img_bbox.append(box_info)
        total_img_bboxes_pre.append(single_img_bbox)
    results_iou = []
    results = []
    # total_imgs_bboxes_gt: (num_image, num_box), num_box: {"points": (4, 2) - (tl, tr, br, bl)
    # total_img_bboxes_pre: (num_image, num_box), num_box: {"points": (4, 2) - (tl, tr, br, bl)

    for i, (gt, pred) in enumerate(list(zip(total_imgs_bboxes_gt, total_img_bboxes_pre))):
        # draw_gt(cv2.imread(total_img_path[i]), gt, "GT_{}".format(i))
        # draw_gt(cv2.imread(total_img_path[i]), pred, "pred_{}".format(i))
        results_iou.append(evaluator.evaluate_image_1(gt, pred))
        results.append(evaluator.evaluate_image(gt, pred))

    iou = evaluator.combine_results_1(results_iou)
    metrics = evaluator.combine_results(results)
    return iou, metrics

def draw_gt(image, info, pred):
    img = image.copy()
    # print(info)
    for i in range(len(info)):
        box = info[i]['points']
        b = np.array(box, dtype=np.int32).reshape(4, 2)
        img = cv2.polylines(img, [b], isClosed=1, color=(0, 255, 0), thickness=2)
        img = cv2.putText(img, str(i), (b[0][0], b[0][1]+2), fontScale=0.35, thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255))
    cv2.imwrite("metric/"+pred+".jpg", img)
    

if __name__ == '__main__':
    args = parser.parse_args()
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load pretrained craft
    craft = CRAFT()
    
    craft.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
    print('Loading weights from checkpoint (' + trained_model + ')')
    evaluator = DetectionIoUEvaluator()
    data_folder = args.data_folder
    text_threshold = args.text_threshold
    link_threshold = args.link_threshold
    low_text = args.low_text
    poly = args.poly
    canvas_size = args.canvas_size
    mag_ratio = args.mag_ratio

    iou, metrics = evaluate(craft, evaluator, device, data_folder, phase="Training" if args.isTraingDataset else "Test")
    print("Metric: ", metrics)
    print("IoU: ", iou)
    metrics["epoch"] = epoch
    iou["epoch"] = epoch
    with open("metric.json", "a") as f:
        json.dump(metrics, f)
        f.write("\n")
    with open("iou.json", "a") as f:
        json.dump(iou, f)
        f.write("\n")
