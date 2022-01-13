import sys
print(sys.path)
import os
import cv2
import numpy as np
import scipy.io as scio
import argparse
import time
import random
import config

import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.autograd import Variable
import torch
import torch.utils.data
import json

from datetime import datetime
import wandb

from net.craft import CRAFT
from net.torch_util import save_train_info, copyStateDict
from dataloader.SynthTextDataset import SynthTextDataset
from dataloader.WordLevelDataset import WordLevelDataset
from log_train import LogTrain
from loss.maploss import MapLoss
from eval import evaluate
from metric.eval_det_iou import DetectionIoUEvaluator

random.seed(43)
torch.manual_seed(43)
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='CRAFT')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--init_lr', '--initial-learning-rate', default=0, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=32, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument("--path_save", default=config.path_saved_train, type=str)
parser.add_argument("--syn_data", default=config.synth_data, type=str)
parser.add_argument("--pretrained", default=config.pretrained_craft, type=str)
parser.add_argument("--word_data", default=config.word_data, nargs='+')
args = parser.parse_args()

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagecraft example:
    # https://github.com/pytorch/examples/blob/master/imagecraft/main.py
    """
    lr = args.lr * (0.8 ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_save_folder():
    now = datetime.now()
    format_time = now.strftime("%Y_%m_%d_%H_%M")
    format_time = format_time + "_lr_%s" % args.lr
    print("format: ", format_time)
    path_save = args.path_save
    saved = os.path.join(path_save, format_time)
    if not os.path.exists(saved):
        os.mkdir(saved)
    return saved

def train():
    batch_size_char = config.batch_size_synthtext
    batch_size_word = config.batch_size_word
    target_size = config.target_size
    epochs_end = config.epochs_end
    nb_epoch_change_lr = config.nb_epochs_change_lr
    syndata_dir = args.syn_data
    vintext_dir = args.word_data
    lr = args.lr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dict_infor = vars(args)

    # Make save train folder
    saved = create_save_folder()
    log_file = LogTrain(saved)
    dict_infor["saved"] = saved

    # Load CRAFT model:
    craft = CRAFT()
    craft.load_state_dict(copyStateDict(torch.load(args.pretrained, map_location=device)))
    craft = craft.to(device)
    craft = torch.nn.DataParallel(craft, device_ids=[0]).to(device)

    # Make SynthText Dataloader
    synthtext_dataset = SynthTextDataset(syndata_dir, target_size=target_size)
    syn_data_loader = torch.utils.data.DataLoader(
        synthtext_dataset,
        batch_size=batch_size_char,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)
    batch_syn = iter(syn_data_loader)
    dict_infor['batch_size_char'] = batch_size_char

    # Make WordLevel Dataloader
    realdata = WordLevelDataset(craft, vintext_dir, target_size=target_size, viz=False)
    real_data_loader = torch.utils.data.DataLoader(
        realdata,
        batch_size=batch_size_word,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)
    dict_infor['batch_size_real'] = batch_size_word

    # Criterion Loss
    criterion = MapLoss(log_file)
    # Evaluator
    evaluator = DetectionIoUEvaluator()

    save_train_info(dict_infor, saved, "config.py")

    try:
        init_epoch = int(os.path.basename(args.pretrained).split("_")[1])
    except Exception as e:
        init_epoch = 0
    if args.init_lr:
        print(1)
        lr = args.init_lr
    # Adam Optimizer
    optimizer = optim.Adam(craft.parameters(), lr=lr, weight_decay=args.weight_decay)
    if args.init_lr == 0:
        adjust_learning_rate(optimizer, args.gamma, init_epoch // nb_epoch_change_lr)

    for epoch in range(init_epoch + 1, epochs_end):
        print("******************    TRAINING EPOCH {}/{}    ******************".format(epoch, epochs_end))

        
        loss_value = 0
        total_loss = 0
        if epoch % nb_epoch_change_lr == 0:
            print("change lr")
            adjust_learning_rate(optimizer, args.gamma, epoch // nb_epoch_change_lr)

        begin = time.time()
        st = time.time()
        print("lr :", optimizer.param_groups[0]['lr'])
        craft.eval()
        for index, (real_images, real_region_scores, real_affinity_scores, real_conf_mask, _) in enumerate(real_data_loader):
            # word_image_out_from_craft.train()
            syn_images, syn_region_scores, syn_affinity_scores, syn_conf_mask, _ = next(batch_syn)
            craft.train()

            # Train real and syn image together
            images = torch.cat((syn_images, real_images), 0)
            region_scores = torch.cat((syn_region_scores, real_region_scores), 0)
            affinity_scores = torch.cat((syn_affinity_scores, real_affinity_scores), 0)
            conf_mask = torch.cat((syn_conf_mask, real_conf_mask), 0)

            # Push to device
            images = Variable(images.type(torch.FloatTensor)).to(device)
            region_scores_gt = Variable(region_scores.type(torch.FloatTensor)).to(device)
            affinity_scores_gt = Variable(affinity_scores.type(torch.FloatTensor)).to(device)
            conf_mask_gt = Variable(conf_mask.type(torch.FloatTensor)).to(device)

            # Output from net
            out, _ = craft(images)
            optimizer.zero_grad()

            region_scores_pred = out[:, :, :, 0].to(device)
            affinity_scores_pred = out[:, :, :, 1].to(device)
            loss = criterion(region_scores_gt, affinity_scores_gt, region_scores_pred, affinity_scores_pred, conf_mask_gt).to(device)

            loss.backward()
            optimizer.step()
            loss_value += loss.item()
            total_loss += loss.item()
            et = time.time()

            if (index % 10 == 0):
                log_info = 'epoch {}:({}/{}) || running_step {}|| time {:.5f} || loss {:5f} || total loss {:5f} '.format(epoch, index, len(real_data_loader), (epoch - 1)*len(real_data_loader) + index, et - st, loss_value, total_loss)
                print(log_info)
                st = time.time()
                loss_value = 0
            
            craft.eval()
            

        torch.save(craft.state_dict(), os.path.join(saved, repr(epoch) + "_" + "{:.6s}".format(
            str(total_loss)) + '.pth'))
        torch.cuda.empty_cache()


if __name__ == '__main__':
    train()
