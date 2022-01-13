import numpy as np
import torch
import torch.nn as nn

THRESH_POSITIVE_REGION = 0.6
THRESH_POSITIVE_AFFINITY = 0.65
THRESH_POSITIVE_LOSS = 0.02
lambda_weight = 2

class MapLoss(nn.Module):
    def __init__(self, log_train, use_gpu=True):

        super(MapLoss, self).__init__()
        self.log_train = log_train

    def single_image_loss(self, pre_loss, label, region=True, thresh_positive=THRESH_POSITIVE_REGION):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1)) * 0
        # print("sum loss: ",sum_loss)
        pre_loss = pre_loss.view(batch_size, -1)
        label = label.view(batch_size, -1)
        for i in range(batch_size):
            mask_pos_pred = pre_loss[i][(label[i] > thresh_positive)]
            mask_neg_pred = pre_loss[i][(label[i] <= thresh_positive)]
            # print("sum loss: ", torch.sum(pre_loss[i]).item(),torch.sum(mask_pos_pred).item())
            positive_pixel = len(mask_pos_pred)
            negative_pixel = len(mask_neg_pred)
            if positive_pixel > 0:
                posi_loss = torch.mean(mask_pos_pred)
                sum_loss += posi_loss
                if negative_pixel < 3 * positive_pixel:
                    if negative_pixel > 0:
                        nega_loss = torch.mean(mask_neg_pred)
                    else:
                        nega_loss = torch.tensor(0.)
                else:
                    nega_loss = torch.mean(torch.topk(mask_neg_pred, 3 * positive_pixel)[0])
                sum_loss += nega_loss
            else:
                posi_loss = torch.tensor(0.)
                nega_loss = torch.mean(torch.topk(pre_loss[i], 500)[0])
                sum_loss += nega_loss + posi_loss
            # sum_loss += loss/average_number
            np_loss = sum_loss.item()
            str_loss = "region loss: " + str(np_loss) if region else "affine loss: " + str(np_loss)
            str_loss += ", neg pixel: " + str(negative_pixel) + ", pos pixel: " + str(positive_pixel)
            str_loss += ", neg loss: " + str(nega_loss.item()) + ", pos loss: " + str(posi_loss.item())
            self.log_train.write(str_loss)
        return sum_loss

    def forward(self, region_score_gt, affinity_score_gt, region_score_pred, affinity_score_pred, mask):
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert region_score_pred.size() == region_score_gt.size() and affinity_score_pred.size() == affinity_score_gt.size()
        log_infor = ""
        # for idx in range((region_score_gt.shape[0])):
        # 	log_infor+="max value: {}, {}, {}, {}\n".format(torch.max(region_score_pred[idx]).item(),torch.max(affinity_score_pred[idx]).item(),torch.max(region_score_gt[idx]).item(),torch.max(affinity_score_gt[idx]).item())
        # self.log_train.write(log_infor)
        # char
        mask_pos_gh = torch.gt(region_score_gt, THRESH_POSITIVE_REGION)
        mask_pos_pred_gh = torch.gt(region_score_pred, 1)
        mask_pos_pred_gh_greater_one = mask_pos_gh & mask_pos_pred_gh
        region_score_pred = torch.where(mask_pos_pred_gh_greater_one, torch.ones_like(region_score_pred), region_score_pred)
        loss1 = loss_fn(region_score_pred, region_score_gt)

        # affine
        mask_pos_gah = torch.gt(affinity_score_gt, THRESH_POSITIVE_AFFINITY)
        mask_pos_pred_gah = torch.gt(affinity_score_pred, 1)
        mask_pos_pred_gah_greater_one = mask_pos_gah & mask_pos_pred_gah
        affinity_score_pred = torch.where(mask_pos_pred_gah_greater_one, torch.ones_like(affinity_score_pred), affinity_score_pred)
        loss2 = loss_fn(affinity_score_pred, affinity_score_gt)
        log_infor = ""
        # for idx in range((region_score_gt.shape[0])):
        # 	log_infor+="max value before: {}, {}, {} \n".format(torch.max(loss1[idx]).item(),torch.max(loss2[idx]).item(),torch.max(mask[idx]).item())
        # self.log_train.write(log_infor)
        loss_g = torch.mul(loss1, mask)
        loss_a = torch.mul(loss2, mask)
        log_infor = ""
        # for idx in range((region_score_gt.shape[0])):
        # 	log_infor+="max value after: {}, {}, {}\n".format(torch.max(loss_g[idx]).item(),torch.max(loss_a[idx]).item(),torch.max(mask[idx]).item())
        # self.log_train.write(log_infor)
        char_loss = self.single_image_loss(loss_g, region_score_gt, region=True, thresh_positive=THRESH_POSITIVE_REGION)
        affi_loss = self.single_image_loss(loss_a, affinity_score_gt, region=False, thresh_positive=THRESH_POSITIVE_AFFINITY)
        # print("shape: ",loss_g.shape[0],loss_a.shape[0])
        return lambda_weight * char_loss / loss_g.shape[0] + affi_loss / loss_a.shape[0]
