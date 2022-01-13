import os
#
# import sys
# print(sys.path)

import torch
from gaussianMap import imgproc
from gaussianMap.gaussian import GaussianTransformer
from net import craft
from net.torch_util import *
from torch.utils.data import Dataset
from watershed import *
from dataloader.Augmentation import *
from PIL import Image
import utils


class BaseDataset(Dataset):
    def __init__(self, target_size=768, viz=False, debug=False):
        super()
        self.target_size = target_size
        self.viz = viz
        self.debug = debug
        gaussian_heatmap_size = config.gaussian_heatmap_size # = 1024
        self.gaussianTransformer = GaussianTransformer(imgSize=gaussian_heatmap_size)

    def load_image_gt_and_confidencemask(self, index):
        return None, None, None, None, None

    def crop_image_by_bbox(self, image, box):
        """
        Crop image by box, after cropping we have image with shape maxWidth, maxHeight
        :param image:
        :param box:
        :return:
        """
        (tl, tr, br, bl) = box
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        # ...and now for the height of our new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        # take the maximum of the width and height values to reach
        # our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        width = maxWidth
        height = maxHeight
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen
        M = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
        maxHeight = height
        maxWidth = width
        warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warp, M

    def resizeGt(self, gtmask):
        """
        Resize gtmask (ori_image/2, ori_image/2)
        :param gtmask:
        :return:
        """
        return cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2))

    def get_image_name(self, index):
        return None

    def saveInput(self, imagename, image, region_scores, affinity_scores, confidence_mask):
        boxes, polys = utils.getDetBoxes(region_scores / 255, affinity_scores / 255, 0.7, 0.4, 0.4, False)
        boxes = np.array(boxes, np.int32) * 2
        if len(boxes) > 0:
            np.clip(boxes[:, :, 0], 0, image.shape[1])
            np.clip(boxes[:, :, 1], 0, image.shape[0])
            for box in boxes:
                cv2.polylines(image, [np.reshape(box, (-1, 1, 2))], True, (0, 0, 255))
        target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores / 255)
        target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores / 255)
        confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)
        gt_scores = np.hstack([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color])
        confidence_mask_gray = np.hstack([np.zeros_like(confidence_mask_gray), confidence_mask_gray])
        output = np.concatenate([gt_scores, confidence_mask_gray],
                                axis=0)
        output = np.hstack([image, output])
        outpath = os.path.join(os.path.join(os.path.dirname(__file__) + '/output'), "%s_input.jpg" % imagename)
        print(outpath)
        if not os.path.exists(os.path.dirname(outpath)):
            os.mkdir(os.path.dirname(outpath))
        cv2.imwrite(outpath, output)

    def saveImage(self, imagename, image, bboxes, affinity_bboxes, region_scores, affinity_scores, confidence_mask):
        output_image = np.uint8(image.copy())
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        if len(bboxes) > 0:
            affinity_bboxes = np.int32(affinity_bboxes)
            for i in range(affinity_bboxes.shape[0]):
                cv2.polylines(output_image, [np.reshape(affinity_bboxes[i], (-1, 1, 2))], True, (255, 0, 0))
            for i in range(len(bboxes)):
                _bboxes = np.int32(bboxes[i])
                for j in range(_bboxes.shape[0]):
                    cv2.polylines(output_image, [np.reshape(_bboxes[j], (-1, 1, 2))], True, (0, 0, 255))

        target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores / 255)
        target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores / 255)
        heat_map = np.concatenate([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color], axis=1)
        confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)
        output = np.concatenate([output_image, heat_map, confidence_mask_gray], axis=1)
        outpath = os.path.join(os.path.join(os.path.dirname(__file__) + '/output'), imagename)

        if not os.path.exists(os.path.dirname(outpath)):
            os.mkdir(os.path.dirname(outpath))
        cv2.imwrite(outpath, output)

    def pull_item(self, index):
        # image: image after transform (random_scale/jitter)
        # character_bboxes: [num_word, num_char, 4, 2]
        # words: [num_word]
        # confidence_mask: [H, W] (H - height of image, W - width of image)
        # confidences: (num_word, ): confidence of each word
        image, character_bboxes, words, confidence_mask, confidences = self.load_image_gt_and_confidencemask(index)

        if len(confidences) == 0:
            # Not have any words in image
            confidences = 1.0
        else:
            # calculate mean of all words' confidence
            confidences = np.array(confidences).mean()

        # shape: [H, W], all pixel score is set to zero
        region_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_bboxes = []

        if len(character_bboxes) > 0:
            affinity_scores, affinity_bboxes = self.gaussianTransformer.generate_affinity(region_scores.shape,
                                                                                          character_bboxes,
                                                                                          words)
            region_scores = self.gaussianTransformer.generate_region(region_scores.shape, character_bboxes)

        if self.viz:
            self.saveImage(self.get_image_name(index), image.copy(), character_bboxes, affinity_bboxes, region_scores,
                           affinity_scores, confidence_mask)
        random_transforms = [image, region_scores, affinity_scores, confidence_mask]
        random_transforms = random_crop(random_transforms, (self.target_size, self.target_size))
        random_transforms = random_horizontal_flip(random_transforms)
        random_transforms = random_rotate(random_transforms)

        cvimage, region_scores, affinity_scores, confidence_mask = random_transforms

        region_scores = self.resizeGt(region_scores)
        affinity_scores = self.resizeGt(affinity_scores)
        confidence_mask = self.resizeGt(confidence_mask)

        if self.viz:
            self.saveInput(self.get_image_name(index), cvimage, region_scores, affinity_scores, confidence_mask)

        image = Image.fromarray(cvimage)
        image = image.convert('RGB')
        # image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)

        image = imgproc.normalizeMeanVariance(np.array(image), mean=(0.485, 0.456, 0.406),
                                              variance=(0.229, 0.224, 0.225))
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        region_scores_torch = torch.from_numpy(region_scores / 255).float()
        affinity_scores_torch = torch.from_numpy(affinity_scores / 255).float()
        confidence_mask_torch = torch.from_numpy(confidence_mask).float()
        return image, region_scores_torch, affinity_scores_torch, confidence_mask_torch, confidences


def test_method_crop_image_by_box(image, boxes):
    for box in boxes:
        print('Box: ', box)
        warp, M = base_dataset.crop_image_by_bbox(image, box)
        print(warp.shape)
        showmat("crop_by_bounding_box", warp)


if __name__ == '__main__':
    base_dataset = BaseDataset(768)
    image = cv2.imread("test_folder/image/im1501.jpg")
    boxes, words = utils.load_boxes_in_gt('test_folder/gt/gt_im1501.txt')

    craft = craft.CRAFT(pretrained="model/pretrained/vgg16_bn-6c64b313.pth")
    craft.load_state_dict(
        copyStateDict(torch.load("model/pretrained/26_35.727.pth", map_location=torch.device("cpu"))))
