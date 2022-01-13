from dataloader.BaseDataset import BaseDataset
from dataloader.Augmentation import *
from gaussianMap import imgproc
from watershed import *
from net import craft, torch_util

import random
import torchvision.transforms as transforms
import glob
import config
import os
import numpy as np
import torch

class WordLevelDataset(BaseDataset):
    def __init__(self, net, vintext_folder, target_size=768, viz=False, debug=False):
        super(WordLevelDataset, self).__init__(target_size, viz, debug)
        self.net = net
        # Contains absolute path of images and gts
        self.image_paths, self.gt_paths = self.make_data_path(vintext_folder)
        # Contains image name
        self.image_names = []
        for image_path in self.image_paths:
            self.image_names.append(os.path.basename(image_path))

    def make_data_path(self, folder):
        imgs = []
        gts = []
        img_folder = os.path.join(folder, 'imgs')
        gt_folder_path = os.path.join(folder, "gt")
        gt_folder = os.listdir(gt_folder_path)
        img_files = glob.glob(os.path.join(img_folder, "*.jpg"))
        for img_path in img_files:
            img_name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
            gt_fname = "gt_{}.txt".format(img_name_without_ext)
            # If gt respective with image not found, ignore this image
            if gt_fname in gt_folder:
                imgs.append(img_path)
                gts.append(os.path.join(gt_folder_path, gt_fname))
        return imgs, gts

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.image_paths)

    def get_image_name(self, index):
        return self.image_names[index]

    def check_valid_box(self, box):
        (tl, tr, br, bl) = box
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        if widthA < 1 or widthB < 1 or heightA < 1 or heightB < 1:
            return False
        return True

    def get_confidence(self, real_len, pursedo_len):
        if pursedo_len == 0:
            return 0.
        return (real_len - min(real_len, abs(real_len - pursedo_len))) / real_len

    def inference_pursedo_bboxes(self, net, image, word_bbox, word, viz=False, idx=-1, height_of_box=64.0
                                 , expand_small_box=5):
        """
        Inference pursedo character box for each word box
        :param net: craft
        :param image: origin image contains word box
        :param word_bbox: coordinates of the word box in iamge
        :param word: word text
        :param viz:
        :param idx:
        :param height_of_box: height of input cropped word of word_image_out_from_craft (64, scale_width) => output (32, scale_width/2)
        :param expand_small_box: expand box if output charbox too small
        :return bboxes: the character boxes in input image, shape: [num_char, 4, 2]
        :return region_scores: of word, shape = [32, scale_width/2]
        :return confidence: confidence scalar of word
        """

        word_image, MM = self.crop_image_by_bbox(image, word_bbox)
        cv2.imwrite('test_folder/vis/word_image.png', word_image)
        # print(word_bbox)
        # showmat("img", word_image)
        real_word_without_space = word.replace('\s', '')
        # cv2.imwrite('./vis_test/word_image'+real_word_without_space+'.png',word_image)
        real_char_nums = len(real_word_without_space)
        input = word_image.copy()
        scale = height_of_box / input.shape[0]
        # resize word box with scale (64/h_word)
        input = cv2.resize(input, None, fx=scale, fy=scale)
        # cv2.imwrite('./vis_test/input_after_resize.png',input)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_torch = torch.from_numpy(imgproc.normalizeMeanVariance(input, mean=(0.485, 0.456, 0.406),
                                                                   variance=(0.229, 0.224, 0.225)))
        img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
        img_torch = img_torch.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            scores, _ = net(img_torch)
        region_scores = scores[0, :, :, 0].cpu().data.numpy()
        region_scores = np.uint8(np.clip(region_scores, 0, 1) * 255)
        # resize to input image
        gray_region_scores = cv2.resize(region_scores, (input.shape[1], input.shape[0]))
        bgr_region_scores = cv2.cvtColor(gray_region_scores, cv2.COLOR_GRAY2BGR)
        cv2.imwrite('test_folder/vis/bgr_region_scores.png', bgr_region_scores)

        # generate coordinates of character boxes in the word_image: pursedo_bboxes
        pursedo_bboxes = watershed(input, bgr_region_scores, idx=idx, viz=False)
        # for i in range(pursedo_bboxes.shape[0]):
        #     print(pursedo_bboxes[i])

        _tmp = []
        for i in range(pursedo_bboxes.shape[0]):
            # numpy.ravel() == numpy.reshape(-1)
            if np.mean(pursedo_bboxes[i].ravel()) > 2:
                _tmp.append(pursedo_bboxes[i])
            else:
                print("filter bboxes", pursedo_bboxes[i])
        # Convert to dtype float 32
        pursedo_bboxes = np.array(_tmp, np.float32)
        # If has more than 1 char box:
        if pursedo_bboxes.shape[0] > 1:
            # np.argsort: sort depend top-left of each box and return the sorting indices in pursedo_bboxes
            index = np.argsort(pursedo_bboxes[:, 0, 0])
            pursedo_bboxes = pursedo_bboxes[index]

        # Calculate Sconf(word)
        confidence = self.get_confidence(real_char_nums, len(pursedo_bboxes))

        bboxes = []
        # if confidence <= 0.5, split word follow len of word
        if confidence <= 0.5:
            width = input.shape[1]
            height = input.shape[0]

            width_per_char = width / len(word)
            for i, char in enumerate(word):
                if char == ' ':
                    continue
                left = i * width_per_char
                right = (i + 1) * width_per_char
                bbox = np.array([[left, 0], [right, 0], [right, height],
                                 [left, height]])
                bboxes.append(bbox)

            bboxes = np.array(bboxes, np.float32)
            confidence = 0.5

        else:
            bboxes = pursedo_bboxes
        bboxes /= scale
        try:
            for j in range(len(bboxes)):
                I = np.matrix(MM).I
                # Reverse from pts of wordbox to pts in image
                bb = bboxes[j]
                bb = bb.reshape((8))
                x1, y1, x2, y2, x3, y3, x4, y4 = bb
                min_x = min(abs(x1 - x2), abs(x3 - x4))
                min_y = min(abs(y1 - y4), abs(y2 - y3))
                if min_x < expand_small_box:
                    x1 = max(0, x1 - expand_small_box)
                    x2 = min(image.shape[1], x2 + expand_small_box)
                    x3 = min(image.shape[1], x3 + expand_small_box)
                    x4 = max(0, x4 - expand_small_box)
                if min_y < expand_small_box:
                    y1 = max(0, y1 - expand_small_box)
                    y2 = max(0, y2 - expand_small_box)
                    y3 = min(image.shape[0], y3 + expand_small_box)
                    y4 = min(image.shape[0], y4 + expand_small_box)
                bb = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape(4, 2)
                bb = bb.reshape(1, bb.shape[0], bb.shape[1])
                bb = cv2.perspectiveTransform(bb, I)
                bboxes[j] = bb.reshape((4, 2))
        except Exception as e:
            print(e, word_bbox, word)

        # Clip each char box into boundary of image
        bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0., image.shape[0] - 1)
        bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0., image.shape[1] - 1)

        return bboxes, region_scores, confidence

    def load_image_gt_and_confidencemask(self, index):
        image_path = self.image_paths[index]
        file_name = self.image_names[index]
        gt_path = self.gt_paths[index]

        # list of word boxes and words
        # word_bboxes: (num_word, 4, 2), words: (num_word, )
        word_bboxes, words = self.load_gt(gt_path)
        word_bboxes = np.float32(word_bboxes)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform image:
        if random.random() < 0.3:
            # Change the brightness, contrast, saturation and hue of image
            jitter = color_jitter_image()
            image = jitter(transforms.ToPILImage()(image))
            image = np.array(image)
        # Change size of image
        image = random_scale(image, word_bboxes, self.target_size)

        # Confidence mask in initial: (H, W), all pixel = 1
        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)
        character_bboxes = []
        new_words = []
        confidences = []
        height_of_box = config.height_of_box
        expand_small_box = config.expand_small_box
        if len(word_bboxes) > 0:
            # For each word box:
            for i in range(len(word_bboxes)):
                # Fill confidence_mask's pixel to 0 if word == '###' or word_box is not valid
                if words[i] == '###' or len(words[i].strip()) == 0:
                    cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (0))
                elif not self.check_valid_box(word_bboxes[i]):
                    cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (0))

            for i in range(len(word_bboxes)):
                if words[i] == '###' or len(words[i].strip()) == 0 or not self.check_valid_box(word_bboxes[i]):
                    continue

                # Inference character boxes and confidence for each word:
                # pursedo_bboxes: (num_char, 4, 2)
                # bbox_region_scores: (32, w_word * scale /2)
                # confidence: confidence of this word
                pursedo_bboxes, bbox_region_scores, confidence = self.inference_pursedo_bboxes(self.net, image,
                                                                                               word_bboxes[i],
                                                                                               words[i],
                                                                                               viz=self.viz, idx=i,
                                                                                               height_of_box=height_of_box,
                                                                                               expand_small_box=expand_small_box)

                confidences.append(confidence)
                # Confidence mask will be more and more filled up by each word
                cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (confidence))
                new_words.append(words[i])
                character_bboxes.append(pursedo_bboxes)
        return image, character_bboxes, new_words, confidence_mask, confidences

    def load_gt(self, gt_path):
        lines = open(gt_path, encoding='utf-8').readlines()
        bboxes = []
        words = []
        for line in lines:
            ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
            box = [int(ori_box[j]) for j in range(8)]
            word = ori_box[8:]
            word = ','.join(word)
            # print(word)
            box = np.array(box, np.int32).reshape(4, 2)
            if word == '###':
                words.append('###')
                bboxes.append(box)
                continue
            if len(word.strip()) == 0:
                continue
            bboxes.append(box)
            words.append(word)
        return bboxes, words

def test_method_inference_pseudo_boxes(word_dataset, net, image, word_bbox, word, viz=False, idx=-1, height_of_box=64.0
                                       , expand_small_box=5):
    word_dataset.inference_pursedo_bboxes(net, image, word_bbox, word, viz=False, idx=-1, height_of_box=64.0
                                          , expand_small_box=5)

if __name__ == '__main__':
    craft = craft.CRAFT(pretrained="model/pretrained/vgg16_bn-6c64b313.pth")
    craft.load_state_dict(
        torch_util.copyStateDict(torch.load("model/pretrained/26_35.727.pth", map_location=torch.device("cpu"))))

    vintext_dataset = WordLevelDataset(craft, config.word_data, target_size=768, viz=True, debug=True)
    print(vintext_dataset[0])