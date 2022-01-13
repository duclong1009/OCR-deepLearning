from dataloader.BaseDataset import BaseDataset
from dataloader.Augmentation import *
import random
import torchvision.transforms as transforms
import glob
import config
import os
import numpy as np
from net import craft, torch_util
import torch
import scipy.io as scio
import re
import itertools

class SynthTextDataset(BaseDataset):
    def __init__(self, synthtext_folder, target_size=768, viz=False, debug=False):
        super(SynthTextDataset, self).__init__(target_size, viz, debug)
        self.synthtext_folder = synthtext_folder
        gt = scio.loadmat(os.path.join(synthtext_folder, 'gt.mat'))
        # shape = [num_image, 2, 4, num_box]
        self.charbox = gt['charBB'][0]
        # shape = [num_image, 1], 1: image_path
        self.image = gt['imnames'][0]
        # shape = [num_image, num_box]
        self.imgtxt = gt['txt'][0]

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.imgtxt)

    def get_image_name(self, index):
        return self.image[index][0]

    def load_image_gt_and_confidencemask(self, index):
        # Read image
        img_path = os.path.join(self.synthtext_folder, self.get_image_name(index))
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Change from (2, 4, num_box) => (num_box, 4, 2)
        _charbox = self.charbox[index].transpose((2, 1, 0))

        # Transform image: Random resize
        image = random_scale(image, _charbox, self.target_size)

        # Load words in image from character
        words = [re.split(' \n|\n |\n| ', t.strip()) for t in self.imgtxt[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]

        character_bboxes = []
        total = 0
        # Store confidence of each word
        confidences = []
        # Store confidence map
        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)
        for i in range(len(words)):
            bboxes = _charbox[total:total + len(words[i])]
            assert (len(bboxes) == len(words[i]))
            total += len(words[i])
            bboxes = np.array(bboxes)
            character_bboxes.append(bboxes)
            confidences.append(1.0)

        # Character_bboxes: [num_word, num_char, 4, 2]
        # Confidences: [num_word], always = 1 because groundtruth is true
        return image, character_bboxes, words, confidence_mask, confidences

if __name__ == '__main__':
    synthtext_dataset = SynthTextDataset(config.synth_data, target_size=768)

    image0_name = synthtext_dataset.get_image_name(0)
    image0_path = os.path.join(config.synth_data, image0_name)
    image = cv2.imread(image0_path)
    # showmat("Original image", image)
    cv2.imwrite("test_folder/synthtext/Image 0.jpg", image)

    synthtext_dataset.load_image_gt_and_confidencemask(0)
    print("SynthText Image 0: \n", synthtext_dataset[0])
