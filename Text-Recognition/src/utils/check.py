from torch.utils.data import DataLoader, Dataset 
import json
import torch
from PIL import Image
import os.path as osp
from utils import resize, process_image
from Vocab import build_vocab
import numpy as np
import cv2
from tqdm.auto import tqdm

class MyDataset(Dataset):
    def __init__(self,label_path,root=None,vocab = None,expected_height=32, min_width=8, max_width= 80,transform=None):
        self.expected_height = expected_height
        self.min_width = min_width
        self.max_width = max_width
        # self.root = root
        self.root = "D:/DLAndApplication/project/data/vietnamese_original/train_images"
        self.transform = transform
        with open("gen-data/" + label_path,'r') as f:
            self.dict_data = json.load(f)
        self.vocab = vocab

    def __len__(self):
        # return len(self.flist)
        return len(self.dict_data)
        return 100

    def processed_img(self, img):
        img = process_image(img, self.expected_height, self.min_width, self.max_width)
        # img = self.convert_to_gray_image(img)
        if self.transform:
            img = self.transform(img)
        # print(img)
        # img = img/255
        # print(img)
        return img

    def convert_to_gray_image(self, img):
        image = np.asarray(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
        out_gray = cv2.divide(image, bg, scale=255)
        out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
        padding = np.ones((out_binary.shape[0], self.max_width - out_binary.shape[1]), dtype=np.uint8) * 255
        out_binary = np.concatenate((out_binary, padding), axis=1)
        three_channel = np.repeat(out_binary[np.newaxis, :, :], 3, axis=0)
        three_channel = three_channel.transpose(1, 2, 0)
        return Image.fromarray(three_channel)

    def __getitem__(self, idx):
        # breakpoint()
        data = self.dict_data[idx]
        img_path = osp.join(self.root, data['image'])
        img = Image.open(img_path).convert("RGB")
        # code crop image
        bbox = data['box']
        img,_ = self.crop_image_by_bbox(img, bbox)
        processed_img = self.processed_img(img)
        label = data['label'].lower()
        encoded_label = self.vocab.encode(label)
        sample = {"img": processed_img, "label" :np.array(encoded_label) }
        return sample

    def crop_image_by_bbox(self, image, box):
        """
        Crop image by box, after cropping we have image with shape maxWidth, maxHeight
        :param image:
        :param box:
        :return:
        """
        # Convert from PIL Image to cv2
        cv_image = np.array(image)
        cv_image = cv_image[:, :, ::-1]
        box = np.array(box)

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
        M = cv2.getPerspectiveTransform(np.array(box, dtype=np.float32), dst)
        maxHeight = height
        maxWidth = width
        warp = cv2.warpPerspective(cv_image, M, (maxWidth, maxHeight))
        warp_pil = Image.fromarray(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB))
        return warp_pil, M

if __name__ == '__main__':
    vocab = build_vocab()
    train_dataset = MyDataset("train.json","src/data/vietnamese/train_images",vocab,32,8,80)
    # my_set = set()
    # for i in tqdm(range(len(train_dataset))):
    #     my_set.add(np.asarray(train_dataset[i]['img']).shape[1])
    # print(my_set)
    # train_dataset[1000]['img'].save('./test.jpg')
    width_dict = dict()
    for i, sample in tqdm(enumerate(train_dataset)):
        width = np.asarray(sample['img']).shape[1]
        if width in width_dict.keys():
            width_dict[width].append(i)
        else:
            width_dict[width] = [i]
    with open('./train-width-cluster.json', 'w') as f:
        json.dump(width_dict, f)
