import cv2
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm.auto import tqdm

def process_label(line):
    split_result = line.split(',', 8)
    label = split_result[-1]
    if label == '###':
        return None
    coordinates = [int(e) for e in split_result[:8]]
    tl = (coordinates[0], coordinates[1])
    tr = (coordinates[2], coordinates[3])
    br = (coordinates[4], coordinates[5])
    bl = (coordinates[6], coordinates[7])
    box = (tl, tr, br, bl)
    return box, label

if __name__ == '__main__':
    root_dir = 'D:/DLAndApplication/project/data/vietnamese_original'
    label_files = os.listdir(os.path.join(root_dir, 'labels'))
    train = list()
    val = list()
    test = list()
    for file in tqdm(label_files[:]):
        index = int(file[3:].split('.', 1)[0])
        with open(os.path.join(root_dir, 'labels', file), 'r', encoding='utf-8') as f:
          lines = [line[:-1] if line.endswith('\n') else line for line in f.readlines()]
        for line in lines:
            try:
                box, label = process_label(line)
                tmp = {
                    'image': "im%04d.jpg"%(index),
                    'label': label,
                    'box': box
                }
                if index <= 1200:
                    train.append(tmp)
                elif index <= 1500:
                    test.append(tmp)
                else:
                    val.append(tmp)
            except:
                pass

    with open('./train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f)
    with open('./val.json', 'w', encoding='utf-8') as f:
        json.dump(val, f)
    with open('./test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f)
