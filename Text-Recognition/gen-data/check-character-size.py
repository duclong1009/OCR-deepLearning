import cv2
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm.auto import tqdm
import sys
import math

def process_label(line):
    split_result = [e for e in line.split(',') if e != '']
    label = split_result[-1]
    if label == '###':
        return None
    coordinates = [int(e) for e in split_result[:-1]]
    xmin = min(coordinates[0], coordinates[6])
    ymin = min(coordinates[1], coordinates[3])
    xmax = max(coordinates[2], coordinates[4])
    ymax = max(coordinates[5], coordinates[7])
    return xmin, ymin, xmax, ymax, label

if __name__ == '__main__':
    root_dir = 'D:/DLAndApplication/project/data/vietnamese_original'
    max_width = 0
    max_height = 0
    myset = set()
    mysecondset = set()
    label_files = os.listdir(os.path.join(root_dir, 'labels'))
    for file in tqdm(label_files[:]):
        index = int(file[3:].split('.', 1)[0])
        with open(os.path.join(root_dir, 'labels', label_files[0]), 'r', encoding='utf-8') as f:
          lines = [line[:-1] if line.endswith('\n') else line for line in f.readlines()]
        for line in lines:
            try:
                xmin, ymin, xmax, ymax, label = process_label(line)
                length = len(label)
                max_width = max(max_width, int(math.ceil((xmax - xmin) / length)))
                max_height = max(max_height, ymax - ymin)
                myset.add(ymax - ymin)
                mysecondset.add(int(math.ceil((xmax - xmin) / length)))
            except:
                pass
    print(max_width)
    print(mysecondset)
    print(max_height)
    print(myset)