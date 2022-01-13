import cv2
import numpy as np
from PIL import Image

imagefilenpath = 'D:/DLAndApplication/project/data/vietnamese_original/train_images/im0010.jpg'
image = cv2.imread(imagefilenpath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
out_gray = cv2.divide(image, bg, scale=255)
out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
MAX_WIDTH = 1000
padding = np.ones((out_binary.shape[0], MAX_WIDTH - out_binary.shape[1]), dtype=np.uint8) * 255
out_binary = np.concatenate((out_binary, padding), axis=1)
three_channel = np.repeat(out_binary[np.newaxis, :, :], 3, axis=0)
print(three_channel.shape)
three_channel = three_channel.transpose(1, 2, 0)
img = Image.fromarray(three_channel)
img.save('./testt.jpg')
cv2.imwrite('./test.jpg', three_channel)