import json
import math 

def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w/round_to)*round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)
    return new_w, expected_height
from PIL import Image
import numpy as np

def process_image(image, image_height, image_min_width, image_max_width):
    img = image.convert('RGB')

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.ANTIALIAS)

    # img = np.asarray(img).transpose(2,0, 1)
    # img = img/255
    return img
import torch
def save_model(model,path):
    print("Saving model")
    checkpoints = {"model_dict": model.state_dict()}
    torch.save(checkpoints, path)