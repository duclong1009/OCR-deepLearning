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
    # breakpoint()
    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)
    
    img = img.resize((new_w, image_height), Image.ANTIALIAS)
    img = np.array(img)
    # img = np.asarray(img).transpose(2,0, 1)
    pad = np.zeros((image_height, image_max_width-new_w,3), dtype=np.float32)
    img = np.concatenate((img, pad), axis=1)
    img = img/255
    return img
import torch
def save_model(model,path):
    print("Saving model")
    checkpoints = {"model_dict": model.state_dict()}
    torch.save(checkpoints, path)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=3, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation acc increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        checkpoints = {"model_dict": model.state_dict()}
        torch.save(checkpoints, self.path)
        self.val_loss_min = val_loss


def load_model(model, checkpoints):
    print("Loading model")
    model.load_state_dict(checkpoints["model_dict"])