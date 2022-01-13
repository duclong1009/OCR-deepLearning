from src.loader.loader import MyDataset
from src.model.seqmodel.seqmodel import Seq2Seq,Encoder,Decoder,Attention
from src.model.backbone.vgg import VGG, vgg11_bn
import torch 
# from src.utils.config import Cfg
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from src.utils.Vocab import build_vocab
# def main():
#     args = parse
if __name__ == '__main__':
    vocab = build_vocab()
    dataset = MyDataset("src/data/vietnamese/train_images",vocab,32,8,80,transforms.ToTensor())
    dataloader = DataLoader(dataset,1)
    for i in dataloader:
        # breakpoint()
        print(i)
        break
    # print(data)