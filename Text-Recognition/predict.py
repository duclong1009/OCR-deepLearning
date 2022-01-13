

from hashlib import md5
from src.model.model import MyModel
from src.loader.loader import MyDataset
from src.model.seqmodel.seqmodel import Seq2Seq, Encoder, Decoder, Attention
from src.model.backbone.vgg import VGG, vgg11_bn
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
from src.utils.option import option
from src.model.seqmodel.seqmodel import Seq2Seq, Encoder, Decoder, Attention
from torch.utils.data import DataLoader
from src.utils.trainer import train, test, eval
from src.utils.Vocab import build_vocab
from src.utils.utils import EarlyStopping, load_model, save_model
import wandb
from torchvision import transforms


if __name__ == '__main__':
    args = option()
    device = torch.device("cpu")
    stride_size = [(2, 2), (2, 2), (2, 2), (2, 1), (2, 1)]
    kernel_size = [(2, 2), (2, 2), (2, 2), (2, 1), (2, 1)]
    vocab = build_vocab()
    cnn = vgg11_bn(stride_size, kernel_size, args.emb_dim, pretrained=False)
    seq2seq = Seq2Seq(
        len(vocab) + 1,
        args.enc_hid_dim,
        args.dec_hid_dim,
        args.emb_dim,
        args.emb_dim,
        args.dropout,
        args.rnn_type,
    )
    model = MyModel(cnn, seq2seq)
    load_model(model, torch.load("checkpoint/model.pt"))
    model.to(device)
    # image = torch.rand(1,3,32,80)
    test_dataset = MyDataset(
        "test.json",
        "src/data/vietnamese/test_image",
        vocab,
        32,
        8,
        80,
        transforms.ToTensor(),
    ) 
    for i in test_dataset:
        image = i["img"]
        label = i['label']
        image = torch.tensor(image).to(device)
        label = torch.tensor(i['label']).to(device)
        image = torch.unsqueeze(image, 0)
        label = torch.unsqueeze(label,0)
        # output = model(image)

        output = model(image, label)
        print(label)
        print(torch.argmax(output, dim=2))
        print("------------------")
        # break
    #     print(image.shape)
    #     image = image.to(device)
    #     image = torch.unsqueeze(image, 0)
    #     cnn_output = model.cnn(image)
    #     print(cnn_output.shape)
    #     output = torch.zeros(cnn_output.shape[0],1)
    #     output = output.to(device)
    # # breakpoint()
    #     hidden,output_encoder = model.seq2seq.forward_encoder(cnn_output)
    #     sequence_length = output_encoder.shape[0]
    #     output = torch.zeros(sequence_length+1,1)
    #     output[0] = 1
    #     for i in range(sequence_length):
    #         output_decoder,(hidden,output_encoder) = model.seq2seq.forward_decoder(output[:i+1],(hidden,output_encoder))
    #         print(output_decoder.shape)
    #         # print(i)
    #         output[i+1] = output_decoder.argmax(-1)
        # print(output.shape)
        # print(output)

