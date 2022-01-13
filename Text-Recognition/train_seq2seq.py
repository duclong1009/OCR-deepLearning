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
from src.utils.utils import EarlyStopping, save_model
import wandb
from torchvision import transforms


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    # device = torch.device("cpu")
    # vocab = Vocab()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    train_dataset = MyDataset(
        "train.json",
        "src/data/vietnamese/train_images",
        vocab,
        32,
        8,
        80,
        transforms.ToTensor(),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = MyDataset(
        "test.json",
        "src/data/vietnamese/test_image",
        vocab,
        32,
        8,
        80,
        transforms.ToTensor(),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    early_stopping = EarlyStopping(
        patience=5, verbose=True, path="checkpoint/model.pt", delta=0.005
    )
    for epoch in range(args.epoch):
        if not early_stopping.early_stop:
            train_loss = train(model, train_loader, optimizer, criterion, device)
            test_loss, y, y_pred = test(model, criterion, test_loader, device)
            acc = eval(y_pred, y)
            save_model(model, "checkpoint/model_fixed_" + str(epoch) + ".pth")
            wandb.log(
                {"loss/train_loss": train_loss, "loss/val_loss": test_loss, "acc": acc}
            )
            wandb.log({"acc": acc})
            early_stopping(acc, model)


if __name__ == "__main__":
    args = option()
    try:
        config = vars(args)
    except IOError as msg:
        args.error(str(msg))
    wandb.init(
        entity="aiotlab",
        project="Spatial_PM2.5",
        group="DL_OCR_fixed",
        name=args.name,
        config=config,
    )
    main(args)
    wandb.finish()
