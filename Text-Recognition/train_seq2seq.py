from src.model.model import MyModel
from src.loader.loader import MyDataset
from src.model.seqmodel.seqmodel import Seq2Seq,Encoder,Decoder,Attention
from src.model.backbone.vgg import VGG, vgg11_bn
import torch 
import torch.nn as nn
import numpy as np
import random
import argparse
from src.utils.option import option
from src.model.seqmodel.seqmodel import Seq2Seq,Encoder,Decoder,Attention
from torch.utils.data import DataLoader
# from src.utils.trainer import train, test, eval
from src.utils.Vocab import build_vocab
from src.utils.utils import save_model
# import wandb
from torchvision import transforms

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    # vocab = Vocab()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # stride_size = [(2,2),(2,2),(2,1),(2,1),(2,1)]
    # kernel_size = [(2,2),(2,2),(2,1),(2,1),(2,1)]
    vocab = build_vocab()
    # cnn = vgg11_bn(stride_size,kernel_size,args.emb_dim,pretrained=False)
    # seq2seq = Seq2Seq(len(vocab)+1,args.enc_hid_dim,args.dec_hid_dim,args.emb_dim,args.emb_dim,args.dropout,args.rnn_type)
    # model = MyModel(cnn,seq2seq)
    # optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    # criterion = nn.CrossEntropyLoss()
    # model.to(device)
    
    train_dataset = MyDataset("train.json","src/data/vietnamese/train_images",vocab,32,8,80,transforms.ToTensor())
    train_dataset[0]
    train_dataset[1]
    print('checking done')
    # train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True)
    # test_dataset = MyDataset("test.json","src/data/vietnamese/test_images",vocab,32,8,80,transforms.ToTensor())
    # test_loader = DataLoader(train_dataset,batch_size=1,shuffle=False)
    # for epoch in range(args.epoch):
    #     train_loss =train(model,train_loader,optimizer,criterion,device)
    #     y, y_pred = test(model, test_loader, device)
    #     acc = eval(y_pred,y)
    #     save_model(model,"checkpoint/model_"+str(epoch)+".pth")
    #     wandb.log({"train_loss" : train_loss})
    #     wandb.log({"acc" : acc})

    
if __name__ == '__main__':
    args =  option()
    # try:
    #     config = vars(args)
    # except IOError as msg:
    #     args.error(str(msg))
    # wandb.init(entity="aiotlab",
    #     project="Spatial_PM2.5",group="OCR", name=args.name,config=config)
    main(args)

    
