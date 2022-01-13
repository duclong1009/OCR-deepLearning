import argparse

def option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride_size",type=list, default=[(2,2),(2,2),(2,1),(2,1),(1,1)])
    parser.add_argument("--kernel_size",type=list, default=[(2,2),(2,2),(2,1),(2,1),(1,1)])
    parser.add_argument("--pretrained",type=bool, default=False)
    parser.add_argument("--emb_dim",type=int,default=320)
    parser.add_argument("--enc_hid_dim",type=int,default=216)
    parser.add_argument("--dec_hid_dim",type=int,default=216)
    parser.add_argument("--dropout",type=float,default=0.3)
    parser.add_argument("--lr",type=float,default=0.005)
    parser.add_argument("--epoch",type=int,default=10)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--num_workers",type=int,default=0)
    parser.add_argument("--max_len",type=int,default=100)
    parser.add_argument("--rnn_type",type=str, default='lstm')
    parser.add_argument("--img_channel",type=int,default=3)
    parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--name",type=str,default='test')
    args = parser.parse_args()
    return args