import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self,cnn,seq2seq):
        super().__init__()
        self.cnn = cnn
        self.seq2seq = seq2seq

    def forward(self,image,trg):
        # breakpoint()
        input = self.cnn(image)
        output = self.seq2seq(input,trg)
        return output

    # def inference(self,image):
    #     input = self.cnn(image)
    #     output = self.seq2seq.inference(input)
    #     return output