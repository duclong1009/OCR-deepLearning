import wandb
from tqdm.auto import tqdm
import torch

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for i in tqdm(dataloader):
        # breakpoint()
        optimizer.zero_grad()
        x = i["img"].to(device)
        y = i["label"].to(device)
        y_pred = model(x,y)
        y_pred = y_pred.reshape(-1,y_pred.shape[-1])
        y = y.reshape(-1)
        loss = criterion(y_pred.float(),y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(dataloader)
import numpy as np
def test(model,criterion,dataloader,device):
    model.eval()
    total_loss = 0
    y_pred_arr = []
    y_grt_arr = []
    with torch.no_grad():
        for i in tqdm(dataloader):
            x = i["img"].to(device)
            y = i["label"].to(device)
            y_pred = model(x,y)
            total_loss += criterion(y_pred.reshape(-1,y_pred.shape[-1]).float(),y.reshape(-1)).item()
            y_pred = torch.argmax(y_pred,dim=2)
            # breakpoint()
            y_pred =  torch.squeeze(y_pred).cpu().detach().numpy()
            y = torch.squeeze(y).cpu().detach().numpy()
            y_pred_arr.append(y_pred)
            y_grt_arr.append(y)
            
    return total_loss/len(dataloader),np.concatenate(y_grt_arr),np.concatenate(y_pred_arr)
from sklearn.metrics import accuracy_score,f1_score

def eval(y_pred,y):
    # breakpoint()
    y= y.reshape(-1)
    y_pred = y_pred.reshape(-1)
    acc = accuracy_score(y,y_pred)
    # f1 =f1_score(y,y_pred,average="macro")
    # result = {"acc":acc, "f1":f1}
    return acc        