#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:06:24 2020

@author: zzy
"""

"---import relavant libraries---"
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm
import csv
import torch.nn.functional as F
from numpy import random
from sklearn.metrics import roc_auc_score
import pandas as pd


"---data_read part---"


def get_data(path):
    if path=="/home/zzy/train_val":
        csv_file=csv.reader(open("/home/zzy/桌面/code/train_val.csv",'r'))
    else:
        csv_file=csv.reader(open("/home/zzy/桌面/code/test.csv",'r'))
    cont=[]
    for line in csv_file:
        cont.append(line)
    data_len=len(cont)
    data_voxel=[]
    data_seg=[]
    label=[]
    
    for i in range(1,data_len):
        data_path=path+'/'+cont[i][0]+".npz"
        tmp=np.load(data_path)
        y=[];h=[]
        seg=tmp['seg'];voxel=tmp['voxel']*(tmp['seg']*0.8+0.2)
        seg=seg.astype(int);voxel=voxel/255
        seg=seg.tolist()
        for j in seg[30:70]:
            x=[]
            for k in j[30:70]:
                x.append(k[30:70])
            y.append(x)
        for j in voxel[30:70]:
            x=[]
            for k in j[30:70]:
                x.append(k[30:70])
            h.append(x)
        data_seg.append(y);data_voxel.append(h)
        la=float(int(cont[i][1])-int('0'))
        label.append(la)
    return data_seg,label,cont,data_voxel


"---data preprocess---"
def Cutout(tube,n_holes,length):
    d = tube.size(2)
    h = tube.size(3)
    w = tube.size(4)

    mask = np.ones((d,h, w), np.float32)

    for n in range(n_holes):
        mask_d = np.random.randint(d-length)
        mask_h = np.random.randint(h-length)
        mask_w = np.random.randint(w-length)

        mask[mask_d:mask_d+length, mask_h:mask_h+length, mask_w:mask_w+length] = 0.

    mask = torch.from_numpy(mask)
    mask = mask.expand_as(tube)
    tube = tube * mask

    return tube


"---produce own net---"
'using Lenet-5'

num_classes = 2
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeNet3D(nn.Module):
    def __init__(self):
        super(LeNet3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(16, 64, kernel_size=3)
        self.fc1 = nn.Linear(32768, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32768)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x=self.sigmoid(x)
        return x

"---test model on test data---"

'using gpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(model,data_test,conte,filename):
    file=open(filename,'w',newline='')
    csv_write=csv.writer(file)
    csv_write.writerow(["Id","Predicted"])
    i=0
    model.eval()
    for data in tqdm(data_test):
        voxel=np.array(data)
        voxel=torch.tensor(voxel).float()
        voxel=voxel.view(-1,1,40,40,40)
        voxel=voxel.to(device)
        out=model(voxel)
        a=out.tolist()
        row=[conte[i+1][0],a[0][1]/(a[0][0]+a[0][1])]
        csv_write.writerow(row)
        i+=1
    file.close()

def test_model1(model,data_test,conte,filename):
    file=open(filename,'w',newline='')
    csv_write=csv.writer(file)
    i=0
    model.eval()
    for data in tqdm(data_test):
        voxel=np.array(data)
        voxel=torch.tensor(voxel).float()
        voxel=voxel.view(-1,1,40,40,40)
        voxel=voxel.to(device)
        out=model(voxel)
        a=out.tolist()
        row=[conte[i+1][0],a[0][1]/(a[0][0]+a[0][1])]
        csv_write.writerow(row)
        i+=1
    file.close()

"---train model on train data---"
num_classes = 2
NUM_EPOCHS = 15
BATCH_SIZE=12

normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def My_mixup(x,y,alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    x1=Variable(lam*x+(1.-lam)*y)
    return x1,lam
"------------------------------------------------- train ------------------------------------------------------------------"
def trainmodel(model,data_train,data_label,c):
    print('-------------------------training-----------------------------------')
    model.train()
    criterion = nn.BCELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    all_loss=[] ; x_axi=0 ; a=[] ; brea_num=0 ; tmp_num=0.
    zhemesize=int(465/BATCH_SIZE+1)
    all_batch=1
    tmp_loss=0.
    for nu in range(zhemesize):
        a.append(nu*BATCH_SIZE)
    for epoch in range(NUM_EPOCHS): 
        wez=np.arange(465)
        weizhi1=wez.tolist(); random.shuffle(weizhi1)
        data_train1=[] ; data_label1=[]
        for dii in range(465):
            data_train1.append(data_train[weizhi1[dii]])
            data_label1.append(data_label[weizhi1[dii]])
        data_train=data_train1
        data_label=data_label1 
        correct=0.0
        total=0.0
        i=0
        for tmp in tqdm(a):
            fen_loss=torch.tensor(0).to(device).float()
            weizhi=[]
            if tmp==a[zhemesize-1]:
                zhege=data_train[tmp:465]
                tmp_label=data_label[tmp:465]
            else:
                zhege=data_train[tmp:tmp+BATCH_SIZE]
                tmp_label=data_label[tmp:tmp+BATCH_SIZE]
            for san in range(len(zhege)):
                weizhi.append(san)
            random.shuffle(weizhi)
            #TODO:nothing
            
            for san in range(len(zhege)):
                x=np.array(zhege[san]) ;x=torch.tensor(x).float() ; x=x.view(-1,1,40,40,40) ; x=x.to(device)
                x_t=tmp_label[san] ; x_la=torch.tensor([[1-float(x_t),float(x_t)]]);x_la=x_la.to(device) 
                out_x=model(x);pre_x=torch.argmax(out_x,1)
                total+=1 ; correct+=(pre_x==x_t).sum().item()
                loss=criterion(out_x,x_la)
                fen_loss+=loss
            loss=fen_loss/len(zhege)
            tmp_loss+=loss.float()
            if all_batch % 18==0:
                tmp_loss=tmp_loss/18
                all_loss.append(tmp_loss)
                tmp_loss=0.
                x_axi+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_batch+=1
            fen_loss=torch.tensor(0).to(device).float()
            
            for san in range(len(zhege)):
                x=np.array(zhege[san]) ;x=torch.tensor(x).float() ; x=x.view(-1,1,40,40,40) 
                x=Cutout(x,4,6) #TODO:Change the cutout size and batchs-------
                x=x.to(device)
                x_t=tmp_label[san] ; x_la=torch.tensor([[1-float(x_t),float(x_t)]]);x_la=x_la.to(device) 
                out_x=model(x);pre_x=torch.argmax(out_x,1)
                total+=1 ; correct+=(pre_x==x_t).sum().item()
                loss=criterion(out_x,x_la)
                fen_loss+=loss
            loss=fen_loss/len(zhege)
            tmp_loss+=loss.float()
            if all_batch % 18==0:
                tmp_loss=tmp_loss/18
                all_loss.append(tmp_loss)
                tmp_loss=0.
                x_axi+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_batch+=1
            fen_loss=torch.tensor(0).to(device).float()
            
            for san in range(len(zhege)):
                x=np.array(zhege[weizhi[san]]) ; y=np.array(zhege[(weizhi[san]+1)%len(zhege)])
                x=torch.tensor(x).float() ; x=x.view(-1,1,40,40,40) ; x=x.to(device)
                y=torch.tensor(y).float() ; y=y.view(-1,1,40,40,40) ; y=y.to(device)

                x_t=tmp_label[weizhi[san]] ; y_t=tmp_label[(weizhi[san]+1)%len(zhege)]
                vox1,lam=My_mixup(x,y)
                x_la=torch.tensor([[1-float(x_t),float(x_t)]]) ; y_la=torch.tensor([[1-float(y_t),float(y_t)]])
                x_la=x_la.to(device) ; y_la=y_la.to(device)
                out=model(vox1);out_x=model(x) ; out_y=model(y)
                pre_x=torch.argmax(out_x,1) ; pre_y=torch.argmax(out_y,1)
                total+=1
                correct+=(lam*(pre_x==x_t).sum().item()+(1-lam)*(pre_y==y_t).sum().item())
                loss=lam * criterion(out,x_la) + (1 - lam) * criterion(out, y_la)
                fen_loss+=loss
            loss=fen_loss/len(zhege)
            tmp_loss+=loss.float()
            if all_batch % 18==0:
                tmp_loss=tmp_loss/18
                all_loss.append(tmp_loss)
                tmp_loss=0.
                x_axi+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_batch+=1
        
        "---early stop---"
        print(epoch+1," accuracy:{}",correct/total)
        if (abs(tmp_num-correct/total)<0.005) | (tmp_num>correct/total):
            brea_num+=1
        else:
            brea_num=0
        if brea_num==2:
            print("-------------Early Stop----------------")
            break
        tmp_num=correct/total
    torch.save(model,'/home/zzy/tmp/model2')
    return model,x_axi,all_loss
