#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:21:36 2020

@author: zzy
"""

"---run this code to get prediction result---"

"---import corresponding libraries---"

import torch
import torchvision.transforms as transforms
from train_test import get_data,LeNet3D,test_model1,trainmodel
import csv
import pandas as pd


normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model1= LeNet3D()
    model1=model1.to(device)

    model2=LeNet3D()
    model1=model2.to(device)
   
    model1=torch.load('./model1')
    data_test,te_label,conte,data_tevoxel=get_data('/home/zzy/test') #将此处的测试集目录更换即可
    filename="Submission1.csv"
    test_model1(model1,data_test,conte,filename)
   
