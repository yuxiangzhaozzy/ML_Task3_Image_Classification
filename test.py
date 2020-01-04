#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:53:43 2020
@author: zzy
"""

'---model merge---'
'---导入库---'
import csv
import pandas as pd
import numpy as np

'---读入两个模型的输出，并进行model averaging---'
csv1=csv.reader(open("Submission2.csv",'r'))  #这里的文件地址要改为模型一代码文件夹里的相应输出地址
csv2=csv.reader(open("Submission1.csv",'r'))  #同上

pred1=pd.DataFrame(csv1, columns = ['ID','Predicted'])
pred2=pd.DataFrame(csv2, columns = ['ID','Predicted'])

name=pred1['ID']
data1=pred1['Predicted']
data2=pred2['Predicted']

y1=[]
for i in range(len(data1)):
    y1.append(float(data1[i]))

y2=[]
for i in range(len(data1)):
    y2.append(float(data2[i]))
    
'---模型融合，赋予权重---'
y3=[]
for i in range(len(data1)):
    y3.append(y1[i]*0.3+y2[i]*0.7)

'---将融合后的结果写入新的csv文件---'
file=open('Submission.csv','w',newline='')
csv_write=csv.writer(file)
csv_write.writerow(["Id","Predicted"])
i=0

for i in range(len(data1)):
    row=[name[i],y3[i]]
    csv_write.writerow(row)
file.close()         


