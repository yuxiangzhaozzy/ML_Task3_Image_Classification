#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:53:43 2020

@author: zzy
"""

'---model merge---'
import csv
import pandas as pd
import numpy as np

csv1=csv.reader(open("/home/zzy/桌面/code/other/Submission2.csv",'r'))
csv2=csv.reader(open("Submission1.csv",'r'))

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
    

y3=[]
for i in range(len(data1)):
    y3.append(y1[i]*0.3+y2[i]*0.7)

print(len(y3)) 

file=open('Submission.csv','w',newline='')
csv_write=csv.writer(file)
csv_write.writerow(["Id","Predicted"])
i=0

for i in range(len(data1)):
    row=[name[i],y3[i]]
    csv_write.writerow(row)
file.close()         



                