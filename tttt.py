# -*- coding: utf-8 -*-
'''
__author__ = 'kongzixiang'
'''

import os

a='sss'
dataset_path='./signatures/'
name='bjy/'
name_path=dataset_path+name

img_path=[]
for root,dirs,files in os.walk(name_path):
    for file in files:
        img_path.append(name_path+file)
print(img_path)