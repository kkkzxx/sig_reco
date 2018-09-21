# -*- coding: utf-8 -*-
'''
__author__ = 'kongzixiang'
'''

from preprocess.img_thre_and_cut import thre_and_cut

path=r'C:\Users\kongzixiang\Desktop\signature_recognition\微信截图_20180917104826.png'
img=thre_and_cut(path)
img.save(path)