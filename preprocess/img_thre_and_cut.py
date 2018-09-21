# -*- coding: utf-8 -*-
'''
__author__ = 'kongzixiang'
'''

import os
import glob
from PIL import Image

def thre_and_cut(path):
    im=Image.open(path)
    Lim=im.convert('L')
    threshold = 90
    table=[]
    for i in range(256):
      if i <threshold:
         table.append(0)
      else :
         table.append(1)

    img = Lim.point(table,'1')
    img=img.convert('RGB')

    pixels=img.load()

    width,height=img.size

    x_region=[]
    y_region=[]

    for y in range(height):
        for x in range(width):
            L=pixels[x, y][0]*0.3+pixels[x, y][1]*0.59+pixels[x, y][2]*0.11
            if L<10:
                x_region.append(x)
                y_region.append(y)

    x_left=min(x_region)
    x_right=max(x_region)
    y_up=min(y_region)
    y_down=max(y_region)

    region_left=x_left-20
    region_right=x_right+20
    region_up=y_up-20
    region_down=y_down+20
    region=(region_left,region_up,region_right,region_down)

    crop_img=img.crop(region)
    crop_img=crop_img.convert('L')
    crop_img=crop_img.resize((220,170))
    return crop_img
