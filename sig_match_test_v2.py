# -*- coding: utf-8 -*-
'''
__author__ = 'kongzixiang'
'''

import numpy as np
from PIL import Image
from scipy.misc import imread
import tensorflow as tf
import tf_signet
import os
import argparse
from tf_cnn_model import TF_CNNModel
from preprocess.normalize import preprocess_signature
import warnings
from preprocess.img_thre_and_cut import thre_and_cut

warnings.filterwarnings("ignore")

model_weight_path = 'models/signetf_lambda0.95.pkl'
dataset_path='./signatures/'

model = TF_CNNModel(tf_signet, model_weight_path)

def get_imgpath(dataset_path,name):
    name_path = dataset_path + name
    img_path = []
    for root, dirs, files in os.walk(name_path):
        for file in files:
            img_path.append(name_path +'/'+ file)
    return img_path

sess = tf.Session()
sess.run(tf.global_variables_initializer())


while True:

    name = '陈慧娴'
    img=input('Input image filename(input close to over):')

    if img=='close':
        break
    else:
        try:
            filepath, tempfilename = os.path.split(img)
            filename, extension = os.path.splitext(tempfilename)
            path = './test_imgs2/%s.jpg' % (filename)

            im=Image.open(img)
            if im.mode!='L' or '.png' in tempfilename :
                im = thre_and_cut(img)
                im.save(path)
            target_sig =[imread(path)]
        except:
            print('Open Error! Try again!')
            continue
        else:
            if os.path.isdir(dataset_path+name):
                print('name is in dataset, calculating similarity....')
                dataset_img_sigs_path=get_imgpath(dataset_path,name)
                dataset_sigs  = [imread(path) for path in dataset_img_sigs_path]

                canvas_size = (500, 300)
                processed_target_sig = np.array([preprocess_signature(sig, canvas_size) for sig in target_sig])
                processed_dataset_sigs = np.array([preprocess_signature(sig, canvas_size) for sig in dataset_sigs])
                target_feature = model.get_feature_vector_multiple(sess,processed_target_sig, layer='fc2')
                dataset_features = model.get_feature_vector_multiple(sess,processed_dataset_sigs, layer='fc2')

                dists = [np.linalg.norm(u1 - u2) for u1 in target_feature for u2 in dataset_features]
                min_similarity=min(dists)
                min_index=dists.index(min_similarity)
                mach_img=dataset_img_sigs_path[min_index]
                print(min_similarity)
                print("最相似的图片是：{},相似度是：{}".format(mach_img,min_similarity))
                match_image=Image.open(mach_img)
                match_image.show()
            else:
                print('name is not in dataset')