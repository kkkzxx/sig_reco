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
import glob
from tf_cnn_model import TF_CNNModel
from preprocess.normalize import preprocess_signature
import warnings
from preprocess.img_thre_and_cut import thre_and_cut

warnings.filterwarnings("ignore")

model_weight_path = 'models/signetf_lambda0.95.pkl'
dataset_path='./signatures/'
model = TF_CNNModel(tf_signet, model_weight_path)

def get_imgpath_and_labels(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    img_path = []
    labels = []
    for idx, folder in enumerate(cate):
        img_list = glob.glob(folder + '/*.jpg')
        img_list += glob.glob(folder + '/*.png')
        for im in img_list:
            img_path.append(im)
            labels.append(idx)
    return img_path

sess = tf.Session()
sess.run(tf.global_variables_initializer())
while True:
    img=input('Input image filename(input close to over):')
    if img=='close':
        break
    else:
        try:
            filepath, tempfilename = os.path.split(img)
            filename, extension = os.path.splitext(tempfilename)
            path='./test_imgs2/%s.jpg'%(filename)
            im=Image.open(img)
            if im.mode!='L' or '.png' in tempfilename :
                im = thre_and_cut(img)
                im.save(path)
            target_sig =[imread(path)]
        except:
            print('Open Error! Try again!')
            continue
        else:
            # dataset_img_path='./sig_dataset/'
            # dataset_img_sigs_path=[os.path.join(dataset_img_path,img) for img in os.listdir(dataset_img_path)]
            dataset_img_sigs_path=get_imgpath_and_labels(dataset_path)
            dataset_sigs  = [imread(path) for path in dataset_img_sigs_path]

            canvas_size = (500, 300)
            processed_target_sig = np.array([preprocess_signature(sig, canvas_size) for sig in target_sig])
            processed_dataset_sigs = np.array([preprocess_signature(sig, canvas_size) for sig in dataset_sigs])
            target_feature = model.get_feature_vector_multiple(sess,processed_target_sig, layer='fc2')
            dataset_features = model.get_feature_vector_multiple(sess,processed_dataset_sigs, layer='fc2')

            # print('Euclidean distance between signature from dataset')
            dists = [np.linalg.norm(u1 - u2) for u1 in target_feature for u2 in dataset_features]
            min_similarity=min(dists)
            min_index=dists.index(min_similarity)
            mach_img=dataset_img_sigs_path[min_index]
            print(min_similarity)
            print("最相似的图片是：{},相似度是：{}".format(mach_img,min_similarity))
            match_image=Image.open(mach_img)
            match_image.show()