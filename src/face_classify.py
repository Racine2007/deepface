#coding: utf-8

from matplotlib import pyplot as plt
import numpy as np
from crawler import *
from tqdm import tqdm
import cv2, os

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model

from model_def import *
from tools import *

from keras.utils.generic_utils import get_custom_objects
loss = smooth_l1
get_custom_objects().update({"smooth_l1": smooth_l1})

def do_train_dataset(file_list, w, h, label):
    X = []
    Y = []

    for i in tqdm(range(len(file_list))):
        fimg = file_list[i]

        img = cv2.imread(fimg.encode('UTF-8'))
        width,height,cn = get_image_size(img)

        img = cv2.resize(img,(int(w),int(h)))
        x = img.reshape(w, h, cn)

        x = x/255.0

        X.append(x)
        Y.append(label)

    return np.array(X), np.array(Y)

def train_process(train_dir_pos, train_dir_neg, model_name, epoch, width, height):
    file_list_pos = []
    file_list_neg = []
    for dir in train_dir_pos:
        file_list = walk(dir,'.jpg')
        for f in file_list:
            file_list_pos.append(f)

    for dir in train_dir_neg:
        file_list = walk(dir,'.jpg')
        for f in file_list:
            file_list_neg.append(f)

    X = []
    Y = []

    for i in tqdm(range(len(file_list_pos))):
        fimg = file_list_pos[i]

        img = cv2.imread(fimg.encode('UTF-8'))
        cn = img.shape[2]

        img = cv2.resize(img,(int(width),int(height)))
        x = img.reshape(width, height, cn)

        x = x/255.0

        X.append(x)
        # print keras.utils.to_categorical(0, 2),keras.utils.to_categorical(1, 2)
        Y.append(keras.utils.to_categorical(0, 2))

    for i in tqdm(range(len(file_list_neg))):
        fimg = file_list_neg[i]

        img = cv2.imread(fimg.encode('UTF-8'))
        cn = img.shape[2]

        img = cv2.resize(img,(int(width),int(height)))
        x = img.reshape(width, height, cn)

        x = x/255.0

        # cv2.imshow('img',x)
        # cv2.waitKey(100)

        X.append(x)
        # print keras.utils.to_categorical(1, 2)
        Y.append(keras.utils.to_categorical(1, 2))

    _X_ = np.array(X)
    _Y_ = np.array(Y)

    train_model_classify(_X_,_Y_,epoch,model_name,width,height)

def test_model(x_test, y_test, file_list_test, model_dir, err_dir, width, height):
    model = load_model(model_dir)
    out = model.predict(x_test)

    folder = 0
    erro_num = 0
    for i in range(len(out)):
        result = out[i]
        # print i, '--->', result
        if result[0]>result[1]:
            erro_num+=1

        # cv2.imshow("x",x_test[i])
        # cv2.waitKey(0)

    print "%f%%(%d / %d)" % (erro_num*100.0/len(out),erro_num,len(out))

def test_process(test_dir,model_name,width,height,error_dir):
    file_list_pos = []
    for dir in test_dir:
        file_list = walk(dir,'.jpg')
        for f in file_list:
            file_list_pos.append(f)

    X = []
    Y = []

    for i in tqdm(range(len(file_list_pos))):
        fimg = file_list_pos[i]

        img = cv2.imread(fimg.encode('UTF-8'))
        cn = img.shape[2]

        img = cv2.resize(img,(int(width),int(height)))
        x = img.reshape(width, height, cn)

        x = x/255.0

        # cv2.imshow("x",x)
        # cv2.waitKey(100)

        X.append(x)
        Y.append(keras.utils.to_categorical(0, 2))

    _X_ = np.array(X)
    _Y_ = np.array(Y)

    test_model(_X_, _Y_, file_list_pos, model_name, error_dir, width, height)

def classify_train(model_name, epoch, width, height):

    folder_range = 25

    train_dir_pos = []
    train_dir_neg = []
    # #1 -----------------------------------------------------------------------
    train_base_dir = u'/home/racine/workdatas/test/db_lift/A_norm_64x64_0.5_0.9'

    for i in range(folder_range):
        dir = os.path.join(train_base_dir,str(i+1))
        train_dir_pos.append(dir)

    folder_range = 2
    train_base_dir = u'/home/racine/workdatas/test/db_lift/B_norm_64x64_0.5_0.9'

    for i in range(folder_range):
        dir = os.path.join(train_base_dir,str(i+1))
        train_dir_pos.append(dir)

    # #2 -----------------------------------------------------------------------
    train_base_dir = u'/home/racine/workdatas/neg/db_lift/A_YDXJ0679.MP4_neg/'
    # train_base_dir = u'/home/racine/workdatas/test/db_lift/B_norm_64x64_0.4_0.9_80'
    folder_range = 25
    for i in range(folder_range):
        dir = os.path.join(train_base_dir,str(i+1))
        train_dir_neg.append(dir)

    train_process(train_dir_pos, train_dir_neg, model_name, epoch, width, height)

def classify_test(model_name, width, height,error_dir):
    test_dir = [
        # u'/home/racine/workdatas/test/db_lift/A_norm_64x64_0.5_0.9/1'
        # u'/home/racine/workdatas/neg/db_lift/A_YDXJ0679.MP4_neg/1'
        # u'/home/racine/workdatas/test/db_lift/B_norm_64x64_0.4_0.9_80/1'
        # u'/home/racine/workdatas/test/db_TaiWan/IMG_4221.MOV_norm_64x64_0.4_0.9_80/20',
        u'/home/racine/workdatas/test/db_lift/B_norm_64x64_0.5_0.9/14',
        # u'/home/racine/workdatas/test/db_lift/B_norm_64x64_0.4_0.9_80/34',
        # u'/home/racine/workdatas/test/db_lift/A_norm_64x64_0.4_0.9_80/63',
    ]

    test_process(test_dir,model_name,width,height,error_dir)

if __name__ == '__main__':

    width = 32
    height = 32

    epoch = 1000
    error_dir = u'/tmp/error_image'

    model_name = u'../model_classify/20180422_1/000155-0.99918610-0.99851944.hdf5'

    classify_train(model_name,epoch,width,height)
    # classify_test(model_name,width,height,error_dir)
