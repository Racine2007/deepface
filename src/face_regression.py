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

# def measure_rect(rect, width, height):
#     x0 = rect.x
#     y0 = rect.y
#     x1 = rect.x + rect.width - 1
#     y1 = rect.y + rect.height - 1
#     return (x0,y0,x1,y1)

def measure_rect(rect, width, height):
    x0 = rect.x
    y0 = rect.y
    x1 = rect.x + rect.width - 1
    y1 = rect.y + rect.height - 1
    return (float(x0)/width,float(y0)/height,float(x1)/width,float(y1)/height)

def measure_center(rect, width, height):
    x0 = rect.x + (rect.width>>1)
    y0 = rect.y + (rect.height>>1)

    maxw = max(rect.width, rect.height)
    # return (float(x0)/width, float(y0)/height, float(maxw)/width)
    return (float(x0)/width, float(y0)/height)

def do_train_dataset(file_list, scale, w, h):
    X = []
    Y = []

    for i in tqdm(range(len(file_list))):
        fimg = file_list[i]
        fxml = os.path.splitext(fimg)[0]+'.xml'

        img = cv2.imread(fimg.encode('UTF-8'))
        width,height,cn = get_image_size(img)

        img = cv2.resize(img,(int(w),int(h)))
        x = img.reshape(w, h, cn)

        x = x/255.0

        source, object, bsucc = xml_parse_get_rect_object(fxml)

        if bsucc == True:
            if len(object) == 1:
                obj = object[0]
                # y = measure_center(obj.rect, width, height)
                y = measure_rect(obj.rect,width,height)

                X.append(x)
                Y.append(y)

    return np.array(X), np.array(Y)

def test_model(x_test, y_test, file_list_test, model_dir, err_dir, scale, width, height):
    print "=============================="
    print model_dir

    model = load_model(model_dir)
    out = model.predict(x_test)

    IOU_list = []

    erro_th = 1.3
    erro_num = 0
    folder = 0
    for i in range(len(out)):
        result = out[i]

        X = cv2.imread(file_list_test[i].encode('UTF-8'), cv2.IMREAD_COLOR)
        width, height, cn = get_image_size(X)
        rect_predict = result_2_rect(result,width,height)
        rect_groundtruth = result_2_rect(y_test[i],width,height)

        # vi_draw_rect_GREEN(X,rect_groundtruth)
        score = calc_IOU(rect_predict,rect_groundtruth)

        # if 0.5 < score and score < 1.01:
        if score < erro_th:
            erro_num += 1

            if erro_num % 900 == 0:
                folder += 1

            save_folder = os.path.join(err_dir,str(folder))
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            # save_fname = os.path.join(err_dir,"%04f_%04d.jpg" % (score,erro_num))
            save_fname = os.path.join(save_folder,os.path.basename(file_list_test[i]))
            vi_draw_rect_GREEN(X,rect_predict)
            cv2.imwrite(save_fname,X)

    print "%f%%(%d / %d)" % (erro_num*100.0/len(out),erro_num,len(out))

def face_regression(train_dir, model_dir, epoch, scale, width, height):
    file_list_train = []
    file_list_test = []
    for dir in train_dir:
        file_list = walk(dir,'.jpg')
        for f in file_list:
            file_list_train.append(f)

    X, Y = do_train_dataset(file_list_train, scale, width, height)
    train_model(X,Y, epoch, model_dir, width, height)

def face_regression_test(test_dir, model_dir, error_dir, scale, width, height):
    file_list_test = []

    for dir in test_dir:
        file_list = walk(dir,'.jpg')
        for f in file_list:
            file_list_test.append(f)

    X, Y = do_train_dataset(file_list_test, scale, width, height)
    test_model(X, Y, file_list_test, model_dir, error_dir, scale, width, height)

def face_train(model_name, epoch, scale, width, height):
    folder_range = 32

    if False == os.path.exists(error_dir):
        os.mkdir(error_dir)

    train_dir = []
    # #1 -----------------------------------------------------------------------
    train_base_dir = u'/home/racine/workdatas/test/db_lift/A_norm_64x64_0.5_0.9'

    for i in range(folder_range):
        dir = os.path.join(train_base_dir,str(i+1))
        train_dir.append(dir)

    # #2 -----------------------------------------------------------------------
    # train_base_dir = u'/home/racine/workdatas/test/db_TaiWan/IMG_4221.MOV_norm_64x64_0.4_0.9_80'
    train_base_dir = u'/home/racine/workdatas/test/db_lift/A_YDXJ0679.MP4_norm_64x64_0.4_0.9_80'
    folder_range = 30
    for i in range(folder_range):
        dir = os.path.join(train_base_dir,str(i+1))
        train_dir.append(dir)

    face_regression(train_dir, model_name, epoch, scale, width, height)

def face_test(model_name, scale, width, height, error_dir):
    test_dir = [
        u'/home/racine/workdatas/test/db_TaiWan/IMG_4221.MOV_norm_64x64_0.4_0.9_80/20',
        # u'/home/racine/workdatas/test/db_lift/B_norm_64x64_0.5_0.9/1',
        # u'/home/racine/workdatas/test/db_lift/B_norm_64x64_0.4_0.9_80/34',
        # u'/home/racine/workdatas/test/db_lift/A_norm_64x64_0.4_0.9_80/63',
    ]

    face_regression_test(test_dir,model_name,error_dir,scale,width,height)

if __name__ == '__main__':

    scale = 1
    # width = 24
    # height = 24
    # width = 18
    # height = 18
    # width = 32
    # height = 32
    width = 32
    height = 32

    epoch = 1000

    error_dir = u'/tmp/error_image'
    if False == os.path.exists(error_dir):
        os.mkdir(error_dir)

    # model_name = u'../model/newdb_center__10folder_24x24_model_easy/smooth_l1_16_bsize512/000641-0.90742222-0.88760000.hdf5'  # best
    # model_name = u'../model/newdb_center__10folder_24x24_model_easy/smooth_l1_20_bsize512/000417-0.91522963-0.87653333.hdf5' # 332x32
    # model_name = u'../model/newdb_center__10folder_24x24_model_easy/smooth_l1_21_bsize512/000907-0.90139259-0.85866667.hdf5'
    # model_name = u'../model/newdb/base0/000982-0.91019259-0.87973333.hdf5'
    # model_name = u'../model/20180411/24x24_0.25_0.9_149f_simple/000995-0.89734203-0.85105416.hdf5'
    # model_name = u'../model/20180412/18x18_0.4_0.9_62f_simple/000992-0.89744086-0.84690323.hdf5'
    # model_name = u'../model/20180410/24x24_0.25_0.9_150f/000199-0.89526381-0.84708324.hdf5'
    # model_name = u'../model_excise/20180412/32x32_5x5_0.5_0.9_32f/000938-0.91373611-0.87512500.hdf5'
    # model_name = u'../model_excise/20180414/32x32_5x5_0.5_0.9_32f/000930-0.87834568-0.84548148.hdf5'
    model_name = u'../model_excise/20180415/32x32_5x5_0.5_0.9_merge_2/000923-0.89875269-0.87787097.hdf5'

    # face_train(model_name, epoch, scale, width, height)
    face_test(model_name, scale, width, height, error_dir)
