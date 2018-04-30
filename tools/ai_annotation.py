#coding:utf-8
from matplotlib import pyplot as plt
import numpy as np
from crawler import *
from tqdm import tqdm
import cv2, os

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model

def smooth_l1(y_true, y_pred):
    th = 1.0
    sub_val = th*0.5
    scale = 24.0/24.0

    regression        = y_pred*7.5*scale
    regression_target = y_true*7.5*scale

    regression_diff = regression - regression_target
    regression_diff = keras.backend.abs(regression_diff)


    regression_loss = tf.where(
        keras.backend.less(regression_diff,th),
            0.5 * keras.backend.pow(regression_diff, 2),
            regression_diff - sub_val
        )

    return regression_loss

from keras.utils.generic_utils import get_custom_objects
loss = smooth_l1
get_custom_objects().update({"smooth_l1": smooth_l1})

def result_2_rect(result, width, height):
    x0 = width*result[0]
    y0 = height*result[1]
    x1 = width*result[2]
    y1 = height*result[3]

    return Rect(int(x0),int(y0),int(x1-x0+1),int(y1-y0+1))

def ai_annotation(image, rect_hype, model):
    w = 32
    h = 32

    subarr = get_subarr(image,rect_hype)
    width,height,cn = get_image_size(subarr)

    img = cv2.resize(subarr,(int(w),int(h)))
    x = img.reshape(w, h, cn)
    x = x/255.0

    out = model.predict(np.array([x]))

    rect_predict = result_2_rect(out[0],width,height)

    rect_predict.x += rect_hype.x
    rect_predict.y += rect_hype.y

    return rect_predict

def DoAnnotation( img_file, img_xml, model):
    x = cv2.imread(img_file.encode('UTF-8'), cv2.IMREAD_COLOR)
    source, object, bsucc = xml_parse_get_rect_object(img_xml)
    if not bsucc:
        return False

    rect_predict = ai_annotation(x, object[0].rect, model)
    object[0].rect = rect_predict

    xml_write_rect_object(img_xml,source,object)

    return True


if __name__ == '__main__':
    dir = u'/data/da/datasets/rxface/image_struct/lift/A/YDXJ0679.MP4/pull_all'
    model_dir = u'../model/000938-0.91373611-0.87512500.hdf5'
    # model_dir = u'../model/000930-0.87834568-0.84548148.hdf5'

    model = load_model(model_dir)

    files = walk(dir,'.jpg')

    for f in files:
        f_xml = os.path.splitext(f)[0]+'.xml'
        if not os.path.exists(f):
            continue

        bsucc = DoAnnotation(f, f_xml, model)
        print f,'--->',bsucc
