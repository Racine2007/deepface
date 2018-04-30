
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

from keras import backend as K

import tensorflow as tf

def smooth_l1(y_true, y_pred):

    # th = 4.0/24.0
    th = 1.0
    sub_val = th*0.5
    scale = 24.0/24.0

    # regression        = y_pred*7.5*scale
    # regression_target = y_true*7.5*scale
    regression        = y_pred*10*scale
    regression_target = y_true*10*scale

    regression_diff = regression - regression_target
    regression_diff = keras.backend.abs(regression_diff)

    # regression_loss = tf.where(
    #     keras.backend.less(regression_diff,th),
    #         0.5 * keras.backend.pow(regression_diff, 2),
    #         regression_diff - sub_val
    #     )

    regression_loss = tf.where(
        keras.backend.less(regression_diff,th),
            0.5 * keras.backend.pow(regression_diff, 2),
            regression_diff - sub_val
        )

    return regression_loss

def create_model_5x5(width, height):

    s = 1
    model = Sequential()

    model.add(Conv2D(int(4*s), (5, 5), activation='relu', input_shape=(width, height, 3)))
    # model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(int(8*s), (5, 5), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(int(16*s), (5, 5), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(int(32*s), activation='relu'))
    # model.add(Dense(int(32), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4, activation='softmax'))
    model.add(Dense(4))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.compile(loss=smooth_l1, optimizer='adam', metrics=['accuracy'])
    return model

def create_model_5x5_classify(width, height):

    s = 1
    model = Sequential()

    model.add(Conv2D(int(4*s), (5, 5), activation='relu', input_shape=(width, height, 3)))
    # model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(int(8*s), (5, 5), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(int(16*s), (5, 5), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(int(32*s), activation='relu'))
    # model.add(Dense(int(32), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4, activation='softmax'))
    model.add(Dense(2, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # model.compile(optimizer='rmsprop',
    #           loss='categorical_crossentropy',
    #           metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def create_model(width, height):

    s = 1
    model = Sequential()

    model.add(Conv2D(int(4*s), (3, 3), activation='relu', input_shape=(width, height, 3)))
    # model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(int(8*s), (3, 3), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(int(16*s), (3, 3), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(int(32*s), activation='relu'))
    # model.add(Dense(int(32), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4, activation='softmax'))
    model.add(Dense(4))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.compile(loss=smooth_l1, optimizer='adam', metrics=['accuracy'])
    return model

def train_model(x_train, y_train, epoch, model_dir, width, height):

    model = create_model_5x5(width,height)
    model.summary()

    # _batch_size_ = 1280
    _batch_size_ = 512

    check_point = keras.callbacks.ModelCheckpoint(model_dir + '/'+'{epoch:06d}-{acc:.8f}-{val_acc:.8f}.hdf5', monitor='acc',
    								verbose=1, save_best_only=True, save_weights_only=False, mode='auto')

    callbacks_list=[check_point]

    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)

    model.fit(x_train, y_train, batch_size=_batch_size_, epochs=epoch, validation_split = 0.1, callbacks = callbacks_list)

    # model.fit(x_train, y_train, batch_size=1280, epochs=epoch, validation_split = 0.1)
    # model.save(model_dir)

def train_model_classify(x_train, y_train, epoch, model_dir, width, height):

    model = create_model_5x5_classify(width,height)
    model.summary()

    # _batch_size_ = 1280
    _batch_size_ = 512

    check_point = keras.callbacks.ModelCheckpoint(model_dir + '/'+'{epoch:06d}-{acc:.8f}-{val_acc:.8f}.hdf5', monitor='acc',
    								verbose=1, save_best_only=True, save_weights_only=False, mode='auto')

    callbacks_list=[check_point]

    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)

    model.fit(x_train, y_train, batch_size=_batch_size_, epochs=epoch, validation_split = 0.1, callbacks = callbacks_list)
