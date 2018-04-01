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

def do_train_dataset(file_list, scale, w=24, h=24):
    X = []
    Y = []
    for i in tqdm(range(len(file_list))):
    # for i in tqdm(range(30000)):
        fimg = file_list[i]
        fxml = os.path.splitext(fimg)[0]+'.xml'

        img = cv2.imread(fimg.encode('UTF-8'))
        img = cv2.resize(img,(int(w),int(h)))
        width,height,cn = get_image_size(img)
        x = img.reshape(height, width, cn)

        x = x/255.0

        X.append(x)
        # continue

        pfile = open(fxml)
        xmlstr = pfile.read()
        pfile.close()

        source, object, bsucc = xml_parse_get_rect_object_str(xmlstr)

        if bsucc == True:
            if len(object) == 1:
                obj = object[0]
                rect = copy.deepcopy(obj.rect)
                bb = rect.to_bb()

                w = (float)(width)
                h = (float)(height)
                # print width, height
                abb = [0,0,0,0]

                # scale = 3
                abb[0] = bb[0]/scale
                abb[1] = bb[1]/scale
                abb[2] = bb[2]/scale
                abb[3] = bb[3]/scale
                bb = abb

                # X.append(x)
                Y.append((bb[0]/w,bb[1]/h,bb[2]/w,bb[3]/h))
                # Y.append( ((bb[0]+bb[2])*0.5/w, (bb[1]+bb[3])*0.5/h) )

    # print len(X[0])
    return np.array(X), np.array(Y)

def create_model_0(width, height):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Dropout(0.5))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(4, activation='softmax'))
    model.add(Dense(4))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def create_model_1(width, height):
    model = Sequential()

    model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(width, height, 3)))
    # model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(16, (3, 3), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4, activation='softmax'))
    model.add(Dense(4))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def create_model(width, height):
    model = Sequential()

    model.add(Conv2D(4, (3, 3), activation='relu', input_shape=(width, height, 3)))
    # model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(8, (3, 3), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(16, (3, 3), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4, activation='softmax'))
    model.add(Dense(4))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def tour_VGG_like(x_train, y_train, x_test, y_test, model_dir, width = 32, height = 32):

    model = create_model(width,height)
    model.summary()
    model.fit(x_train, y_train, batch_size=1280, epochs=200, validation_split = 0.1)
    model.save(model_dir)

    # score = model.evaluate(x_test, y_test, batch_size=128)
    #
    # print score
    #
    #
    # score = model.predict(x_test)
    #
    # for i in range(len(score)):
    #     print score[i], '--->', y_test[i]

def calc_IOU(rA, rB):
    ax0 = rA.x
    ay0 = rA.y
    ax1 = rA.x+rA.width-1
    ay1 = rA.y+rA.height-1

    bx0 = rB.x
    by0 = rB.y
    bx1 = rB.x+rB.width-1
    by1 = rB.y+rB.height-1

    ox0 = max(ax0,bx0)
    oy0 = max(ay0,by0)
    ox1 = min(ax1,bx1)
    oy1 = min(ay1,by1)

    areaA = rA.width*rA.height
    areaB = rB.width*rB.height
    areaC = (ox1-ox0+1)*(oy1-oy0+1)

    # print areaA, areaB, areaC
    return float(areaC)/float(areaA+areaB-areaC)

def Test_VGG_like(x_test, y_test, model_dir, err_dir, scale, width = 32, height = 32):
    # model = create_model(width,height)
    # model = model.load_model(model_dir)
    model = load_model(model_dir)
    out = model.predict(x_test)

    IOU_list = []

    erro_th = 0.3
    erro_num = 0
    for i in range(len(out)):
        result = out[i]
        # x = x_test[i]
        x = x_test[i].reshape(height, width, 3)

        _scale = 1
        rect = Rect()
        rect.x = (int)(width*result[0]*_scale)
        rect.y = (int)(height*result[1]*_scale)
        rect.width = (int)(width*result[2]*_scale) - rect.x + 1
        rect.height = (int)(height*result[3]*_scale) - rect.y + 1

        rect_gth = Rect()
        rect_gth.x = (int)(width*y_test[i][0])
        rect_gth.y = (int)(height*y_test[i][1])
        rect_gth.width = (int)(width*y_test[i][2]) - rect_gth.x + 1
        rect_gth.height = (int)(height*y_test[i][3]) - rect_gth.y + 1

        # rect_gth.x = rect_gth.x/2
        # rect_gth.y = rect_gth.y/2
        # rect_gth.width = rect_gth.width/2
        # rect_gth.height = rect_gth.height/2


        val_IOU = calc_IOU(rect,rect_gth)
        IOU_list.append(val_IOU)

        rect = Rect()
        rect.x = (int)(width*result[0]*scale)
        rect.y = (int)(height*result[1]*scale)
        rect.width = (int)(width*result[2]*scale) - rect.x + 1
        rect.height = (int)(height*result[3]*scale) - rect.y + 1

        # if 0.5< val_IOU and val_IOU < 1.1:
        if 0.1< val_IOU and val_IOU < 0.5:
            save_dir = os.path.join(err_dir, "%04f_%02d.jpg"%(val_IOU,i))

            x_predict = copy.deepcopy(x)
            vi_draw_rect_GREEN(x_predict,rect)
            # vi_draw_rect_GREEN(x,rect_gth)

            bigx_predict = cv2.resize(x_predict,(64*8,64*8))
            # bigx_gth = cv2.resize(x,(64*8,64*8))

            cv2.imwrite(save_dir,bigx_predict*255)

            erro_num+=1

            # cv2.imshow("x",bigx_predict)
            # cv2.waitKey(0)

            # print save_dir

        # print val_IOU
        #
        # x_predict = copy.deepcopy(x)
        # vi_draw_rect_GREEN(x_predict,rect)
        # vi_draw_rect_GREEN(x,rect_gth)
        #
        # bigx_predict = cv2.resize(x_predict,(64*8,64*8))
        # bigx_gth = cv2.resize(x,(64*8,64*8))
        # # cv2.imshow("x",x)
        # # cv2.imshow("bigx_gth",bigx_gth)
        # cv2.imshow("bigx_predict",bigx_predict)
        # cv2.imshow("x",bigx_gth)
        # cv2.waitKey(0)


        # print result, "---->", y_test[i]

    # plt.figure("IOU")
    # n, bins, patches = plt.hist(IOU_list, bins=30, normed=0, alpha=0.75)
    # print n
    # print bins
    # plt.show()

    print "%f%%(%d / %d)" % (erro_num*100.0/len(out),erro_num,len(out))

def face_regression(train_dir, test_dir, model_dir, scale, width, height):
    file_list_train = []
    file_list_test = []
    for dir in train_dir:
        file_list = walk(dir,'.jpg')
        for f in file_list:
            file_list_train.append(f)

    # for dir in test_dir:
    #     file_list = walk(dir,'.jpg')
    #     for f in file_list:
    #         file_list_test.append(f)

    X, Y = do_train_dataset(file_list_train, scale, width, height)
    # tX, tY = do_train_dataset(file_list_test)

    tour_VGG_like(X,Y,X,Y,model_dir, width, height)

    # tour_VGG_like(X,Y,tX,tY,40,40)
    # Test_VGG_like(X,Y, model_dir)

def face_regression_test(test_dir, model_dir, error_dir, scale, width, height):
    file_list_test = []

    for dir in test_dir:
        file_list = walk(dir,'.jpg')
        for f in file_list:
            file_list_test.append(f)

    X, Y = do_train_dataset(file_list_test, width, height)
    Test_VGG_like(X,Y, model_dir, error_dir, scale, width, height)

if __name__ == '__main__':

    # folder_range = 200
    folder_range = 10*5

    width = 24
    height = 24

    error_dir = u'/tmp/error_image'

    if False == os.path.exists(error_dir):
        os.mkdir(error_dir)

    train_dir = []
    train_base_dir = u'/home/racine/workdatas/test/pull/norm_24x24_0.5_0.9'
    for i in range(folder_range):
        dir = os.path.join(train_base_dir,str(i+1))
        train_dir.append(dir)

    # train_base_dir = u'/home/racine/workdatas/test/pull/norm_widerface_32x32_big'
    #
    # folder_range = 100
    # for i in range(folder_range):
    #     dir = os.path.join(train_base_dir,str(i+1))
    #     train_dir.append(dir)

    test_dir = [
        # u'/home/racine/workdatas/test/pull/norm_32x32/700',
        # u'/home/racine/workdatas/test/pull/norm_24x24_0.5_0.9/700' #88.84
        u'/home/racine/workdatas/test/pull/norm_24x24_0.5_0.9/800', #88.84
        u'/home/racine/workdatas/test/pull/norm_24x24_0.5_0.9/700', #88.84
        u'/home/racine/workdatas/test/pull/norm_24x24_0.5_0.9/600', #88.84
        u'/home/racine/workdatas/test/pull/norm_24x24_0.5_0.9/500', #88.84
        u'/home/racine/workdatas/test/pull/norm_24x24_0.5_0.9/400', #88.84
        # u'/home/racine/workdatas/test/pull/norm_32x32/102',
        # u'/data/da/datasets/vggface2/test/n000001'
        # u'/home/racine/workdatas/test/pull/norm_32x32_big/940'
        # u"/home/racine/workdatas/test/pull/norm_widerface_32x32_big/144"
    ]

    # model_name = u'../model/my_model_slow.h5'
    # model_name = u'../model/model_fast_400_dense32_200folder_big.h5'
    # model_name = u'../model/model_fast_400_dense32_200folder.h5'
    # model_name = u'../model/model_fast_400_dense32_200folder_100folder_merge.h5'
    # model_name = u'../model/model_fast_400_dense32_400folder_merge.h5'
    # model_name = u'../model/model_fast_200_dense32_200folder_24x24_0.5_0.9.h5'
    # model_name = u'../model/model_fast_200_dense32_200folder_24x24_0.5_0.9_2out_s3.h5'
    # model_name = u'../model/model_fast_s2_50.h5'
    model_name = u'../model/model_fast_s2.h5'
    # train_dir = u'/home/racine/workdatas/test/pull/norm_64x64/2'
    # test_dir = u'/home/racine/workdatas/test/pull/norm_64x64/1'

    scale = 2
    # face_regression(train_dir,test_dir, model_name, scale, width, height)
    face_regression_test(test_dir,model_name,error_dir,scale,width,height)
