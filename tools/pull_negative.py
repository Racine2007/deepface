#coding utf-8

from crawler import *
from tqdm import tqdm
import cv2, os
import numpy as np

def RectInRect(rectA, rectB):
    (ax0,ay0,ax1,ay1) = rectA.to_bb()
    (bx0,by0,bx1,by1) = rectB.to_bb()

    # print (ax0,ay0,ax1,ay1), (bx0,by0,bx1,by1)

    if bx0<=ax0 and ax1<=bx1 and by0<=ay0 and ay1<=by1:
        return True
    else:
        return False

def collect_negative( x, object ):

    max_num = 100
    width,height,cn = get_image_size(x)

    minmax_w = (100,width*0.4)
    minmax_h = (100,height*0.4)

    th = 0.2
    iou_th = 0.1

    pw = np.random.randint(minmax_w[0],minmax_w[1],max_num)
    ph = np.random.randint(minmax_h[0],minmax_h[1],max_num)

    neg_rect = []

    for i in range(max_num):
        # image = copy.deepcopy(x)
        x0 = np.random.randint(0,width-pw[i])
        x1 = np.random.randint(0,height-ph[i])

        rect = Rect(x0,x1,pw[i],ph[i])

        bhit_rect = False
        hit_obj = None
        for obj in object:
            iou = calc_IOU(obj.rect,rect)

            if iou>=iou_th:
                bhit_rect = True
                hit_obj = obj
                break
            else:
                bhit_rect = False
                bIn = RectInRect(obj.rect,rect)
                hit_obj = obj
                if bIn:
                    if obj.rect.width*obj.rect.height>=rect.width*rect.height*th:
                        bhit_rect = True

        # if bhit_rect:
        #     vi_draw_rect_GREEN(image,rect)
        #     vi_draw_rect_GREEN(image,hit_obj.rect)
        #
        #     cv2.imshow("x",image)
        #     cv2.waitKey(0)

        if not bhit_rect:
            neg_rect.append(rect)

    return neg_rect

def pull_negative(package_dir, pull_dir):

    db = DB_Image_Annotation()
    db.open(package_dir)

    records_num = db.get_records_num()
    num, source_dir = read_info_file(package_dir)

    if num != records_num:
        print("Records num is not match!")
        return

    total_export = 0
    total_samples = 0

    folder_idx = 0
    max_folder_num = 2500

    for i in tqdm(range(records_num)):
        if db.next() == True:
            (key, val_i, val_a) = db.item()
            # Read annotation
            source, object, bsucc = xml_parse_get_rect_object_str(val_a)

            if bsucc == True:
                basename = os.path.splitext(source.filename)[0]
                xml_name = basename+'.xml'

                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(val_i)
                flat_x = np.fromstring(datum.data, dtype=np.uint8)
                x = flat_x.reshape(datum.height, datum.width, datum.channels)

                neg_rect = collect_negative(x, object)

                for m in range(len(neg_rect)):
                    neg = neg_rect[m]
                    subarr = get_subarr(x, neg)

                    if total_export % max_folder_num == 0:
                        folder_idx+=1

                    total_export += 1

                    foldername = os.path.join(pull_dir,(str)(folder_idx))
                    if False == os.path.exists(foldername):
                        os.mkdir(foldername)

                    # Write image and annotation
                    write_basename = "%s_%03d.jpg" % (os.path.join(foldername,basename),m)
                    # print write_basename
                    cv2.imwrite(write_basename,subarr)

                    # img = copy.deepcopy(x)
                    # vi_draw_rect_GREEN(img,neg)
                    # cv2.imshow("x",img)
                    # cv2.waitKey(50)



if __name__ == "__main__":
    package_dir = u'/home/racine/datasets/rxface/package/image_struct/lift/A/YDXJ0679.MP4'
    pull_dir = u'/home/racine/workdatas/neg/db_lift/A_YDXJ0679.MP4_neg'

    if False == os.path.exists(pull_dir):
        os.mkdir(pull_dir)

    pull_negative(package_dir,pull_dir)
