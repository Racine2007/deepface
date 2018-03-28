#coding: utf-8

from crawler import *
from tqdm import tqdm
import cv2, os
import numpy as np

def image_2_rect_obj_with_jitter(img, single_obj,save_dir):
    obj = single_obj

    # range0 = 0.25
    # range1 = 0.9
    range0 = 0.25
    range1 = 0.9
    num = 100

    scale_x = np.random.rand(1,num)*(range1-range0)+range0
    scale_y = np.random.rand(1,num)*(range1-range0)+range0

    zone_scale_x = 1.0/scale_x
    zone_scale_y = 1.0/scale_y
    pos_x = np.random.rand(1,num)*(1.0-scale_x)
    pos_y = np.random.rand(1,num)*(1.0-scale_y)

    zone_width = obj.rect.width*zone_scale_x
    zone_height = obj.rect.height*zone_scale_y


    for i in range(num):

        max_zone_size = max(zone_width[0][i],zone_height[0][i])
        zone_x = obj.rect.x - pos_x[0][i]*max_zone_size
        zone_y = obj.rect.y - pos_y[0][i]*max_zone_size

        rect = Rect(int(zone_x),int(zone_y),int(max_zone_size),int(max_zone_size))
        rect.clip_rect(img)

        subarr = get_subarr(img, rect)
        crope = cv2.resize(subarr,(32,32))
        cv2.imshow("img",crope)
        cv2.waitKey(100)

def select_face_from_package_with_jitter(package_dir, pull_dir):
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
            # (key, val_i, val_a) = db.item(image=False)
            (key, val_i, val_a) = db.item()
            source, object, bsucc = xml_parse_get_rect_object_str(val_a)
            if bsucc == True:
                basename = os.path.splitext(source.filename)[0]
                xml_name = basename+'.xml'

                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(val_i)
                flat_x = np.fromstring(datum.data, dtype=np.uint8)
                x = flat_x.reshape(datum.height, datum.width, datum.channels)

                for m in range(len(object)):
                    total_samples+=1
                    obj = object[m]
                    whrate = (float)(obj.rect.width)/obj.rect.height

                    subarr, zone, subrect = image_2_rect_obj(x,obj)

                    if obj.rect.width > 25 and 0.8<=whrate and whrate<=1.1:

                        if total_export % max_folder_num == 0:
                            folder_idx+=1

                        foldername = os.path.join(pull_dir,(str)(folder_idx))
                        if False == os.path.exists(foldername):
                            os.mkdir(foldername)

                        image_2_rect_obj_with_jitter(x,obj,pull_dir)

                        total_export+=1
                        # exit()

    print "total_export: %d / %d (%f%%)" % (total_export,total_samples,total_export*100.0/total_samples)

    db.close()


if __name__ == '__main__':
    package_dir = u'/home/racine/datasets/rxface/package/VzenithFace_30k'
    pull_dir = u'/tmp/test/pull'

    if False == os.path.exists(pull_dir):
        os.mkdir(pull_dir)

    # image_2_rect_obj_with_jitter(None,None)
    select_face_from_package_with_jitter(package_dir,pull_dir)
    # pull_face_from_package(package_dir,pull_dir)
