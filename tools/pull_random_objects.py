#coding: utf-8

from crawler import *
from tqdm import tqdm
import cv2, os
import numpy as np

def image_2_rect_obj_with_jitter(img, single_obj, save_dir, range0 = 0.25, range1 = 0.9, num = 40):
    obj = single_obj

    jitter_rect = []

    eRect = obj.rect.margined(img,0.2,0.2)

    scale_x = np.random.rand(1,num)*(range1-range0)+range0
    scale_y = np.random.rand(1,num)*(range1-range0)+range0

    scale_x = np.random.rand(1,num)*(range1-range0)+range0
    scale_y = np.random.rand(1,num)*(range1-range0)+range0

    zone_scale_x = 1.0/scale_x
    zone_scale_y = 1.0/scale_y
    pos_x = np.random.rand(1,num)*(1.0-scale_x)
    pos_y = np.random.rand(1,num)*(1.0-scale_y)

    zone_width = eRect.width*zone_scale_x
    zone_height = eRect.height*zone_scale_y

    total_samples = 0
    for i in range(num):

        max_zone_size = max(zone_width[0][i],zone_height[0][i])
        zone_x = eRect.x - pos_x[0][i]*max_zone_size
        zone_y = eRect.y - pos_y[0][i]*max_zone_size

        rect = Rect(int(zone_x),int(zone_y),int(max_zone_size),int(max_zone_size))
        bsucc = rect.squeeze_rect(img)
        if False==bsucc:
            continue

        wrate = (float)(eRect.width)/rect.width
        hrate = (float)(eRect.height)/rect.height

        if range0<=wrate and wrate<=range1 and range0<=hrate and hrate<=range1:
            total_samples+=1

            jitter_rect.append(rect)

    return jitter_rect

def RectClipRect(rectA, rectB):
    x0 = max(rectA.x,rectB.x)
    y0 = max(rectA.y,rectB.y)
    x1 = min(rectA.x+rectA.width,rectB.x+rectB.width)-1
    y1 = min(rectA.y+rectA.height,rectB.y+rectB.height)-1

    rect = Rect(x0,y0,x1-x0+1,y1-y0+1)
    return rect

def extract_obj_from_jitterzone(objects_list, zone, overlap_th = 0.5):

    overlap_list = []
    overlap_obj = []

    # print "++++++++++++++++++++++++++++++"
    for obj in objects_list:
        overRect = RectClipRect(obj.rect,zone)
        if overRect.width <= 0 or overRect.height <= 0:
            continue

        area_overRect = overRect.width*overRect.height
        area_A = obj.rect.width*obj.rect.height
        area_B = zone.width*zone.height

        overlap_A = (float)(area_overRect) / area_A
        overlap_B = (float)(area_overRect) / area_B

        overlap = max(overlap_A,overlap_B)

        if overlap > overlap_th:
            overlap_list.append(overlap)
            overlap_obj.append(obj)

        # print overlap_A, overlap_B

    # print "============================", len(overlap_list)
    # if len(overlap_list)>1:
    #     for overlap in overlap_list:
    #         print "***",overlap

    return overlap_obj


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

                # Draw annotation
                # for obj in object:
                #      vi_draw_rect_GREEN(x,obj.rect)

                for m in range(len(object)):
                    total_samples+=1
                    obj = object[m]
                    whrate = (float)(obj.rect.width)/obj.rect.height

                    subarr, zone, subrect = image_2_rect_obj(x,obj)

                    if obj.rect.width > 25 and 0.8<=whrate and whrate<=1.1:
                        # print foldername, total_export, max_folder_num, folder_idx, total_export % max_folder_num

                        jitter_rect = image_2_rect_obj_with_jitter(x,obj,pull_dir,0.25,0.9,100)

                        # print "\t\t >>>>>>>>>>>>>", len(jitter_rect)

                        everymaxnum = 40
                        cur_count = 0
                        for jrect in jitter_rect:
                            cur_count+=1
                            if cur_count > everymaxnum:
                                break

                            subarr = get_subarr(x, jrect)
                            # crope = cv2.resize(subarr,(32*4,32*4))

                            obj = extract_obj_from_jitterzone(object,jrect,0.2)
                            if len(obj) == 1:

                                if total_export % max_folder_num == 0:
                                    folder_idx+=1

                                foldername = os.path.join(pull_dir,(str)(folder_idx))
                                if False == os.path.exists(foldername):
                                    os.mkdir(foldername)

                                savename = "%s.jpg" % total_export
                                savedir = os.path.join(foldername,(str)(savename))

                                cv2.imwrite(savedir,subarr)

                                total_export+=1
                                # print total_export

    print "total_export: %d / %d (%f%%)" % (total_export,total_samples,total_export*100.0/total_samples)

    db.close()


if __name__ == '__main__':
    package_dir = u'/home/racine/datasets/rxface/package/VzenithFace_30k'
    pull_dir = u'/tmp/test/pull'

    if False == os.path.exists(u'/tmp/test'):
        os.mkdir(u'/tmp/test/')

    if False == os.path.exists(pull_dir):
        os.mkdir(pull_dir)

    # image_2_rect_obj_with_jitter(None,None)
    select_face_from_package_with_jitter(package_dir,pull_dir)
    # pull_face_from_package(package_dir,pull_dir)
