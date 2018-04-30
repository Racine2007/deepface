#coding: utf-8

from crawler import *
from tqdm import tqdm
import cv2, os
import numpy as np

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

    return overlap_obj

def image_2_rect_obj_with_jitter(img, single_obj, save_dir, range0 = 0.25, range1 = 0.9, num = 40):
    obj = single_obj

    jitter_rect = []

    # eRect = obj.rect.margined(img,0.2,0.2)
    eRect = obj.rect

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

def write_image_and_annotation(image, basename, parent_source, parent_obj,
        cur_zone, save_width, save_height, bShow = True):
    savename_jpg = basename + ".jpg"
    savename_xml = basename + ".xml"

    saveshortname_jpg = os.path.basename(savename_jpg)
    saveshortname_xml = os.path.basename(savename_xml)

    # write source info
    s = copy.deepcopy(parent_source)
    s.folder = os.path.dirname(savename_jpg)
    s.filename = saveshortname_jpg
    s.path = savename_jpg

    s.parent = parent_source.path
    s.fingerprint = ""

    # write object info
    for i in range(len(parent_obj)):
        obj = copy.deepcopy(parent_obj[i])

        obj.rect.x -= cur_zone[i].x
        obj.rect.y -= cur_zone[i].y
        obj.parent_zone = cur_zone[i]

        subarr = get_subarr(image, obj.parent_zone)

        # if bOrignal == True:
        #     s.width, s.height, s.depth = get_image_size(subarr)
        #     if bShow == True:
        #         vi_draw_rect_GREEN(subarr,obj.rect)
        #     cv2.imwrite(savename_jpg,subarr)
        # else:
        scale_x = (float)(save_width)/obj.parent_zone.width
        scale_y = (float)(save_height)/obj.parent_zone.height

        rect = Rect()
        rect.x = (int)(obj.rect.x*scale_x)
        rect.y = (int)(obj.rect.y*scale_y)
        rect.width = (int)(obj.rect.width*scale_x)
        rect.height = (int)(obj.rect.height*scale_y)

        save_patch = subarr
        if obj.rect.x != rect.x or obj.rect.y != rect.y or obj.rect.width != rect.width or obj.rect.height != rect.height:
            obj.rect = rect
            save_patch = cv2.resize(subarr,(save_width,save_height))

        s.width, s.height, s.depth = get_image_size(save_patch)
        if bShow == True:
            vi_draw_rect_GREEN(save_patch,obj.rect)

        cv2.imwrite(savename_jpg,save_patch)
        xml_write_rect_object(savename_xml,s,[obj])
        # cv2.imshow("subarr",subarr)
        # cv2.waitKey(1000)

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

                # Draw annotation
                # for obj in object:
                #      vi_draw_rect_GREEN(x,obj.rect)

                for m in range(len(object)):
                    total_samples+=1
                    obj = object[m]
                    whrate = (float)(obj.rect.width)/(obj.rect.height+0.00001)

                    # if obj.rect.width > 25 and 0.8<=whrate and whrate<=1.1:
                    if True:
                        jitter_rect = image_2_rect_obj_with_jitter(x,obj,pull_dir,0.4,0.9,100*5)
                        # jitter_rect = image_2_rect_obj_with_jitter(x,obj,pull_dir,0.5,0.95,100)
                        # jitter_rect = image_2_rect_obj_with_jitter(x,obj,pull_dir,0.7,0.9,100)

                        everymaxnum = 40*2
                        cur_count = 0
                        for jrect in jitter_rect:
                            cur_count+=1
                            if cur_count > everymaxnum:
                                break

                            subarr = get_subarr(x, jrect)

                            obj = extract_obj_from_jitterzone(object,jrect,0.2)
                            if len(obj) == 1:

                                if total_export % max_folder_num == 0:
                                    folder_idx+=1

                                foldername = os.path.join(pull_dir,(str)(folder_idx))
                                if False == os.path.exists(foldername):
                                    os.mkdir(foldername)

                                # Write image and annotation
                                write_basename = "%s_%02d_%03d" % (os.path.join(foldername,basename),m,cur_count)

                                # write_image_and_annotation(x,write_basename,
                                #     source,obj,[jrect],
                                #     jrect.width,jrect.height,
                                #     False)

                                write_image_and_annotation(x,write_basename,
                                    source,obj,[jrect],
                                    # 24, 24,
                                    64, 64,
                                    False)

                                total_export+=1

    print "total_export: %d / %d (%f%%)" % (total_export,total_samples,total_export*100.0/total_samples)

    db.close()


if __name__ == '__main__':
    # package_dir = u'/home/racine/datasets/rxface/package/VzenithFace_30k'
    # pull_dir = u'/home/racine/workdatas/test/pull/norm_64x64_0.5_0.9'
    package_dir = u'/home/racine/datasets/rxface/package/image_struct/lift/A/YDXJ0679.MP4'
    pull_dir = u'/home/racine/workdatas/test/db_lift/A_YDXJ0679.MP4_norm_64x64_0.4_0.9_80'

    # package_dir = u"/home/racine/datasets/rxface/package/wider_face/WIDER_val/"
    # pull_dir = u'/home/racine/workdatas/test/pull/norm_widerface_24x24_0.5_0.9_easy_val'
    # pull_dir = u'/tmp/test/pull/org'

    # if False == os.path.exists(u'/home/racine/workdatas/test'):
    #     os.mkdir(u'/home/racine/workdatas/test/')
    #
    # if False == os.path.exists(u'/home/racine/workdatas/test/pull'):
    #     os.mkdir(u'/home/racine/workdatas/test/pull')

    if False == os.path.exists(pull_dir):
        os.mkdir(pull_dir)

    # image_2_rect_obj_with_jitter(None,None)
    select_face_from_package_with_jitter(package_dir,pull_dir)
    # pull_face_from_package(package_dir,pull_dir)
