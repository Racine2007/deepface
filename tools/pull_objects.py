#coding: utf-8

from crawler import *
from tqdm import tqdm
import cv2, os

def select_face_from_package(package_dir, pull_dir, crop_scale_x = 1.0, crop_scale_y = 1.0):
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
                    # print whrate
                    # if obj.rect.width > 40:
                    #
                    #     pull_single(source_dir,source,obj,x,pull_dir)

                    subarr, zone, subrect = image_2_rect_obj(x,obj)
                    save_name = "%04f_%d_0_%s" % (whrate,m,source.filename)
                    save_name2 = "%04f_%d_1_%s" % (whrate,m,source.filename)
                    save_name3 = "%04f_%d_2_%s" % (whrate,m,source.filename)

                    if obj.rect.width > 25 and 0.8<=whrate and whrate<=1.1:

                        if total_export % max_folder_num == 0:
                            folder_idx+=1

                        foldername = os.path.join(pull_dir,(str)(folder_idx))
                        if False == os.path.exists(foldername):
                            os.mkdir(foldername)
                        # foldername = os.path.join(foldername,(str)(total_samples))
                        # if False == os.path.exists(foldername):
                        #     os.mkdir(foldername)

                        # vi_draw_rect_GREEN(subarr,subrect)
                        # cv2.imwrite(os.path.join(foldername,save_name),subarr)
                        # cv2.imwrite(os.path.join(foldername,save_name),subarr)
                        image = cv2.resize(subarr,(32,32))
                        cv2.imwrite(os.path.join(foldername,save_name2),image)
                        # image = cv2.resize(subarr,(24,24))
                        # cv2.imwrite(os.path.join(foldername,save_name3),image)
                        total_export+=1

    print "total_export: %d / %d (%f%%)" % (total_export,total_samples,total_export*100.0/total_samples)

    db.close()

def pull_face_from_package(package_dir, pull_dir, crop_scale_x = 1.0, crop_scale_y = 1.0):
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

                    if obj.rect.width > 25 and 0.8<=whrate and whrate<=1.1:

                        if obj.blur == 0 and \
                           obj.expression == 0 and \
                           obj.illumination == 0 and \
                           obj.occlusion == 0 and \
                           obj.pose == 0 and \
                           obj.invalid == 0:

                            if total_export % max_folder_num == 0:
                                folder_idx+=1

                            foldername = os.path.join(pull_dir,(str)(folder_idx))
                            if False == os.path.exists(foldername):
                                os.mkdir(foldername)

                            pull_single(source_dir,source,obj,x,foldername)
                            total_export+=1

    print "total_export: %d / %d (%f%%)" % (total_export,total_samples,total_export*100.0/total_samples)

    db.close()

if __name__ == '__main__':
    package_dir = u'/home/racine/datasets/rxface/package/VzenithFace_30k'
    pull_dir = u'/tmp/test/pull'
    # package_dir = u'/data/da/datasets/rxface/image_struct/lift/A/YDXJ0677.MP4/'
    # pull_dir = u'/data/da/datasets/rxface/image_struct/lift/A/pull'

    if False == os.path.exists(pull_dir):
        os.mkdir(pull_dir)

    select_face_from_package(package_dir,pull_dir)
    # pull_face_from_package(package_dir,pull_dir)
