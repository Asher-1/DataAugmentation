# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/15 17:40
# @Author  : ludahai
# @FileName: remove_according_annotations.py
# @Software: PyCharm

import os
import shutil
import file_processing
from tqdm import tqdm

if __name__ == '__main__':
    # ROOT_PATH = 'E:/dataset/images/CrowdHuman/'
    ROOT_PATH = '/media/yons/data/dataset/images/helmet_worker/VOC2020/'
    #
    # # videoPath = ROOT_PATH + 'test/test_videos/'  # 视频地址
    # # EXTRACT_FOLDER = ROOT_PATH + 'test/test_images/'  # 存放帧图片的位置
    #
    # xml_path = ROOT_PATH + 'data/test/temp_annotations/'  # 视频地址
    # save_dir = ROOT_PATH + 'data/test/temp_images'
    # # input_dir = ROOT_PATH + 'data/images_raw/'
    # input_dir = ROOT_PATH + 'data/CrowdHuman/images_raw2/'

    # xml_path = ROOT_PATH + 'worker_2019052291318_annotations/'
    # save_dir = ROOT_PATH + 'worker_2019052291318_partial'
    # # input_dir = ROOT_PATH + 'data/images_raw/'
    # input_dir = ROOT_PATH + 'worker_2019052291318/'

    # input_dir = ROOT_PATH + 'images_raw2'
    # xml_path = ROOT_PATH + 'temp_annotations-modification/'
    # save_dir = ROOT_PATH + 'images_raw2_partial'

    input_dir = ROOT_PATH + 'JPEGImages'
    xml_path = ROOT_PATH + 'Annotations/'
    save_dir = ROOT_PATH + 'tmp'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    xml_save_dir = os.path.join(save_dir, 'xml')
    if not os.path.exists(xml_save_dir):
        os.makedirs(xml_save_dir)

    filePath_list = file_processing.get_files_list(xml_path, postfix='ALL')
    nrof_images = len(filePath_list)
    print("Number of xml files: {}".format(nrof_images))
    for file in tqdm(filePath_list):
        image_name = os.path.splitext(os.path.basename(file))[0]
        input_image = os.path.join(input_dir, '{}.{}'.format(image_name, 'jpg'))
        if not os.path.exists(input_image):
            assert FileNotFoundError(input_image)
            continue
        shutil.copy(input_image, save_dir)
        shutil.copy(file, os.path.join(xml_save_dir, os.path.basename(file)))
