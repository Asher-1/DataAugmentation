# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/7 11:30
# @Author  : ludahai
# @FileName: extract_helmets.py
# @Software: PyCharm

# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/4/26 9:06
# @Author  : ludahai
# @FileName: MergePersonAndHelmet.py
# @Software: PyCharm

import file_processing
import os
from xml.dom import minidom
from tqdm import tqdm
import numpy as np
from scipy import misc
import copy
import time


def parse_bboxes_for_workers(src_doc, img_shape, labels=('helmet',), margin=4, mini_size=6):
    src_object = src_doc.getElementsByTagName('object')
    bbs = []
    for ind, object_node in enumerate(src_object):
        name_node = object_node.getElementsByTagName("name")
        name = name_node[0].firstChild.data
        if name in labels:
            bndbox_node = object_node.getElementsByTagName("bndbox")[0]
            x_min = int(bndbox_node.getElementsByTagName("xmin")[0].firstChild.data)
            y_min = int(bndbox_node.getElementsByTagName("ymin")[0].firstChild.data)
            x_max = int(bndbox_node.getElementsByTagName("xmax")[0].firstChild.data)
            y_max = int(bndbox_node.getElementsByTagName("ymax")[0].firstChild.data)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(x_min - margin, 0)  # x_min
            bb[1] = np.maximum(y_min - margin, 0)  # y_min
            bb[2] = np.minimum(x_max + margin, img_shape[1])  # x_max
            bb[3] = np.minimum(y_max + margin, img_shape[0])  # y_max
            if bb[2] - bb[0] >= mini_size and bb[3] - bb[1] >= mini_size:
                tuple_obj = (name, bb) if name == 'helmet' else ('head_with_helmet', bb)
                bbs.append(tuple_obj)
    return bbs


def parse_bboxes_for_crowdHuman(src_doc, img_shape, labels=('head',), margin=0, mini_size=6):
    src_object = src_doc.getElementsByTagName('object')
    bbs = []
    for ind, object_node in enumerate(src_object):
        name_node = object_node.getElementsByTagName("name")
        name = name_node[0].firstChild.data
        if name in labels:
            bndbox_node = object_node.getElementsByTagName("bndbox")[0]
            x_min = int(bndbox_node.getElementsByTagName("xmin")[0].firstChild.data)
            y_min = int(bndbox_node.getElementsByTagName("ymin")[0].firstChild.data)
            x_max = int(bndbox_node.getElementsByTagName("xmax")[0].firstChild.data)
            y_max = int(bndbox_node.getElementsByTagName("ymax")[0].firstChild.data)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(x_min - margin, 0)  # x_min
            bb[1] = np.maximum(y_min - margin, 0)  # y_min
            bb[2] = np.minimum(x_max + margin, img_shape[1])  # x_max
            bb[3] = np.minimum(y_max + margin, img_shape[0])  # y_max
            if bb[2] - bb[0] >= mini_size and bb[3] - bb[1] >= mini_size:
                tuple_obj = ('head_with_' + prefix, bb) if name == 'head' else name == (prefix, bb)
                bbs.append(tuple_obj)
                bb2 = copy.deepcopy(bb)
                bb2[3] = bb2[1] + (bb2[3] - bb2[1]) / 2.0
                bbs.append((prefix, bb2))
    return bbs


def MiniDomParseForWorker(src_file, img_shape):
    assert os.path.isfile(src_file)
    src_head = minidom.parse(src_file)
    src_doc = src_head.documentElement
    return parse_bboxes_for_workers(src_doc, img_shape, labels=('helmet', 'head',))


def MiniDomParseForCrowdHuman(src_file, img_shape):
    assert os.path.isfile(src_file)
    src_head = minidom.parse(src_file)
    src_doc = src_head.documentElement
    bbs = parse_bboxes_for_crowdHuman(src_doc, img_shape, labels=('head',))
    return bbs


if __name__ == '__main__':
    ROOT_PATH = 'D:/develop/workstations/resource/datasets/'

    # src_xml_path = ROOT_PATH + 'helmet_db/data/helmet_db/annotations_with_helmet_and_head/'
    # image_path = ROOT_PATH + 'helmet_db/data/helmet_db/compressed/'
    # prefix = 'helmet_'
    # parse_func = MiniDomParseForWorker

    # src_xml_path = ROOT_PATH + 'helmet_db/data/CrowdHuman/annotations/'
    # image_path = ROOT_PATH + 'helmet_db/data/CrowdHuman/images_raw/'
    # prefix = 'no_helmet_'
    # parse_func = MiniDomParseForCrowdHuman

    image_path = ROOT_PATH + 'helmet_db/data/helmet_classification_data/images_raw/2019-05-09_images/'
    src_xml_path = ROOT_PATH + 'helmet_db/data/helmet_classification_data/images_raw/2019-05-09XML/'

    prefix = 'helmet6'
    parse_func = MiniDomParseForCrowdHuman

    save_path = ROOT_PATH + 'helmet_db/data/helmet_classification_data/images/'
    image_size = 256
    filePath_list = []
    filePath_list.extend(file_processing.get_files_list(src_xml_path, postfix='xml'))
    nrof_images = len(filePath_list)
    print("Number of xml annotation files: {}".format(nrof_images))
    index = 0
    for src_xml_file in tqdm(filePath_list):
        index += 1
        image_name = '{}.{}'.format(os.path.splitext(os.path.basename(src_xml_file))[0], 'jpg')
        img_file = os.path.join(image_path, image_name)
        img = misc.imread(img_file, mode='RGB')
        bbs = parse_func(src_xml_file, img.shape)
        for i, bb in enumerate(bbs):
            class_name = bb[0]
            bb = bb[1]
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            try:
                aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            except Exception as e:
                print(e)
                continue
            save_dir = os.path.join(save_path, class_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            time.sleep(0.01)
            out_file_name = os.path.join(save_dir, (prefix + '_{}_{}.{}').format(str(index), str(i), 'jpg'))
            misc.imsave(out_file_name, aligned)
    print('processing end')
