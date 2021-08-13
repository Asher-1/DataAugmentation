# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/4/26 9:06
# @Author  : ludahai
# @FileName: MergePersonAndHelmet.py
# @Software: PyCharm

import os
import copy
from tqdm import tqdm
import file_processing
from xml.dom import minidom


def generate_new_xml(src_doc, modify_label, add_label='', remove_label=''):
    src_object = src_doc.getElementsByTagName('object')
    add_object = []
    for ind, object_node in enumerate(src_object):
        name_node = object_node.getElementsByTagName("name")
        name = name_node[0].firstChild.data
        if name in modify_label:
            name_node[0].firstChild.data = 'no_helmet'
        if add_label != '':
            object_node_copy = copy.deepcopy(object_node)
            object_node_copy.getElementsByTagName("name")[0].firstChild.data = add_label
            bndbox_node = object_node_copy.getElementsByTagName("bndbox")[0]
            y_min = int(bndbox_node.getElementsByTagName("ymin")[0].firstChild.data)
            y_max = int(bndbox_node.getElementsByTagName("ymax")[0].firstChild.data)
            y_new_max = int(y_min + (y_max - y_min) / 2.0)
            bndbox_node.getElementsByTagName("ymax")[0].firstChild.data = str(y_new_max)
            add_object.append(object_node_copy)
        if remove_label != '':
            if name == remove_label:
                src_doc.removeChild(object_node)
    for node in add_object:
        src_doc.appendChild(node)


def MiniDomParse(src_file, dst_path):
    assert os.path.isfile(src_file) and os.path.isdir(dst_path)
    dst_file = os.path.join(dst_path, os.path.basename(src_file))

    src_head = minidom.parse(src_file)
    src_doc = src_head.documentElement
    generate_new_xml(src_doc, modify_label=['head'], add_label='', remove_label='')
    with open(dst_file, 'w') as f:
        src_head.writexml(f, addindent='', newl='', encoding='utf-8')


if __name__ == '__main__':
    # ROOT_PATH = 'D:/develop/workstations/resource/datasets/'
    # src_xml_path = ROOT_PATH + 'helmet_db/data/helmet_classification_data/images_raw/2019-05-05-hl-rename-human-annotations/'
    # dst_xml_path = ROOT_PATH + 'helmet_db/data/helmet_classification_data/images_raw/2019-05-05-hl-rename-human-annotations/'
    # if not os.path.exists(dst_xml_path):
    #     os.makedirs(dst_xml_path)

    ROOT_PATH = '/media/yons/develop/develop/workstations/resource/datasets/'
    # ROOT_PATH = 'D:/develop/workstations/resource/datasets/'
    src_xml_path = ROOT_PATH + 'helmet_db/data/test/temp_annotations/'
    dst_xml_path = ROOT_PATH + 'helmet_db/data/test/temp_annotations-modification/'
    if not os.path.exists(dst_xml_path):
        os.makedirs(dst_xml_path)
    # src_xml_path = ROOT_PATH + 'helmet_db/data/helmet_db/annotations_helmet/'
    # dst_xml_path = ROOT_PATH + 'helmet_db/data/helmet_db/annotations_with_helmet_and_head/'

    filePath_list = []
    filePath_list.extend(file_processing.get_files_list(src_xml_path, postfix='xml'))
    nrof_images = len(filePath_list)
    print("Number of xml annotation files: {}".format(nrof_images))
    for src_xml_file in tqdm(filePath_list):
        MiniDomParse(src_xml_file, dst_xml_path)
    print('processing end')
