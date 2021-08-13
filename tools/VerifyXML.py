# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/4/26 9:06
# @Author  : ludahai
# @FileName: MergePersonAndHelmet.py
# @Software: PyCharm

import os
import shutil
from tqdm import tqdm
import file_processing
from xml.dom import minidom


def generate_new_xml(src_doc, verify_labels):
    src_object = src_doc.getElementsByTagName('object')
    validation = True
    for ind, object_node in enumerate(src_object):
        name_node = object_node.getElementsByTagName("name")
        name = name_node[0].firstChild.data

        size_object = src_doc.getElementsByTagName('size')[0]
        img_width = int(size_object.getElementsByTagName("width")[0].firstChild.data)
        img_height = int(size_object.getElementsByTagName("height")[0].firstChild.data)

        if name not in verify_labels:
            validation = False
            print('find unknown label: {}'.format(name))
        else:
            bndbox_node = object_node.getElementsByTagName("bndbox")[0]
            x_min = int(bndbox_node.getElementsByTagName("xmin")[0].firstChild.data)
            y_min = int(bndbox_node.getElementsByTagName("ymin")[0].firstChild.data)
            x_max = int(bndbox_node.getElementsByTagName("xmax")[0].firstChild.data)
            y_max = int(bndbox_node.getElementsByTagName("ymax")[0].firstChild.data)
            if x_min < 0 or y_min < 0 or x_max >= img_width or y_max >= img_height:
                validation = False
                # print("find annotations out of bounds...{}".format([x_min, x_max, y_min, y_max]))

                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, img_width - 1)
                y_max = min(y_max, img_height - 1)
                if abs(x_max-img_width) > 5 or abs(y_max - img_height) > 5:
                    assert "invalid data..."
                bndbox_node.getElementsByTagName("xmin")[0].firstChild.data = str(x_min)
                bndbox_node.getElementsByTagName("ymin")[0].firstChild.data = str(y_min)
                bndbox_node.getElementsByTagName("xmax")[0].firstChild.data = str(x_max)
                bndbox_node.getElementsByTagName("ymax")[0].firstChild.data = str(y_max)
    return validation


def MiniDomParse(src_file, dst_path):
    global invalid_number
    assert os.path.isfile(src_file) and os.path.isdir(dst_path)
    dst_file = os.path.join(dst_path, os.path.basename(src_file))

    src_head = minidom.parse(src_file)
    src_doc = src_head.documentElement
    # invalidation = generate_new_xml(src_doc, verify_labels=['person', 'helmet', 'no_helmet'])
    validation = generate_new_xml(src_doc, verify_labels=['person', 'helmet', 'no_helmet'])
    if validation:
        shutil.copy(src_file, dst_file)
    else:
        # print("invalid file : " + src_file)
        invalid_number += 1
        with open(dst_file, 'w') as f:
            src_head.writexml(f, addindent='', newl='', encoding='utf-8')


if __name__ == '__main__':
    # ROOT_PATH = '/media/yons/data/dataset/images/helmet_voc/VOC2021/'
    # ROOT_PATH = '/media/yons/data/dataset/images/helmet_worker/VOC2021/'
    ROOT_PATH = '/media/yons/data/dataset/images/helmet_worker/VOC2021/'
    src_xml_path = ROOT_PATH + 'Annotations/'
    dst_xml_path = ROOT_PATH + 'tmp/'
    invalid_number = 0
    if not os.path.exists(dst_xml_path):
        os.makedirs(dst_xml_path)

    filePath_list = []
    filePath_list.extend(file_processing.get_files_list(src_xml_path, postfix='xml'))
    nrof_images = len(filePath_list)
    print("Number of xml annotation files: {}".format(nrof_images))
    for src_xml_file in tqdm(filePath_list):
        MiniDomParse(src_xml_file, dst_xml_path)

    print('processing end\n ')
    print('#' * 50 + " find {} invalid xml files ".format(invalid_number) + "#" * 50)
