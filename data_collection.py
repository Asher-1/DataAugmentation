import shutil
import os
from tqdm import tqdm
from os.path import basename
from os.path import splitext
from data_aug.file_util import get_files_list

if __name__ == '__main__':
    INPUT_PATH = '/media/yons/data/dataset/images/helmet_worker'
    OUT_PATH = '/media/yons/data/dataset/images/helmet_voc/VOC2021'
    save_pic_root_path = os.path.join(OUT_PATH, 'JPEGImages')
    save_xml_root_path = os.path.join(OUT_PATH, 'Annotations')
    if not os.path.exists(save_pic_root_path):
        os.makedirs(save_pic_root_path)
    if not os.path.exists(save_xml_root_path):
        os.makedirs(save_xml_root_path)

    start = 2014
    for i in range(8):
        sub_path = 'VOC{}'.format(start + i)
        # source_pic_path = os.path.join(INPUT_PATH, sub_path, 'JPEGImages_target')
        # source_xml_path = os.path.join(INPUT_PATH, sub_path, 'Annotations_target')
        #
        # pic_list = get_files_list(source_pic_path)
        # xml_list = get_files_list(source_xml_path)
        #
        # for pic, xml in tqdm(list(zip(pic_list, xml_list))):
        #     pic_name = splitext(basename(pic))[0]
        #     xml_name = splitext(basename(xml))[0]
        #     assert pic_name == xml_name
        #     shutil.copy(pic, save_pic_root_path)
        #     shutil.copy(xml, save_xml_root_path)

        source_pic_path = os.path.join(INPUT_PATH, sub_path, 'JPEGImages')
        source_xml_path = os.path.join(INPUT_PATH, sub_path, 'Annotations')

        pic_list = get_files_list(source_pic_path)
        xml_list = get_files_list(source_xml_path)

        for pic, xml in tqdm(list(zip(pic_list, xml_list))):
            pic_name = splitext(basename(pic))[0]
            xml_name = splitext(basename(xml))[0]
            assert pic_name == xml_name
            shutil.copy(pic, save_pic_root_path)
            shutil.copy(xml, save_xml_root_path)
        print('*' * 50 + 'successfully copy {}'.format(sub_path) + '*' * 50)
