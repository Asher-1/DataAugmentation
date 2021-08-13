import os
from tqdm import tqdm
import shutil
from collections import Counter
from data_aug.xml_helper import parse_xml
from data_aug.file_util import get_files_list

if __name__ == '__main__':
    ROOT_PATH = '/media/yons/data/dataset/images/helmet_voc/VOC2021'
    xml_root_path = os.path.join(ROOT_PATH, 'Annotations')
    pic_root_path = os.path.join(ROOT_PATH, 'JPEGImages')
    xml_list = get_files_list(xml_root_path)

    labels = []
    for xml_file in tqdm(xml_list):
        img_name = os.path.splitext(os.path.basename(xml_file))[0] + '.jpg'
        assert os.path.exists(os.path.join(pic_root_path, img_name))
        gtbox_label = parse_xml(xml_file)  # 解析得到box信息，格式为[[x_min,y_min,x_max,y_max,name]]
        if len(gtbox_label) == 0:
            print('detect empty xml file: {}, remove'.format(xml_file))
            shutil.move(xml_file, ROOT_PATH)
            shutil.move(os.path.join(pic_root_path, img_name), ROOT_PATH)
        # for bbs in gtbox_label:
        #     if len(bbs) == 0:
        #         print('detect empty xml file: {}'.format(xml_file))
        labels.extend([bbs[-1] for bbs in gtbox_label])
    res_dict = Counter(labels)

    total_pics = len(xml_list)
    total_labels = sum(res_dict.values()) * 1.0
    print('total pictures: {}'.format(total_pics))
    print('total labels: {}'.format(total_labels))
    for label, label_num in res_dict.items():
        ratio = round(label_num / total_labels, 4) * 100
        print('{}-->\t number: {}, ratio: {}%'.format(label, label_num, ratio))
