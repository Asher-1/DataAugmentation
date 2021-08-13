import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_aug.data_aug import *
from data_aug.bbox_util import *
from data_aug.xml_helper import parse_xml
from data_aug.xml_helper import generate_xml

NAME_LABEL_MAP = {
    'back_ground': 0,
    'person': 1,
    'no_helmet': 2,
    'helmet': 3
}


def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict


def save_pic(img, save_name, save_path):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    cv2.imwrite(os.path.join(save_path, save_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    vis = False
    need_aug_num = 2
    LABEl_NAME_MAP = get_label_name_map()

    # ROOT_PATH = './test'
    ROOT_PATH = '/media/yons/data1/dataset/images/helmet_worker/VOC2017'
    source_pic_root_path = os.path.join(ROOT_PATH, 'JPEGImages')
    source_xml_root_path = os.path.join(ROOT_PATH, 'Annotations')

    target_pic_root_path = os.path.join(ROOT_PATH, 'JPEGImages_target')
    target_xml_root_path = os.path.join(ROOT_PATH, 'Annotations_target')
    if not os.path.exists(target_pic_root_path):
        os.makedirs(target_pic_root_path)
    if not os.path.exists(target_xml_root_path):
        os.makedirs(target_xml_root_path)

    # create data augmentation sequence
    seq = Sequence([RandomHSV(40, 40, 30),
                    RandomHorizontalFlip(p=0.5),
                    RandomNoise(p=0.4, mode='gaussian'),
                    RandomLight(p=0.5, light=(0.5, 1.5)),
                    RandomRotate(10),
                    RandomScale(scale=0.1, diff=True),
                    RandomTranslate(diff=True),
                    RandomShear(0.1)])

    for parent, _, files in os.walk(source_pic_root_path):
        for file in tqdm(files):
            cnt = 0

            pic_path = os.path.join(parent, file)
            img = cv2.imread(pic_path)[:, :, ::-1]

            xml_path = os.path.join(source_xml_root_path, file[:-4] + '.xml')
            gtbox_label = parse_xml(xml_path)  # 解析得到box信息，格式为[[x_min,y_min,x_max,y_max,name]]

            # convert to numpy float64 format
            for bbs in gtbox_label:
                bbs[-1] = NAME_LABEL_MAP[bbs[-1]]
            gtbox_label = np.array(gtbox_label).astype(np.float64)

            while cnt < need_aug_num:
                cnt += 1
                # AUGMENTATION
                auged_img, auged_bboxes = seq(img.copy(), gtbox_label.copy())

                # visualization
                if vis:
                    plotted_img = draw_rect(auged_img, auged_bboxes)
                    plt.imshow(plotted_img)
                    plt.show()

                # map label string by indexing
                auged_bboxes = auged_bboxes.astype(int).tolist()
                for bbs in auged_bboxes:
                    bbs[-1] = LABEl_NAME_MAP[bbs[-1]]

                name, extention = os.path.splitext(file)
                target_name = '{}_{}{}'.format(name, cnt, extention)

                # save augmented image
                save_pic(auged_img, target_name, target_pic_root_path)
                # save augmented bboxes
                generate_xml(target_name, auged_bboxes, auged_img.shape, target_xml_root_path)
