from .data_aug import *
from .bbox_util import *
from .xml_helper import parse_xml
from .xml_helper import generate_xml
from .file_util import get_files_list

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


def task(pic_path):
    img = cv2.imread(pic_path)[:, :, ::-1]

    name, extention = os.path.splitext(os.path.basename(pic_path))

    xml_path = os.path.join(source_xml_root_path, name + '.xml')
    gtbox_label = parse_xml(xml_path)  # 解析得到box信息，格式为[[x_min,y_min,x_max,y_max,name]]
    # convert to numpy float64 format
    for bbs in gtbox_label:
        bbs[-1] = NAME_LABEL_MAP[bbs[-1]]
    gtbox_label = np.array(gtbox_label).astype(np.float64)
    cnt = 0
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

        target_name = '{}_{}{}'.format(name, cnt, extention)

        # save augmented image
        save_pic(auged_img, target_name, target_pic_root_path)
        # save augmented bboxes
        generate_xml(target_name, auged_bboxes, auged_img.shape, target_xml_root_path)
    yield pic_path


vis = False
need_aug_num = 0
LABEl_NAME_MAP = get_label_name_map()

ROOT_PATH = None
source_pic_root_path = None
source_xml_root_path = None

target_pic_root_path = None
target_xml_root_path = None
seq = None


def config_setting(paths, sequence_engine, aug_num_per_img, visualizaiton=False):
    global source_pic_root_path, source_xml_root_path, target_pic_root_path, target_xml_root_path, \
        seq, need_aug_num, vis
    source_pic_root_path, source_xml_root_path, target_pic_root_path, target_xml_root_path = paths
    seq = sequence_engine
    need_aug_num = aug_num_per_img
    vis = visualizaiton
    if not os.path.exists(target_pic_root_path):
        os.makedirs(target_pic_root_path)
    if not os.path.exists(target_xml_root_path):
        os.makedirs(target_xml_root_path)
    files_lists = get_files_list(source_pic_root_path)
    return files_lists