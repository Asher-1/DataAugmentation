from data_aug.data_aug import *
from data_aug.multi_yielder import *
from data_aug.timer_wrapper import timer_wrapper
from data_aug.base_task import task, config_setting


@timer_wrapper
def multi_generator(files_lists):
    for item in multi_yield(files_lists, task, process_mode, 16):
        print('Image : {} augmentation completely successfully...'.format(item))


if __name__ == '__main__':
    # process_mode : 188s
    # thread_mode : 361s

    need_aug_num = 1
    ROOT_PATH = '/media/yons/data/dataset/images/helmet_worker/VOC2018'

    # create data augmentation sequence
    seq = Sequence([RandomHSV(30, 30, 20),
                    RandomHorizontalFlip(p=0.5),
                    RandomNoise(p=0.4, mode='gaussian'),
                    RandomLight(p=0.5, light=(0.5, 1.5)),
                    RandomRotate(5),
                    RandomScale(scale=0.05, diff=False),
                    RandomTranslate(translate=0.05, diff=False)])

    source_pic_root_path = os.path.join(ROOT_PATH, 'JPEGImages')
    source_xml_root_path = os.path.join(ROOT_PATH, 'Annotations')
    target_pic_root_path = os.path.join(ROOT_PATH, 'JPEGImages_target')
    target_xml_root_path = os.path.join(ROOT_PATH, 'Annotations_target')
    paths = [source_pic_root_path, source_xml_root_path, target_pic_root_path, target_xml_root_path]

    files_lists = config_setting(paths, seq, need_aug_num, visualizaiton=False)
    multi_generator(files_lists)
