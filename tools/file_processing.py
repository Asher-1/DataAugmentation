# -*-coding: utf-8 -*-
"""
    @Project: faceRecognition
    @File   : image_processing.py
    @Author : Asher
    @E-mail : ludahai19@163.com
    @Date   : 2018-12-07 10:10:27
"""

import configparser
import os
import json
import numpy as np
import glob
from collections import OrderedDict
import pickle


def write_data(file, content_list, model):
    with open(file, mode=model) as f:
        for line in content_list:
            f.write(os.path.basename(line) + "\n")


def read_data(file):
    with open(file, mode="r") as f:
        content_list = f.readlines()
        content_list = [content.rstrip() for content in content_list]
    return content_list


def getFilePathList(file_dir):
    '''
    获取file_dir目录下，所有文本路径，包括子目录文件
    :param rootDir:
    :return:
    '''
    filePath_list = []
    for walk in os.walk(file_dir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list


def get_files_list(file_dir, postfix='ALL'):
    '''
    获得file_dir目录下，后缀名为postfix所有文件列表，包括子目录
    :param file_dir:
    :param postfix:
    :return:
    '''
    postfix = postfix.split('.')[-1]
    file_list = []
    filePath_list = getFilePathList(file_dir)
    if postfix == 'ALL':
        file_list = filePath_list
    else:
        for file in filePath_list:
            basename = os.path.basename(file)  # 获得路径下的文件名
            postfix_name = basename.split('.')[-1]
            if postfix_name == postfix:
                file_list.append(file)
    file_list.sort()
    return file_list


def gen_files_labels(files_dir, postfix='ALL'):
    '''
    获取files_dir路径下所有文件路径，以及labels,其中labels用子级文件名表示
    files_dir目录下，同一类别的文件放一个文件夹，其labels即为文件的名
    :param files_dir:
    :return:filePath_list所有文件的路径,label_list对应的labels
    '''
    # filePath_list = getFilePathList(files_dir)
    filePath_list = get_files_list(files_dir, postfix=postfix)
    print("files nums:{}".format(len(filePath_list)))
    # 获取所有样本标签
    label_list = []
    for filePath in filePath_list:
        label = filePath.split(os.sep)[-2]
        label_list.append(label)

    labels_set = list(set(label_list))
    print("labels:{}".format(labels_set))

    # 标签统计计数
    # print(pd.value_counts(label_list))
    return filePath_list, label_list


class PersonClass(object):
    "Stores the paths to images for a given class"

    def __init__(self, id, name, image_paths):
        self.id = id
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.id + ', ' + self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


class GroupClass(object):
    def __init__(self, group, person_classes, image_db=None):
        self.group_name = group
        self.person_set = person_classes
        self.image_db = image_db

    def __str__(self):
        return self.group_name + ', ' + str(len(self.person_set)) + ' person'

    def __len__(self):
        return len(self.person_set)


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def getDatasetWithGroups(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    groups = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
    groups.sort()
    nrof_groups = len(groups)
    for i in range(nrof_groups):
        group_name = groups[i]
        id_dir = os.path.join(path_exp, group_name)
        classes = [path for path in os.listdir(id_dir) if os.path.isdir(os.path.join(id_dir, path))]
        nrof_classes = len(classes)
        persons = []
        for j in range(nrof_classes):
            class_name = classes[j]
            facedir = os.path.join(id_dir, class_name)
            image_paths = get_image_paths(facedir)
            name = os.path.basename(image_paths[0]).split('_')[1]
            persons.append(PersonClass(class_name, name, image_paths))
        dataset.append(GroupClass(group_name, persons))
    return dataset


def to_json(output_path, *args):
    origin_name = os.path.splitext(os.path.basename(output_path))[0]
    image_id = "_".join(origin_name.split("_")[-2:])
    with open(output_path, "w") as json_writer:
        boxes, boxes_name, distances, rotate_angle = args
        total_number = len(boxes)
        assert total_number == len(boxes_name) == len(distances)
        result = -1
        data = OrderedDict()
        index = 1
        missing_math_num = 0
        for labels, box, dis in zip(boxes_name, boxes, distances):
            if labels == '未识别':
                label_list = labels.split('_')
                group = None
                name = None
                user_id = None
                result = 0
                missing_math_num += 1
            else:
                label_list = labels.split('_')
                group = label_list[0]
                name = label_list[1]
                user_id = label_list[2]
                result = -1
            box_with = box[2] - box[0]
            box_height = box[3] - box[1]
            box_left = box[0]
            box_top = box[1]
            probability = min(1 - round(dis, 2) + 0.5, 1)

            face = {"user_id": user_id,
                    "group": group,
                    "probability": str(probability),
                    "result": str(result),
                    "face_rectangle": {
                        "width": str(box_with),
                        "top": str(box_top),
                        "left": str(box_left),
                        "height": str(box_height)
                    }}
            face_key = "{}{}".format("faces", str(index))
            index += 1
            data[face_key] = face
        data["image_id"] = image_id
        data["rotate_angle"] = str(rotate_angle)
        data["total_number"] = str(total_number)
        data["match_number"] = str(total_number - missing_math_num)
        data["match_rate"] = str(round((total_number - missing_math_num) * 1.0 / total_number, 4))
        json_writer.write(json.dumps(data, ensure_ascii=False, sort_keys=False, indent=4))


def convert_to_json(image_id, *args):
    boxes, boxes_name, distances, rotate_angle = args
    total_number = len(boxes)
    assert total_number == len(boxes_name) == len(distances)
    result = -1
    data = OrderedDict()
    index = 1
    missing_math_num = 0
    for labels, box, dis in zip(boxes_name, boxes, distances):
        if labels == '未识别':
            label_list = labels.split('_')
            group = None
            name = None
            user_id = None
            result = 0
            missing_math_num += 1
        else:
            label_list = labels.split('_')
            group = label_list[0]
            name = label_list[1]
            user_id = label_list[2]
            result = -1
        box_with = box[2] - box[0]
        box_height = box[3] - box[1]
        box_left = box[0]
        box_top = box[1]
        probability = min(1 - round(dis, 2) + 0.5, 1)

        face = {"user_id": user_id,
                "group": group,
                "probability": str(probability),
                "result": str(result),
                "face_rectangle": {
                    "width": str(box_with),
                    "top": str(box_top),
                    "left": str(box_left),
                    "height": str(box_height)
                }}
        face_key = "{}{}".format("faces", str(index))
        index += 1
        data[face_key] = face
    data["image_id"] = image_id
    data["rotate_angle"] = str(rotate_angle)
    data["total_number"] = str(total_number)
    data["match_number"] = str(total_number - missing_math_num)
    data["match_rate"] = str(round((total_number - missing_math_num) * 1.0 / total_number, 4))
    return data


def load_dataset(dataset_path, group_name=None):
    '''
    加载人脸数据库
    :param dataset_path: embedding.npy文件（faceEmbedding.npy）
    :param group_name: the group type
    :return:
    '''

    with open(os.path.join(dataset_path, 'labels.pkl'), 'rb') as f:
        label_dict = pickle.load(f)
    names_dict = {}
    if group_name:
        names_dict[group_name] = label_dict[group_name]
    else:
        names_dict = label_dict
    feature_dicts = {}
    for group_labels, names_list in names_dict.items():
        group_name, _ = os.path.splitext(os.path.basename(group_labels))
        compare_emb = np.load(os.path.join(dataset_path, '{}{}'.format(group_name, '.npy')))
        assert len(compare_emb) == len(names_list)
        feature_dicts[group_name] = (names_list, compare_emb)
    return feature_dicts


def log_distances(dis_writer, pic_name, pred_label, distances, box_probs):
    '''
    log distances
    :param dis_writer:
    :param pic_name:
    :param pred_label:
    :param distances:
    :param box_probs:
    :return:
    '''
    assert len(pred_label) == len(distances) == len(box_probs)
    dis_writer.write(pic_name)
    dis_writer.write('-->\t')
    for i in range(len(pred_label)):
        text = '{}({}:{}-{}:{})\t'.format(pred_label[i], 'box_prob',
                                          round(box_probs[i], 4), 'dis', round(distances[i], 2))
        dis_writer.write(text)
    dis_writer.write('\n')


def load_configs(config_path):
    # 第一步：创建conf对象
    conf = configparser.ConfigParser()
    conf.read(config_path)
    secs = conf.sections()
    if len(secs) == 0:
        error_messages = 'cannot find the config file :{}'.format(config_path)
        raise FileNotFoundError(error_messages)
    config_dict = {}
    for sec in secs:
        if sec == 'Section_path':
            ROOT_PATH = conf.get(sec, 'ROOT_PATH')
            for key, value in conf.items(sec):
                config_dict[key.upper()] = os.path.join(ROOT_PATH, value)
        else:
            for key, value in conf.items(sec):
                if type(eval(value)) == int:
                    value = int(value)
                elif type(eval(value)) == float:
                    value = float(value)
                elif type(eval(value)) == bool:
                    value = bool(eval(value))
                else:
                    pass
                config_dict[key.upper()] = value
    return config_dict


if __name__ == '__main__':
    file_dir = 'JPEGImages'
    file_list = get_files_list(file_dir)
    for file in file_list:
        print(file)
