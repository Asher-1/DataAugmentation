import os
import random

trainval_percent = 0.95
train_percent = 0.95
# ROOT_PATH = '/media/yons/data/dataset/images/helmet_voc/VOC2021/'
ROOT_PATH = '/media/yons/data/dataset/images/helmet_worker/VOC_TEST/'
xmlfilepath = ROOT_PATH + 'Annotations'
txtsavepath = ROOT_PATH + 'ImageSets/Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
print("train_number:{}\t val_number:{}\t test_number:{}".format(tr, tv-tr, num-tv))
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(ROOT_PATH + 'ImageSets/Main/trainval.txt', 'w')
ftest = open(ROOT_PATH + 'ImageSets/Main/test.txt', 'w')
ftrain = open(ROOT_PATH + 'ImageSets/Main/train.txt', 'w')
fval = open(ROOT_PATH + 'ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
