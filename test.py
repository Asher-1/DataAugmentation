from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("messi.jpg")[:, :, ::-1]  # OpenCV uses BGR channels
bboxes = pkl.load(open("messi_ann.pkl", "rb"))
print(bboxes)

transforms = Sequence([RandomHorizontalFlip(1), RandomScale(0.2, diff=True), RandomRotate(10)])

img, bboxes = transforms(img, bboxes)

plt.imshow(draw_rect(img, bboxes))
plt.show()


seq = Sequence(
    [RandomHSV(40, 40, 30), RandomHorizontalFlip(), RandomScale(diff=True), RandomTranslate(), RandomRotate(10), RandomShear()])
img_, bboxes_ = seq(img.copy(), bboxes.copy())

plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()