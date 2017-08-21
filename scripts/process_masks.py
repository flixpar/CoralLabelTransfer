import os
import glob
import cv2
import numpy as np


n_classes = 9
base_path = "/home/ml/felix_ws/labeling/masks/"
mask_fn_pattern = base_path + "*.png"

filelist = glob.glob(mask_fn_pattern)
names = [fn.split('/')[-1].split('_')[0] for fn in filelist]
masks = [cv2.imread(fn, 0) for fn in filelist]

for i in range(n_classes):
    dir_path = base_path + "class" + str(i) + "/"
    os.mkdir(dir_path)
    for mask, name in zip(masks, names):
        m = (mask == i).astype(np.unit8) * 255
        fn = dir_path + name + "_mask.png"
        cv2.imwrite(fn, m)
