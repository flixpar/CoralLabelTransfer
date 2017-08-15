# classes with > 1% of total area:
# 0: 54.72%
# 3: 2.98%
# 11: 1.20%
# 17: 10.53%
# 25: 2.95%
# 26: 1.59%
# 31: 3.06%
# 33: 4.50%
# 38: 6.78%

import cv2
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt

import os
import glob

# setup file paths
base_path = "/home/ml/felix_ws/labeling/imgs/"
fn_pattern = "*mask.png"

# find all mask images
file_paths = []
file_paths += glob.glob(os.path.join(base_path, fn_pattern))

# setup lookup table for input class -> output class
num_input_classes = 58
frequent_classes = [0, 3, 11, 17, 25, 26, 31, 33, 38]
lookup = [0] * num_input_classes

# fill lookup table
used = 0
for i in frequent_classes:
    lookup[i] = used
    used += 1

# modify the files
print("modifying %d files..." % len(file_paths))
for fn in file_paths:

    img = cv2.imread(fn, 0)
    out = np.zeros(img.shape, dtype=np.int)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j] = lookup[img[i,j]]

    cv2.imwrite(fn, out)
