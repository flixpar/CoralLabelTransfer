import cv2
import numpy as np
import scipy.misc
from scipy import ndimage
import matplotlib.pyplot as plt

img = cv2.imread("../imgs/region_raw.png")

cropped = img[200:300, 200:300]

img_height, img_width, img_channels = img.shape
height, width, channels = (100, 100, 3)

angle = 0
theta = np.deg2rad(angle)
sin_theta = np.sin(theta)
cos_theta = np.cos(theta)

new_img_height = int(height * (sin_theta + cos_theta))
new_img_width = int(width * (sin_theta + cos_theta))

print((new_img_height, new_img_width))

# center = (50, 50)
center = (64, 64)

rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])

top_left = np.array([200, 200])
bottom_right = np.array([300, 300])

top_left = np.dot(top_left, rotMatrix)
bottom_right = np.dot(bottom_right, rotMatrix)

print(top_left)
print(bottom_right)

top_left = (int(top_left[0]), int(top_left[1]))
bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

scale = 1.15
# scale = new_img_height / height
print(scale)

# M = cv2.getRotationMatrix2D(center, angle, scale)
# rotated = cv2.warpAffine(cropped, M, (new_img_width, new_img_height))

# rotated = ndimage.rotate(img, 20, reshape=False)
# rotated = rotated[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

rotated = ndimage.rotate(cropped, angle, reshape=False)


# display the results
fig, ax = plt.subplots(1, 2)

ax[0].set_title("Cropped")
ax[1].set_title("Rotated")

ax[0].imshow(cropped)
ax[1].imshow(rotated)

plt.show()
