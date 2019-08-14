import numpy as np
import cv2 as cv
import os

IMG_ROWS = 128
IMG_COLS = 64
NUM_CLASSES = 1501

# Creating an array of the required shape
x = np.zeros((12936, 128, 64, 3), dtype=np.uint8)

# Training data preparation
fds = sorted(os.listdir('bounding_box_train/'))
i = 0
for name in fds:
    if (name[len(name) - 4:] == '.jpg'):
        img = cv.imread('bounding_box_train/{}'.format(name), 1)
        x[i] += img
        i += 1

cv.imshow('image', x[12935])
cv.waitKey(0)
cv.destroyAllWindows()
