import cv2
import numpy as np
import copy
from math import sqrt

from skimage.morphology import skeletonize

def expand_mask(shape, crop_corners, mask):
    ## shape: [heigh/y, width/x]
    ## crop_corners: ((x1, y1), (x2, y2))
    expanded = np.zeros(shape,dtype=np.uint8)

    [[x1, y1], [x2, y2]] = crop_corners
    resized_mask = cv2.resize(mask, (x2-x1,y2-y1))

    for iy in range(y2-y1):
        for ix in range(x2-x1):
            if resized_mask[iy,ix] > 100:
                expanded[iy+y1, ix+x1] = 255

    return expanded

def rope_grow(rope_seed, feature_mask):
    ## rope_seed is your dl network output
    ## feature_mask can be obtained from hue channel
    height = feature_mask.shape[0]
    width = feature_mask.shape[1]

    mixed = copy.deepcopy(rope_seed)

    for iy in range(height-1):
        for ix in range(width):
            if mixed[iy, ix] > 100:
                if feature_mask[iy+1, ix] > 100:
                    mixed[iy+1, ix] = 255

    for iy in range(height-1, 0, -1):
        for ix in range(width):
            if mixed[iy, ix] > 100:
                if feature_mask[iy-1, ix] > 100:
                    mixed[iy-1, ix] = 255

    return mixed

def get_ropes(shape, crop_corners, dl_mask, feature_mask):
    rope_mask = rope_grow(dl_mask, feature_mask)

    flesh = np.where(rope_mask>100, 1, 0)
    skeleton = skeletonize(flesh)
    ropes = (np.where(skeleton==True, 255, 0)).astype(np.uint8)

    full_mask = expand_mask(shape, crop_corners, ropes)

    return full_mask

def find_ropes(img):
    rope = []
    for iy in range():
        ...


class rope_info():
    def __init__(self):
        link = []
        length = 0
