import cv2
import numpy as np
import copy

from math import sqrt

import matplotlib.pyplot as plt

from rope_pre_process import hue_detection
from quality_check import helix_len_mask, check_adv

def img_binary(img, color_range):
    # input should be a single channel image
    height = img.shape[0]
    width = img.shape[1]
    output_img = np.zeros((height,width), dtype=np.uint8)
    for iy in range(height):
        for ix in range(width):
            if (img[iy, ix] > color_range[0]) and (img[iy, ix] < color_range[1]):
                output_img[iy, ix] = 255
            else:
                output_img[iy, ix] = 0

    return output_img

def draw_mask(img, mask):
    height = img.shape[0]
    width = img.shape[1]
    output = copy.deepcopy(img)
    for iy in range(height):
        for ix in range(width):
            if mask[iy, ix] > 100:
                rgb = output[iy, ix]
                if rgb[1] + 100 > 255:
                    rgb[1] = 255
                else:
                    rgb[1] += 100

    return output

def img_diff(img1, img2):
    ## assume images have the same size
    [height, width] = img1.shape

    output = np.zeros((height,width), dtype=np.uint8)
    for iy in range(height):
        for ix in range(width):
            if not img1[iy,ix] == img2[iy, ix]:
                output[iy, ix] = 255

    return output

if __name__ == '__main__': 
    img_list = ["./quality/09-06-10-27-48.png",\
                "./quality/09-06-10-28-31.png",\
                "./quality/09-06-10-29-12.png",\
                "./quality/09-06-10-30-12.png"]

    i = 1
    img1 = cv2.imread(img_list[i])
    img2 = cv2.imread(img_list[i+1])

    poly = np.array([[923, 391],[508,370],[512,310],[927,331]])

    ## extend the size:
    rope_hue = hue_detection(img1, poly)

    p = check_adv(img1, img2, poly, rope_hue)
    print(p)

