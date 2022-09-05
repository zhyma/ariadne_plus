import cv2
import numpy as np
import copy

from scipy.stats import norm
from math import sqrt

import matplotlib.pyplot as plt

def apply_crop(img, crop_corners):
    ## extract feature_map from img by using the 2d bounding box
    height = img.shape[0]
    width = img.shape[1]
    x1 = crop_corners[0,0]
    x2 = crop_corners[1,0]
    y1 = crop_corners[0,1]
    y2 = crop_corners[2,1]
    feature_map = []
    if len(img.shape) == 2:
        cropped_image = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    else:
        cropped_image = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
    for iy in range(height):
        for ix in range(width):
            j = cv2.pointPolygonTest(crop_corners, (ix,iy), False)
            if j > 0:
                cropped_image[iy-y1, ix-x1] = img[iy, ix]

    return cropped_image


def draw_dl_mask(img, crop_corners, mask, color='g'):
    [[x1, y1], [x2, y2]] = crop_corners
    output = copy.deepcopy(img)
    resized_mask = cv2.resize(mask, (x2-x1,y2-y1))

    if color == 'g':
        c = 1
    elif color == 'r':
        c = 2
    else:
        c = 0

    shape = len(resized_mask.shape)
    selected = False

    for iy in range(y2-y1):
        for ix in range(x2-x1):
            if shape ==2:
                if resized_mask[iy,ix] > 100:
                    selected = True
                else:
                    selected = False
            elif shape == 3:
                if resized_mask[iy,ix][0] > 100:
                    selected = True
                else:
                    selected = False
            if selected:
                rgb = output[iy+y1, ix+x1]
                rgb[0] = rgb[0]/2
                rgb[1] = rgb[1]/2
                rgb[2] = rgb[2]/2
                if rgb[c] + 100 > 255:
                    rgb[c] == 255
                else:
                    rgb[c] += 100

    return output

if __name__ == '__main__': 
    img = cv2.imread("./image1.jpg")
    # img = cv2.imread("./helix/08-31-15-38-15.png")
    # img = cv2.imread("./helix/08-31-15-39-10.png")
    # img = cv2.imread("./helix/08-31-15-39-42.png")
    # img = cv2.imread("./helix/08-31-15-40-08.png")
    # img = cv2.imread("./helix/08-31-15-40-22.png")

    corners = np.array([[943, 378],[530,363],[533,304],[945,318]])
    # corners = np.array([[923, 391],[508,370],[512,310],[927,331]])

    mask = hue_masked(img, corners)

    # output = draw_mask(img, mask)
    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
    plt.show()
    # display_img(binary_img)
    # dog = dog_edge(binary_img)
    # display_img(dog)

