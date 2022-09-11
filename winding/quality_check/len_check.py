import cv2
import numpy as np
import copy

from math import sqrt

import matplotlib.pyplot as plt

from adv_check import helix_adv_mask, find_all_contours

def helix_len_mask(h_img, poly, color_range):
    ## extract feature_map from img by using the 2d bounding box
    [height, width] = h_img.shape

    sort_y = poly[poly[:,1].argsort()]
    sort_x = poly[poly[:,0].argsort()]
    y1 = sort_y[0,1]
    y2 = sort_y[-1,1]+30
    x1 = sort_x[0,0]
    x2 = sort_x[-1,0]

    output = np.zeros((y2-y1+1, x2-x1+1), dtype=np.uint8)

    for iy in range(0, y2-y1+1):
        for ix in range(0, x2-x1+1):
            # print("ix+x1, iy+y1: {}, {}, {}, {}".format(ix, x1, iy, y1))
            # j = cv2.pointPolygonTest(poly, (int(ix+x1),int(iy+y1)), False)
            j = 1
            if (j>0) and (h_img[iy+y1, ix+x1] > color_range[0]) and (h_img[iy+y1, ix+x1] < color_range[1]):
                output[iy, ix] = 255
            else:
                output[iy, ix] = 0

    kernel = np.ones((3, 3), np.uint8)
    output = cv2.erode(output, kernel, iterations=1)
    output = cv2.dilate(output, kernel, iterations=1)

    line = [[sort_y[-2][0]-x1, sort_y[-2][1]-y1], [sort_y[-1][0]-x1, sort_y[-1][1]-y1]]

    return output, line

def find_rope_width(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rope_piece = []
    size_max = -1
    idx_max = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area >= size_max:
            size_max = area
            idx_max = i

    # hull = cv2.convexHull(contours[idx_max])
    rect = cv2.minAreaRect(contours[idx_max])
    width = rect[1][1]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return int(width), box

def remove_active(img, rope_width, line):

    ## need to know the rope_width, and the bottom line of the rod
    [height, width] = img.shape

    new_mask = np.zeros((height, width), dtype=np.uint8)

    cont = find_all_contours(img)
    cv2.fillPoly(new_mask, pts=[i for i in cont], color=255)

    [x1, y1] = line[0]
    [x2, y2] = line[1]
    a = (y2-y1)/(x2-x1)
    b = y1-a*x1

    for ix in range(width):
        new_mask[0, ix] = 0

    for iy in range(1, height):
        new_mask[iy, width-1] = 0

        ix = width-2
        ## skip empty part
        while ix >= 1:
            if new_mask[iy, ix] < 100:
                ix-=1
                continue
            else:
                break

        ## remove the active end
        cnt_n = 0
        while (cnt_n < rope_width) and (ix >= 1):
            new_mask[iy, ix] = 0
            cnt_n += 1
            ix -= 1

        ## skip gap (if exist)
        while (ix >= 0) and new_mask[iy, ix] < 100:
            ix -= 1

        ## keep the wrap we want to examine
        cnt_n = 0
        while (cnt_n < rope_width) and (ix >= 1):
            if new_mask[iy, ix] > 100:
                cnt_n += 1
                ix -= 1
            else:
                break

        ## remove the rest
        while (ix >= 1):
            ## any of the top three pixels are marked
            ## and below the bottom line of the rod
            j = iy - (a*ix+b)
            or_op = int(new_mask[iy-1, ix-1]) + int(new_mask[iy-1, ix]) + int(new_mask[iy-1, ix+1])
            if (j > 0) and (or_op > 100):
                ...
            else:
                new_mask[iy, ix] = 0
            ix -= 1
 
    # cont = find_all_contours(new_mask)

    contours, hierarchy = cv2.findContours(new_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    x = 0
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < 50:
            continue
        M = cv2.moments(contours[i])
        # if M['m00'] == 0:
        #     continue
        cx = int(M['m10']/M['m00'])
        if cx > x:
            x = cx
            idx = i

    output = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(output, pts=[contours[idx]], color=255)
    # cv2.fillPoly(output, pts=[i for i in contours], color=255)

    return output

def check_len(img1, img2, poly, hue):
    mask1 = helix_len_mask(cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)[:,:,0], poly, hue)
    mask2 = helix_len_mask(cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)[:,:,0], poly, hue)

    hull1 = get_single_hull(mask1)
    hull2 = get_single_hull(mask2)

    [height, width] = mask1.shape
    output = np.zeros((height, width), dtype=np.uint8)
    for iy in range(height):
        for ix in range(width):
            j = cv2.pointPolygonTest(hull1, (ix,iy), False)
            # pixel location within hull1, AND (XOR if pixel is in mask1 and mask2)
            if (j > 0) and (mask1[iy, ix] != mask2[iy, ix]):
                output[iy, ix] = 255

    return output

if __name__ == '__main__': 
    from rope_pre_process import hue_detection

    fig = plt.figure(figsize=(16,8))
    ax = []
    for i in range(3):
        for j in range(4):
            ax.append(plt.subplot2grid((3,4),(i,j)))

    img_list = ["./quality/09-06-10-27-48.png",\
                "./quality/09-06-10-28-31.png",\
                "./quality/09-06-10-29-12.png",\
                "./quality/09-06-10-30-12.png"]

    img = []
    for i in range(4):
        img.append(cv2.imread(img_list[i]))
        ax[i].imshow(cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB))

    poly = np.array([[923, 391],[508,370],[512,310],[927,331]])

    ## find the rope in the first image and estimate its' property:
    
    rope_hue = hue_detection(img[0], poly)
    mask0 = helix_adv_mask(cv2.cvtColor(img[0], cv2.COLOR_BGR2HSV)[:,:,0], poly, rope_hue)
    rope_width, box0 = find_rope_width(mask0)

    ax[4].imshow(mask0)
    box0_img = np.zeros(mask0.shape, dtype=np.uint8)
    cv2.drawContours(box0_img, [box0],-1,255,1)
    ax[8].imshow(box0_img)

    mask = []
    contours = []
    inter_step_img = []
    for i in range(3):
        m, l = helix_len_mask(cv2.cvtColor(img[i+1], cv2.COLOR_BGR2HSV)[:,:,0], poly, rope_hue)
        mask.append(m)
        new_mask = remove_active(mask[i], rope_width, l)
        inter_step_img.append(new_mask)
        # cv2.drawContours(inter_step_img[i], [contours[i]],-1,255,1)

        ax[i+5].imshow(mask[i])
        ax[i+9].imshow(inter_step_img[i])

    plt.tight_layout()

    plt.show()