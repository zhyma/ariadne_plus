import cv2
import numpy as np
import copy
from math import sqrt

from skimage.morphology import skeletonize
import operator

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

    kernel = np.ones((5, 5), np.uint8)
    feature_dilate = cv2.dilate(feature_mask, kernel, iterations=1)

    for iy in range(height-1):
        for ix in range(width):
            if mixed[iy, ix] > 100:
                if feature_dilate[iy+1, ix] > 100:
                    mixed[iy+1, ix] = 255

    for iy in range(height-1, 0, -1):
        for ix in range(width):
            if mixed[iy, ix] > 100:
                if feature_dilate[iy-1, ix] > 100:
                    mixed[iy-1, ix] = 255

    return mixed

def get_rope_mask(shape, crop_corners, dl_mask, feature_mask):
    rope_mask = rope_grow(dl_mask, feature_mask)

    flesh = np.where(rope_mask>100, 1, 0)
    skeleton = skeletonize(flesh)
    ropes = (np.where(skeleton==True, 255, 0)).astype(np.uint8)

    full_mask = expand_mask(shape, crop_corners, ropes)

    return full_mask

def find_ropes(img):
    
    ropes = []
    [height, width] = img.shape
    # iy = 719
    # for ix in range(width):
    #     if img[iy, ix] > 100:
    #         ropes.append(rope_info())
    #         ropes[-1].link = [iy, ix]
    #         ropes[-1].length = 1

    for iy in range(height-1, 0, -1):
        for ix in range(width):
            if img[iy, ix] > 100:
                ## a pixel belong to a piece of rope
                connected = []
                for j in ropes:
                    if abs(iy-j.link[0][0]) + abs(ix-j.link[0][1]) <= 2:
                        ## is connected to one of the rope
                        connected.append(j)
                if len(connected) > 0:
                    for k in connected:
                        k.link = [[iy, ix]] + k.link
                        k.length += 1
                else:
                    ## no previous section, create a new one
                    ropes.append(rope_info())
                    ropes[-1].link = [[iy, ix]]
                    ropes[-1].length = 1
            else:
                continue
    
    for r in ropes:
        print(r.length, end=',')
    print('')

    ropes.sort(key=operator.attrgetter('length'), reverse=True)
    ropes = ropes[:2]

    return ropes


    # top2 = [] ## the longest one and the 2nd longest one
    # for i in skeletons:

class rope_info():
    def __init__(self):
        self.link = []
        self.center = None
        self.length = 0
