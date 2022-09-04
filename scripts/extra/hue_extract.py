import cv2
import numpy as np
import copy

def mask_or(hue_mask, dl_mask):
    height = hue_mask.shape[0]
    width = hue_mask.shape[1]

    mixed = copy.deepcopy(dl_mask)

    for iy in range(height-1):
        for ix in range(width):
            if mixed[iy, ix] > 100:
                if hue_mask[iy+1, ix] > 100:
                    mixed[iy+1, ix] = 255

    for iy in range(height-1, 0, -1):
        for ix in range(width):
            if mixed[iy, ix] > 100:
                if hue_mask[iy-1, ix] > 100:
                    mixed[iy-1, ix] = 255

    return mixed