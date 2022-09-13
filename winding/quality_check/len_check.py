import cv2
import numpy as np
import copy

from math import sqrt

import matplotlib.pyplot as plt

from adv_check import helix_adv_mask, find_all_contours

from skimage.morphology import skeletonize


def helix_len_mask(h_img, poly, color_range):
    ## extract feature_map from img by using the 2d bounding box
    [height, width] = h_img.shape

    sort_y = poly[poly[:,1].argsort()]
    sort_x = poly[poly[:,0].argsort()]
    y1 = sort_y[0,1]
    y2 = sort_y[-1,1]+30
    x1 = sort_x[0,0]
    x2 = sort_x[-1,0]

    offset = [x1, y1]

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

    bottom_edge = [[sort_y[-2][0]-x1, sort_y[-2][1]-y1], [sort_y[-1][0]-x1, sort_y[-1][1]-y1]]

    return output, offset, bottom_edge

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

def string_search(img, bottom_edge):
    ## search for the longest string(path) within a given image
    [height, width] = img.shape

    [x1, y1] = bottom_edge[0]
    [x2, y2] = bottom_edge[1]
    a = (y2-y1)/(x2-x1)
    b = y1-a*x1

    rope_top = []
    ## rope position near the top edge of the rod, take as the expected position of the wrap
    for iy in range(height):
        for ix in range(width):
            if img[iy, ix] > 100:
                ## a pixel belong to a piece of rope
                rope_top = [ix, iy]
                break
        if len(rope_top) > 0:
            break

    ## find the extra length
    ## breath first search, always prune the shortest branch greedily
    ## find the intersection between the skeleton and the 
    
    search = True
    node0 = node(rope_top, None)
    visit_list = [node0]
    frontier = [node0]
    ## 8 connection
    search_dir = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    while search:
        l = len(frontier)
        ## search on all frontier nodes, 
        ## move down one level (if it's child exist),
        ## or delete the frontier node (if no child, prune greadily)
        for i in range(l-1, -1, -1):
            curr_node = frontier[i]
            print("parent node: [{}]: ".format(curr_node.xy), end=",")
            for next in search_dir:
                x = curr_node.xy[0] + next[0]
                y = curr_node.xy[1] + next[1]
                n_children = 0
                
                visited = False
                for j in visit_list:
                    if [x,y] == j.xy:
                        visited = True

                if visited:
                    ## skip any visited
                    continue

                ## search for its' kids
                if (x < 0) or (y < 0) or (x > width-1) or (y > height-1):
                    ## skip those out of the boundary
                    continue
                elif img[y, x] < 100:
                    ## skip those not being marked
                    continue
                
                ## those are the children of the current node
                n_children += 1
                new_node = node([x,y], curr_node)
                frontier.append(new_node)
                visit_list.append(new_node)
                print("add node: [{}]".format([x,y]), end=', ')
                # visited.append(node([x,y], curr_node))

                if n_children < 1:
                    ## reach the edge of the image, does not have a child  
                    curr_node.n_children = -1

                else:
                    curr_node.n_children = n_children

            print('')

            if len(frontier) > 1:
                ## more than one frontier node left, the other one must has the same length
                frontier.pop(i)
            else:
                ## no other frontier node left, stop searching
                search = False

        print("====number of frontiers: {}====".format(len(frontier)))

    print(frontier[0].len)

    new_mask = np.zeros((height, width), dtype=np.uint8)
    string = []
    i_node = frontier[0]
    while i_node.parent is not None:
        string.append(i_node.xy)
        i_node = i_node.parent

    for i in string:
        new_mask[i[1], i[0]] = 255

    return rope_top, frontier[0].len, new_mask

def remove_active(img, rope_width, bottom_edge):

    ## need to know the rope_width, and the bottom edge of the rod
    [height, width] = img.shape

    new_mask = np.zeros((height, width), dtype=np.uint8)

    cont = find_all_contours(img)
    cv2.fillPoly(new_mask, pts=[i for i in cont], color=255)

    [x1, y1] = bottom_edge[0]
    [x2, y2] = bottom_edge[1]
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
            ## and below the bottom edge of the rod
            j = iy - (a*ix+b)
            or_op = int(new_mask[iy-1, ix-1]) + int(new_mask[iy-1, ix]) + int(new_mask[iy-1, ix+1])
            if (j > 0) and (or_op > 100):
                ...
            else:
                new_mask[iy, ix] = 0
            ix -= 1

    ## assume the largest contours is the piece of rope we want to check (there will be a small piece)
    contours, hierarchy = cv2.findContours(new_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    size = 0
    for i in range(len(contours)):
        i_size = cv2.contourArea(contours[i])
        if i_size > size:
            size = i_size
            idx = i

    output = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(output, pts=[contours[idx]], color=255)

    flesh = np.where(output>100, 1, 0)
    skeleton = skeletonize(flesh)
    rope_skeleton = (np.where(skeleton==True, 255, 0)).astype(np.uint8)

    return rope_skeleton

class node():
    def __init__(self, xy, parent):
        self.xy = xy
        self.parent = None
        self.len = 0
        if parent is not None:
            self.parent = parent
            self.len = parent.len + 1
            
        self.n_children = 0

def draw_cutting_line(img, line):
    output = copy.deepcopy(img)
    [height, width] = img.shape

    [x1, y1] = line[0]
    [x2, y2] = line[1]
    a = (y2-y1)/(x2-x1)
    b = y1-a*x1

    for iy in range(height):
        for ix in range(width):
            j = iy - (a*ix+b)
            if (j < 0) and (output[iy,ix] > 100):
                output[iy, ix] = 80
    cv2.line(output, line[0], line[1], 255, 2)
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
        sub_mask, offset, bottom_edge = helix_len_mask(cv2.cvtColor(img[i+1], cv2.COLOR_BGR2HSV)[:,:,0], poly, rope_hue)
        mask.append(sub_mask)
        new_mask = remove_active(mask[i], rope_width, bottom_edge)
        _, _, filtered_string = string_search(new_mask, bottom_edge)
        # inter_step_img.append(new_mask)
        # cv2.drawContours(inter_step_img[i], [contours[i]],-1,255,1)

        ax[i+5].imshow(mask[i])
        # inter_step_img.append(draw_cutting_line(new_mask, bottom_edge))
        inter_step_img.append(draw_cutting_line(filtered_string, bottom_edge))
        ax[i+9].imshow(inter_step_img[i])

    plt.tight_layout()

    plt.show()