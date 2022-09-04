import cv2
import os

from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

# ariadne
from scripts.ariadne import AriadnePlus

from winding.create_mask import hue_masked, apply_crop, draw_dl_mask, expand_mask
import numpy as np

from winding.post_process import rope_grow
from skimage.morphology import skeletonize

# from scripts.extra.post_process import mask_or

file_list = []
for filename in os.listdir('./winding'):
    if filename.endswith('png'):
        file_list.append(filename)


num_segments = 30
show_result = True

main_folder = os.getcwd()

##################################
# Initializing class
##################################
ariadne = AriadnePlus(main_folder, num_segments)

##################################
# Loading Input Image
##################################

if show_result:
    file_list = [file_list[-1]]

for img_name in file_list:

    img_path = os.path.join(main_folder, 'winding/'+img_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # corners = np.array([[943, 378],[530,363],[533,304],[945,318]])
    corners = np.array([[923, 391],[508,370],[512,310],[927,331]])

    ## Use hue range to generate a feature map
    mask_by_hue = hue_masked(img, corners)
    feature_mask = cv2.resize(mask_by_hue, (640,480))

    ##################################
    # Run pipeline
    ##################################

    ## extract 4:3 image that includes ropes.
    sort1 = corners[corners[:,1].argsort()]
    sort2 = corners[corners[:,0].argsort()]
    y1 = sort1[0,1]-10
    y2 = 720
    x1 = sort2[0,0]-10
    x2 = sort2[-1,0]+10

    if (x2-x1)/(y2-y1) > 640/480:
            ## too rectangle
            y1 = int(y2-(x2-x1)*480/640)
    else:
        ## too square
        xc = (x2+x1)/2
        width = (y2-y1)*640/480
        x1 = int(xc-width/2)
        x2 = int(xc+width/2)

    crop_corners= np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])

    cropped_img = cv2.resize(img[y1:y2,x1:x2], (640,480))

    rv = ariadne.runAriadne(cropped_img, debug=True)

    rope_mask = rope_grow(rv["img_mask"], feature_mask)

    flesh = np.where(rope_mask>100, 1, 0)
    skeleton = skeletonize(flesh)
    ropes = (np.where(skeleton==True, 255, 0)).astype(np.uint8)

    full_mask = expand_mask(img.shape, crop_corners, ropes)

    ##################################
    # Check the result
    ################################## 

    kernel = np.ones((5, 5), np.uint8)
    mask_dilate = cv2.dilate(full_mask, kernel, iterations=1)
    new_img = draw_dl_mask(img, np.array([[0, 0],[1279,0],[1279,719],[0, 719]]), mask_dilate, color='b')
    cv2.polylines(new_img,[corners],True,(0,255,255))

    if show_result: 
        ##################################
        # Show result
        ##################################
        # cv2.imshow("img_input", img)
        # new_img = draw_dl_mask(img, crop_corners, rv["img_mask"], color='b')
        
        cv2.imshow("result", new_img)
        cv2.waitKey(0)
        # plt.imshow(ropes)
        # plt.show()

    else:
        cv2.imwrite('winding/'+img_name[:-4]+'_output.jpg', new_img)