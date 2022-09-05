import cv2
import os

from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

# ariadne
from scripts.ariadne import AriadnePlus

from winding.create_mask import apply_crop, draw_dl_mask
import numpy as np

from winding.rope_pre_process import get_subimg
from winding.rope_post_process import get_ropes


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


    ##################################
    # Run pipeline
    ##################################

    ## extract 4:3 image that includes ropes.
    crop_corners, cropped_img, feature_mask = get_subimg(img, corners)

    rv = ariadne.runAriadne(cropped_img, debug=True)

    # new_img = draw_dl_mask(img, crop_corners, feature_mask, color='b')

    full_mask = get_ropes(img.shape, crop_corners, rv["img_mask"], feature_mask)

    # ##################################
    # # Check the result
    # ################################## 

    kernel = np.ones((5, 5), np.uint8)
    mask_dilate = cv2.dilate(full_mask, kernel, iterations=1)
    new_img = draw_dl_mask(img, np.array([[0, 0],[1279,719]]), mask_dilate, color='b')
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