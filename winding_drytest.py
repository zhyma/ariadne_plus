import cv2
import os

# ariadne
from scripts.ariadne import AriadnePlus

from winding.create_mask import hue_masked, apply_crop, draw_dl_mask
import numpy as np

from scripts.extra.hue_extract import mask_or

file_list = []
for filename in os.listdir('./winding'):
    if filename.endswith('png'):
        file_list.append(filename)

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--img_path', required=True, help='relative path of the input image')
# parser.add_argument('--num_segments', default=20, help='number superpixels segments')
# parser.add_argument('--show', action = "store_true", help='show result computation')
# args = parser.parse_args()


# num_segments = int(args.num_segments)
# <arg name="num_superpixels" default="30"/>
num_segments = 30
# show_result = bool(args.show)
show_result = False

main_folder = os.getcwd()

##################################
# Initializing class
##################################
ariadne = AriadnePlus(main_folder, num_segments)

##################################
# Loading Input Image
##################################
# img_path = os.path.join(main_folder, args.img_path)
# img_path = 'winding/image1.jpg'
# img_path = 'winding/08-31-15-38-15.png'
# img_path = 'winding/08-31-15-39-10.png'
# img_path = 'winding/08-31-15-39-42.png'
# img_path = 'winding/08-31-15-40-08.png'

for img_name in file_list:

    img_path = os.path.join(main_folder, 'winding/'+img_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # corners = np.array([[943, 378],[530,363],[533,304],[945,318]])
    corners = np.array([[923, 391],[508,370],[512,310],[927,331]])

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

    # cropped_img = cv2.resize(apply_crop(img, crop_corners), (640,480))
    cropped_img = cv2.resize(img[y1:y2,x1:x2], (640,480))

    mask_by_hue = hue_masked(img, corners)
    feature_mask = cv2.resize(mask_by_hue, (640,480))

    # img = cv2.resize(cropped_img, (640,480)) # resize necessary for the network model

    ##################################
    # Run pipeline
    ##################################


    # corners = np.array([[923, 391],[508,370],[512,310],[927,331]])
    rv = ariadne.runAriadne(cropped_img, feature_mask, debug=True)

    new_mask = mask_or(feature_mask, rv['img_mask'])


    # show_feature = draw_dl_mask(img, crop_corners, feature_mask, color='g')
    # show_mask = draw_dl_mask(img, crop_corners, rv["img_mask"], color='b')
    # hori = np.concatenate((show_feature, show_mask), axis=1)

    cv2.imwrite('winding/'+img_name[:-4]+'_output.jpg', draw_dl_mask(img, crop_corners, new_mask, color='b'))

    if show_result: 
        ##################################
        # Show result
        ##################################
        # cv2.imshow("img_input", img)
        # cv2.imshow("result", draw_dl_mask(img, crop_corners, rv["img_mask"], color='b'))
        # cv2.imshow("img_final", rv["img_final"])
        # cv2.imshow("hue_masked", hori)
        # cv2.imshow("img_mask", rv["img_mask"])
        cv2.imshow("img_mask", draw_dl_mask(img, crop_corners, new_mask, color='b'))
        cv2.waitKey(0)
