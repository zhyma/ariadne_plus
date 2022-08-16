#! /usr/bin/python3

import numpy as np
import cv2, os
from PIL import Image
from cv_bridge import CvBridge
# ros
import rospy
import sensor_msgs.msg
import rospkg

from ariadne_plus.srv import getSplines, getSplinesRequest, getSplinesResponse

def display_img(img):
    cv2.imshow('image', img)
    show_window = True
    while show_window:
        k = cv2.waitKey(0) & 0xFF
        if k == 27:#ESC
            cv2.destroyAllWindows()
            show_window = False

def generateImage(img_np):
    img = Image.fromarray(img_np).convert("RGB") 
    msg = sensor_msgs.msg.Image()
    msg.header.stamp = rospy.Time.now()
    msg.height = img.height
    msg.width = img.width
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = 3 * img.width
    msg.data = np.array(img).tobytes()
    return msg

if __name__ == '__main__':
    
    bridge = CvBridge()
    rospy.init_node('test_ariadne_service')
    rospy.sleep(1)

    ##################################
    # Loading Input Image
    ##################################
    img = cv2.imread('test_images/simple_0.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640,480)) # resize necessary for the network model
    
    img_msg = generateImage(img)

    rospy.wait_for_service('get_splines')
    try:
        get_cable = rospy.ServiceProxy('get_splines', getSplines)
        req = getSplinesRequest()
        req.input_image = img_msg
        resp1 = get_cable(req)
        # print("get cable:")
        print(resp1.tck)
        cv_image = bridge.imgmsg_to_cv2(resp1.mask_image, desired_encoding='passthrough')
        display_img(cv_image)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
