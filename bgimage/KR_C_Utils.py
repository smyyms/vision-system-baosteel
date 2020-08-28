from __future__ import print_function
from __future__ import division
import time
import copy

import logging
import logging.config
import glob
import csv
import math
from math import atan2, cos, sin, sqrt, pi

import numpy as np
import pandas as pd

import cv2
import imghdr
import skimage
from skimage import io, filters, morphology, util, measure, color, segmentation
from skimage.measure import compare_ssim as ssim
import PIL
import imutils
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import optimize

import socket
import struct
import ctypes
import datetime
import threading

from pypylon import pylon


camera_ip_address = "192.168.0.122"

# 参数
KR_C_source_file = "./../sources/KR_C"
KR_C_temp_file = "./../sources/KR_C/temp"
KR_C_demo_filename = "test.bmp"



left_top = [500, 1000]
right_bottom = [950, 1600]


GAUSS_SIGMA = 0.0
GAUSS_KERNEL_SIZE = 3
AREA_MIN, AREA_MAX = 10000, 600000
SHIFT_X, SHIFT_Y = 100, 500
SHIFT_XX, SHIFT_YY = 0, 100
SHIFT_END = 15
MSE_THRES = 1500
SSIM_THRES = 0.55


# TCP/IP
HOST = socket.gethostname()
# PORT = 12356
# HOST = '192.168.0.1'
PORT = 2001
ADDR = (HOST, PORT)
BUFSIZE = 1024
SEND_SIZE = 1024
RECV_SIZE = 4
FLAG_CONNECT = True



class SendStruct(object):
    def __init__(self):
        self.heartbeat = 1 # 0xff: disconnect;   1: keep connection.
        self.camera_id = 2 # 1: acA5472-5gc(仓架前方相机);  2: acA2040-35gm(测温枪前方相机)
        self.program_id = 1
        self.status = 0 # 0 means no-ops, 1 means ready, 2 means busy, 3 means fault.
        # self.reserve = 0 # Reserved
        self.sendstatus = 1
        self.z = 0


class RecvStruct(object):
    def __init__(self):
        self.heartbeat = 1 # 0xff: disconnect;   1: keep connection.
        self.camera_id = 1 # 1: acA5472-5gc(仓架前方相机);  2: acA2040-35gm(测温枪前方相机)
        self.status = 0 # 0: no-ops; 1: operation;  2: RST
        self.reserve = 0 # Reserved

def check(data_buff):
    flag = True
    heartbeat_check = [0, 1]
    camera_id_check = [1, 2, 3]
    status_check = [0, 1, 2]
    reserve_check = [0]
    if len(data_buff) == RECV_SIZE:
        heartbeat = int(data_buff[0])
        if heartbeat not in heartbeat_check:
            #print('heartbeat不属于',camera_id_check)
            flag = False
        camera_id = int(data_buff[1])
        if camera_id != 2:
            #print('camera_id 不是 2')
            flag = False
        status = int(data_buff[2])
        if status not in status_check:
            #print('status不属于',status_check)
            flag = False
        reserve = int(data_buff[3])
        if reserve not in reserve_check:
            #print('reserve不是0')
            flag = False
        #print('接收数据为：',[heartbeat,camera_id,status,reserve])
    else:
        #print('接收数据长度应为',RECV_SIZE,'实际为：',len(data_buff))
        flag = False
    return flag


def read_image_from_file(filename):

    return skimage.io.imread(filename)


def read_image_from_camera(_camera_ip_address = camera_ip_address):
    image_res = None

    info = pylon.DeviceInfo()
    info.SetPropertyValue('IpAddress', _camera_ip_address)
    # connecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(info))
    # connecting to the first available camera
    # camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Grabbing Continue(video) with minimal delay
    # camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    numberOfImagesToGrab = 10
    camera.StartGrabbingMax(numberOfImagesToGrab)
    converter = pylon.ImageFormatConverter()

    # converting to OpenCV bgr format
    converter.OutputPixelFormat = pylon.PixelType_RGB8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            if image is not None:
                image_res = image.GetArray()
        grabResult.Release()

    # Releasing the resource
    camera.StopGrabbing()

    return image_res

def rgb2gray(image, method='cv'):
    if image.ndim > 2:
        if method == 'cv':
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif method == 'sk':
            return skimage.color.rgb2gray(image)
    else:
        return image
def template_match(image_roi, image_tmp):
    image_roi_gray = rgb2gray(image_roi)
    image_tmp_gray = rgb2gray(image_tmp)

    result = cv2.matchTemplate(image_roi_gray,image_tmp_gray, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    h, w = image_tmp_gray.shape
    top_left = (top_left[0], top_left[1])
    bottom_right = (top_left[0] + w, top_left[1] + h)
    candidate_box = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
    image_candidate = reduce_domain(image_roi, candidate_box)
    image_candidate_gray = rgb2gray(image_candidate)

    mse_value = mse(image_candidate_gray, image_tmp_gray)
    ssim_value = ssim(image_candidate_gray, image_tmp_gray)
    #print(mse_value, ssim_value)
    # if mse_value < MSE_THRES and ssim_value > SSIM_THRES:
    if mse_value < 10*MSE_THRES and ssim_value > 0.5*SSIM_THRES:
        # top_left_ori = (top_left[0] + roi_box[0], top_left[1] + roi_box[1])
        # bottom_right_ori = (top_left_ori[0] + w, top_left_ori[1] + h)
        # cv2.rectangle(image_display, top_left_ori, bottom_right_ori, (255, 0, 0), 10)
        # match_range_ori = (top_left_ori[0], top_left_ori[1], bottom_right_ori[0], bottom_right_ori[1])
        match_range_roi = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
        cv2.rectangle(image_roi, top_left, bottom_right, (255, 0, 0), 10)
        return match_range_roi
    else:
        return None


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def reduce_domain(image, bbox_roi):
    image_roi = image[bbox_roi[1]:bbox_roi[3], bbox_roi[0]:bbox_roi[2]]
    return image_roi



def KR_C_process(image_ori):
   
    res = [1,0]
    # image_ori = read_image_from_camera()
    #image_ori = read_image_from_file(KR_C_source_file+"/"+KR_C_demo_filename)
    image_tmp = read_image_from_file(KR_C_temp_file + "/" + "temp.bmp")
    image_display = copy.deepcopy(image_ori)
    match_range = template_match(image_ori, image_tmp)
    if match_range == None:
        res[0] = 0
    else:
        mid_point = ((match_range[0]+match_range[2])//2, match_range[3])
        z_value = mid_point[1]
        if z_value > left_top[1] and z_value < right_bottom[1]:
            res[1] = 1
        #print("Z value: ", z_value)
    cv2.putText(image_ori, "OK",
                    (left_top[0]-10, left_top[1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 6)  
    cv2.rectangle(image_ori, (left_top[0], left_top[1]),
                          (right_bottom[0], right_bottom[1]), (0, 0, 255), 5)
    cv2.imwrite("./../sources/KR_C_display.png", image_ori)
    # cv2.imshow("display",image_display)
    # cv2.waitKey()
    # skimage.io.imshow(image_display)
    # plt.show()
    return res

