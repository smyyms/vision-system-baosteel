# 导入库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import sys
import argparse
import os
import time
import copy
import math

import logging
import logging.config
import glob
import csv

import numpy as np
from numpy.linalg import eig, inv
import pandas as pd

import cv2
import imghdr
import skimage
from skimage import io, filters, morphology, util, measure, color, segmentation
from skimage.measure import compare_ssim as ssim
import PIL
from PIL import Image, ImageEnhance
import imutils
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import optimize
import bgimage
import bgimage.hf as hf

import socket
import struct
import ctypes
import datetime
import threading



# 参数
KR_A_source_file = "./../sources/KR_A/"
KR_A_temp_file = "./../sources/KR_A/temp"
KR_A_demo_filename = "rack_front_0.bmp"

from pypylon import pylon

GAUSS_SIGMA = 0.0
GAMMA = 1.05  # 1.05
GAMMA_TMP = 0.6  # 0.6
GAUSS_KERNEL_SIZE = 3
PROBE_NUM = 15
# MSE_THRES = 1000
# SSIM_THRES = 0.55
MSE_THRES = 1500
SSIM_THRES = 0.55



camera_ip_address = "192.168.0.121"



LAYER_X, LAYER_Y = [461, 725, 1013], [320, 1740, 3110]
L_X_WID, L_Y_WID = 800, 400

C_POINT_X = 2752

Y_COORD = [-220, 0, 440]
PPM = 0.2429

target_centroid = []

ORDER = [0, 1, 5, 2, 6, 10, 3, 7, 11, 4, 8, 12, 9, 13, 14]

# TCP/IP
HOST = socket.gethostname()
# PORT = 12356
# HOST = '192.168.0.1'
PORT = 2000
ADDR = (HOST, PORT)
BUFSIZE = 1024
SEND_SIZE = 1024
RECV_SIZE = 4
FLAG_CONNECT = True


class SendStruct(object):
    def __init__(self):
        self.heartbeat = 1 # 0xff: disconnect;   1: keep connection.
        self.camera_id = 1 # 1: acA5472-5gc(仓架前方相机);  2: acA2040-35gm(测温枪前方相机)
        self.program_id = 1
        self.status = 0 # 0 means no-ops, 1 means ready, 2 means busy, 3 means fault.
        # self.reserve = 0 # Reserved
        self.data = [[i + 1, -1, -1, -1] for i in range(PROBE_NUM)]
        self.target = [-1, -1, 0]


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
    string = ""
    if len(data_buff) == RECV_SIZE:
        heartbeat = int(data_buff[0])
        if heartbeat not in heartbeat_check:
            string += 'heartbeat不属于'+str(camera_id_check) + ";"
            #print('heartbeat不属于',camera_id_check)
            flag = False
        camera_id = int(data_buff[1])
        if camera_id != 1:
            string += 'camera_id 不是 1' + ";"
            #print('camera_id 不是 1')
            flag = False
        status = int(data_buff[2])
        if status not in status_check:
            string += 'status不属于' + str(status_check) + ";"
            print('status不属于',status_check)
            flag = False
        reserve = int(data_buff[3])
        if reserve not in reserve_check:
            string += 'reserve不是0' + ";"
            #print('reserve不是0')
            flag = False
        #print('接收数据为：',[heartbeat,camera_id,status,reserve])
    else:
        string += '接收数据长度应为' + str(RECV_SIZE) + '实际为：' + str(len(data_buff))
        #print('接收数据长度应为',RECV_SIZE,'实际为：',len(data_buff))
        flag = False
    return string, flag


def read_image_from_camera(camera_ip_address):

    image_res = None

    info = pylon.DeviceInfo()
    info.SetPropertyValue('IpAddress', camera_ip_address)
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


def read_image_from_file(filename):

    return skimage.io.imread(filename)


def gaussian_blur(image, sigma=0.0, kernel_size=3):
    return skimage.filters.gaussian(image, sigma=sigma, truncate=1 / kernel_size)


def binary(image):
    thresh = skimage.filters.threshold_otsu(image)
    dst = (image >= 0.7 * thresh) * 1.0
    dst = skimage.morphology.closing(dst, skimage.morphology.disk(10))
    dst = skimage.util.invert(dst)

    return dst


def fill_hole(image):
    im_th = copy.deepcopy(image)
    th, im_th = cv2.threshold(im_th, 0.5, 255, cv2.THRESH_BINARY)
    im_th = im_th.astype(np.uint8)
    im_flood_fill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    fill_x, fill_y = 100, 100
    cv2.floodFill(im_flood_fill, mask, (fill_x, fill_y), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    image_res = im_th | im_flood_fill_inv

    return image_res


def connection(image):
    cleared = image.copy()
    skimage.segmentation.clear_border(cleared)
    labels = skimage.measure.label(cleared, connectivity=2)
    borders = np.logical_xor(image, cleared)
    labels[borders] = -1
    labels_rgb = skimage.color.label2rgb(labels)

    return labels, labels_rgb


def reduce_domain(image, bbox_roi):
    image_roi = image[bbox_roi[1]:bbox_roi[3], bbox_roi[0]:bbox_roi[2]]
    return image_roi


def format_convert(image, arrow):
    if arrow == 'cv2pil':
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif arrow == 'pil2cv':
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image


def image_enhance(image):
    flag = False
    if type(image) == np.ndarray:
        flag = True
        image = format_convert(image, 'cv2pil')

    enh_bri = ImageEnhance.Brightness(image)
    enh_brightness = 1.5
    image_brightened = enh_bri.enhance(enh_brightness)

    enh_col = ImageEnhance.Color(image)
    enh_color = 1.5
    image_colored = enh_col.enhance(enh_color)

    enh_con = ImageEnhance.Contrast(image)
    enh_contrast = 2.0
    image_contrasted = enh_con.enhance(enh_contrast)

    image = image_contrasted
    enh_sha = ImageEnhance.Sharpness(image)
    enh_sharpness = 2.0
    image_sharped = enh_sha.enhance(enh_sharpness)

    if flag:
        image_sharped = format_convert(image_sharped, 'pil2cv')
    return image_sharped


def image_adjust(image, gamma):
    image = skimage.exposure.adjust_gamma(image, gamma)
    h, w = image.shape
    image_co = np.zeros([h, w], image.dtype)
    image = cv2.addWeighted(image, 1.2, image_co, 1 - 1.2, 5.0)
    if skimage.exposure.is_low_contrast(image):
        image = skimage.exposure.rescale_intensity(image)
    # image = skimage.exposure.equalize_hist(image)
    return image


def rgb2gray(image, method='cv'):
    if image.ndim > 2:
        if method == 'cv':
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif method == 'sk':
            return skimage.color.rgb2gray(image)
    else:
        return image


def _reduce_domain(image, bbox_roi):
    image_roi = image[bbox_roi[0]:bbox_roi[2], bbox_roi[1]:bbox_roi[3]]
    shift = bbox_roi
    return image_roi


def select_circle(image, labels, area_min, area_max, dia_min, dia_max, eccentricity_max=0.5, circu_min=0.85):
    bboxs = []
    centroids = []
    radius = []
    circularity = []
    ellipse_contour = []
    height = image.shape[0]
    width = image.shape[1]
    c_flag = False

    for region in measure.regionprops(labels):  # , coordinates='xy'):
        if region.area < area_min or region.area > area_max:
            continue
        if region.euler_number != 1 and region.euler_number != 0:
            continue
        if region.equivalent_diameter > dia_max or region.equivalent_diameter < dia_min:
            continue
        if region.eccentricity > eccentricity_max:
            continue
        # print(region.area)

        F = region.area
        local_centroid = region.local_centroid
        centroid = region.centroid
        bbox = region.bbox
        diameter = region.equivalent_diameter
        offset_up = offset_down = offset_left = offset_right = 0
        if bbox[0] > 0:
            offset_up = -1
        if bbox[1] > 0:
            offset_left = -1
        if bbox[2] < height:
            offset_down = 1
        if bbox[3] < width:
            offset_right = 1
        # offset = (offset_up, offset_left, offset_down, offset_right)
        bbox_add_shift = [bbox[0] + offset_up, bbox[1] + offset_left, bbox[2] + offset_down, bbox[3] + offset_right]
        roi = _reduce_domain(image, bbox_add_shift)
        min_r = max(roi.shape[0], roi.shape[1])
        max_r = 0
        contours = measure.find_contours(roi, 0.01, fully_connected='low', positive_orientation='low')

        contour = None
        for len_contours in range(len(contours)):
            if len(contours[len_contours]) < 0.7 * math.pi * diameter or len(
                    contours[len_contours]) > 2 * math.pi * diameter:
                continue
            contour = contours[len_contours]
        if contour is None:
            # c_flag = False
            continue
            # return c_flag, bboxs, centroids, radius, circularity, ellipse_contour

        lc = (local_centroid[0] + abs(offset_up), local_centroid[1] + abs(offset_left))
        for i in range(contour.shape[0]):
            r = math.sqrt(np.square(lc[0] - contour[i][0]) + np.square(lc[1] - contour[i][1]))
            if r > max_r:
                max_r = r
            if r < min_r:
                min_r = r
        c_ = F / (np.square(max_r) * math.pi)
        if c_ > 1:
            raise (Exception, "Invalid level!")
        c = min(c_, 1)
        if c < circu_min:
            continue
        bboxs.append(bbox)
        centroids.append(centroid)
        radius.append(diameter / 2)
        circularity.append(c)
        ellipse_contour.append(contour)
    if len(bboxs) <= 0:
        c_flag = False
        return c_flag, bboxs, centroids, radius, circularity, ellipse_contour
    else:
        c_flag = True
        return c_flag, bboxs[0], centroids[0], radius[0], circularity[0], ellipse_contour[0]


def fitEllipse(x, y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2;
    C[1, 1] = -1
    E, V = eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    return a


def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])


def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return 0.5 * np.arctan(2 * b / (a - c))


def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])


def ellipse_angle_of_rotation2(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi / 2
    else:
        if a > c:
            return np.arctan(2 * b / (a - c)) / 2
        else:
            return np.pi / 2 + np.arctan(2 * b / (a - c)) / 2


def pre_process(image):
    image_enh = image_enhance(image)
    image_gray = image_adjust(rgb2gray(image_enh), GAMMA)
    image_blur = gaussian_blur(image_gray, GAUSS_SIGMA, GAUSS_KERNEL_SIZE)
    image_dst = binary(image_blur)
    image_dst_labels, image_dst_labels_rgb = connection(image_dst)
    # skimage.io.imshow(image_dst_labels_rgb)
    # plt.show()

    return image_dst, image_dst_labels


def pre_process_tmp(image):
    image_enh = image_enhance(image)
    image_gray = image_adjust(rgb2gray(image_enh), GAMMA_TMP)
    image_blur = gaussian_blur(image_gray, GAUSS_SIGMA, GAUSS_KERNEL_SIZE)
    image_dst = binary(image_blur)
    image_dst_labels, image_dst_labels_rgb = connection(image_dst)
    # skimage.io.imshow(image_dst_labels_rgb)
    # plt.show()

    return image_dst, image_dst_labels



def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def template_match(image_roi, image_tmp, image_display, roi_box, index):
    # image_roi_display = copy.deepcopy(image_roi)
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
    # print(mse_value, ssim_value)
    # if mse_value < MSE_THRES and ssim_value > SSIM_THRES:
    if mse_value < 10*MSE_THRES and ssim_value > 0.5*SSIM_THRES:
        top_left_ori = (top_left[0] + roi_box[0], top_left[1] + roi_box[1])
        bottom_right_ori = (top_left_ori[0] + w, top_left_ori[1] + h)
        cv2.rectangle(image_display, top_left_ori, bottom_right_ori, (255, 0, 0), 10)
        match_range_ori = (top_left_ori[0], top_left_ori[1], bottom_right_ori[0], bottom_right_ori[1])
        match_range_roi = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
        return match_range_ori, match_range_roi
    else:
        return None, None


def HF(image, rh=1, rl=1):
    # image = image / 255
    # image = rgb2gray(image)
    rows, cols = image.shape
    d0 = 9
    c = 3
    # rh, rl, cutoff = 1.1, 0.9, 32
    # y_log = np.log(image + 0.01)
    y_log = np.log(image + 0.1)
    y_fft = np.fft.fft2(y_log)
    # y_fft_shift = np.fft.fftshift(y_fft)
    # DX = cols / cutoff
    G = np.ones((rows, cols))
    for i in range(rows):
        for j in range(cols):
            # G[i][j] = ((rh - rl) * (1 - np.exp(-((i - rows / 2) ** 2 + (j - cols / 2) ** 2) / (2 * DX ** 2)))) + rl
            G[i][j] = ((rh - rl) * (np.exp(c * (-((i - rows / 2) ** 2 + (j - cols / 2) ** 2)) / (d0 ** 2)))) + rl
    result_filter = G * y_fft
    # result_interm = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))
    # result = np.exp(result_interm)
    result = np.real(np.exp(np.fft.ifft2(result_filter)))

    a = result
    a_min, a_max = a.min(), a.max()
    for i in range(rows):
        for j in range(cols):
            result[i][j] = (result[i][j] - a_min) / (a_max - a_min) * 255
    result = np.uint8(result)
    return result


def calibrate_image2robot(C_POINT_X, PPM, Y_COORD, coord_image):
    if coord_image is None or len(coord_image) != 15:
        return None
    coord_robot = []
    for i in range(len(coord_image)):
        x, y = coord_image[i]
        if x == -1 and y == -1:
            x_r, y_r = x, y
            coord_robot.append([x_r, y_r])
            continue
        else:
            x_r = int(PPM * (x - C_POINT_X))
            if i <= 4:
                y = Y_COORD[0]
            elif i > 4 and i <= 9:
                y = Y_COORD[1]
            else:
                y = Y_COORD[2]
            y_r = y
            coord_robot.append([x_r, y_r])
    return coord_robot

def process(image_dst, image_dst_labels, image_roi, image_display, roi_box, match_box, index):
    AREA_MIN = 6000
    AREA_MAX = 13000
    DIA_MIN = 80
    DIA_MAX = 170
    ECC_MAX = 0.8
    CIR_MIN = 0.5
    flag = False
    c_flag, tar_bbox, tar_centroids, tar_radius, tar_circularity, tar_contour = \
        select_circle(image_dst, image_dst_labels, area_min=AREA_MIN,
                              area_max=AREA_MAX, dia_min=DIA_MIN, dia_max=DIA_MAX,
                              eccentricity_max=ECC_MAX, circu_min=CIR_MIN)
    if not c_flag:
        cv2.putText(image_display, "Failed",
                    (roi_box[0]+30, roi_box[1]+80), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 0), 5)
        return None
    else:
        cv2.putText(image_display, "OK",
                    (roi_box[0] + 30, roi_box[1] + 80), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 255, 0), 6)
        flag = True

    image_dst_inv = util.invert(image_dst)
    contour_cv = np.array(tar_contour, dtype=int)
    box = cv2.minAreaRect(contour_cv)

    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    box[:, 0] = box[:, 0] + tar_bbox[1]
    box[:, 1] = box[:, 1] + tar_bbox[0]
    # cv2.drawContours(image_roi, [box.astype("int")], -1, (0, 255, 0), 2)
    # for (x, y) in box:
    #     cv2.circle(image_roi, (int(x), int(y)), 2, (0, 0, 255), -1)
    cv2.rectangle(image_roi, (int(tar_bbox[1]-2), int(tar_bbox[0]-2)),
                  (int(tar_bbox[3]+2), int(tar_bbox[2]+2)), (255, 0, 0), 5)
    cv2.rectangle(image_display, (int(tar_bbox[1]-8+match_box[0]), int(tar_bbox[0]-8+match_box[1])),
                  (int(tar_bbox[3]+8+match_box[0]), int(tar_bbox[2]+8+match_box[1])), (0, 255, 0), 12)

    arc = 2
    R = np.arange(0, arc * np.pi, 0.01)
    x = tar_contour[:, 0]
    y = tar_contour[:, 1]
    a = fitEllipse(x, y)
    center = ellipse_center(a)
    center_i = [center[0] + tar_bbox[0], center[1] + tar_bbox[1]]
    center_ii = [center_i[0] + match_box[1], center_i[1] + match_box[0]]
    cv2.circle(image_display, (int(center_ii[1]), int(center_ii[0])), 1, (255, 0, 0), 10)
    cv2.line(image_display, (int(center_ii[1])-30, int(center_ii[0])), (int(center_ii[1])+30, int(center_ii[0])), (255, 0, 0), 8)
    cv2.line(image_display, (int(center_ii[1]), int(center_ii[0])-30), (int(center_ii[1]), int(center_ii[0])+30), (255, 0, 0), 8)
    # print("centroid of target %d" % (index + 1), ": ",(int(center_ii[1]), int(center_ii[0])))
    phi = ellipse_angle_of_rotation2(a)
    axes = ellipse_axis_length(a)

    a_axes, b_axes = axes
    xx = center[0] + a_axes * np.cos(R) * np.cos(phi) - b_axes * np.sin(R) * np.sin(phi)
    yy = center[1] + a_axes * np.cos(R) * np.sin(phi) + b_axes * np.sin(R) * np.cos(phi)
    cv2.ellipse(image_roi, (int(center_i[1]), int(center_i[0])), (int(a_axes), int(b_axes)), -phi, 0, 360,
                (0, 255, 0), 5)
    # cv2.ellipse(image_display, (int(center_ii[1]), int(center_ii[0])), (int(a_axes), int(b_axes)), -phi, 0, 360,
    #             (0, 255, 0), 10)
    # center_ii_text = [float('%.4f' % center_ii[0]), float('%.4f' % center_ii[1])]
    # axes_text = [float('%.4f' % axes[0]), float('%.4f' % axes[1])]
    # text_1 = 'slot ' + str(i) + ' : ' + str(flag)
    # cv2.putText(image, text_1, (1400, 60 + int(i * d_text_height)),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),
    #             1)
    # if flag:
    #     text_2 = 'center: ' + str(center_ii_text)
    #     cv2.putText(image, text_2, (1400, 60 + int((i + 1 / 4) * d_text_height)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),
    #                 1)
    #     text_3 = 'd1= ' + str(axes_text[0]) + ', ' + 'd2= ' + str(axes_text[1])
    #     cv2.putText(image, text_3, (1400, 60 + int((i + 2 / 4) * d_text_height)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),
    #                 1)

    # skimage.io.imshow(image_roi)
    # plt.show()

    return [int(center_ii[1]), int(center_ii[0])]


def KR_A_process(image_ori):

    """
    :param image_ori:
    :return: [[bool,x,y]]*PROBE_NUM
    """
    result = [[i+1, 0, -1, -1] for i in range(PROBE_NUM)]
    # t0 = time.clock()
    #image_ori = read_image_from_camera()
    #image_ori = read_image_from_file(KR_A_source_file + "/" + KR_A_demo_filename)
    #image_display = copy.deepcopy(image_ori)

    ROI_PROBE = []
    for j in range(3):
        for i in range(5):
            tmp = (LAYER_X[j] + i*L_X_WID, LAYER_Y[j], LAYER_X[j] + (i+1)*L_X_WID, LAYER_Y[j] + L_Y_WID)
            cv2.rectangle(image_ori, (tmp[0], tmp[1]),
                          (tmp[2], tmp[3]), (0, 0, 255), 5)
            ROI_PROBE.append(tmp)
    coord_image = []
    for i in range(PROBE_NUM):
        coord_image_tmp = [-1, -1]
        image_roi = reduce_domain(image_ori, ROI_PROBE[i])
        image_roi = rgb2gray(image_roi)
        image_HF = HF(image_roi, 2, 0.8)
        image_roi = cv2.cvtColor(image_HF, cv2.COLOR_GRAY2BGR)
        image_tmp = read_image_from_file(KR_A_temp_file + "/" + "temp_" + str(i) + ".bmp")
        match_box_ori, match_box_roi = template_match(image_roi, image_tmp, image_ori, ROI_PROBE[i], i)
        if match_box_roi is not None:
            # print("Index: ", i);
            image_roi_new = reduce_domain(image_roi, match_box_roi)
            image_dst, image_dst_labels = pre_process(image_roi_new)
            coord_image_tmp_pro = process(image_dst, image_dst_labels, image_roi_new, image_ori, ROI_PROBE[i], match_box_ori, i)
            if coord_image_tmp_pro is None:
                coord_image.append(coord_image_tmp)
                # result[i][1] = 0
            else:
                coord_image_tmp = coord_image_tmp_pro
                coord_image.append(coord_image_tmp)
                # result[i][1] = 1
        else:
            cv2.putText(image_ori, "Failed",
                        (ROI_PROBE[i][0] + 30, ROI_PROBE[i][1] + 80), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 0, 0), 5)
            coord_image.append(coord_image_tmp)
            # result[i][1] = 0
        # skimage.io.imshow(image_roi)
        # plt.show()
    cv2.imwrite("./../sources/KR_A_display.png",image_ori)
    #skimage.io.imshow(image_display)
    #plt.show()

    # print(time.clock() - t0)
    # image_display = cv2.cvtColor(image_display, cv2.COLOR_RGB2BGR)
    # cv2.namedWindow('display_image', cv2.WINDOW_FREERATIO)
    # cv2.imshow('display_image', image_display)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    coord_robot = calibrate_image2robot(C_POINT_X, PPM, Y_COORD, coord_image)
    # print(coord_image)
    if coord_robot is not None:
        for i in range(PROBE_NUM):
            if coord_robot[i][0] != -1 and coord_robot[i][1] != -1:
                result[i][1] = 1
                result[i][2], result[i][3] = coord_robot[i][0], coord_robot[i][1]
            else:
                result[i][1] = 0
    return result
