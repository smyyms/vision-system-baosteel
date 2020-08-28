'''
Created on June 21, 2018
@author: W.Liu
'''

import cv2
import numpy as np
import os
import re

from scipy.signal import fftconvolve
from scipy.ndimage import filters

import skimage
import skimage.io as io
import skimage.transform
from skimage import data, util, measure, color, morphology, feature, segmentation, filters, draw
from skimage.measure import label

from PIL import Image
from itertools import groupby

_template_path =  '../sources/template/'
_template_path_raw = _template_path + '/raw/'
_template_path_rawMask = _template_path + '/raw_mask/'
_template_path_mask  = _template_path + 'mask/'


def matching(image, offset):
    maxima = []
    max_positions = []
    max_signal = 0
    r = 9
    targetID = -1
    circle_x = 0
    circle_y = 0

    count = 0
    for root, _, files in os.walk(_template_path_mask):
        for file in files:
            count = count + 1
    result = np.zeros((count, image.shape[0], image.shape[1]))

    iter = 0
    for root, _, files in os.walk(_template_path_mask):
        for file in files:
            maskModelName = file
            maskModelPath = os.path.join(root, maskModelName)
            maskModelID = re.sub('.jpg', '', re.sub('model_', '', maskModelName))

            modelImage = cv2.imread(maskModelPath, cv2.IMREAD_GRAYSCALE)
            result[iter, :, :] = fftconvolve(image, modelImage, 'same')

            max_positions.append(np.unravel_index(result[iter].argmax(), result[iter].shape))
            maxima.append(result[iter].max())
            signal = maxima[iter] / np.sqrt(float(r))

            if signal > max_signal:
                max_signal = signal
                (circle_y, circle_x) = max_positions[iter]
                targetID = maskModelID

            radius = r
            print("Maximum signal for maskModelID: %s, radius %d: %d %s, normal signal: %f" % (maskModelID, r, maxima[iter], max_positions[iter], signal))

            iter = iter + 1

    circle_y = circle_y + offset[0]
    circle_x = circle_x + offset[1]
    print("\n\n-----------------------RESULT----------------------")
    print("Final model mask ID: %s, the center: (%d, %d), the norlmal signal: %f" % (targetID, circle_x, circle_y, max_signal))
    return circle_x, circle_y

def temp_jud(image, signal_min = 2000):
    flag = False
    maxima = []
    max_positions = []
    max_signal = 0
    count = 0
    r = 10000
    for root, _, files in os.walk(_template_path_mask):
        for file in files:
            count = count + 1
    result = np.zeros((count, image.shape[0], image.shape[1]))

    iter = 0
    for root, _, files in os.walk(_template_path_mask):
        for file in files:
            temp_mask_name = file
            temp_mask_path = os.path.join(root, temp_mask_name)
            temp_mask_id= re.sub('.jpg', '', re.sub('model_', '', temp_mask_name))

            temp_image = cv2.imread(temp_mask_path, cv2.IMREAD_GRAYSCALE)
            result[iter, :, :] = fftconvolve(image, temp_image, 'same')

            max_positions.append(np.unravel_index(result[iter].argmax(), result[iter].shape))
            maxima.append(result[iter].max())
            signal = maxima[iter] / np.sqrt(float(r))

            if signal > signal_min:
                print("signal: ",signal)
                flag = True
                break
    return flag

def fake_reject():
    pass



