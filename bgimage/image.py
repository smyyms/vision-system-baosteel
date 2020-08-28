'''
Created on June 21, 2018
@author: W.Liu
'''

import sys
import os
import cv2
import copy
import numpy as np
import skimage
import skimage.io as io
from skimage import data, util, measure, color, morphology, feature, segmentation, filters, draw, exposure
import PIL
from PIL import Image, ImageEnhance

_DEBUG = True


def read_single(image_file):
    image = skimage.io.imread(image_file)
    tag = True
    if image is None:
        tag = False
    if _DEBUG and tag:
        file_path, file_name, ext = get_info(image_file)
        print("input image name: %s%s " % (file_name, ext))
        print("input image shape: ", image.shape)
    return image


def get_info(image_file):
    file_path, temp_filename = os.path.split(image_file)
    file_name, ext = os.path.splitext(temp_filename)
    return file_path, file_name, ext


def get_size(image):
    if 2 == image.ndim or 3 == image.ndim:
        height = image.shape[0]
        width = image.shape[1]
        if _DEBUG:
            print("input image dimension: %d" % image.ndim)
    else:
        print("error: unrecognized data! dimension not equals to 2 or 3.\n")
        sys.exit(1)
    size = (height, width)
    return size


def rgb2gray(image, method='cv'):
    if image.ndim > 2:
        if method == 'cv':
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif method == 'sk':
            return skimage.color.rgb2gray(image)


def format_convert(image, arrow):
    if arrow == 'cv2pil':
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif arrow == 'pil2cv':
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image


def gaussian_blur(image, side=3):
    return cv2.GaussianBlur(image, (side, side), 0)


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


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
    h, w= image.shape
    image_co = np.zeros([h, w], image.dtype)
    image = cv2.addWeighted(image, 1.2, image_co, 1-1.2, 5.0)
    if skimage.exposure.is_low_contrast(image):
        image = skimage.exposure.rescale_intensity(image)
    # image = skimage.exposure.equalize_hist(image)
    return image


def noise_filter(image, sigma):
    mc = False
    if image.ndim == 2:
        mc = False
    else:
        mc = True
    image = skimage.filters.gaussian(image, sigma, multichannel=mc, preserve_range=False)
    return image


def binary_otsu(image):
    thresh = skimage.filters.threshold_otsu(image)
    dst = (image <= thresh) * 1.0
    return thresh, dst


def binary_adaptive(image, block_size=33):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 0)


def binary_manual(image, threshold=35, target=255, mode='inv'):
    if mode == 'inv':
        thresh, binary_image = cv2.threshold(image, threshold, target, cv2.THRESH_BINARY_INV)
    else:
        thresh, binary_image = cv2.threshold(image, threshold, target, cv2.THRESH_BINARY)
    return thresh, binary_image


def threshold_image(image, block_size=(64, 64), step_size=(32, 32), min_range=32, min_value=64):
    binary = np.zeros_like(image)
    for row in range(0, image.shape[0], step_size[0]):
        for col in range(0, image.shape[1], step_size[1]):
            im = image[row:row + block_size[0], col:col + block_size[1]]
            if np.ptp(im) < min_range:
                if np.min(im) > min_value:
                    binary[row:row + block_size[0], col:col + block_size[1]] = 255
            else:
                _, bin = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                binary[row:row + block_size[0], col:col + block_size[1]] = bin
    return binary


def clear_border(image):
    if image.ndim == 2:
        return skimage.segmentation.clear_border(image)


def connection(image):
    cleared = image.copy()
    skimage.segmentation.clear_border(cleared)
    labels = skimage.measure.label(cleared, connectivity=2)
    borders = np.logical_xor(image, cleared)
    labels[borders] = -1
    labels_rgb = skimage.color.label2rgb(labels)
    return labels, labels_rgb


# def edge_extract(image, threshold):
#     edges = skimage.filters.sobel(image, 0) ** 2 + skimage.filters.sobel(image, 1) ** 2
#     edges -= edges.min()
#     edges = edges > edges.max() * threshold
#     edges.dtype = np.int8
#
#     edge_list = np.array(edges.nonzero())
#     density = float(edge_list[0].size) / edges.size
#     # if DEBUG:
#     #     print("Signal density:", density)
#     if density > 0.25:
#         pass
#     return edges, density


def edge_extract(image, method="canny", param=2):
    if method == "canny":
        return skimage.feature.canny(image, sigma=param)
    else:
        pass


def reduce_domain(image, bbox_roi):
    image_roi = image[bbox_roi[0]:bbox_roi[2], bbox_roi[1]:bbox_roi[3]]
    shift = bbox_roi
    return image_roi


def bbox_shift(size, bbox, shift_x=5, shift_y=5):
    height = size[0]
    width = size[1]
    bbox_shift = copy.deepcopy(bbox)
    if (bbox[0] - shift_y) < 1:
        bbox_shift[0] = 1
    else:
        bbox_shift[0] = bbox[0] - shift_y
    if (bbox[1] - shift_x) < 1:
        bbox_shift[1] = 1
    else:
        bbox_shift[1] = bbox[1] - shift_x
    if (bbox[2] + shift_y) > height-1:
        bbox_shift[2] = height - 1
    else:
        bbox_shift[2] = bbox[2] + shift_y
    if (bbox[3] + shift_x) > width-1:
        bbox_shift[3] = width-1
    else:
        bbox_shift[3] = bbox[3] + shift_x
    return bbox_shift


def dilation(image, oper='disk', size=3):
    if oper == 'square':
        image = skimage.morphology.dilation(image, morphology.square(size))
    elif oper == 'disk':
        image = skimage.morphology.dilation(image, morphology.disk(size))
    return image


def ROI_fusion(image, bboxs, shift_x=100, shift_y=100):
    height = image.shape[0]
    width = image.shape[1]
    min_y = height-1
    min_x = width-1
    max_y = 1
    max_x = 1
    for i in range(len(bboxs)):
        if bboxs[i][0] < min_y:
            min_y = bboxs[i][0]
        if bboxs[i][1] < min_x:
            min_x = bboxs[i][1]
        if bboxs[i][2] > max_y:
            max_y = bboxs[i][2]
        if bboxs[i][3] > max_x:
            max_x = bboxs[i][3]
    if (min_y - shift_y) < 1:
        min_y = 1
    else:
        min_y = min_y - shift_y
    if (min_x - shift_x) < 1:
        min_x = 1
    else:
        min_x = min_x - shift_x
    if (max_y + shift_y) > height-1:
        max_y = height - 1
    else:
        max_y = max_y + shift_y
    if (max_x + shift_x) > width-1:
        max_x = width-1
    else:
        max_x = max_x + shift_x
    roi = [min_y, min_x, max_y, max_x]
    offset = [min_y, min_x]
    return roi, offset

