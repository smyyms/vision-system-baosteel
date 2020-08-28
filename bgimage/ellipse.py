'''
Created on June 21, 2018
@author: W.Liu
'''

import numpy as np
from numpy.linalg import eig, inv
import skimage
from skimage import measure
import math

arc = 2
R = np.arange(0,arc*np.pi, 0.01)

def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a


def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])


def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2


def _reduce_domain(image, bbox_roi):
    image_roi = image[bbox_roi[0]:bbox_roi[2], bbox_roi[1]:bbox_roi[3]]
    shift = bbox_roi
    return image_roi


def select_circle(image, labels, area_min=100, dia_min=40, dia_max=72, eccentricity_max=0.5, circu_min=0.85):
    bboxs = []
    centroids = []
    radius = []
    circularity = []
    ellipse_contour = []
    height = image.shape[0]
    width = image.shape[1]

    for region in measure.regionprops(labels):
        if region.area < area_min:
            continue
        if region.euler_number != 1 and region.euler_number != 0:
            continue
        if region.equivalent_diameter > dia_max or region.equivalent_diameter < dia_min:
            continue
        if region.eccentricity > eccentricity_max:
            continue

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

        for len_contours in range(len(contours)):
            if len(contours[len_contours]) < 0.7 * math.pi * diameter or len(contours[len_contours]) > 2 * math.pi * diameter:
                continue
            contour = contours[len_contours]

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
    return bboxs[0], centroids[0], radius[0], circularity[0], ellipse_contour[0]

