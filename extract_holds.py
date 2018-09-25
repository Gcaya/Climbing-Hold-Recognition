#%%
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image


#%%
def change_hsv(cv_image, hue_rotation):

    image = cv_image.copy()
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for x in range(0, hsv_image.shape[0]):
        for y in range(0, hsv_image.shape[1]):
            pixel = hsv_image[x, y]
            h = int(pixel[0])
            s = int(pixel[1])
            v = int(pixel[2])

            hsv_image[x, y] = [int(h + hue_rotation) % 180, s, v]

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

#%%  
def mser_color(cv_image, lower_color_bound, upper_color_bound):

    lower_bound = np.array(lower_color_bound)
    upper_bound = np.array(upper_color_bound)

    mask = cv2.inRange(cv_image, lower_bound, upper_bound)
    mask_rgb = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

    cv_image = cv_image & mask_rgb

    mser = cv2.MSER_create(_min_area=250, _max_area=2000, _max_evolution=50000)
    regions , _ = mser.detectRegions(cv_image)

    for p in regions:
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)
        cv2.rectangle(cv_image, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)

    return cv_image

#%%
cv_image = cv2.imread(r'D:\Climbing-Hold-Recognition\Sample-Data\\3.png')

image = change_hue(cv_image, 90)
cv2.imwrite(r'D:\Climbing-Hold-Recognition\Sample-Result\result.png', image)

lower_blue_color_bounds = ([100, 50, 10], [130, 50, 10])
upper_blue_color_bounds = ([255, 180, 100], [255, 180, 100])

lower_yellow_color_bounds = ([2, 140, 180], [2, 130, 160], [2, 90, 140], [2, 90, 140])
upper_yellow_color_bounds = ([118, 172, 202], [118, 172, 202], [110, 180, 210], [150, 190, 210])

for x in range(0, 4):
    mser_image = mser_color(cv_image, lower_yellow_color_bounds[x], upper_yellow_color_bounds[x])
    cv2.imwrite(r'D:\Climbing-Hold-Recognition\Sample-Result\result' + str(x) + r'.png',mser_image)


