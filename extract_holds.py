#%%
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image


#%%
def remove_gray(cv_image):

    cv_image = cv2.resize(cv_image, (int(cv_image.shape[1]/5), int(cv_image.shape[0]/5)))
    tolerance = 12

    for x in range(0, cv_image.shape[0] - 1):
        for y in range(0, cv_image.shape[1] - 1):
            pixel = cv_image[x, y]
            r = int(pixel[0])
            g = int(pixel[1])
            b = int(pixel[2])

            if abs(r - g) <= tolerance and abs(g - b) <= tolerance:
                cv_image[x, y] = [0, 0, 0]

    return cv_image

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
cv_image = cv2.imread(r'C:\Users\Guillaume\Documents\Climbing-Hold-Recognition\Sample-Data\\1.png')

#lower_color_bounds = (, [164, 131, 92], [160, 108, 71], [159, 113, 65], [138, 122, 110])
#upper_color_bounds = ([150, 138, 128], [144, 93,  43], [127, 106, 98], [110, 84,  70], )

lower_blue_color_bounds = ([100, 50, 10], [130, 50, 10])
upper_blue_color_bounds = ([255, 180, 100], [255, 180, 100])

lower_yellow_color_bounds = ([2, 140, 180], [2, 130, 160], [2, 90, 140], [2, 90, 140])
upper_yellow_color_bounds = ([118, 172, 202], [118, 172, 202], [110, 180, 210], [150, 190, 210])

for x in range(0, 4):
    mser_image = mser_color(cv_image, lower_yellow_color_bounds[x], upper_yellow_color_bounds[x])
    cv2.imwrite(r'C:\Users\Guillaume\Documents\Climbing-Hold-Recognition\Sample-Result\result' + str(x) + r'.png',mser_image)



#plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
#plt.imshow(mser_image)
