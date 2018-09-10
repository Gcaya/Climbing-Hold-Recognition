import os
import sys
import cv2
import numpy as np

#from PIL import Image


def mser(cv_image):
    vis = cv_image.copy()
    mser = cv2.MSER_create(_min_area= 2000)
    #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

    regions, _ = mser.detectRegions(cv_image)
    for p in regions:
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)
        cv2.rectangle(cv_image, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
    return cv_image


def color_based_mser(cv_image):

    cv_image = cv2.resize(cv_image, (int(cv_image.shape[1]/5), int(cv_image.shape[0]/5)))

    lower_color_bounds = np.array([100, 50, 11])
    upper_color_bounds = np.array([255,190,170])

    mask = cv2.inRange(cv_image,lower_color_bounds,upper_color_bounds)
    mask_rgb = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

    cv_image = cv_image & mask_rgb

    mser = cv2.MSER_create()
    regions , _ = mser.detectRegions(cv_image)

    for p in regions:
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)
        cv2.rectangle(cv_image, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)

    return mask
    

def main(argv):
    file_path = argv[1]
    save_path = argv[2]

    #cv2.imwrite('1_' + save_path, mser(cv2.imread(file_path)))
    cv2.imwrite('2_' + save_path, color_based_mser(cv2.imread(file_path)))


if __name__=='__main__':
    main(sys.argv)