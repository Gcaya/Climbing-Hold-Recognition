#%%
import os
import sys
import cv2
import uuid
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from os import listdir
from os.path import isfile, join


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

    mser = cv2.MSER_create(_min_area=250, _max_area=50000, _max_evolution=50000)
    regions , _ = mser.detectRegions(cv_image)

    for p in regions:
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)
        cv2.rectangle(cv_image, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)

    return cv_image

#%%
def main(args):

    source_path = args[1]
    dest_path = args[2]

    images = []
    lower_blue_color_bounds = [[100, 50, 10], [130, 50, 10]]
    upper_blue_color_bounds = [[255, 180, 100], [255, 180, 100]]
    holds_hsv_transformations = [('yellow', 90), ('green', 70), ('red', 60)]

    picture_files = [p for p in listdir(source_path) if isfile(join(source_path, p))]

    for image_name in picture_files:

        cv_image = cv2.imread(os.path.join(source_path, image_name))
        file_extension = os.path.splitext(image_name)[1]

        # Transform images so that holds can be detected with blue threshold
        for trans in holds_hsv_transformations:
            images.append(change_hsv(cv_image, trans[1]))

        for image in images:
            for x in range(0, 2):
                mser_image = mser_color(cv_image, lower_blue_color_bounds[x], upper_blue_color_bounds[x])
                cv2.imwrite(os.path.join(dest_path, str(uuid.uuid4()) + file_extension), mser_image)

if __name__=='__main__':
    main(sys.argv)



