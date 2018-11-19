#%%
import os
import sys
import cv2
import uuid
import math
import numpy as np
from Region import Region
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
def mser_extract_regions(cv_image, lower_color_bound, upper_color_bound):

    image = cv_image.copy()
    lower_bound = np.array(lower_color_bound)
    upper_bound = np.array(upper_color_bound)

    mask = cv2.inRange(image, lower_bound, upper_bound)
    mask_rgb = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

    masked_image = image & mask_rgb

    mser = cv2.MSER_create(_min_area=250, _max_area=50000, _max_evolution=50000)
    regions , _ = mser.detectRegions(masked_image)

    return regions

#%%
def combine_regions(image, detected_regions):

    regions = []
    final_regions = []

    for p in detected_regions:
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)

        regions.append([Region(xmax, xmin, ymax, ymin), 0])

    while True:
        region = None

        # Find a region that hasn't been used yet
        for x in range(0, len(regions)):
            if(regions[x][1] == 0 and region is None):
                region = regions[x][0]
                regions[x][1] = 1 
                break

        # If we used all the regions 
        if region is None:
            break

        # Try and combine the region with others
        for x in range(0, len(regions)):

            if(regions[x][1] == 0):
                potential_region = regions[x][0]

                if(region.try_combine(potential_region)):
                    regions[x][1] = 1

        final_regions.append(region)

    return final_regions

#%%
def crop_regions_from_image(original_image, regions_to_crop:[Region]):

    image = original_image.copy()

    for p in regions_to_crop:
        x_center, y_center = p.get_center_point()
        radius = p.get_radius()

        image = cv2.circle(image, (x_center, y_center), int(radius), (0, 255, 0), 1)

    return image            


#%%
def main(args):

    source_path = args[1]
    dest_path = args[2]

    hsv_modified_images = []
    lower_blue_color_bounds = [130, 50, 10]
    upper_blue_color_bounds = [255, 180, 100]
    holds_hsv_transformations = [('blue', 0), ('yellow', 90), ('green', 70), ('red', 115)]

    picture_files = [p for p in listdir(source_path) if isfile(join(source_path, p))]

    # Get every hue variation of an image
    for image_name in picture_files:
        original_image = cv2.imread(os.path.join(source_path, image_name))
        file_extension = os.path.splitext(image_name)[1]

        # Transform original image's hue so that holds can be detected with blue threshold
        for trans in holds_hsv_transformations:
            hsv_modified_image = change_hsv(original_image, trans[1])
            hsv_modified_images.append((hsv_modified_image, file_extension))

    # Extract holds on every image
    for image_infos in hsv_modified_images:
        image = image_infos[0]
        detected_regions = mser_extract_regions(image, lower_blue_color_bounds, upper_blue_color_bounds)
        final_regions = combine_regions(image, detected_regions)
        result_image = crop_regions_from_image(image, final_regions)
        cv2.imwrite(os.path.join(dest_path, str(uuid.uuid4()) + image_infos[1]), result_image)

if __name__=='__main__':
    main(sys.argv)

