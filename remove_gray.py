import os
import sys
import cv2
import numpy as np



def remove_gray(cv_image):

    cv_image = cv2.resize(cv_image, (int(cv_image.shape[1]/5), int(cv_image.shape[0]/5)))
    tol = 15

    for x in range(0, cv_image.shape[0] - 1):
        for y in range(0, cv_image.shape[1] - 1):
            pixel = cv_image[x, y]
            r = int(pixel[0])
            g = int(pixel[1])
            b = int(pixel[2])

            if abs(r - g) <= tol and abs(g - b) <= tol:
                cv_image[x, y] = [0, 0, 0]

    return cv_image

    lower_color_bounds = np.array([100, 50, 11])
    upper_color_bounds = np.array([255,190,170])

    mask = cv2.inRange(cv_image,lower_color_bounds,upper_color_bounds)
    mask_rgb = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

    cv_image = cv_image & mask_rgb

    mser = cv2.MSER_create(_min_area=5, _max_area=1000)
    regions , _ = mser.detectRegions(cv_image)

    for p in regions:
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)
        cv2.rectangle(cv_image, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)

    return cv_image
    

def main(argv):
    file_path = argv[1]
    save_path = argv[2]

    cv2.imwrite('2_' + save_path, remove_gray(cv2.imread(file_path)))


if __name__=='__main__':
    main(sys.argv)