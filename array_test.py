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


source_path = r'C:\Users\Guillaume\Documents\Climbing-Hold-Recognition\Sample-Data'

transformation = [90, 1, 1]

test =  [[[187, 129, 63],
          [187, 129, 63]],

          [[187, 129, 63],
           [187, 129, 63],
           [187, 129, 63]]]

picture_files = [p for p in listdir(source_path) if isfile(join(source_path, p))]

# Get every hue variation of an image
for image_name in picture_files:
    original_image = cv2.imread(os.path.join(source_path, image_name))

    print(test)
    print(original_image)
    break

       


