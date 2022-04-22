#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:47:29 2022

@author: jekim
"""

import os 
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import cv2
from PIL import Image

path_img = '/home/jekim/workspace/calib_extri/cal_data_sample/extri_data/images_ori/3'
save_img = './cal_data_sample/extri_data/images_resize/3' # use relative path
list_img = os.listdir(path_img)

for ind in os.listdir(path_img):
    img = cv2.imread(os.path.join(path_img,ind))
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(1998, 1125), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(save_img,ind),img)