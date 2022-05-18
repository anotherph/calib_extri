#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:53:52 2022

@author: jekim

use fuction of goodFeaturesToTrack

"""

import cv2
import matplotlib.pyplot as plt

src = cv2.imread("/home/jekim/workspace/calib_extri/cal_data_0510/extri_data/images/4/000000.jpg")
dst = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 5, blockSize=3, useHarrisDetector=True, k=0.03)
a=corners.squeeze()

img = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
 
plt.scatter(a[:,0],a[:,1],s=5,c='r')
plt.imshow(img)
plt.show

# for i in corners:
#     cv2.circle(dst, tuple(i[0]), 3, (0, 0, 255), 2)

# cv2.imshow("dst", dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

 # plt.scatter(corners[:,0],corners[:,1])