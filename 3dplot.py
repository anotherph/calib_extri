#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:59:29 2022

@author: jekim
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# n = 100
# xmin, xmax, ymin, ymax, zmin, zmax = 0, 20, 0, 20, 0, 50
# cmin, cmax = 0, 2

# xs = np.array([(xmax - xmin) * np.random.random_sample() + xmin for i in range(n)])
# ys = np.array([(ymax - ymin) * np.random.random_sample() + ymin for i in range(n)])
# zs = np.array([(zmax - zmin) * np.random.random_sample() + zmin for i in range(n)])
# color = np.array([(cmax - cmin) * np.random.random_sample() + cmin for i in range(n)])

xs= kpts_repro[:,0]
ys= kpts_repro[:,1]
zs= kpts_repro[:,2]

# xs= a[:,0]
# ys= a[:,1]
# zs= a[:,2]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, marker='o', s=15, cmap='Greens')

plt.show()

plt.imshow(img)
plt.scatter(k2d[:,0],k2d[:,1])
plt.show()

k2p=np.array(annots['keypoints2d'])
k3p=np.array(annots['keypoints3d'])

# img1 = cv2.resize(img, dsize=(int(1279), int(720)), interpolation=cv2.INTER_AREA) 
cv2.imshow('vis', img)
cv2.waitKey()
cv2.destroyAllWindows()