#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:35:58 2022

mark the corner of chess board using red/yellow/magenta/blue point
remove the background using mark

@author: jekim
"""

import os 
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import cv2
from PIL import Image
import matplotlib.pyplot as plt

path_img = '/home/jekim/workspace/calib_extri/cal_data_sample/extri_data/image_mark/3'
save_img = './cal_data_sample/extri_data/image_1_noback/3' # use relative path
list_img = os.listdir(path_img)

for ind in os.listdir(path_img):
    
    img = cv2.imread(os.path.join(path_img,ind))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    
    marg_R = 5
    marg_B = 5
    marg_G = 5
    
    ind_red=np.where((img==[254,0,0]).all(axis=2)) # red color
    ind_blue = np.where((img==[0,0,254]).all(axis=2)) # blue color
    ind_mag = np.where((img==[255,0,254]).all(axis=2)) # magenta color
    ind_yel = np.where((img==[255,255,1]).all(axis=2)) # yellow color
    canvas = np.zeros(shape=img.shape, dtype=np.uint8)
    canvas.fill(255)
    
    # canvas[ind_red]=[0,0,0]
    canvas[ind_blue]=[0,0,0]
    # canvas[ind_mag]=[0,0,0]
    # canvas[ind_yel]=[0,0,0]
    
    plt.imshow(canvas)
    
    thr=100
    
    mean_red=[ind_red[1].mean(), ind_red[0].mean()]
    mean_blue=[ind_blue[1].mean(), ind_blue[0].mean()]
    mean_mag=[ind_mag[1].mean(), ind_mag[0].mean()]
    mean_yel=[ind_yel[1].mean(), ind_yel[0].mean()]
    
    pt_temp=np.array([mean_red,mean_mag,mean_blue,mean_yel],np.int32)
    plt.scatter(pt_temp[:,0],pt_temp[:,1],color='r')
    
    pt_temp_index=np.argsort(pt_temp,axis=0)[:,0]
    pt_temp_sort=pt_temp[list(pt_temp_index),:]
    
    pt_temp_l=pt_temp_sort[:2,:]
    pt_temp_r=pt_temp_sort[2:,:]
    
    if pt_temp_l[0,1]>pt_temp_l[1,1]:
        mean_lb=pt_temp_l[0,:]
        mean_lt=pt_temp_l[1,:]
    else:
        mean_lb=pt_temp_l[1,:]
        mean_lt=pt_temp_l[0,:]
        
    if pt_temp_r[0,1]>pt_temp_r[1,1]:
        mean_rb=pt_temp_r[0,:]
        mean_rt=pt_temp_r[1,:]
    else:
        mean_rb=pt_temp_r[1,:]
        mean_rt=pt_temp_r[0,:]
    
    # mean_lb=np.sort(pt_temp_l,axis=1)[0,:]
    # mean_lt=np.sort(pt_temp_l,axis=1)[1,:]
    # mean_rb=np.sort(pt_temp_r,axis=1)[0,:]
    # mean_rt=np.sort(pt_temp_r,axis=1)[1,:]
    
    # corner_red= np.array(mean_red) + [-thr,-thr]
    # corner_blue =np.array(mean_blue)  + [+thr,+thr]
    # corner_mag=np.array(mean_mag) + [-thr,+thr]
    # corner_yel=np.array(mean_yel) + [+thr,-thr]
    
    corner_lt= np.array(mean_lt) + [-thr,-thr]
    corner_rb =np.array(mean_rb)  + [+thr,+thr]
    corner_lb=np.array(mean_lb) + [-thr,+thr]
    corner_rt=np.array(mean_rt) + [+thr,-thr]
    
    pt1=np.array([corner_lt,corner_lb,corner_rb,corner_rt],np.int32)
    plt.scatter(pt1[:,0],pt1[:,1],color='b')
    
    img_noback=cv2.fillConvexPoly(canvas,pt1,(0,0,0))
    plt.imshow(img_noback)
    
    mask = np.zeros(shape=img.shape, dtype=np.uint8)
    ind_mask =np.where((img_noback==[0,0,0]).all(axis=2)) # red color
    mask[ind_mask]=1
    plt.imshow(mask*img)
    a=1
    cv2.imwrite(os.path.join(save_img,ind),cv2.cvtColor(mask*img,cv2.COLOR_RGB2BGR))

    

    

    
