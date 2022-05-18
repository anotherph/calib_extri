#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step 1) extract image file for calibration

python3 step1_.py --path /home/jekim/workspace/calib_extri/KETI_cal

@author: jekim
"""

import os, sys
import cv2
from os.path import join
from tqdm import tqdm
from glob import glob
import numpy as np
import argparse

def extract_image(path, args):
    # dir_base ='/home/jekim/workspace/calib_extri/extri_data'
    dir_base=path
    dir_video = os.path.join(dir_base,'extri_data/video')
    dir_image=os.path.join(dir_base,'extri_data/images')
    os.mkdir(dir_image)
    
    if not os.path.exists(dir_video):
        print('directory of video is incorrect!')
        sys.exit()
    list_video = os.listdir(dir_video)
    if list_video==[]:
        print('No video-files in the directroy!')
        sys.exit()
    
    for list_ in list_video:  
        outpath=os.path.join(dir_image,os.path.splitext(list_)[0])
        os.mkdir(outpath)
        name_video=os.path.join(dir_video,list_)
        video = cv2.VideoCapture(name_video)
        totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for cnt in range(totalFrames):
            if cnt == int(round(totalFrames/2)):
                ret, frame = video.read()
                cv2.imwrite(join(outpath, '{:06d}.jpg'.format(0)), frame[:,:int(frame.shape[1]/2),:])
                break
        video.release()   
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    extract_image(args.path, args=args)
