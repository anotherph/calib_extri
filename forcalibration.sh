#!/bin/bash

:<<'END'
Created on Wed May 18 10:32:01 2022

input: mp4 file
output: calibration result

bash file 
step 1) extract image file for calibration
step 2) detection_chessboard by user-self
step 3) calculate extrinsic parameter
step 4) check the calibration result

@author: jekim

END

v_path="/home/jekim/workspace/calib_extri/KETI_cal"
v_pattern="4,4"
v_grid="0.045"

python3 step1_.py --path $v_path
python3 step2_.py --path $v_path --pattern $v_pattern --grid $v_grid
python3 step3_.py --path $v_path
python3 step4_.py --path $v_path --pattern $v_pattern --grid $v_grid



