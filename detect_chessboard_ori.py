'''
  @ Date: 2022-04-20
  @ Author: jekim
  
  original code for detecting the chessboard
  
'''
# detect the corner of chessboard
# from easymocap.annotator.file_utils import getFileList, read_json, save_json
from tqdm import tqdm
# from easymocap.annotator import ImageFolder
# from easymocap.annotator.chessboard import getChessboard3d, findChessboardCorners
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import json
import cv2
import os

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json(file, data):
    if file is None:
        return 0
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def getFileList(root, ext='.jpg', max=-1, ret_full=False):
    files = []
    dirs = sorted(os.listdir(root))
    while len(dirs) > 0:
        path = dirs.pop()
        fullname = join(root, path)
        if os.path.isfile(fullname) and fullname.endswith(ext):
            if ret_full:
                files.append(fullname)
            else:
                files.append(path)
        elif os.path.isdir(fullname):
            names = sorted(os.listdir(fullname))
            if max != -1 and os.path.isfile(join(fullname, names[0])):
                names = names[:max]
            for s in names:
                newDir = join(path, s)
                dirs.append(newDir)
    files = sorted(files)
    return files

class ImageFolder:
    def __init__(self, path, sub=None, image='images', annot='annots', no_annot=False, ext='.jpg', remove_tmp=True) -> None:
        self.root = path
        self.image = image
        self.annot = annot
        self.image_root = join(path, self.image)
        self.annot_root = join(path, self.annot)
        if not os.path.exists(self.annot_root):
            no_annot = True
        self.annot_root_tmp = join(path, self.annot + '_tmp')
        if os.path.exists(self.annot_root_tmp) and remove_tmp:
            shutil.rmtree(self.annot_root_tmp)
        if sub is None:
            self.imgnames = getFileList(self.image_root, ext=ext)
            if not no_annot:
                self.annnames = getFileList(self.annot_root, ext='.json')
        else:
            self.imgnames = getFileList(join(self.image_root, sub), ext=ext)
            self.imgnames = [join(sub, name) for name in self.imgnames]
            if not no_annot:
                self.annnames = getFileList(join(self.annot_root, sub), ext='.json')
                self.annnames = [join(sub, name) for name in self.annnames]
                length = min(len(self.imgnames), len(self.annnames))
                self.imgnames = self.imgnames[:length]
                self.annnames = self.annnames[:length]
                # assert len(self.imgnames) == len(self.annnames)
        self.isTmp = True
        self.no_annot = no_annot
    
    def __getitem__(self, index):
        imgname = join(self.image_root, self.imgnames[index])
        if self.no_annot:
            annname = None
        else:
            if self.isTmp:
                annname = join(self.annot_root_tmp, self.annnames[index])
            else:
                annname = join(self.annot_root, self.annnames[index])
        return imgname, annname
    
    def __len__(self):
        return len(self.imgnames)
    
    def __str__(self) -> str:
        return '{}: {} images'.format(self.root, len(self))
    
def getChessboard3d(pattern, gridSize):
    object_points = np.zeros((pattern[1]*pattern[0], 3), np.float32)
    # 注意：这里为了让标定板z轴朝上，设定了短边是x，长边是y
    object_points[:,:2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1,2)
    object_points[:, [0, 1]] = object_points[:, [1, 0]]
    object_points = object_points * gridSize
    return object_points

colors_chessboard_bar = [
    [0, 0, 255],
    [0, 128, 255],
    [0, 200, 200],
    [0, 255, 0],
    [200, 200, 0],
    [255, 0, 0],
    [255, 0, 250]
]

def _findChessboardCorners(img, pattern):
    "basic function"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    retval, corners = cv2.findChessboardCorners(img, pattern, 
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)
    if not retval:
        return False, None
    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
    corners = corners.squeeze()
    return True, corners

def _findChessboardCornersAdapt(img, pattern):
    "Adapt mode"
    img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,21, 2)
    return _findChessboardCorners(img, pattern)

def findChessboardCorners(img, annots, pattern):
    conf = sum([v[2] for v in annots['keypoints2d']])
    if annots['visited'] and conf > 0:
        return True
    elif annots['visited']:
        return None
    annots['visited'] = True
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    for func in [_findChessboardCornersAdapt, _findChessboardCorners]:
        ret, corners = func(gray, pattern)
        if ret:break
    else:
        return None
    # found the corners
    show = img.copy()
    show = cv2.drawChessboardCorners(show, pattern, corners, ret)
    assert corners.shape[0] == len(annots['keypoints2d'])
    corners = np.hstack((corners, np.ones((corners.shape[0], 1))))
    annots['keypoints2d'] = corners.tolist()
    return show

def create_chessboard(path, pattern, gridSize, ext):
    print('Create chessboard {}'.format(pattern))
    keypoints3d = getChessboard3d(pattern, gridSize=gridSize)
    keypoints2d = np.zeros((keypoints3d.shape[0], 3))
    imgnames = getFileList(path, ext=ext)
    template = {
        'keypoints3d': keypoints3d.tolist(),
        'keypoints2d': keypoints2d.tolist(),
        'visited': False
    }
    for imgname in tqdm(imgnames, desc='create template chessboard'):
        annname = imgname.replace('images', 'chessboard').replace(ext, '.json')
        annname = join(path, annname)
        if os.path.exists(annname):
            # 覆盖keypoints3d
            data = read_json(annname)
            data['keypoints3d'] = template['keypoints3d']
            save_json(annname, data)
        else:
            save_json(annname, template)
    

def detect_chessboard(path, out, pattern, gridSize, args):
    create_chessboard(path, pattern, gridSize, ext=args.ext)
    dataset = ImageFolder(path, annot='chessboard', ext=args.ext)
    dataset.isTmp = False
    if args.silent:
        trange = range(len(dataset))
    else:
        trange = tqdm(range(len(dataset)))
    for i in trange:
        imgname, annotname = dataset[i]
        img = cv2.imread(imgname)
        annots = read_json(annotname)
        show = findChessboardCorners(img, annots, pattern)
        save_json(annotname, annots)        
        if show is None:
            if args.debug:
                print('Cannot find {}'.format(imgname))
            continue
        outname = join(out, imgname.replace(path + '/images/', ''))
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, show)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('path', type=str)
    parser.add_argument('--path', type=str,default='/home/jekim/workspace/calib_extri/cal_data_sample/extri_data')
    # parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--out', type=str,default='/home/jekim/workspace/calib_extri/cal_data_sample/extri_data/output/calibration')
    parser.add_argument('--ext', type=str, default='.jpg', choices=['.jpg', '.png'])
    parser.add_argument('--pattern', type=lambda x: (int(x.split(',')[0]), int(x.split(',')[1])),
        help='The pattern of the chessboard', default=(10, 6))
    parser.add_argument('--grid', type=float, default=0.107, 
        help='The length of the grid size (unit: meter)')
    parser.add_argument('--max_step', type=int, default=50)
    parser.add_argument('--min_step', type=int, default=0)
    
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seq', action='store_true')
    args = parser.parse_args()
    detect_chessboard(args.path, args.out, pattern=args.pattern, gridSize=args.grid, args=args)
    
    # if args.seq:
    #     detect_chessboard_sequence(args.path, args.out, pattern=args.pattern, gridSize=args.grid, args=args)
    # else:
    #     detect_chessboard(args.path, args.out, pattern=args.pattern, gridSize=args.grid, args=args)