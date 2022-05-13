'''
  @ Date: 2022-05-10
  @ Author: jekim
  
  # Detect the corner of chessboard (Ver.4)
  
  1. make chess-board
  2. detect the corner of chessboard by hand
  
'''

from tqdm import tqdm
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

def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:   # visible point 
        cv2.circle(data['image'],(x,y),5,(0,0,255),-1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(data['image'],str(data['number']),(x,y),font,0.5,(255,0,0),2)
        cv2.imshow('image',data['image'])
        
        if len(data['points'])<(data['pattern'][0]*data['pattern'][1]):
            data['points'].append([x,y,1])
            data['number']+=1
    elif event == cv2.EVENT_RBUTTONDOWN: # invisible point
        data['points'].append([0,0,0])
    else:
        pass

def get_points(image,pattern):
    data ={}
    data['image']=image.copy()
    data['points']=[]
    data['pattern']=pattern
    data['number']=1
    
    cv2.imshow('image',image)
    cv2.setMouseCallback("image", mouse_handler, data)
    cv2.waitKey()
    
    points=np.array(data['points'],dtype=float)
    
    return points

def findChessboardCorners(img, annots, pattern):
    conf = sum([v[2] for v in annots['keypoints2d']])
    if annots['visited'] and conf > 0:
        return True
    elif annots['visited']:
        return None
    annots['visited'] = True
    
    points_src=get_points(img,pattern)
    annots['keypoints2d']=points_src.tolist()
    
    return None  

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

    
def detect_chessboard(path, pattern, invalid, gridSize, args):
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
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('path', type=str)
    parser.add_argument('--path', type=str,default='/home/jekim/workspace/calib_extri/cal0513/extri_data')
    parser.add_argument('--ext', type=str, default='.png', choices=['.jpg', '.png'])
    parser.add_argument('--pattern', type=lambda x: (int(x.split(',')[0]), int(x.split(',')[1])),
        help='The pattern of the chessboard', default=(4, 4))
    parser.add_argument('--grid', type=float, default=0.045, 
        help='The length of the grid size (unit: meter)')
    parser.add_argument('--invalid',type=str,default='True')
    parser.add_argument('--max_step', type=int, default=50)
    parser.add_argument('--min_step', type=int, default=0)
    
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seq', action='store_true')
    args = parser.parse_args()
    detect_chessboard(args.path, pattern=args.pattern, invalid=args.invalid, gridSize=args.grid, args=args)
    cv2.destroyAllWindows()
