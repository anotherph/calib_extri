'''
  @ Date: 2022-04-20
  @ Author: jekim
  
  # Detect the corner of chessboard (Ver.3)
  
  same as Ver.2, add 1 parser and parameterize of image_size
  
  1. detect cross-point on chess-board 
	if detection is failed, change the size of image (factor of 1, 1.5, 2, 2.5)
	return the 2d keypoints and the size of image
  2. rearrange:flip the 2d-keypoint to match the 3d-keypoint of real-world
  3. rescale the 2d-keypoint of resized image to that of the original image
  4. remove the outputs which does not have chess board result with respect to all views
  
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

def _findChessboardCorners(img, pattern):
    "basic function"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    retval, corners = cv2.findChessboardCorners(img, pattern, 
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)
    if not retval:
        return False, None
    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
    corners = corners.squeeze()
    # plt.scatter(corners[:,0],corners[:,1])
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
    factor=[1, 1.5, 2, 2.5]
    for factor_ in factor:
        img_size=img.shape # the size of original image
        img = cv2.resize(img, dsize=(int(img_size[1]*factor_), int(img_size[0]*factor_)), interpolation=cv2.INTER_AREA) 
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        for func in [_findChessboardCornersAdapt, _findChessboardCorners]:
            ret, corners = func(gray, pattern)
            if ret:
                corners=rearrange(corners,img,pattern)
                show = img.copy()
                show = cv2.drawChessboardCorners(show, pattern, corners, ret)
                assert corners.shape[0] == len(annots['keypoints2d'])
                corners = np.hstack((corners, np.ones((corners.shape[0], 1))))
                temp_2dkey=rescaling(corners.tolist(),[img_size[1],img_size[0]],[img.shape[1],img.shape[0]])
                annots['keypoints2d']=temp_2dkey.tolist()
                return show
    if not ret: # if the last ret is false, we declare it is failed to detect chess board and return nothing
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

def rearrange(corner,img,pattern):
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    ind_red=np.where((img==[254,0,0]).all(axis=2)) # red color in real 3d world [0,0,0]
    ind_blue = np.where((img==[0,0,254]).all(axis=2)) # blue color in real 3d world[5,9,0]
    ind_mag = np.where((img==[255,0,254]).all(axis=2)) # magenta color in real 3d world[5,0,0]
    ind_yel = np.where((img==[255,255,1]).all(axis=2)) # yellow color in real 3d world [0,9,0]
    
    mean_red=[ind_red[1].mean(), ind_red[0].mean()]
    mean_blue=[ind_blue[1].mean(), ind_blue[0].mean()]
    mean_mag=[ind_mag[1].mean(), ind_mag[0].mean()]
    mean_yel=[ind_yel[1].mean(), ind_yel[0].mean()]
    mean_pt=np.array([mean_red,mean_blue,mean_mag,mean_yel])
    
    # key3d=annots['keypoints3d']
    # key2d=np.array(annots['keypoints2d'])[:,:2]
    key2d=corner
    # key2d_mat=key2d.reshape(-1,2)
    # corner=key2d.reshape(10,-1)
    corner_=key2d[0,:].reshape(-1,2)
    corner_=np.append(corner_,key2d[pattern[0]-1,:].reshape(-1,2),axis=0)
    corner_=np.append(corner_,key2d[pattern[0]*(pattern[1]-1),:].reshape(-1,2),axis=0)
    corner_=np.append(corner_,key2d[pattern[0]*pattern[1]-1,:].reshape(-1,2),axis=0)
    
    temp=np.zeros(4)
    for corners in corner_:
        for ind, means in enumerate(mean_pt):
            # corners=np.repeat(corners.reshape(-1,2),4,axis=0)
            temp[ind]=np.linalg.norm(corners-means)
        if temp.argmin()==0:
            corner_red=corners
        elif temp.argmin()==1:
            corner_blue=corners
        elif temp.argmin()==2:
            corner_mag=corners
        else:
            corner_yel=corners
    # plt.scatter(corner[:,1],corner[:,0])
    corner_ideal=np.array([corner_red, corner_yel, corner_mag, corner_blue]) #ideal color of corners
    corner_flip=np.flipud(corner_ideal)
    
    # match [0,0,0] to red corner
    
    # if corner_.tolist()==corner_ideal.tolist():
    #     pass
    # elif corner_.tolist()==corner_flip.tolist():
    #     corner=np.flipud(key2d)
    
    # match [0,0,0] to yellow corner
    
    if corner_.tolist()==corner_ideal.tolist():
        pass
    elif corner_.tolist()==corner_flip.tolist():
        corner=np.flipud(key2d)
    
    temp_x=corner[:,0].reshape(pattern[1],-1)
    temp_y=corner[:,1].reshape(pattern[1],-1)
    for i in range(pattern[1]):
        corner[i*pattern[0]:(i+1)*pattern[0],0]=np.flipud(temp_x[i,:])
        corner[i*pattern[0]:(i+1)*pattern[0],1]=np.flipud(temp_y[i,:])
        
    # match [0,0,0] to blue corner   
    
    # if corner_.tolist()==corner_ideal.tolist():
    #     pass
    # elif corner_.tolist()==corner_flip.tolist():
    #     corner=np.flipud(key2d)
    
    # match [0,0,0] to magenta corner
    
    # if corner_.tolist()==corner_flip.tolist():
    #     pass
    # elif corner_.tolist()==corner_ideal.tolist():
    #     corner=np.flipud(key2d)
    
    # temp_x=corner[:,0].reshape(pattern[1],-1)
    # temp_y=corner[:,1].reshape(pattern[1],-1)
    # for i in range(pattern[1]):
    #     corner[i*pattern[0]:(i+1)*pattern[0],0]=np.flipud(temp_x[i,:])
    #     corner[i*pattern[0]:(i+1)*pattern[0],1]=np.flipud(temp_y[i,:])
        
    return corner

def rescaling(corners,size_ori,size_img):
    corners=np.array(corners)
    size_ori=np.array(size_ori)
    size_img=np.array(size_img)
    corners[:,0]=corners[:,0]*size_ori[0]/size_img[0]
    corners[:,1]=corners[:,1]*size_ori[1]/size_img[1]
    
    return corners
    
def detect_chessboard(path, out, pattern, invalid, gridSize, args):
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
    
    if invalid:
        # find the non-valid detection-results
        name_cam=os.listdir(os.path.join(path,'images'))
        name_cam.sort()
        file_list=os.listdir(os.path.join(path,'images',name_cam[0]))
        file_list_images = np.array([file for file in file_list if file.endswith(".jpg")])
        file_list_images.sort()
        name_result=np.zeros([len(name_cam),len(file_list_images)])
        for ind_cam, cam in enumerate(name_cam):
            cam_list=np.array(os.listdir(os.path.join(out,cam)))
            for ind_file_, file_ in enumerate(file_list_images):
                if True in (cam_list==file_):
                    name_result[ind_cam,ind_file_]=1
        index_invalid=np.where(name_result==0)
        
        #get rid of the invalid chessboard data
        index_invalid_img=file_list_images[np.unique(index_invalid[1])] # image_index of return None
        for index in index_invalid_img:
            file_name=os.path.splitext(index)[0]
            for cam in name_cam:
                path_invalid=os.path.join(path,'chessboard',cam,file_name+'.json')
                os.remove(path_invalid)
    

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
    parser.add_argument('--invalid',type=str,default='True')
    parser.add_argument('--max_step', type=int, default=50)
    parser.add_argument('--min_step', type=int, default=0)
    
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seq', action='store_true')
    args = parser.parse_args()
    detect_chessboard(args.path, args.out, pattern=args.pattern, invalid=args.invalid, gridSize=args.grid, args=args)
    
