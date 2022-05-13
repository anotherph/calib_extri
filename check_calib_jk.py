'''
  @ Date: 2021-03-27 19:13:50
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-04-15 22:53:23
  @ FilePath: /EasyMocap/apps/calibration/check_calib.py
'''
import cv2
import numpy as np
import os
from os.path import join
# from easymocap.mytools import read_json, merge
# from easymocap.mytools import read_camera, plot_points2d
# from easymocap.mytools import batch_triangulate, projectN3, Undistort
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

POINTS_SQUARE = np.array([
    [0., 0., 0.],
    [1., 0., 0.],
    [1., 1., 0.],
    [0., 1., 0.]
])

LINES_SQUARE = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0]
])

row_col_ = {
    2: (2, 1),
    7: (2, 4),
    8: (2, 4),
    9: (3, 3),
    26: (4, 7)
}
def get_row_col(l):
    if l in row_col_.keys():
        return row_col_[l]
    else:
        from math import sqrt
        row = int(sqrt(l) + 0.5)
        col = int(l/ row + 0.5)
        if row*col<l:
            col = col + 1
        if row > col:
            row, col = col, row
        return row, col

class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        else:
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    def __del__(self):
        cv2.FileStorage.release(self.fs)

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            cv2.FileStorage.write(self.fs, key, value)
        elif dt == 'list':
            if self.major_version == 4: # 4.4
                self.fs.startWriteStruct(key, cv2.FileNode_SEQ)
                for elem in value:
                    self.fs.write('', elem)
                self.fs.endWriteStruct()
            else: # 3.4
                self.fs.write(key, '[')
                for elem in value:
                    self.fs.write('none', elem)
                self.fs.write('none', ']')

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def merge(images, row=-1, col=-1, resize=False, ret_range=False, **kwargs):
    if row == -1 and col == -1:
        row, col = get_row_col(len(images))
    height = images[0].shape[0]
    width = images[0].shape[1]
    ret_img = np.zeros((height * row, width * col, images[0].shape[2]), dtype=np.uint8) + 255
    ranges = []
    for i in range(row):
        for j in range(col):
            if i*col + j >= len(images):
                break
            img = images[i * col + j]
            # resize the image size
            img = cv2.resize(img, (width, height))
            ret_img[height * i: height * (i+1), width * j: width * (j+1)] = img
            ranges.append((width*j, height*i, width*(j+1), height*(i+1)))
    if resize:
        min_height = 1000
        if ret_img.shape[0] > min_height:
            scale = min_height/ret_img.shape[0]
            ret_img = cv2.resize(ret_img, None, fx=scale, fy=scale)
    if ret_range:
        return ret_img, ranges
    return ret_img

def read_camera(intri_name, extri_name, cam_names=[]):
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    intri = FileStorage(intri_name)
    extri = FileStorage(extri_name)
    cams, P = {}, {}
    cam_names = intri.read('names', dt='list')
    for cam in cam_names:
        # 内参只读子码流的
        cams[cam] = {}
        cams[cam]['K'] = intri.read('K_{}'.format( cam))
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        Rvec = extri.read('R_{}'.format(cam))
        Tvec = extri.read('T_{}'.format(cam))
        R = cv2.Rodrigues(Rvec)[0]
        RT = np.hstack((R, Tvec))

        cams[cam]['RT'] = RT
        cams[cam]['R'] = R
        cams[cam]['T'] = Tvec
        P[cam] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['P'] = P[cam]

        cams[cam]['dist'] = intri.read('dist_{}'.format(cam))
    cams['basenames'] = cam_names
    return cams

def plot_points2d(img, points2d, lines, lw=4, col=(0, 255, 0), putText=True):
    # 将2d点画上去
    if points2d.shape[1] == 2:
        points2d = np.hstack([points2d, np.ones((points2d.shape[0], 1))])
    for i, (x, y, v) in enumerate(points2d):
        if v < 0.01:
            continue
        c = col
        plot_cross(img, x, y, width=10, col=c, lw=lw)
        if putText:
            font_scale = img.shape[0]/2000
            cv2.putText(img, '{}'.format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, c, 2)
    for i, j in lines:
        if points2d[i][2] < 0.01 or points2d[j][2] < 0.01:
            continue
        plot_line(img, points2d[i], points2d[j], 2, col)
        
def batch_triangulate(keypoints_, Pall, keypoints_pre=None, lamb=1e3):
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    v = (keypoints_[:, :, -1]>0).sum(axis=0)
    valid_joint = np.where(v > 1)[0]
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0)/v[valid_joint]
    # P2: P矩阵的最后一行：(1, nViews, 1, 4)
    P0 = Pall[None, :, 0, :]
    P1 = Pall[None, :, 1, :]
    P2 = Pall[None, :, 2, :]
    # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.hstack([Au, Av])
    if keypoints_pre is not None:
        # keypoints_pre: (nJoints, 4)
        B = np.eye(4)[None, :, :].repeat(A.shape[0], axis=0)
        B[:, :3, 3] = -keypoints_pre[valid_joint, :3]
        confpre = lamb * keypoints_pre[valid_joint, 3]
        # 1, 0, 0, -x0
        # 0, 1, 0, -y0
        # 0, 0, 1, -z0
        # 0, 0, 0,   0
        B[:, 3, 3] = 0
        B = B * confpre[:, None, None]
        A = np.hstack((A, B))
    u, s, v = np.linalg.svd(A)
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = np.zeros((keypoints_.shape[1], 4))
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d
    return result

def projectN3(kpts3d, Pall):
    # kpts3d: (N, 3)
    nViews = len(Pall)
    kp3d = np.hstack((kpts3d[:, :3], np.ones((kpts3d.shape[0], 1))))
    kp2ds = []
    for nv in range(nViews):
        kp2d = Pall[nv] @ kp3d.T
        kp2d[:2, :] /= kp2d[2:, :]
        kp2ds.append(kp2d.T[None, :, :])
    kp2ds = np.vstack(kp2ds)
    kp2ds[..., -1] = kp2ds[..., -1] * (kpts3d[None, :, -1] > 0.)
    return kp2ds

class Undistort:
    @staticmethod
    def image(frame, K, dist):
        return cv2.undistort(frame, K, dist, None)

    @staticmethod
    def points(keypoints, K, dist):
        # keypoints: (N, 3)
        assert len(keypoints.shape) == 2, keypoints.shape
        kpts = keypoints[:, None, :2]
        kpts = np.ascontiguousarray(kpts)
        kpts = cv2.undistortPoints(kpts, K, dist, P=K)
        keypoints[:, :2] = kpts[:, 0]
        return keypoints
    
    @staticmethod
    def bbox(bbox, K, dist):
        keypoints = np.array([[bbox[0], bbox[1], 1], [bbox[2], bbox[3], 1]])
        kpts = Undistort.points(keypoints, K, dist)
        bbox = np.array([kpts[0, 0], kpts[0, 1], kpts[1, 0], kpts[1, 1], bbox[4]])
        return bbox

def plot_cross(img, x, y, col, width=-1, lw=-1):
    if lw == -1:
        lw = max(1, int(round(img.shape[0]/1000)))
        width = lw * 5
    cv2.line(img, (int(x-width), int(y)), (int(x+width), int(y)), col, lw)
    cv2.line(img, (int(x), int(y-width)), (int(x), int(y+width)), col, lw)

def plot_line(img, pt1, pt2, lw, col):
    cv2.line(img, (int(pt1[0]+0.5), int(pt1[1]+0.5)), (int(pt2[0]+0.5), int(pt2[1]+0.5)),
        col, lw)
    
def load_cube():
    points3d = 0.045*2*np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 1.],
        [1., 1., 1.],
        [0., 1., 1.]
    ])
    
    # points3d = np.array([
    #     [0., 0., 0.],
    #     [-1., 0., 0.],
    #     [-1., 1., 0.],
    #     [0., 1., 0.],
    #     [0., 0., 1.],
    #     [-1., 0., 1.],
    #     [-1., 1.,1.],
    #     [0., 1.,1.]
    # ])
    
    # points3d = np.array([
    #     [0., 0., 0.],
    #     [-1., 0., 0.],
    #     [-1., 1., 0.],
    #     [0., 1., 0.],
    #     [0., 0., 1.],
    #     [-1., 0., 1.],
    #     [-1., 1., 1.],
    #     [0., 1., 1.]
    # ])
    
    lines = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7]
    ], dtype=np.int)
    
    
    points3d = np.hstack((points3d, np.ones((points3d.shape[0], 1))))
    return points3d, lines

def merge_points_lines(points3d, lines):
    dist = np.linalg.norm(points3d[:, None, :] - points3d[None, :, :], axis=-1)
    mapid = np.arange(points3d.shape[0])
    for i in range(dist.shape[0]):
        if mapid[i] != i:
            continue
        equal = np.where(dist[i] < 1e-3)[0]
        for j in equal:
            if j == i:
                continue
            mapid[j] = i
    newid = sorted(list(set(mapid)))
    newpoints = points3d[newid]
    for i, newi in enumerate(newid):
        mapid[mapid==newi] = i
    return newpoints, mapid[lines]

def load_grid(xrange=10, yrange=10):
    start = np.array([0., 0., 0.])
    xdir = np.array([1., 0., 0.])
    ydir = np.array([0., 1., 0.])
    stepx = 1.
    stepy = 1.
    points3d, lines = [], []
    for i in range(xrange):
        for j in range(yrange):
            base = start + xdir*i*stepx + ydir*j*stepy
            points3d.append(POINTS_SQUARE+base)
            lines.append(LINES_SQUARE+4*(i*yrange+j))
    points3d = np.vstack(points3d)
    lines = np.vstack(lines)
    return merge_points_lines(points3d, lines)

def check_calib(path, out, vis=False, show=False, debug=False):
    if vis:
        out_dir = join(out, 'check')
        os.makedirs(out_dir, exist_ok=True)
    cameras = read_camera(join(out, 'intri.yml'), join(out, 'extri.yml'))
    cameras.pop('basenames')
    total_sum, cnt = 0, 0
    for nf in tqdm(range(1)):
        imgs = []
        k2ds = []
        for cam, camera in cameras.items():
            if vis:
                # imgname = join(path, 'images', cam, '{:06d}.jpg'.format(nf))
                dir_img=os.path.join(path,'images',cam)
                file_list=os.listdir(dir_img)
                file_list_images = np.array([file for file in file_list if file.endswith(".jpg")])
                file_list_images.sort()
                imgname=os.path.join(dir_img,file_list_images[0])
                assert os.path.exists(imgname), imgname
                img = cv2.imread(imgname)
                img = Undistort.image(img, camera['K'], camera['dist'])
                imgs.append(img)
            # annname = join(path, 'chessboard', cam, '{:06d}.json'.format(nf))
            dir_annos=os.path.join(path,'chessboard',cam)
            file_list=os.listdir(dir_annos)
            file_list_annos = np.array([file for file in file_list if file.endswith(".json")])
            file_list_annos.sort()
            annname=os.path.join(dir_annos,file_list_annos[0])
            if not os.path.exists(annname):
                break
            data = read_json(annname)
            k2d = np.array(data['keypoints2d'], dtype=np.float32)
            k2d = Undistort.points(k2d, camera['K'], camera['dist'])
            k2ds.append(k2d)
        if len(k2ds) == 0:
            break
        Pall = np.stack([camera['P'] for camera in cameras.values()])
        k2ds = np.stack(k2ds)
        k3d = batch_triangulate(k2ds, Pall)
        kpts_repro = projectN3(k3d, Pall)
        for nv in range(len(k2ds)):
            conf = k2ds[nv][:, -1]
            dist = conf * np.linalg.norm(kpts_repro[nv][:, :2] - k2ds[nv][:, :2], axis=1)
            total_sum += dist.sum()
            cnt += conf.sum()
            if debug:
                print('{:2d}-{:2d}: {:6.2f}/{:2d}'.format(nf, nv, dist.sum(), int(conf.sum())))
            if vis:
                plot_points2d(imgs[nv], kpts_repro[nv], [], col=(0, 0, 255), lw=1, putText=False)
                plot_points2d(imgs[nv], k2ds[nv], [], lw=1, putText=False)
                if show:
                    cv2.imshow('vis', imgs[nv])
                    cv2.waitKey()
                    cv2.destroyAllWindows()
        if vis:
            imgout = merge(imgs, resize=False)
            outname = join(out, 'check', '{:06d}.jpg'.format(nf))
            cv2.imwrite(outname, imgout)
    print('{:.2f}/{} = {:.2f} pixel'.format(total_sum, int(cnt), total_sum/cnt))

def check_scene(path, out, points3d, lines):
    cameras = read_camera(join(out, 'intri.yml'), join(out, 'extri.yml'))
    cameras.pop('basenames')
    nf = 0
    for cam, camera in cameras.items():
        # imgname = join(path, 'images', cam, '{:06d}.jpg'.format(nf))
        dir_img=os.path.join(path,'images',cam)
        file_list=os.listdir(dir_img)
        file_list_images = np.array([file for file in file_list if file.endswith(".png")])
        file_list_images.sort()
        imgname=os.path.join(dir_img,file_list_images[0])
        assert os.path.exists(imgname), imgname
        img = cv2.imread(imgname)
        img = Undistort.image(img, camera['K'], camera['dist'])
        kpts_repro = projectN3(points3d, camera['P'][None, :, :])[0]
        plot_points2d(img, kpts_repro, lines, col=(0, 0, 255), lw=1, putText=True)
        cv2.imshow('vis', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

def check_match(path, out):
    os.makedirs(out, exist_ok=True)
    cameras = read_camera(join(path, 'intri.yml'), join(path, 'extri.yml'))
    cams = cameras.pop('basenames')
    annots = read_json(join(path, 'calib.json'))
    points_global = annots['points_global']
    points3d = np.ones((len(points_global), 4))
    # first triangulate
    points2d = np.zeros((len(cams), len(points_global), 3))
    for i, record in enumerate(points_global):
        for cam, (x, y) in record.items():
            points2d[cams.index(cam), i] = (x, y, 1)
    # 2. undistort
    for nv in range(points2d.shape[0]):
        camera = cameras[cams[nv]]
        points2d[nv] = Undistort.points(points2d[nv], camera['K'], camera['dist'])
    Pall = np.stack([cameras[cam]['P'] for cam in cams])
    points3d = batch_triangulate(points2d, Pall)
    lines = []
    nf = 0
    for cam, camera in cameras.items():
        imgname = join(path, 'images', cam, '{:06d}.jpg'.format(nf))
        assert os.path.exists(imgname), imgname
        img = cv2.imread(imgname)
        img = Undistort.image(img, camera['K'], camera['dist'])
        kpts_repro = projectN3(points3d, camera['P'][None, :, :])[0]
        plot_points2d(img, kpts_repro, lines, col=(0, 0, 255), lw=1, putText=True)
        plot_points2d(img, points2d[cams.index(cam)], lines, col=(0, 255, 0), lw=1, putText=True)
        outname = join(out, cam+'.jpg')
        cv2.imwrite(outname, img)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('path', type=str, 
        # help='the directory contains the extrinsic images')
    # parser.add_argument('--path', type=str,default='/home/jekim/workspace/calib_extri/cal_data_1001_SGU_corner/extri_data')
    parser.add_argument('--path', type=str,default='/home/jekim/workspace/calib_extri/cal0513/extri_data')
    # parser.add_argument('--path', type=str,default='/home/jekim/workspace/calib_extri/extrinsic/extri_data')
    # parser.add_argument('--out', type=str,
    #     help='with camera parameters')
    # parser.add_argument('--out', type=str,default='/home/jekim/workspace/calib_extri/cal_data_1001_SGU_corner/intri_data/output')
    parser.add_argument('--out', type=str,default='/home/jekim/workspace/calib_extri/cal0513/intri_data/output')
    # parser.add_argument('--out', type=str,default='/home/jekim/workspace/calib_extri/extrinsic/intri_data/output')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--calib', action='store_true')
    
    # for vis
    # parser.add_argument('--vis', default=True)
    # parser.add_argument('--show', default=True)
    # parser.add_argument('--cube', action='store_true')
    
    # for cube
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--cube', default=True)
    
    args = parser.parse_args()
    if args.cube:
        points, lines = load_cube()
        check_scene(args.path, args.out, points, lines)
    elif args.grid:
        points, lines = load_grid(xrange=15, yrange=14)
        check_scene(args.path, args.out, points, lines)
    elif args.calib:
        check_match(args.path, args.out)
    else:
        check_calib(args.path, args.out, args.vis, args.show, args.debug)