'''
step 3) calculate extrinsic parameter

python3 step3_.py --path /home/jekim/workspace/calib_extri/KETI_cal

@author: jekim
'''
import os
from glob import glob
from os.path import join
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

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
        
def read_intri(intri_name):
    assert os.path.exists(intri_name), intri_name
    intri = FileStorage(intri_name)
    camnames = intri.read('names', dt='list')
    cameras = {}
    for key in camnames:
        cam = {}
        cam['K'] = intri.read('K_{}'.format(key))
        cam['invK'] = np.linalg.inv(cam['K'])
        cam['dist'] = intri.read('dist_{}'.format(key))
        cameras[key] = cam
    return cameras

def write_extri(extri_name, cameras):
    extri = FileStorage(extri_name, True)
    results = {}
    camnames = list(cameras.keys())
    extri.write('names', camnames, 'list')
    for key_, val in cameras.items():
        key = key_.split('.')[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])
    return 0

def read_json(path):
    assert os.path.exists(path), path
    with open(path) as f:
        data = json.load(f)
    return data

def calib_extri(path, intriname):
    assert os.path.exists(intriname), intriname
    intri = read_intri(intriname)
    camnames = list(intri.keys())
    extri = {}
    for ic, cam in enumerate(camnames):
        imagenames = sorted(glob(join(path, 'images', cam, '*.jpg')))
        chessnames = sorted(glob(join(path, 'chessboard', cam, '*.json')))
        chessname = chessnames[0]
        data = read_json(chessname)
        k3d = np.array(data['keypoints3d'], dtype=np.float32)
        # k3d[:, 0] *= -1 # original
        # k3d[:, 1] *= -1
        k2d = np.array(data['keypoints2d'], dtype=np.float32)
        k2d = np.ascontiguousarray(k2d[:, :-1])
        ret, rvec, tvec = cv2.solvePnP(k3d, k2d, intri[cam]['K'], intri[cam]['dist'])
        extri[cam] = {}
        extri[cam]['Rvec'] = rvec
        extri[cam]['R'] = cv2.Rodrigues(rvec)[0]
        extri[cam]['T'] = tvec
        center = - extri[cam]['R'].T @ tvec
        print('{} center => {}'.format(cam, center.squeeze()))
    write_extri(join(os.path.dirname(intriname), 'extri.yml'), extri)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    base=args.path
    args.path=os.path.join(base,'extri_data')
    args.intri=os.path.join(base,'intri_data/output/intri.yml')
    calib_extri(args.path, intriname=args.intri)