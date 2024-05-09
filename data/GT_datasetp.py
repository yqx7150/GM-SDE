import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass

from scipy.io import loadmat
class GTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.GT_paths = None
        self.GT_env = None  # environment for lmdb
        self.GT_size = opt["GT_size"]
        #print(opt["data_type"])

        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.GT_paths, self.GT_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )
        elif opt["data_type"] == "img":    #@@@@@@@@@@@@@@@@add
            self.GT_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )  # GT list
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.GT_paths, "Error: GT paths are empty."

        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt["dataroot_GT"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
    def k2wgt(self,X,W):
        Y = np.multiply(X,W) 
        return Y

    def wgt2k(self,X,W,DC):
        Y = np.multiply(X,1./W)
        Y[W==0] = DC[W==0] 
        return Y

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if self.GT_env is None:
                self._init_lmdb()

        GT_path = None
        GT_size = self.opt["GT_size"]

        # get GT image
        GT_path = self.GT_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.GT_sizes[index].split("_")]
        else:
            resolution = None
        '''
        img_GT = util.read_img(
            self.GT_env, GT_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]
        '''
        w_root = self.opt["weight_root"]
        weight = loadmat(os.path.join(w_root, 'weight1_GEBrain.mat'))['weight']
        assert weight is not None , 'no weight'
        
        img_GT = loadmat(GT_path)['SlcRaw']
        img_GT = img_GT[:,7:263,:]
        for i in range(12):
            img_GT[:,:,i] = np.fft.ifftshift(np.fft.ifft2(img_GT[:,:,i]))
        # img_GT = loadmat(GT_path)['DATA']
        # img_GT = loadmat(GT_path)['Img2']
        # img_GT = loadmat(GT_path)['Img']
        

        #print(self.opt["phase"])
        #print(img_GT_k.shape)
        '''

        if self.opt["phase"] == "train":#@@@@@@@@@@@@@@@@add
            H, W, C = w_k.shape

            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            w_k = w_k[rnd_h : rnd_h + GT_size, rnd_w : rnd_w + GT_size, :]

            # augmentation - flip, rotate
            w_k = util.augment(
                w_k,
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
            )

            # GT_size和图像相同：仅数据增强；GT_size<图像大小：先随机剪切，再增强
        '''



        '''
        # change color space if necessary
        if self.opt["color"]:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[
                0
            ]
        '''

        # BGR to RGB, HWC to CHW, numpy to tensor
        '''
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        '''
        
        # w_k = torch.from_numpy(
        #     np.ascontiguousarray(np.transpose(w_k, (2, 0, 1)))
        # ).float()

        return {"GT": img_GT, "GT_path": GT_path}
        #return img_GT

    def __len__(self):
        return len(self.GT_paths)
