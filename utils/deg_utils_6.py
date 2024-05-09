import os
import cv2
import abc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils


########### denoising ###############
def add_noise(tensor, sigma):
    sigma = sigma / 255 if sigma > 1 else sigma
    return tensor + torch.randn_like(tensor) * sigma


######## inpainting ###########
def mask_to(tensor, mask_root='data/datasets/gt_keep_masks/genhalf', mask_id=-1, n=100):
    batch = tensor.shape[0]
    if mask_id < 0:
        mask_id = np.random.randint(0, n, batch)
        masks = []
        for i in range(batch):
            masks.append(cv2.imread(os.path.join(mask_root, f'{mask_id[i]:06d}.png'))[None, ...] / 255.)
        mask = np.concatenate(masks, axis=0)
    else:
        mask = cv2.imread(os.path.join(mask_root, f'{mask_id:06d}.png'))[None, ...] / 255.

    mask = torch.tensor(mask).permute(0, 3, 1, 2).float()
    # for images are clipped or scaled
    mask = F.interpolate(mask, size=tensor.shape[2:], mode='nearest')
    masked_tensor = mask * tensor
    return masked_tensor + (1. - mask)
def mri_mask_to(tensor, mask_root='data/datasets/gt_keep_masks/genhalf', mask_id=-1, n=10):
    from scipy.io import loadmat
    mask_name = os.listdir(mask_root)
    batch = tensor.shape[0]
    if mask_id < 0:
        mask_id = np.random.randint(0, n, batch)
        masks = []
        for i in range(batch):
            mask_temp = loadmat(os.path.join(mask_root, mask_name[mask_id[i]]))['mask']
            # mask_temp = np.stack([mask_temp, mask_temp], axis= 0)#2,256,256
            mask_temp = np.repeat(mask_temp[None,...],6,0)
            masks.append(mask_temp)

        mask = np.stack(masks, axis=0)#batch,2,256,256
    else:
        loop_id = mask_id % n
        mask = loadmat(os.path.join(mask_root, mask_name[loop_id]))['mask']
        # mask = np.stack([mask, mask], 0)[None, ...]
        mask = np.repeat(mask[None,...],6,0)[None, ...]
        #mask = cv2.imread(os.path.join(mask_root, f'{mask_id:06d}.png'))[None, ...] / 255.

    mask = torch.tensor(mask).float()
    # for images are clipped or scaled
    #mask = F.interpolate(mask, size=tensor.shape[2:], mode='nearest')
    # print(mask.shape,tensor.shape)
    masked_tensor = mask * tensor
    #return masked_tensor + (1. - mask)
    return masked_tensor
######## super-resolution ###########

def upscale(tensor, scale=4, mode='bicubic'):
    tensor = F.interpolate(tensor, scale_factor=scale, mode=mode)
    return tensor




