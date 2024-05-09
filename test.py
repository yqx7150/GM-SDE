import argparse
import logging
import os.path
import os.path as osp
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
from IPython import embed
import lpips
import math
import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from utils import sde_utils_mri, sde_utils_mri_single
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr
from scipy.io import loadmat
import cv2
try:
    from skimage.measure import compare_psnr,compare_ssim
except:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
def write_images(x, image_save_path):
    maxvalue = np.max(x)
    if maxvalue < 128:
        x = np.array(x*255.0,dtype=np.uint8)
        cv2.imwrite(image_save_path, x.astype(np.uint8))
def k2wgt(X,W):
    Y = np.multiply(X,W) 
    return Y

def wgt2k(X,W,DC):
    Y = np.multiply(X,1./W)
    Y[W==0] = DC[W==0] 
    return Y

def write_Data_p(model_num,psnr,ssim):
    filedir="result_psnr_main.txt"
    with open(osp.join('./Res/',filedir),"w+") as f:#a+
        f.writelines(str(model_num)+' '+'['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.write('\n')
def write_Data_s(model_num,psnr,ssim):
    filedir="result_ssim_main.txt"
    with open(osp.join('./Res/',filedir),"w+") as f:#a+
        f.writelines(str(model_num)+' '+'['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.write('\n')        
def write_Data2(psnr,ssim):
    filedir="PC.txt"
    with open(osp.join('./Res/',filedir),"a+") as f:#a+
        f.writelines('['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.write('\n')

def im2row(im,winSize):
    size = (im).shape
    out = np.zeros(((size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),winSize[0]*winSize[1],size[2]),dtype=np.complex64)
    count = -1
    for y in range(winSize[1]):
        for x in range(winSize[0]):
            count = count + 1                 
            temp1 = im[x:(size[0]-winSize[0]+x+1),y:(size[1]-winSize[1]+y+1),:]
            temp2 = np.reshape(temp1,[(size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),1,size[2]],order = 'F')
            out[:,count,:] = np.squeeze(temp2)          
            
    return out
def row2im(mtx,size_data,winSize):
    size_mtx = mtx.shape 
    sx = size_data[0]
    sy = size_data[1] 
    sz = size_mtx[2] 
    
    res = np.zeros((sx,sy,sz),dtype=np.complex64)
    W = np.zeros((sx,sy,sz),dtype=np.complex64)
    out = np.zeros((sx,sy,sz),dtype=np.complex64)
    count = -1
    
    for y in range(winSize[1]):
        for x in range(winSize[0]):
            count = count + 1
            res[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] = res[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] + np.reshape(np.squeeze(mtx[:,count,:]),[sx-winSize[0]+1,sy-winSize[1]+1,sz],order = 'F')  
            W[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] = W[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] + 1
            

    out = np.multiply(res,1./W)
    return out

def get_muti_mask():

    # data_size = data.shape

    mask_muti = np.zeros((4, 256, 256), dtype=np.complex64)
    mask_muti[0, 0:192, :] = mask_muti[0, 0:192, :] + 1
    mask_muti[1, 64:256, :] = mask_muti[1, 64:256, :] + 1
    mask_muti[2, :, 0:192] = mask_muti[2, :, 0:192] + 1
    mask_muti[3, :, 64:256] = mask_muti[3, :, 64:256] + 1
    mask_sum = mask_muti[0, :, :] + mask_muti[1, :, :] + mask_muti[2, :, :] + mask_muti[3, :, :]

    return mask_muti,mask_sum

def L_SVD(A, k):
    svd_input  = torch.tensor(A, dtype=torch.complex64)
    U,S,V = torch.svd(svd_input)
    
    #svd_input  = np.array(A, dtype=np.complex64)
    #U,S,V = np.linalg.svd(svd_input)
    S = torch.diag(S)
    U = np.array(U.resolve_conj().numpy(),dtype=np.complex64)
    S = np.array(S.resolve_conj().numpy(),dtype=np.complex64)
    V = np.array(V.resolve_conj().numpy(),dtype=np.complex64)          
    uu = U[:, 0:k]
    ss = S[0:k, 0:k]
    vv = V[:, 0:k]
    return uu, ss, vv
#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)


#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")
if not os.path.exists("./result"):
    os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")


util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device

# sde = sde_utils_mri.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde = sde_utils_mri_single.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)

sde.set_model(model.model)
lpips_fn = lpips.LPIPS(net='alex').to(device)

scale = opt['degradation']['scale']
mask_root = opt['degradation']['mask_root']
weight_root = opt['degradation']['weight_root']
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []
    test_results["lpips"] = []
    test_results["psnr_zf"] = []
    test_results["ssim_zf"] = []
    test_times = []

    for i, test_data in enumerate(test_loader):
        

        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []
        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        

        

        #### input dataset_LQ
        GT = test_data["GT"].squeeze().cpu().numpy()#1,256,256,12->256,256,12
        imgsize = GT.shape
        coil = imgsize[2]
        ori_data = GT/np.max(np.abs(GT))

        ori_data = np.transpose(ori_data,(2,0,1))#12,256,256

        # for i in range(coil):
        #     ori_data[i,:,:] = np.rot90(ori_data[i,:,:], 1)
        
        mask = loadmat(os.path.join(mask_root, 'r4.mat'))['mask']
        # mask = np.rot90(mask, 1)
        mask = np.repeat(mask[None,...],coil,0)
        write_images(np.abs(mask[0,:,:]),osp.join('./Res/','mask1'+'.png'))

        ww = loadmat(os.path.join(weight_root, 'weight1_GEBrain.mat'))['weight']
        weight = np.repeat(ww[None,...],coil,0)



        
        # write_images(abs(zeorfilled_data_sos),osp.join('./result/parallel_12ch/'+'Zeorfilled_'+str(round(psnr_zero, 2))+str(round(ssim_zero, 4))+'.png'))
        # write_images(np.abs(zeorfilled_data_sos), os.path.join(dataset_dir,img_name+'_zf_'+str(round(psnr_zf, 2))+str(round(ssim_zf, 4))+'.png'))
        # write_images(np.abs(zeorfilled_data_sos),osp.join('./Res/',img_name+'_zf_'+str(round(psnr_zf, 2))+str(round(ssim_zf, 4))+'.png'))
        Kdata = np.zeros((coil,256,256),dtype=np.complex64)
        Ksample = np.zeros((coil,256,256),dtype=np.complex64)
        zeorfilled_data = np.zeros((coil,256,256),dtype=np.complex64)
        k_w = np.zeros((coil,256,256),dtype=np.complex64)
        Kdata_w = np.zeros((coil,256,256),dtype=np.complex64)

        for i in range(coil):
            Kdata[i,:,:] = np.fft.fftshift(np.fft.fft2(ori_data[i,:,:]))
            # Kdata[i, :, :] = np.fft.fft2(ori_data[i, :, :])
            Kdata_w[i,:,:] = k2wgt(Kdata[i,:,:],weight[i,:,:])
            Ksample[i,:,:] = np.multiply(mask[i,:,:],Kdata[i,:,:])
            k_w[i,:,:] = k2wgt(Ksample[i,:,:],weight[i,:,:])           
            zeorfilled_data[i,:,:] = np.fft.ifft2(Ksample[i,:,:])

        mask_muti, _ = get_muti_mask()
        k_GT_4_partial = np.zeros((coil, 4, 256, 256), dtype=np.complex64)
        k_w_8_partial = np.zeros((coil, 8, 256, 256), dtype=np.complex64)
        k_GT_4_ori = np.zeros((coil, 4, 256, 256), dtype=np.complex64)
        k_w_8_ori = np.zeros((coil, 8, 256, 256), dtype=np.complex64)
        for j in range(coil):
            for i in range(4):
                k_GT_4_partial[j, i, :, :] = np.multiply(mask_muti[i, :, :], Kdata_w[j, :, :])
                k_w_8_partial[j, i, :, :] = np.real(k_GT_4_partial[j, i, :, :])
                k_w_8_partial[j, i + 4, :, :] = np.imag(k_GT_4_partial[j, i, :, :])
                k_GT_4_ori[j, i, :, :] = np.multiply(mask_muti[i, :, :], k_w[j, :, :])
                k_w_8_ori[j, i, :, :] = np.real(k_GT_4_ori[j, i, :, :])
                k_w_8_ori[j, i + 4, :, :] = np.imag(k_GT_4_ori[j, i, :, :])
                # k_GT_4_ori[j, i, :, :] = np.multiply(mask_muti[i, :, :], k_w[j, :, :])
                # k_w_8_ori[j, i, :, :] = np.real(k_GT_4_ori[j, i, :, :])
                # k_w_8_ori[j, i + 4, :, :] = np.imag(k_GT_4_ori[j, i, :, :])
        x_input = k_w_8_ori
        

        # x_input=np.stack((np.real(k_w),np.imag(k_w)),1)
        x_mean = torch.tensor(x_input, dtype=torch.float32).cuda()#12,2,256,256
        GT_ = torch.tensor(k_w_8_partial, dtype=torch.float32).cuda()

        #############
        # import test_fuction

        # test_fuction.run(ori=ori_data,mask=mask,weight=weight,opt=opt,model=model)



        # a=1
        # assert a==0

        #############
        
        noisy_state = sde.noise_state(x_mean)
        model.feed_data(noisy_state, x_mean,GT_)
        tic = time.time()
        data=model.test(sde, save_states=True,ori=ori_data,mask=mask,weight=weight,opt=opt,model=model)
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        # SR_img = visuals["Output"]
        # output = util.tensor2img(SR_img.squeeze())  # uint8
        # LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
        # GT_ = util.tensor2img(visuals["GT"].squeeze())  # uint8

        # print(x_mean.shape)
        x_mean = visuals['Output'].squeeze().float().cpu().numpy()#w12,2,256,256
        # print(x_mean.shape)

        max_psnr = data["psnr"]
        max_psnr_ssim = data["ssim"]
        psnr_zf = data["zf_psnr"]
        ssim_zf = data["zf_ssim"]
        
        test_results["psnr"].append(max_psnr)
        test_results["ssim"].append(max_psnr_ssim)
        test_results["psnr_zf"].append(psnr_zf)
        test_results["ssim_zf"].append(ssim_zf)


        logger.info(
            "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}*****  零填充: PSNR: {:.6f} dB; SSIM: {:.6f}".format(
                img_name, max_psnr, max_psnr_ssim, psnr_zf, ssim_zf
            )
        )


        '''
        if need_GT:
            gt_img = GT_ / 255.0
            sr_img = output / 255.0

            crop_border = opt["crop_border"] if opt["crop_border"] else scale
            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]
                cropped_gt_img = gt_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]

            psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
            lp_score = lpips_fn(
                GT.to(device) * 2 - 1, SR_img.to(device) * 2 - 1).squeeze().item()

            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)
            test_results["lpips"].append(lp_score)

            if len(gt_img.shape) == 3:
                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    if crop_border == 0:
                        cropped_sr_img_y = sr_img_y
                        cropped_gt_img_y = gt_img_y
                    else:
                        cropped_sr_img_y = sr_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                        cropped_gt_img_y = gt_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                    psnr_y = util.calculate_psnr(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )
                    ssim_y = util.calculate_ssim(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )

                    test_results["psnr_y"].append(psnr_y)
                    test_results["ssim_y"].append(ssim_y)

                    logger.info(
                        "img{:3d}:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; LPIPS: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.".format(
                            i, img_name, psnr, ssim, lp_score, psnr_y, ssim_y
                        )
                    )
            else:
                logger.info(
                    "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}".format(
                        img_name, psnr, ssim
                    )
                )

                test_results["psnr_y"].append(psnr)
                test_results["ssim_y"].append(ssim)
        else:
            logger.info(img_name)
        '''


    # ave_lpips = sum(test_results["lpips"]) / len(test_results["lpips"])
    ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
    ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
    ave_psnr_zf = sum(test_results["psnr_zf"]) / len(test_results["psnr_zf"])
    ave_ssim_zf = sum(test_results["ssim_zf"]) / len(test_results["ssim_zf"])
    logger.info(
        "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}*****  零填充: PSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
            test_set_name, ave_psnr, ave_ssim, ave_psnr_zf, ave_ssim_zf
        )
    )
    '''
    if test_results["psnr_y"] and test_results["ssim_y"]:
        ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
        ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
        logger.info(
            "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n".format(
                ave_psnr_y, ave_ssim_y
            )
        )

    logger.info(
            "----average LPIPS\t: {:.6f}\n".format(ave_lpips)
        )
    '''
    print(f"average test time: {np.mean(test_times):.4f}")
