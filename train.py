import argparse
import logging
import math
import os
import random
import sys
import copy

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# from IPython import embed

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
from scipy.io import loadmat
from data.util import bgr2ycbcr

try:
    from skimage.measure import compare_psnr,compare_ssim
except:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
# torch.autograd.set_detect_anomaly(True)

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group
def k2wgt(X,W):
    Y = np.multiply(X,W) 
    return Y

def get_muti_mask(data=None):

    # data_size = data.shape

    mask_muti = np.zeros((4, 256, 256), dtype=np.complex64)
    mask_muti[0, 0:192, :] = mask_muti[0, 0:192, :] + 1
    mask_muti[1, 64:256, :] = mask_muti[1, 64:256, :] + 1
    mask_muti[2, :, 0:192] = mask_muti[2, :, 0:192] + 1
    mask_muti[3, :, 64:256] = mask_muti[3, :, 64:256] + 1
    mask_sum = mask_muti[0, :, :] + mask_muti[1, :, :] + mask_muti[2, :, :] + mask_muti[3, :, :]

    return mask_muti,mask_sum

def wgt2k(X,W,DC):
    Y = np.multiply(X,1./W)
    Y[W==0] = DC[W==0] 
    return Y
def write_images(x, image_save_path):
    maxvalue = np.max(x)
    if maxvalue < 128:
        x = np.array(x*255.0,dtype=np.uint8)
        cv2.imwrite(image_save_path, x.astype(np.uint8))
def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")


    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt) 
    device = model.device

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    scale = opt['degradation']['scale']
    mask_root = opt['degradation']['mask_root']
    weight_root = opt['degradation']['weight_root']

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value('b', False)
    # a=1      

    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1

            print('current step:',current_step)

            if current_step > total_iters:
                break

            GT = train_data["GT"]#w

            mask_muti, mask_sum = get_muti_mask()

            k_w_8_partial = np.zeros((int(GT.shape[0]),8, 256, 256), dtype=np.float32)
            k_w_8_ori = np.zeros((int(GT.shape[0]),8, 256, 256), dtype=np.float32)
            for j in range(int(GT.shape[0])):
                for i in range(4):
                    k_w_8_partial[j,i, :, :] = np.multiply(mask_muti[i, :, :], GT[j,0, :, :])
                    k_w_8_partial[j, i+4, :, :] = np.multiply(mask_muti[i, :, :], GT[j, 1, :, :])
                    # k_w_8_ori[j, i, :, :] = GT[j,0, :, :]
                    # k_w_8_ori[j, i+4, :, :] = GT[j, 1, :, :]


            # k_w_8_ori = torch.tensor(k_w_8_ori, dtype=torch.float32).cuda()

            LQ = util.multi_mri_mask_to(k_w_8_partial, mask_root)["black"]
            k_w_8_partial = torch.tensor(k_w_8_partial, dtype=torch.float32).cuda()
            LQ = torch.tensor(LQ, dtype=torch.float32).cuda()


            # LQ = util.multi_mri_mask_to(GT, mask_root)["black"]
            timesteps, states = sde.generate_random_states(x0=k_w_8_partial, mu=LQ)

            model.feed_data(states, LQ, k_w_8_partial) # xt, mu, x0
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            
            if current_step in [10,100,200,300,400]:
                img_4 = GT.cpu().numpy()
                img_2_com_batch0 = img_4[0,0,:,:]+1j*img_4[0,1,:,:]
                img_2_com_batch1 = img_4[1,0,:,:]+1j*img_4[1,1,:,:]

                LQ_numpy = LQ.squeeze().float().cpu().numpy()

                complex1 = LQ_numpy[0,0,:,:]+1j*LQ_numpy[0,1,:,:]
                complex2 = LQ_numpy[1,0,:,:]+1j*LQ_numpy[1,1,:,:]
                # temp1 = np.fft.ifft2(complex1)
                # temp1 = temp1/np.max(np.abs(temp1))
                # temp2 = np.fft.ifft2(complex2)
                # temp2 = temp2/np.max(np.abs(temp2))

                write_images(np.abs(img_2_com_batch0), os.path.join('/zw_code/IR_SDE/lingshi_test/codes/config/inpainting/trainImg', f'{current_step}ori_k_w_batch0'+'.png'))
                write_images(np.abs(img_2_com_batch1), os.path.join('/zw_code/IR_SDE/lingshi_test/codes/config/inpainting/trainImg', f'{current_step}ori_k_w_batch1'+'.png'))
                write_images(np.abs(complex1), os.path.join('/zw_code/IR_SDE/lingshi_test/codes/config/inpainting/trainImg', f'{current_step}sample_k_w_batch0'+'.png'))
                write_images(np.abs(complex2), os.path.join('/zw_code/IR_SDE/lingshi_test/codes/config/inpainting/trainImg', f'{current_step}sample_K_w_batch1'+'.png'))

            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation, to produce ker_map_list(fake)
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
            # if a == 1:
                avg_psnr = 0.0
                idx = 0
                for _, val_data in enumerate(val_loader):
                    # print('ok')
                    GT_k = val_data["GT"]#w
                    k_w_8_partial = np.zeros((int(GT_k.shape[0]), 8, 256, 256), dtype=np.float32)
                    k_w_8_ori = np.zeros((int(GT_k.shape[0]), 8, 256, 256), dtype=np.float32)
                    for j in range(int(GT_k.shape[0])):
                        for i in range(4):
                            k_w_8_partial[j, i, :, :] = np.multiply(mask_muti[i, :, :], GT_k[j, 0, :, :])
                            k_w_8_partial[j, i + 4, :, :] = np.multiply(mask_muti[i, :, :], GT_k[j, 1, :, :])
                            # k_w_8_ori[j, i, :, :] = GT_k[j, 0, :, :]
                            # k_w_8_ori[j, i + 4, :, :] = GT_k[j, 1, :, :]






                    LQ_k = util.multi_mri_mask_to(k_w_8_partial, mask_root, mask_id = -1)["black"]
                    k_w_8_partial = torch.tensor(k_w_8_partial, dtype=torch.float32).cuda()
                    LQ_k = torch.tensor(LQ_k, dtype=torch.float32).cuda()
                    # LQ_k = util.multi_mri_mask_to(GT_k, mask_root, mask_id = idx)["black"]
                    noisy_state = sde.noise_state(LQ_k)

                    # valid Predictor
                    model.feed_data(noisy_state, LQ_k, k_w_8_partial)
                    model.test(sde,mode='val')
                    visuals = model.get_current_visuals()

                    output_k = visuals['Output'].squeeze().float().cpu().numpy()# 2,256,256

                    output_k_complex = np.zeros((256, 256), dtype=np.complex64)
                    output_k_4_complex = np.zeros((int(output_k.shape[0] / 2), 256, 256), dtype=np.complex64)
                    for i in range(output_k_4_complex.shape[0]):
                        output_k_4_complex[i, :, :] = output_k[i, :, :] + 1j * output_k[i + 4, :, :]
                        output_k_complex = output_k_4_complex[i, :, :] + output_k_complex

                    output_k_complex = np.multiply(output_k_complex, 1 / mask_sum)

                    # print(type(visuals['Output'].squeeze()))
                    # print(type(output_k))
                    # print(output_k.shape)
                    # output_k_complex = output_k[0, :, :] + 1j*output_k[1, :, :]

                    ori_k_no_w_complex = val_data["GT_k_no_w"].squeeze()#batch,256,256->256,256
                    # print(type(ori_k_no_w_complex))

                    ori_k_no_w = torch.stack((torch.real(ori_k_no_w_complex), torch.imag(ori_k_no_w_complex)), 0)
                    sample_k_no_w = util.multi_mri_mask_to(ori_k_no_w, mask_root, mask_id=idx)["black"].squeeze().float().cpu().numpy()
                    # sample_k_no_w = util.multi_mri_mask_to(ori_k_no_w, mask_root, mask_id=idx)["black"].squeeze().float().cpu().numpy()
                    sample_k_no_w_complex = sample_k_no_w[0, :, :] + 1j*sample_k_no_w[1, :, :]
                    # print(ori_k_no_w.shape,sample_k_no_w.shape,sample_k_no_w_complex.shape)
                    zf_img = np.fft.ifft2(sample_k_no_w_complex)

                    weight = loadmat(os.path.join(weight_root, 'weight1_GEBrain.mat'))['weight']
                    assert weight is not None , 'no weight'
                    # print(output_k_complex.shape,weight.shape,sample_k_no_w_complex.shape)

                    output_k_complex_no_w = wgt2k(output_k_complex, weight, sample_k_no_w_complex)
                    # output_k_complex_no_w = sample_k_no_w_complex + (1 - mask)
                    output_img = np.fft.ifft2(output_k_complex_no_w)
                    # output_img = output_img/np.max(np.abs(output_img))#complex;256,256

                    # gt_k = visuals['GT'].squeeze().float().cpu().numpy()# 2,256,256
                    # gt_k_complex = gt_k[0, :, :] + 1j * gt_k[1, :, :]
                    gt_img = np.fft.ifft2(ori_k_no_w_complex.cpu().numpy())
                    # gt_img = gt_img/np.max(np.abs(gt_img))#complex;256,256

                    # print(output_img.shape,gt_img.shape)


                    #output = util.tensor2img(visuals["Output"].squeeze())  # uint8
                    #gt_img = util.tensor2img(visuals["GT"].squeeze())  # uint8

                    # calculate PSNR
                    avg_psnr += compare_psnr(255*abs(output_img),255*abs(gt_img),data_range=255)
                    #print(avg_psnr)
                    #avg_psnr += util.calculate_psnr(output, gt_img)
                    idx += 1
                    write_images(np.abs(output_img), os.path.join(
                        '/zw_code/IR_SDE/lingshi_test/codes/config/inpainting/val_img/' + 'output_img' + '.png'))

                    if current_step in [5,10000,60000,130000]:
                        write_images(np.abs(gt_img), os.path.join(
                            '/zw_code/IR_SDE/lingshi_test/codes/config/inpainting/val_img/' + '_ori' + '.png'))
                        write_images(np.abs(zf_img), os.path.join(
                            '/zw_code/IR_SDE/lingshi_test/codes/config/inpainting/val_img/' + '_zf' + '.png'))
                        write_images(np.abs(output_img), os.path.join(
                            '/zw_code/IR_SDE/lingshi_test/codes/config/inpainting/val_img/' + '_rec' + '.png'))

                avg_psnr = avg_psnr / idx

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step

                # log
                logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, best_psnr, best_iter))
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    )
                )
                print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    ))
                # tensorboard logger
                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)

            if error.value:
                sys.exit(0)
            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    # model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


if __name__ == "__main__":
    main()
