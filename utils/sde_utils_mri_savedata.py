import math
import torch
import abc
from tqdm import tqdm
import torchvision.utils as tvutils
import os
from scipy import integrate
import cv2
try:
    from skimage.measure import compare_psnr,compare_ssim
except:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
import os.path
import os.path as osp
import numpy as np
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
    with open(osp.join('./datasets/rec_kdata_train/log/',filedir),"w+") as f:#a+
        f.writelines(str(model_num)+' '+'['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.write('\n')
def write_Data_s(model_num,psnr,ssim):
    filedir="result_ssim_main.txt"
    with open(osp.join('./datasets/rec_kdata_train/log/',filedir),"w+") as f:#a+
        f.writelines(str(model_num)+' '+'['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.write('\n')        
def write_Data2(psnr,ssim):
    filedir="PC.txt"
    with open(osp.join('./datasets/rec_kdata_train/log/',filedir),"a+") as f:#a+
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

def L_SVD(A, k):
    svd_input  = torch.tensor(A, dtype=torch.complex64)
    U,S,V = torch.svd(svd_input)
    
    #svd_input  = np.array(A, dtype=np.complex64)
    #U,S,V = np.linalg.svd(svd_input)
    S = torch.diag(S)
    # U = np.array(U.resolve_conj().numpy(),dtype=np.complex64)
    # S = np.array(S.resolve_conj().numpy(),dtype=np.complex64)
    # V = np.array(V.resolve_conj().numpy(),dtype=np.complex64)
    U = np.array(U,dtype=np.complex64)
    S = np.array(S,dtype=np.complex64)
    V = np.array(V,dtype=np.complex64)      
    uu = U[:, 0:k]
    ss = S[0:k, 0:k]
    vv = V[:, 0:k]
    return uu, ss, vv

class SDE(abc.ABC):
    def __init__(self, T, device=None):
        self.T = T
        self.dt = 1 / T
        self.device = device

    @abc.abstractmethod
    def drift(self, x, t):
        pass

    @abc.abstractmethod
    def dispersion(self, x, t):
        pass

    @abc.abstractmethod
    def sde_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def ode_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def score_fn(self, x, t):
        pass

    ################################################################################

    def forward_step(self, x, t):
        return x + self.drift(x, t) + self.dispersion(x, t)

    def reverse_sde_step_mean(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t)

    def reverse_sde_step(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t) - self.dispersion(x, t)

    def reverse_ode_step(self, x, score, t):
        return x - self.ode_reverse_drift(x, score, t)

    def forward(self, x0, T=-1):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

        return x

    def reverse_sde(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_sde_step(x, score, t)

        return x

    def reverse_ode(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)

        return x


#############################################################################


class IRSDE(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''
    def __init__(self, max_sigma, T=100, schedule='cosine', eps=0.01,  device=None):
        super().__init__(T, device)
        self.max_sigma = max_sigma / 255 if max_sigma >= 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule, eps)

    def _initialize(self, max_sigma, T, schedule, eps=0.01):

        def constant_theta_schedule(timesteps, v=1.):
            """
            constant schedule
            """
            print('constant schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            return torch.ones(timesteps, dtype=torch.float32)

        def linear_theta_schedule(timesteps):
            """
            linear schedule
            """
            print('linear schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        def cosine_theta_schedule(timesteps, s = 0.008):
            """
            cosine schedule
            """
            print('cosine schedule')
            timesteps = timesteps + 2 # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma**2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma**2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))
            
        if schedule == 'cosine':
            thetas = cosine_theta_schedule(T)
        elif schedule == 'linear':
            thetas = linear_theta_schedule(T)
        elif schedule == 'constant':
            thetas = constant_theta_schedule(T)
        else:
            print('Not implemented such schedule yet!!!')

        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0] # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)
        
        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

        self.mu = 0.
        self.model = None

    #####################################

    # set mu for different cases
    def set_mu(self, mu):
        self.mu = mu

    # set score model for reverse process
    def set_model(self, model):
        self.model = model

    #####################################

    def mu_bar(self, x0, t):
        return self.mu + (x0 - self.mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def drift(self, x, t):
        return self.thetas[t] * (self.mu - x) * self.dt

    def sde_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - self.sigmas[t]**2 * score) * self.dt

    def ode_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - 0.5 * self.sigmas[t]**2 * score) * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def score_fn(self, x, t):
        # need to pre-set mu and score_model
        noise = self.model(x, self.mu, t)
        # noise = self.model(x, self.mu, t)

        return self.get_score_from_noise(noise, t)


    def noise_fn(self, x, t):
        # need to pre-set mu and score_model
        return self.model(x, self.mu, t)

    # optimum x_{t-1}
    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t-1] * self.dt)

        term1 = A * (1 - C**2) / (1 - B**2)
        term2 = C * (1 - A**2) / (1 - B**2)

        return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def get_real_noise(self, xt, x0, t):
        return (xt - self.mu_bar(x0, t)) / self.sigma_bar(t)

    def get_real_score(self, xt, x0, t):
        return -(xt - self.mu_bar(x0, t)) / self.sigma_bar(t)**2

    # forward process to get x(T) from x(0)
    def forward(self, x0, T=-1, save_dir='forward_state'):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{t}.png', normalize=False)
        return x

    def reverse_sde(self, xt, T=-1, save_states=False, save_dir='sde_state',ori = None, sample_k = None, weight = None,
                    opt=None,model=None,img_name=None,mask = None):
        
        device = model.device
        # img_name = os.path.splitext(os.path.basename(opt["forgetname"]))[0]
        T = self.T if T < 0 else T
        x_mean = xt.clone()
        '''
        ori_data = ori
        coil = ori_data.shape[0]

        ori_data_sos = np.sqrt(np.sum(np.square(np.abs(ori_data)),axis=0)) 
        ori_data_sos = ori_data_sos/np.max(np.abs(ori_data_sos))
        # write_images(np.abs(ori_data_sos), os.path.join(dataset_dir, img_name + '_ori'+'.png'))
        write_images(np.abs(ori_data_sos),osp.join('./Res/',img_name+'_ori'+'.png'))

        Kdata = np.zeros((coil,256,256),dtype=np.complex64)
        Ksample = np.zeros((coil,256,256),dtype=np.complex64)
        zeorfilled_data = np.zeros((coil,256,256),dtype=np.complex64)
        k_w = np.zeros((coil,256,256),dtype=np.complex64)
        Kdata_w = np.zeros((coil,256,256),dtype=np.complex64)

        for i in range(coil):
            Kdata[i,:,:] = np.fft.fftshift(np.fft.fft2(ori_data[i,:,:]))
            Kdata_w[i,:,:] = k2wgt(Kdata[i,:,:],weight[i,:,:])
            Ksample[i,:,:] = np.multiply(mask[i,:,:],Kdata[i,:,:])
            k_w[i,:,:] = k2wgt(Ksample[i,:,:],weight[i,:,:])           
            zeorfilled_data[i,:,:] = np.fft.ifft2(Ksample[i,:,:])
        zeorfilled_data_sos = np.sqrt(np.sum(np.square(np.abs(zeorfilled_data)),axis=0))
        zeorfilled_data_sos = zeorfilled_data_sos/np.max(np.abs(zeorfilled_data_sos))
        psnr_zf=compare_psnr(255*abs(zeorfilled_data_sos),255*abs(ori_data_sos),data_range=255)
        ssim_zf=compare_ssim(abs(zeorfilled_data_sos),abs(ori_data_sos),data_range=1)
        print('psnr_zero: ',psnr_zf,'ssim_zero: ',ssim_zf)
        # write_images(abs(zeorfilled_data_sos),osp.join('./result/parallel_12ch/'+'Zeorfilled_'+str(round(psnr_zero, 2))+str(round(ssim_zero, 4))+'.png'))
        # write_images(np.abs(zeorfilled_data_sos), os.path.join(dataset_dir,img_name+'_zf_'+str(round(psnr_zf, 2))+str(round(ssim_zf, 4))+'.png'))
        write_images(np.abs(zeorfilled_data_sos),osp.join('./Res/',img_name+'_zf_'+str(round(psnr_zf, 2))+str(round(ssim_zf, 4))+'.png'))
        '''
        zf_img=np.fft.ifft2(sample_k)
        zf_img=zf_img/np.max(np.abs(zf_img))

        psnr_zf=compare_psnr(255*abs(zf_img),255*abs(ori),data_range=255)
        ssim_zf=compare_ssim(abs(zf_img),abs(ori),data_range=1)
        print('psnr_zero: ',psnr_zf,'ssim_zero: ',ssim_zf)
        max_psnr = 0
        max_ssim = 0



        # x_input=np.stack((np.real(k_w),np.imag(k_w)),1)
        # x_mean = torch.tensor(x_input, dtype=torch.float32).cuda()#12,2,256,256



        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x_mean, t)
            x_mean = self.reverse_sde_step(x_mean, score, t)

            x_mean = x_mean.squeeze().cpu().numpy()

            x_complex = x_mean[0,:,:]+1j*x_mean[1,:,:]

            rec_k = wgt2k(x_complex,weight,sample_k)
            rec_k = sample_k + rec_k * (1-mask)

            x_temp = k2wgt(rec_k,weight)
            x_temp = np.stack((np.real(x_temp), np.imag(x_temp)),0)

            x_mean = torch.tensor(x_temp[None, ...],dtype=torch.float32).cuda()    
            x_mean = x_mean.to(device)

            rec_img = np.fft.ifft2(rec_k)

            rec_img = rec_img/np.max(np.abs(rec_img))

            
            # write_images(np.abs(rec_Image_sos), os.path.join(dataset_dir, img_name + '_rec'+'.png'))


            psnr = compare_psnr(255*np.abs(rec_img),255*np.abs(ori),data_range=255)
            ssim = compare_ssim(np.abs(rec_img),np.abs(ori),data_range=1)

            print(' PSNR:', psnr,' SSIM:', ssim) 
            # write_Data2(psnr,ssim)#@@@@

            if max_ssim <= ssim:
                max_ssim = ssim
                max_ssim_psnr = psnr
                # write_Data_s('checkpoint',psnr,ssim) #@@@@
                # write_images(np.abs(rec_img),osp.join('./datasets/rec_kdata_train/png/',img_name+'Rec_ssim_main'+'.png'))
                #savemat(osp.join('./Res/'+'rec_s.mat'),{'Img':rec_Image})
            if max_psnr <= psnr:
                max_psnr = psnr
                max_psnr_ssim = ssim
                # write_Data_p('checkpoint',psnr,ssim)#@@@@
                # write_images(np.abs(rec_img),osp.join('./datasets/rec_kdata_train/png/',img_name+'Rec_psnr_main'+'.png'))#@@@@
                #savemat(osp.join('./Res/'+'rec_p.mat'),{'Img':rec_Image})
                savedata_k = rec_k

            if save_states: # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x_mean.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x_mean,{"psnr":max_psnr,"ssim":max_psnr_ssim,"zf_psnr":psnr_zf,"zf_ssim":ssim_zf,"kdata":savedata_k}

    def reverse_ode(self, xt, T=-1, save_states=False, save_dir='ode_state'):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)

            if save_states: # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x


    # sample ode using Black-box ODE solver (not used)
    def ode_sampler(self, xt, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3,):
        shape = xt.shape

        def to_flattened_numpy(x):
          """Flatten a torch tensor `x` and convert it to numpy."""
          return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
          """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
          return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x):
            t = int(t)
            x = from_flattened_numpy(x, shape).to(self.device).type(torch.float32)
            score = self.score_fn(x, t)
            drift = self.ode_reverse_drift(x, score, t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(xt),
                                     rtol=rtol, atol=atol, method=method)

        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        return x

    def optimal_reverse(self, xt, x0, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            x = self.reverse_optimum_step(x, x0, t)

        return x

    ################################################################

    def weights(self, t):
        return torch.exp(-self.thetas_cumsum[t] * self.dt)

    # sample states for training
    def generate_random_states(self, x0, mu):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

        state_mean = self.mu_bar(x0, timesteps)
        noises = torch.randn_like(state_mean)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + state_mean

        return timesteps, noisy_states.to(torch.float32)

    def noise_state(self, tensor):
        return tensor + torch.randn_like(tensor) * self.max_sigma
    # def noise_state(self, tensor):
    #     return torch.randn_like(tensor) * self.max_sigma




################################################################################
################################################################################
############################ Denoising SDE ##################################
################################################################################
################################################################################


class DenoisingSDE(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''
    def __init__(self, max_sigma, T, schedule='cosine', device=None):
        super().__init__(T, device)
        self.max_sigma = max_sigma / 255 if max_sigma > 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule)

    def _initialize(self, max_sigma, T, schedule, eps=0.04):

        def linear_beta_schedule(timesteps):
            timesteps = timesteps + 1
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)

        def cosine_beta_schedule(timesteps, s = 0.008):
            """
            cosine schedule
            as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
            """
            timesteps = timesteps + 2
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype = torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            # betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma**2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma**2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))
        
        if schedule == 'cosine':
            thetas = cosine_beta_schedule(T)
        else:
            thetas = linear_beta_schedule(T)    
        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0]
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)
        
        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

        self.mu = 0.
        self.model = None

    # set noise model for reverse process
    def set_model(self, model):
        self.model = model

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def mu_bar(self, x0, t):
        return x0

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def drift(self, x, x0, t):
        return self.thetas[t] * (x0 - x) * self.dt

    def sde_reverse_drift(self, x, score, t):
        A = torch.exp(-2 * self.thetas_cumsum[t] * self.dt)
        return -0.5 * self.sigmas[t]**2 * (1 + A) * score * self.dt

    def ode_reverse_drift(self, x, score, t):
        A = torch.exp(-2 * self.thetas_cumsum[t] * self.dt)
        return -0.5 * self.sigmas[t]**2 * A * score * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def get_init_state_from_noise(self, x, noise, t):
        return x - self.sigma_bar(t) * noise

    def get_init_state_from_score(self, x, score, t):
        return x + self.sigma_bar(t)**2 * score

    def score_fn(self, x, t):
        # need to preset the score_model
        noise = self.model(x, t)
        return self.get_score_from_noise(noise, t)

    ############### reverse sampling ################

    def get_real_noise(self, xt, x0, t):
        return (xt - self.mu_bar(x0, t)) / self.sigma_bar(t)

    def get_real_score(self, xt, x0, t):
        return -(xt - self.mu_bar(x0, t)) / self.sigma_bar(t)**2

    def reverse_sde(self, xt, x0=None, T=-1, save_states=False, save_dir='sde_state'):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            if x0 is not None:
                score = self.get_real_score(x, x0, t)
            else:
                score = self.score_fn(x, t)
            x = self.reverse_sde_step(x, score, t)

            if save_states:
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    def reverse_ode(self, xt, x0=None, T=-1, save_states=False, save_dir='ode_state'):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            if x0 is not None:
                real_score = self.get_real_score(x, x0, t)

            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)

            if save_states:
                interval = self.T // 100
                if t % interval == 0:
                    state = x.clone()
                    if x0 is not None:
                        state = torch.cat([x, score, real_score], dim=0)
                    os.makedirs(save_dir, exist_ok=True)
                    idx = t // interval
                    tvutils.save_image(state.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    def ode_sampler(self, xt, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3,):
        shape = xt.shape

        def to_flattened_numpy(x):
          """Flatten a torch tensor `x` and convert it to numpy."""
          return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
          """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
          return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x):
            t = int(t)
            x = from_flattened_numpy(x, shape).to(self.device).type(torch.float32)
            score = self.score_fn(x, t)
            drift = self.ode_reverse_drift(x, score, t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(xt),
                                     rtol=rtol, atol=atol, method=method)

        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        return x

    def get_optimal_timestep(self, sigma, eps=1e-6):
        sigma = sigma / 255 if sigma > 1 else sigma
        thetas_cumsum_hat = -1 / (2 * self.dt) * math.log(1 - sigma**2/self.max_sigma**2 + eps)
        T = torch.argmin((self.thetas_cumsum - thetas_cumsum_hat).abs())
        return T


    ##########################################################
    ########## below functions are used for training #########
    ##########################################################

    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t-1] * self.dt)

        term1 = A * (1 - C**2) / (1 - B**2)
        term2 = C * (1 - A**2) / (1 - B**2)

        return term1 * (xt - x0) + term2 * (x0 - x0) + x0

    def optimal_reverse(self, xt, x0, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            x = self.reverse_optimum_step(x, x0, t)

        return x

    def weights(self, t):
        # return 0.1 + torch.exp(-self.thetas_cumsum[t] * self.dt)
        return self.sigmas[t]**2

    def generate_random_states(self, x0):
        x0 = x0.to(self.device)

        batch = x0.shape[0]
        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

        noises = torch.randn_like(x0, dtype=torch.float32)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + x0

        return timesteps, noisy_states

