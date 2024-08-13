# GM-SDE

**Paper**: Diffusion model based on generalized map for accelerated MRI

**Authors**: Zengwei Xiao#, Yujuan Lu#, Binzhong He, Pinhuang Tan, Shanshan Wang, Xiaoling Xu*, Qiegen Liu*   

NMR in Biomedicine, https://doi.org/10.1002/nbm.5232   

Date : May-22-2024  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2024, Department of Electronic Information Engineering, Nanchang University.  

In recent years, diffusion models have made significant progress in accelerating magnetic resonance imaging. Nevertheless, it still has inherent limitations, such as prolonged iteration times and sluggish convergence rates. In this work, we present a novel generalized map generation model based on mean-reverting SDE, called GM-SDE, to alleviate these shortcomings. Notably, the core idea of GM-SDE is optimizing the initial values of the iterative algorithm. Specifically, the training process of GM-SDE diffuses the original k-space data to an intermediary degraded state with fixed Gaussian noise, while the reconstruction process generates the data by reversing this process. Based on the generalized map, three variants of GM-SDE are proposed to learn k-space data with different structural characteristics to improve the effectiveness of model training. GM-SDE also exhibits flexibility, as it can be integrated with traditional constraints, thereby further enhancing its overall performance. Experimental results showed that the proposed method can reduce reconstruction time and deliver excellent image reconstruction capabilities compared to the complete diffusion-based method.    


## Training
```bash
python train.py -opt=options/train/ir-sde.yml
```

## Test
```bash
python test.py -opt=options/test/ir-sde.yml
```


## Graphical representation
 <div align="center"><img src="https://github.com/yqx7150/GM-SDE/blob/main/png/fig1.png" width = "900" height = "540">  </div>
 
Different constructions of <i>ϕ<sup>{1,2,3}</sup><sub>0</sub>(x<sub>w</sub>)</i> and <i>ϕ<sup>{1,2,3}</sup><sub>T</sub>(x<sub>w</sub>)</i> correspond to the three variants of GM-SDE. <i>ϕ<sup>{1,2,3}</sup><sub>0</sub>(x<sub>w</sub>)</i> denotes the input of the network comprising the ground truth (GT) original data, and <i>ϕ<sup>{1,2,3}</sup><sub>T</sub>(x<sub>w</sub>)</i> represents the degraded image (LQ) augmented with Gaussian noise distribution. <i>w</i> stands for the weight operator. The proposed method diffuses <i>ϕ<sup>{1,2,3}</sup><sub>0</sub>(x<sub>w</sub>)</i> into the <i>ϕ<sup>{1,2,3}</sup><sub>T</sub>(x<sub>w</sub>)</i> data by gradually injecting noise and reconstructs the data by reversing the process.

</br>

 <div align="center"><img src="https://github.com/yqx7150/GM-SDE/blob/main/png/Fig2.png" width = "1000" height = "540"> </div>

The pipeline of the prior learning and iterative reconstruction procedure in GM-SDE. Top: The training process involves learning k-space priors through noise networks. Bottom: The reconstruction process is characterized by iteratively eliminating aliasing artifacts and reconstructing intricate details using a numeri-cal solver for reverse SDE, low-rank constraint, and data consistency.


## Reconstruction Results Compared to DL-based Methods at R=4 using 2D Poisson Sampling Mask.
<div align="center"><img src="https://github.com/yqx7150/GM-SDE/blob/main/png/Fig6.png" width = "804" height = "552"> </div>

The reconstruction results of T2 Transverse Brain data at 2D poisson sam-pling pattern with acceleration factors of 4. The first row shows the full-sampled, under-sampled, and the reconstruction of EBMRec, HGGDP, and GM-SDE. The second row shows the corresponding error maps of the re-construction.


## Reconstruction Results Compared to SDE-based Methods at R=10 using 2D Poisson Sampling Mask.
<div align="center"><img src="https://github.com/yqx7150/GM-SDE/blob/main/png/Fig7.png" width = "804" height = "552"> </div>

Results by full-sampled, under-sampled, WKGM, HFS-SDE, and GM-SDE on T1 GE Brain image at R=10 using 2D poisson sampling mask.


## Other Related Projects
  * Multi-Channel and Multi-Model-Based Autoencoding Prior for Grayscale Image Restoration  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8782831)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MEDAEP)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Highly Undersampled Magnetic Resonance Imaging Reconstruction using Autoencoding Priors  
[<font size=5>**[Paper]**</font>](https://cardiacmr.hms.harvard.edu/files/cardiacmr/files/liu2019.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDAEPRec)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide) [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * High-dimensional Embedding Network Derived Prior for Compressive Sensing MRI Reconstruction  
 [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300815?via%3Dihub)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDMSPRec)
 
  * Denoising Auto-encoding Priors in Undecimated Wavelet Domain for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S0925231221000990) [<font size=5>**[Paper]**</font>](https://arxiv.org/ftp/arxiv/papers/1909/1909.01108.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/WDAEPRec)

  * Complex-valued MRI data from SIAT--test31 [<font size=5>**[Data]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/test_data_31)
  * More explanations with regard to the MoDL test datasets, we use some data from the test dataset in "dataset.hdf5" file, where the image slice numbers are 40,48,56,64,72,80,88,96,104,112(https://drive.google.com/file/d/1qp-l9kJbRfQU1W5wCjOQZi7I3T6jwA37/view)
  * DDP Method Link [<font size=5>**[DDP Code]**</font>](https://github.com/kctezcan/ddp_recon)
  * MoDL Method Link [<font size=5>**[MoDL code]**</font>](https://github.com/hkaggarwal/modl)
  * Complex-valued MRI data from SIAT--SIAT_MRIdata200 [<font size=5>**[Data]**</font>](https://github.com/yqx7150/SIAT_MRIdata200)  
  * Complex-valued MRI data from SIAT--SIAT_MRIdata500-singlecoil [<font size=5>**[Data]**</font>](https://github.com/yqx7150/SIAT500data-singlecoil)   
  * Complex-valued MRI data from SIAT--SIAT_MRIdata500-12coils [<font size=5>**[Data]**</font>](https://github.com/yqx7150/SIAT500data-12coils)    
 
  * Learning Multi-Denoising Autoencoding Priors for Image Super-Resolution  
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S1047320318302700)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MDAEP-SR)

  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9703672)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)

  * Universal Generative Modeling for Calibration-free Parallel MR Imaging  
[<font size=5>**[Paper]**</font>](https://biomedicalimaging.org/2022/)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/UGM-PI)   [<font size=5>**[Poster]**</font>](https://github.com/yqx7150/UGM-PI/blob/main/paper%20%23160-Poster.pdf)

* Progressive Colorization via Interative Generative Models  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9258392)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

* Joint Intensity-Gradient Guided Generative Modeling for Colorization
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2012.14130)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/JGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

* Diffusion Models for Medical Imaging
[<font size=5>**[Paper]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT)  
    
 * One-shot Generative Prior in Hankel-k-space for Parallel Imaging Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10158730)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HKGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT)
    
* Lens-less imaging via score-based generative model (基于分数匹配生成模型的无透镜成像方法)
[<font size=5>**[Paper]**</font>](https://www.opticsjournal.net/M/Articles/OJf1842c2819a4fa2e/Abstract)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/LSGM)





