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
python3 separate_siat.py --exe SIAT_TRAIN --config siat_config.yml --checkpoint your save path
```

## Test
```bash
python3 separate_siat.py --exe SIAT_MULTICHANNEL --config siat_config.yml --model hggdp --test
```


## Graphical representation
 <div align="center"><img src="https://github.com/yqx7150/GM-SDE/blob/main/png/fig1.png" width = "1000" height = "600">  </div>
 
Performance exhibition of “multi-view noise” strategy. (a) Training sliced score matching (SSM) loss and validation loss for each iteration. (b) Image quality comparison on the brain dataset at 15% radial sampling: Reconstruction images, error maps (Red) and zoom-in results (Green).
</br>

 <div align="center"><img src="https://github.com/yqx7150/GM-SDE/blob/main/png/Fig2.png"> </div>

The pipeline of the prior learning and iterative reconstruction procedure in GM-SDE. Top: The training process involves learning k-space priors through noise networks. Bottom: The reconstruction process is characterized by iteratively eliminating aliasing artifacts and reconstructing intricate details using a numeri-cal solver for reverse SDE, low-rank constraint, and data consistency.


## Reconstruction Results Compared to DL-based Methods at R=4 using 2D Poisson Sampling Mask.
<div align="center"><img src="https://github.com/yqx7150/GM-SDE/blob/main/png/Fig6.png" width = "800" height = "800"> </div>

The reconstruction results of T2 Transverse Brain data at 2D poisson sam-pling pattern with acceleration factors of 4. The first row shows the full-sampled, under-sampled, and the reconstruction of EBMRec, HGGDP, and GM-SDE. The second row shows the corresponding error maps of the re-construction.


## Reconstruction Results Compared to SDE-based Methods at R=10 using 2D Poisson Sampling Mask.
<div align="center"><img src="https://github.com/yqx7150/GM-SDE/blob/main/png/Fig7.png" width = "800" height = "800"> </div>

Results by full-sampled, under-sampled, WKGM, HFS-SDE, and GM-SDE on T1 GE Brain image at R=10 using 2D poisson sampling mask.







