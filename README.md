# GM-SDE

**Paper**: Diffusion model based on generalized map for accelerated MRI

**Authors**: Zengwei Xiao#, Yujuan Lu#, Binzhong He, Pinhuang Tan, Shanshan Wang, Xiaoling Xu*, Qiegen Liu*   

NMR in Biomedicine, https://doi.org/10.1002/nbm.5232   

Date : May-22-2024  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2024, Department of Electronic Information Engineering, Nanchang University.  

In recent years, diffusion models have made significant progress in accelerating magnetic resonance imaging. Nevertheless, it still has inherent limitations, such as prolonged iteration times and sluggish convergence rates. In this work, we present a novel generalized map generation model based on mean-reverting SDE, called GM-SDE, to alleviate these shortcomings. Notably, the core idea of GM-SDE is optimizing the initial values of the iterative algorithm. Specifically, the training process of GM-SDE diffuses the original k-space data to an intermediary degraded state with fixed Gaussian noise, while the reconstruction process generates the data by reversing this process. Based on the generalized map, three variants of GM-SDE are proposed to learn k-space data with different structural characteristics to improve the effectiveness of model training. GM-SDE also exhibits flexibility, as it can be integrated with traditional constraints, thereby further enhancing its overall performance. Experimental results showed that the proposed method can reduce reconstruction time and deliver excellent image reconstruction capabilities compared to the complete diffusion-based method.    


## Graphical representation
 <div align="center"><img src="https://github.com/yqx7150/GM-SDE/blob/main/png/Fig2.png" width = "400" height = "450">  </div>
 
Performance exhibition of “multi-view noise” strategy. (a) Training sliced score matching (SSM) loss and validation loss for each iteration. (b) Image quality comparison on the brain dataset at 15% radial sampling: Reconstruction images, error maps (Red) and zoom-in results (Green).

 <div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig7.png"> </div>

Pipeline of sampling from the high-dimensional noisy data distribution with multi-view noise and intermediate samples. (a) Conceptual dia-gram of the sampling on high-dimensional noisy data distribution with multi-view noise. (b) Intermediate samples of annealed Langevin dynamics.


## Reconstruction Results by Various Methods at 85% 2D Random Undersampling.
<div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig11.png"> </div>

Reconstruction comparison on pseudo radial sampling at acceleration factor 6.7 . Top: Reference, reconstruction by DLMRI, PANO, FDLCP; Bottom: Reconstruction by NLR-CS, DC-CNN, EDAEPRec, HGGDPRec. Green and red boxes illustrate the zoom in results and error maps, respectively.






