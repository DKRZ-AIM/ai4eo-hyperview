# AI4EO Hyperview

This repository contains the contribution of the team EagleEyes to the ![AI4EO Hyperview Machine Learning Challenge](https://platform.ai4eo.eu/seeing-beyond-the-visible)

**Overview**

The objective of the AI4EO HYPERVIEW challenge is to predict agriculturally relevant soil pa-
rameters (K, Mg, P2O5, pH) from airborne hyperspectral images. We present a hybrid model fusing
Random Forest and K-nearest neighbor regressors that exploit the average spectral reflectance, as
well as derived features such as gradients, wavelet coeﬀicients, and Fourier transforms. The solution
is computationally lightweight and improves upon the challenge baseline by 21%.

**This Repository contains the following:**

* A jupyter notebook containing the final solution can be found in ![final-submission](final-submission).
* The folder ![notebooks](notebooks) contains some jupyter notebooks for data exploration and first simple models
* We explored several other approaches, which can be found in ![hyperview](hyperview)
    * Different Neural Network architectures (based on keras) ![NN keras](hyperview/keras])
    * ![PSELTAE model](https://github.com/VSainteuf/pytorch-psetae) (based on pytorch-lightning) ![PSELTAE](hyperview/pytorch_lightning) 
    * ![Random Forest and XGBoost models](random-forest)
