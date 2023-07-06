# DL-NICE

## Dual-loss nonlinear independent component estimation
Code accompanying the paper "Dual-loss nonlinear independent component estimation for augmenting explainable vibration samples of rotating machinery faults" by Xiaoyun Gong, Mengxuan Hao, Chuan Li1, Wenliao Du, Wenbin He and Ziqiang Pu (Ready to be submitted for publication).

* Tensorflow 2.0 implementation
* Inspired by Laurent **_et al_**.
[nice.py](https://github.com/bojone/flow/blob/master/nice.py): NICE (Non-linear Independent Components Estimation, [intro]( https://kexue.fm/archives/5776))


# Requirements

* Python 3.6.0
* Keras == 2.8.0 
* Tensorflow == 2.8.0

# File description




* `DL-NICE`:  The model we build is the NICE loading time domain frequency domain dual loss.
* `F-NICE`:   The model we build is the NICE loading frequency domain loss only
* `T-NICE`:   The model we build is the NICE loading time domain loss only  
* `t_SNE_utils`:  t-SNE package for 2-D visualization.

# Implementation details

* The proposed method was implemented in the TensorFlow framework. The DLNICE model was trained using the provided training set. 
* During training, a batch size of 32 and a total of 30 iterations were employed. The optimizer used was Adam, with a learning rate of 0.002. 
* The activation function utilized in the model was the Leaky Rectified Lin-ear Unit (Leaky ReLU).

# Usage

## Model description
| Architecture                                                                                                 | Description |
| -----------                                                                                                  | ----------- |
| ![image](https://github.com/123MHao/DL-NICE/assets/102200358/fb72ce41-a64c-404c-b84d-19e8afd00b2e)           | The generator in NICE consists of an encoder and a decoder. During the decoding pro-cess, a new sample is generated by sampling the latent variable space and then mapping it back to the sample space using an inverse transformation, resulting in the generation of Gaussian noise.     |
| ![双栏老照片](https://github.com/123MHao/DL-NICE/assets/102200358/3d8ba0a1-ef93-4b8c-9e71-dada043f2710)       | The time domain loss is very sensitive to impulsiveness, while the frequency domain loss can effectively characterize cyclostationarity. Therefore, to fully express the fault character-istics of rotating machinery, it is better to express both impulsiveness and cyclostationarity, and DLNICE uses a dual loss function to solve this problem.        |

## Results on T-NICE, F-NICE and DL-NICE


|                               | T-NICE | F-NICE  | DL-NICE  |
| -----------            | ----------- |----------- | ----------- |
|  t-SNE     |  ![13 ](https://github.com/123MHao/DL-NICE/assets/102200358/fa6773f8-8a3e-459b-8aee-2e2e8b3af7ed) |![14 ](https://github.com/123MHao/DL-NICE/assets/102200358/fc0dffd3-4e1b-47f8-9aa8-1d26ace3f92b) |![15 ](https://github.com/123MHao/DL-NICE/assets/102200358/4cd3b98b-c224-490e-ba77-0c7e9e369a38) |    









# Acknowledgments

This work is supported in part by the National Natural Science Foundation of China (52175080, 52275138)，the Advanced Programs for Overseas Researchers of Henan Province (Grant No. 20221803), the Key R&D Projects in Henan Province (Grant No. 221111240200), and the Key Science and Technology Research Project of the Henan Province (Grant No. 232102221039).
