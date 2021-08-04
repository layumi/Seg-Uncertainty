## Seg_Uncertainty
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

![](https://github.com/layumi/Seg_Uncertainty/blob/master/Visual.jpg)

In this repo, we provide the code for the two papers, i.e., 

- MRNet：[Unsupervised Scene Adaptation with Memory Regularization in vivo](https://arxiv.org/pdf/1912.11164.pdf), IJCAI (2020)

- MRNet+Rectifying: [Rectifying Pseudo Label Learning via Uncertainty Estimation for Domain Adaptive Semantic Segmentation](https://arxiv.org/pdf/2003.03773.pdf), IJCV (2021) [[中文介绍]](https://zhuanlan.zhihu.com/p/130220572)

## Table of contents
* [CommonQ&A](#Common-Q&A)
* [Prerequisites](#prerequisites)
* [Prepare Data](#prepare-data)
* [Training](#training)
* [Testing](#testing)
* [Trained Model](#trained-model)
* [The Key Code](#the-key-code)
* [Related Works](#related-works)
* [Citation](#citation)

### Common Q&A 
1. Why KLDivergence is always non-negative (>=0)?

Please check the wikipedia at (https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Properties) . It provides one good demonstration. 

2. Why both log_sm and sm are used?

You may check the pytorch doc at https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html?highlight=nn%20kldivloss#torch.nn.KLDivLoss. 
I follow the discussion at https://discuss.pytorch.org/t/kl-divergence-loss/65393

### Prerequisites
- Python 3.6
- GPU Memory >= 11G (e.g., GTX2080Ti or GTX1080Ti)
- Pytorch or [Paddlepaddle](https://www.paddlepaddle.org.cn/)


### Prepare Data
Download [GTA5] and [Cityscapes] to run the basic code.
Alternatively, you could download extra two datasets from [SYNTHIA] and [OxfordRobotCar].

- Download [The GTA5 Dataset]( https://download.visinf.tu-darmstadt.de/data/from_games/ )

- Download [The SYNTHIA Dataset]( http://synthia-dataset.net/download/808/)  SYNTHIA-RAND-CITYSCAPES (CVPR16)

- Download [The Cityscapes Dataset]( https://www.cityscapes-dataset.com/ )

- Download [The Oxford RobotCar Dataset]( http://www.nec-labs.com/~mas/adapt-seg/adapt-seg.html
