### Seg_Uncertainty

![](https://github.com/layumi/Seg_Uncertainty/blob/master/Visual.jpg)

In this repo, we provide the code for the two papers, i.e., 

- MRNet：[Unsupervised Scene Adaptation with Memory Regularization in vivo](https://arxiv.org/pdf/1912.11164.pdf), IJCAI (2020)

- MRNet+Rectifying: [Rectifying Pseudo Label Learning via Uncertainty Estimation for Domain Adaptive Semantic Segmentation](https://arxiv.org/pdf/2003.03773.pdf), arXiv (2020)

### Prerequisites
- Python 3.6
- GPU Memory >= 11G (e.g., GTX2080Ti or GTX1080Ti)
- Pytorch 


### Prepare Data
Download [GTA5] and [Cityscapes] to run the basic code.
Alternatively, you could download extra two datasets from [SYNTHIA] and [OxfordRobotCar].

- Download [The GTA5 Dataset]( https://download.visinf.tu-darmstadt.de/data/from_games/ )

- Download [The SYNTHIA Dataset]( http://synthia-dataset.net/download-2/ )

- Download [The Cityscapes Dataset]( https://www.cityscapes-dataset.com/ )

- Download [The imagenet pretraind model]( https://drive.google.com/open?id=13kjtX481LdtgJcpqD3oROabZyhGLSBm2 )

The data folder is structured as follows:
```
├── data/
│   ├── Cityscapes/  
|   |   ├── data/
|   |       ├── gtFine/
|   |       ├── leftImg8bit/
│   ├── GTA5/
|   |   ├── images/
|   |   ├── labels/
|   |   ├── ...
│   ├── synthia/ 
|   |   ├── RGB/
|   |   ├── GT/
|   |   ├── Depth/
|   |   ├── ...
│   └── Oxford_Robot_ICCV19
|   |   ├── train/
|   |   ├── ...
```

### Training 
Stage-I:
```bash
python train_ms.py --snapshot-dir ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5  --drop 0.1 --warm-up 5000 --batch-size 2 --learning-rate 2e-4 --crop-size 1024,512 --lambda-seg 0.5  --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001   --lambda-me-target 0  --lambda-kl-target 0.1  --norm-style gn  --class-balance  --only-hard-label 80  --max-value 7  --gpu-ids 0,1  --often-balance  --use-se  
```

Generate Pseudo Label:
```bash
python generate_plabel_cityscapes.py  --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5/GTA5_25000.pth
```

Stage-II (with recitfying pseudo label):
```bash
python train_ft.py --snapshot-dir ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.8_Noaug --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5/GTA5_25000.pth --drop 0.2 --warm-up 5000 --batch-size 9 --learning-rate 1e-4 --crop-size 512,256 --lambda-seg 0.5 --lambda-adv-target1 0 --lambda-adv-target2 0 --lambda-me-target 0 --lambda-kl-target 0 --norm-style gn --class-balance --only-hard-label 80 --max-value 7 --gpu-ids 0,1,2 --often-balance  --use-se  --input-size 1280,640  --train_bn  --autoaug False
```
*** If you want to run the code without recitfying pseudo label, please change [[this line]](https://github.com/layumi/Seg_Uncertainty/blob/master/train_ft.py#L20) to 'from trainer_ms import AD_Trainer', which would apply the conventional pseudo label learning. ***

### Testing
```bash
python evaluate_cityscapes.py --restore-from ./snapshots/1280x640_restore_ft_GN_batchsize9_512x256_pp_ms_me0_classbalance7_kl0_lr1_drop0.2_seg0.5_BN_80_255_0.8_Noaug/GTA5_25000.pth
```

### Trained Model
I will provide the link soon. 

### The Key Code
Core code is relatively simple, and could be directly applied to other works. 
- Memory in vivo:  https://github.com/layumi/Seg_Uncertainty/blob/master/trainer_ms.py#L232

- Recitfying Pseudo label:  https://github.com/layumi/Seg_Uncertainty/blob/master/trainer_ms_variance.py#L166

