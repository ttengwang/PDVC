# PDVC
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/end-to-end-dense-video-captioning-with/dense-video-captioning-on-youcook2)](https://paperswithcode.com/sota/dense-video-captioning-on-youcook2?p=end-to-end-dense-video-captioning-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/end-to-end-dense-video-captioning-with/dense-video-captioning-on-activitynet)](https://paperswithcode.com/sota/dense-video-captioning-on-activitynet?p=end-to-end-dense-video-captioning-with)

Official implementation for End-to-End Dense Video Captioning with Parallel Decoding (ICCV 2021) 

[[paper]](https://arxiv.org/abs/2108.07781) [[valse论文速递(Chinese)]](https://www.bilibili.com/video/BV1Xq4y1B7p7) 

**This repo supports:**
* two video captioning tasks: dense video captioning and video paragraph captioning
* two datasets: ActivityNet Captions and YouCook2
* video features containing C3D, TSN, and TSP.
* visualization of the generated captions of your own videos

**Table of Contents:**
* [Updates](#updates)
* [Introduction](#introduction)
* [Preparation](#preparation)
* [Running PDVC on Your Own Videos](#running-pdvc-on-your-own-videos)
* [Training and Validation](#training-and-validation)
  + [Download Video Features](#download-video-features)
  + [Dense Video Captioning](#dense-video-captioning)
  + [Video Paragraph Captioning](#video-paragraph-captioning)
* [Performance](#performance)
  + [Dense video captioning](#dense-video-captioning)
  + [Video paragraph captioning](#video-paragraph-captioning)
* [Citation](#citation)
* [Acknowledgement](#acknowledgement)



## Updates
- (2021.11.19) **add code for running PDVC on raw videos and visualize the generated captions (support Chinese and other non-English languages)**
- (2021.11.19) add pretrained models with [TSP features](https://github.com/HumamAlwassel/TSP). It achieves 9.03 METEOR(2021) and 6.05 SODA_c, a very competitive result on ActivityNet Captions without self-critical sequence training.
- (2021.08.29) add TSN pretrained models and support YouCook2

## Introduction
PDVC is a simple yet effective framework for end-to-end dense video captioning with parallel decoding (PDVC), by formulating the dense caption generation as a set prediction task. Without bells and whistles, extensive experiments on ActivityNet Captions and YouCook2 show that PDVC is capable of producing high-quality captioning results, surpassing the state-of-the-art methods when its localization accuracy is on par with them.
![pdvc.jpg](pdvc.jpg)

## Preparation
Environment: Linux,  GCC>=5.4, CUDA >= 9.2, Python>=3.7, PyTorch>=1.5.1

1. Clone the repo
```bash
git clone --recursive https://github.com/ttengwang/PDVC.git
```

2. Create virtual environment by conda
```bash
conda create -n PDVC python=3.7
source activate PDVC
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
conda install ffmpeg
pip install -r requirement.txt
```

3. Compile the deformable attention layer (requires GCC >= 5.4). 
```bash
cd pdvc/ops
sh make.sh
```

## Running PDVC on Your Own Videos
Download a pretrained model ([GoogleDrive](https://drive.google.com/drive/folders/1Y34puRNE0lpbz3i38k1nh8d9E1i3S0i4?usp=drive_link)) with [TSP](https://github.com/HumamAlwassel/TSP) features  and put it into `./save`. Then run:
```bash
video_folder=visualization/videos
output_folder=visualization/output
pdvc_model_path=save/anet_tsp_pdvc/model-best.pth
output_language=en
bash test_and_visualize.sh $video_folder $output_folder $pdvc_model_path $output_language
```
check the `$output_folder`, you will see a new video with embedded captions. 
Note that we generate non-English captions by translating the English captions by GoogleTranslate. 
To produce Chinese captions, set `output_language=zh-cn`. 
For other language support, find the abbreviation of your language at this [url](https://github.com/lushan88a/google_trans_new/blob/main/constant.py), and you also may need to download a font supporting your language and put it into `./visualization`.

![demo.gif](visualization/xukun_en.gif)![demo.gif](visualization/xukun_cn.gif)

## Training and Validation

### Download Video Features

```bash
cd data/anet/features
bash download_anet_c3d.sh
# bash download_anet_tsn.sh
# bash download_i3d_vggish_features.sh
# bash download_tsp_features.sh
```
The preprocessed C3D and TSN features have been uploaded to [google drive](https://drive.google.com/drive/folders/15Owgdwc1gJWYf_B9PcDiemq_478r9VJZ?usp=sharing)

### Dense Video Captioning
1. PDVC with learnt proposals
```
# Training
config_path=cfgs/anet_c3d_pdvc.yml
python train.py --cfg_path ${config_path} --gpu_id ${GPU_ID}
# The script will evaluate the model for every epoch. The results and logs are saved in `./save`.

# Evaluation
eval_folder=anet_c3d_pdvc # specify the folder to be evaluated
python eval.py --eval_folder ${eval_folder} --eval_transformer_input_type queries --gpu_id ${GPU_ID}
```
2. PDVC with ground-truth proposals

```
# Training
config_path=cfgs/anet_c3d_pdvc_gt.yml
python train.py --cfg_path ${config_path} --gpu_id ${GPU_ID}

# Evaluation
eval_folder=anet_c3d_pdvc_gt
python eval.py --eval_folder ${eval_folder} --eval_transformer_input_type gt_proposals --gpu_id ${GPU_ID}
```


### Video Paragraph Captioning

1. PDVC with learnt proposals
```bash
# Training
config_path=cfgs/anet_c3d_pdvc.yml
python train.py --cfg_path ${config_path} --criteria_for_best_ckpt pc --gpu_id ${GPU_ID} 

# Evaluation
eval_folder=anet_c3d_pdvc # specify the folder to be evaluated
python eval.py --eval_folder ${eval_folder} --eval_transformer_input_type queries --gpu_id ${GPU_ID}
```
2. PDVC with ground-truth proposals
```
# Training
config_path=cfgs/anet_c3d_pdvc_gt.yml
python train.py --cfg_path ${config_path} --criteria_for_best_ckpt pc --gpu_id ${GPU_ID}

# Evaluation
eval_folder=anet_c3d_pdvc_gt
python eval.py --eval_folder ${eval_folder} --eval_transformer_input_type gt_proposals --gpu_id ${GPU_ID}
```

## Performance
### Dense video captioning (with learnt proposals)

|  Model | Features | config_path |   Url   | Recall | Precision |    BLEU4   | METEOR2018 | METEOR2021 |  CIDEr | SODA_c |
|  ----  |  ----    |   ----  |  ----  |  ----   |  ----  |   ----  |  ----  |  ----  |  ----  | ---- |
| PDVC_light   | C3D  | cfgs/anet_c3d_pdvcl.yml | [Google Drive](https://drive.google.com/drive/folders/16ZEYpEwRXOpCYD9_jjctqzSrIz0EsAyF?usp=drive_link)  |  55.30   |  58.42  | 1.55  |  7.13  |  7.66 | 24.80  |  5.23  |
| PDVC   | C3D  | cfgs/anet_c3d_pdvc.yml |  [Google Drive](https://drive.google.com/drive/folders/1o7zJncJYF1S3jLeuFTk2JE8rljSNq6Sl?usp=drive_link)  |  55.20   |  57.36  | 1.82  |  7.48  |  8.09  | 28.16  |  5.47  |
| PDVC_light   | TSN | cfgs/anet_tsn_pdvcl.yml | [Google Drive](https://drive.google.com/drive/folders/1btfP9i6VDz1Hle2cpVqBzxizlc_Ecrg-?usp=drive_link)  |  55.34   |  57.97  | 1.66  |  7.41  |  7.97 | 27.23  |  5.51  |
| PDVC   | TSN  | cfgs/anet_tsn_pdvc.yml | [Google Drive](https://drive.google.com/drive/folders/1J0Z0pjhb6x-fAhxEUKuPApsWUFdObFMo?usp=sharing)  |  56.21   |  57.46  | 1.92  |  8.00  |  8.63 | 29.00  |  5.68  |
| PDVC_light   | TSP | cfgs/anet_tsp_pdvcl.yml | [Google Drive](https://drive.google.com/drive/folders/1f1GGtoT1kPFP6ZS-RKkORiMlRXBX06Ug?usp=drive_link)  |  55.24   |  57.78  | 1.77  |  7.94  |  8.55 | 28.25  |  5.95  |
| PDVC   | TSP  | cfgs/anet_tsp_pdvc.yml | [Google Drive](https://drive.google.com/drive/folders/1Y34puRNE0lpbz3i38k1nh8d9E1i3S0i4?usp=drive_link)  |  55.79   |  57.39  | 2.17  |  8.37  |  9.03 | 31.14  |  6.05  |


Notes:
* In the paper, we follow the most previous methods to use the [evaluation toolkit in ActivityNet Challenge 2018](https://github.com/ranjaykrishna/densevid_eval/tree/deba7d7e83012b218a4df888f6c971e21cfeea33). Note that the latest [evluation tookit](https://github.com/ranjaykrishna/densevid_eval/tree/9d4045aced3d827834a5d2da3c9f0692e3f33c1c) (METEOR2021) gives the same CIDEr/BLEU4 but a higher METEOR score. 
* In the paper, we use an [old version of SODA_c implementation](https://github.com/fujiso/SODA/tree/22671b3570e088217139bcb1e4de7a3499c30294), while here we use an [updated version](https://github.com/fujiso/SODA/tree/9cb3e2c5a73c4e320a38c72f320b63bbef4aa798) for convenience.

### Video paragraph captioning (with learnt proposals)
|  Model | Features | config_path | BLEU4 | METEOR | CIDEr |
|  ----  |  ----    |   ----  |  ----  |  ----  |   ----  |
| PDVC   | C3D  | cfgs/anet_c3d_pdvc.yml | 9.67   |  14.74  | 16.43  |  
| PDVC   | TSN  | cfgs/anet_tsn_pdvc.yml | 10.18   |  15.96  | 20.66  | 
| PDVC   | TSP  | cfgs/anet_tsp_pdvc.yml | 10.46 | 16.42 | 20.91 |

Notes:
* Paragraph-level scores are evaluated on the ActivityNet Entity ae-val set.


## Citation
If you find this repo helpful, please consider citing:
```
@inproceedings{wang2021end,
  title={End-to-End Dense Video Captioning with Parallel Decoding},
  author={Wang, Teng and Zhang, Ruimao and Lu, Zhichao and Zheng, Feng and Cheng, Ran and Luo, Ping},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6847--6857},
  year={2021}
}
```
```
@ARTICLE{wang2021echr,
  author={Wang, Teng and Zheng, Huicheng and Yu, Mingjing and Tian, Qian and Hu, Haifeng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Event-Centric Hierarchical Representation for Dense Video Captioning}, 
  year={2021},
  volume={31},
  number={5},
  pages={1890-1900},
  doi={10.1109/TCSVT.2020.3014606}}
```

## Acknowledgement

The implementation of Deformable Transformer is mainly based on [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). 
The implementation of the captioning head is based on [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch).
We thanks the authors for their efforts.
