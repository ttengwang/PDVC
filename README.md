# PDVC
Code for End-to-End Dense Video Captioning with Parallel Decoding (ICCV 2021) [[arxiv](https://arxiv.org/abs/2108.07781)]

![pdvc.png](pdvc.jpg)


# Preparation
Environment: Linux,  GCC>=5.4, CUDA >= 9.2, Python>=3.7, PyTorch>=1.5.1,

1. Create vitual environment by conda
```bash
conda create -n PDVC python=3.7; 
source activate PDVC; 
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
pip install -r requirement.txt
```

2. Clone the repo
```bash
git clone --recursive https://github.com/ttengwang/PDVC.git
```

3. Prepare the video features.
```bash
cd data/features;
bash download_anet_c3d.sh
bash download_anet_tsn.sh
```

4. Compile the deformable attention layer (requires gcc >= 5.4). 
```bash
cd models/ops
sh make.sh
```

# Usage
### PDVC
- Training
```bash
python train.py --cfg_path cfgs/anet_c3d_pdvc.yml --gpu_id ${GPU_ID}
```
The script will print the log and evaluate the model every epoch.

- Evaluation
```bash
python eval.py --eval_folder $eval_folder --eval_transformer_input_type queries --gpu_id ${GPU_ID}
```

### PDVC with gt proposals

- Training
```bash
python train.py --cfg_path cfgs/anet_c3d_pdvc_gt.yml --gpu_id ${GPU_ID}
```
- Evaluation
```bash
python eval.py --eval_folder $eval_folder --eval_transformer_input_type gt_proposals --gpu_id ${GPU_ID}
```

# Performance

|  Model | Features |   Url   | Recall |Precision|    BLEU4   | METEOR2018 | METEOR2021 |  CIDEr | SODA_c | Para METEOR |
|  ----  |  ----    |   ----  |  ----   |  ----  |   ----  |  ----  |  ----  |  ----  | ---- | ---- |
| PDVC   | TSN  | [Google Drive](https://drive.google.com/drive/folders/1v2Xj0Qjt3Te_SgVyySKEofRaZsSw_rjs?usp=sharing)  |  56.21   |  57.46  | 1.92  |  8.00  |  8.63 | 29.00  |  5.68  | 15.85 |

Some notes:
* In the paper, we foloow the most previous methods to use the [evaluation tookit in ActivityNet Challenge 2018](https://github.com/ranjaykrishna/densevid_eval/tree/deba7d7e83012b218a4df888f6c971e21cfeea33). Note that the latest [evluation tookit](https://github.com/ranjaykrishna/densevid_eval/tree/9d4045aced3d827834a5d2da3c9f0692e3f33c1c) (METEOR2021) gives a higher score.
* PARA METEOR ar evaluated on the ActivityNet ae-val set, while others are on standard validation set.

### TODO

- [ ] more pretrained models
- [ ] support youcook2

# Citation
If you find this repo helpful, please consider citing:
```
@article{wang2021end,
  title={End-to-End Dense Video Captioning with Parallel Decoding},
  author={Wang, Teng and Zhang, Ruimao and Lu, Zhichao and Zheng, Feng and Cheng, Ran and Luo, Ping},
  journal={arXiv preprint},
  year={2021}
```
```
@article{wang2020dense,
  title={Dense-Captioning Events in Videos: SYSU Submission to ActivityNet Challenge 2020},
  author={Wang, Teng and Zheng, Huicheng and Yu, Mingjing},
  journal={arXiv preprint arXiv:2006.11693},
  year={2020}
}
```
