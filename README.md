# PDVC
Official implementation for End-to-End Dense Video Captioning with Parallel Decoding (ICCV 2021) [[arxiv](https://arxiv.org/abs/2108.07781)].

![pdvc.jpg](pdvc.jpg)

Dense video captioning aims to generate multiple associated captions with their temporal locations from the video. Previous methods follow a sophisticated "localize-then-describe" scheme, which heavily relies on numerous hand-crafted components. In this paper, we proposed a simple yet effective framework for end-to-end dense video captioning with parallel decoding (PDVC), by formulating the dense caption generation as a set prediction task. In practice, through stacking a newly proposed event counter on the top of a transformer decoder, the PDVC precisely segments the video into a number of event pieces under the holistic understanding of the video content, which effectively increases the coherence and readability of predicted captions. Compared with prior arts, the PDVC has several appealing advantages: (1) Without relying on heuristic non-maximum suppression or a recurrent event sequence selection network to remove redundancy, PDVC directly produces an event set with an appropriate size; (2) In contrast to adopting the two-stage scheme, we feed the enhanced representations of event queries into the localization head and caption head in parallel, making these two sub-tasks deeply interrelated and mutually promoted through the optimization; (3) Without bells and whistles, extensive experiments on ActivityNet Captions and YouCook2 show that PDVC is capable of producing high-quality captioning results, surpassing the state-of-the-art two-stage methods when its localization accuracy is on par with them. 

# Preparation
Environment: Linux,  GCC>=5.4, CUDA >= 9.2, Python>=3.7, PyTorch>=1.5.1

1. Clone the repo
```bash
git clone --recursive https://github.com/ttengwang/PDVC.git
```

2. Create vitual environment by conda
```bash
conda create -n PDVC python=3.7
source activate PDVC
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
pip install -r requirement.txt
```


3. Prepare the video features of ActivityNet Captions and YouCook2.
```bash
cd data/anet/features
bash download_anet_c3d.sh
# bash download_anet_tsn.sh
# bash download_i3d_vggish_features.sh

```

4. Compile the deformable attention layer (requires GCC >= 5.4). 
```bash
cd models/ops
sh make.sh
```

# Performance
### Dense video captioning

|  Model | Features | config_path |   Url   | Recall | Precision |    BLEU4   | METEOR2018 | METEOR2021 |  CIDEr | SODA_c |
|  ----  |  ----    |   ----  |  ----  |  ----   |  ----  |   ----  |  ----  |  ----  |  ----  | ---- |
| PDVC_light   | C3D  | cfgs/anet_c3d_pdvcl.yml | [Google Drive](https://drive.google.com/drive/folders/1JKOJrm5QMAkso-VJnzGnksIVqNYt8BSI?usp=sharing)  |  55.30   |  58.42  | 1.55  |  7.13  |  7.66 | 24.80  |  5.23  |
| PDVC_light   | TSN | cfgs/anet_tsn_pdvcl.yml | [Google Drive](https://drive.google.com/drive/folders/1hImJ7sXABzS-ycErruLFCE_pkWEHzFSV?usp=sharing)  |  55.34   |  57.97  | 1.66  |  7.41  |  7.97 | 27.23  |  5.51  |
| PDVC   | C3D  | cfgs/anet_c3d_pdvc.yml |  [Google Drive](https://drive.google.com/drive/folders/1I77miVvThdMenmprgozfRsXDVoc-9TxY?usp=sharing)  |  55.20   |  57.36  | 1.82  |  7.48  |  8.09  | 28.16  |  5.47  |
| PDVC   | TSN  | cfgs/anet_tsn_pdvc.yml | [Google Drive](https://drive.google.com/drive/folders/1v2Xj0Qjt3Te_SgVyySKEofRaZsSw_rjs?usp=sharing)  |  56.21   |  57.46  | 1.92  |  8.00  |  8.63 | 29.00  |  5.68  |

Notes:
* In the paper, we follow the most previous methods to use the [evaluation toolkit in ActivityNet Challenge 2018](https://github.com/ranjaykrishna/densevid_eval/tree/deba7d7e83012b218a4df888f6c971e21cfeea33). Note that the latest [evluation tookit](https://github.com/ranjaykrishna/densevid_eval/tree/9d4045aced3d827834a5d2da3c9f0692e3f33c1c) (METEOR2021) gives the same CIDEr/BLEU4 but a higher METEOR score. 
* In the paper, we use an [old version of SODA_c implementation](https://github.com/fujiso/SODA/tree/22671b3570e088217139bcb1e4de7a3499c30294), while here we use an [updated version](https://github.com/fujiso/SODA/tree/9cb3e2c5a73c4e320a38c72f320b63bbef4aa798) for convenience.

### Video paragraph captioning
|  Model | Features | config_path | BLEU4 | METEOR | CIDEr |
|  ----  |  ----    |   ----  |  ----  |  ----  |   ----  |
| PDVC   | C3D  | cfgs/anet_c3d_pdvc.yml |  9.67   |  14.74  | 16.43  |  
| PDVC   | TSN  | cfgs/anet_tsn_pdvc.yml |  10.18   |  15.96  | 20.66  | 

Notes:
* Paragraph-level scores are evaluated on the ActivityNet Entity ae-val set.

# Usage
### Dense Video Captioning
1. PDVC with learnt proposal
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
config_path=cfgs/anet_c3d_pdvc.yml
python train.py --cfg_path ${config_path} --gpu_id ${GPU_ID}

# Evaluation
eval_folder=anet_c3d_pdvc_gt
python eval.py --eval_folder ${eval_folder} --eval_transformer_input_type gt_proposals --gpu_id ${GPU_ID}
```


# Video Paragraph Captioning

1. PDVC with learnt proposal
```bash
# Training
config_path=cfgs/anet_c3d_pdvc.yml
python train.py --cfg_path ${config_path} --criteria_for_best_ckpt pc --gpu_id ${GPU_ID} 

# Evaluation
eval_folder=anet_c3d_pdvc # specify the folder to be evaluated
python eval.py --eval_folder ${eval_folder} --eval_transformer_input_type queries --gpu_id ${GPU_ID}
```
2. PDVC with ground-truth proposal
```
# Training
config_path=cfgs/anet_c3d_pdvc.yml
python train.py --cfg_path ${config_path} --criteria_for_best_ckpt pc --gpu_id ${GPU_ID}

# Evaluation
eval_folder=anet_c3d_pdvc_gt
python eval.py --eval_folder ${eval_folder} --eval_transformer_input_type gt_proposals --gpu_id ${GPU_ID}
```

# TODO
- [x] add more pretrained models
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

# Acknowledgement

The implementation of Deformable Transformer is mainly based on [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). 
The implementation of the captioning head is based on [ImageCaptioning.pyotrch](https://github.com/ruotianluo/ImageCaptioning.pytorch).
We thanks the authors for their efforts.
