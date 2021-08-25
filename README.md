# PDVC
Official implementation for End-to-End Dense Video Captioning with Parallel Decoding (ICCV 2021) [[arxiv](https://arxiv.org/abs/2108.07781)].

![pdvc.jpg](pdvc.jpg)

Dense video captioning aims to generate multiple associated captions with their temporal locations from the video. Previous methods follow a sophisticated "localize-then-describe" scheme, which heavily relies on numerous hand-crafted components. In this paper, we proposed a simple yet effective framework for end-to-end dense video captioning with parallel decoding (PDVC), by formulating the dense caption generation as a set prediction task. In practice, through stacking a newly proposed event counter on the top of a transformer decoder, the PDVC precisely segments the video into a number of event pieces under the holistic understanding of the video content, which effectively increases the coherence and readability of predicted captions. Compared with prior arts, the PDVC has several appealing advantages: (1) Without relying on heuristic non-maximum suppression or a recurrent event sequence selection network to remove redundancy, PDVC directly produces an event set with an appropriate size; (2) In contrast to adopting the two-stage scheme, we feed the enhanced representations of event queries into the localization head and caption head in parallel, making these two sub-tasks deeply interrelated and mutually promoted through the optimization; (3) Without bells and whistles, extensive experiments on ActivityNet Captions and YouCook2 show that PDVC is capable of producing high-quality captioning results, surpassing the state-of-the-art two-stage methods when its localization accuracy is on par with them. 

# Preparation
Environment: Linux,  GCC>=5.4, CUDA >= 9.2, Python>=3.7, PyTorch>=1.5.1,

1. Clone the repo
```bash
git clone --recursive https://github.com/ttengwang/PDVC.git
```

2. Create vitual environment by conda
```bash
conda create -n PDVC python=3.7; 
source activate PDVC; 
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

# Usage
### PDVC
- Training
```bash
python train.py --cfg_path cfgs/anet_c3d_pdvc.yml --gpu_id ${GPU_ID}
```
The script will print the log and evaluate the model for every epoch. The results and logs are saved in `./save/args.id`.

- Evaluation
```bash
eval_folder=anet_c3d_pdvc # the folder name you want to evaluate
python eval.py --eval_folder ${eval_folder} --eval_transformer_input_type queries --gpu_id ${GPU_ID}
```

### PDVC with gt proposals

- Training
```bash
python train.py --cfg_path cfgs/anet_c3d_pdvc_gt.yml --gpu_id ${GPU_ID}
```
- Evaluation
```bash
eval_folder=anet_c3d_pdvc_gt
python eval.py --eval_folder ${eval_folder} --eval_transformer_input_type gt_proposals --gpu_id ${GPU_ID}
```

# Performance

|  Model | Features |   Url   | Recall |Precision|    BLEU4   | METEOR2018 | METEOR2021 |  CIDEr | SODA_c | METEOR (Para-level) |
|  ----  |  ----    |   ----  |  ----   |  ----  |   ----  |  ----  |  ----  |  ----  | ---- | ---- |
| PDVC   | TSN  | [Google Drive](https://drive.google.com/drive/folders/1v2Xj0Qjt3Te_SgVyySKEofRaZsSw_rjs?usp=sharing)  |  56.21   |  57.46  | 1.92  |  8.00  |  8.63 | 29.00  |  5.68  | 15.85 |

Some notes:
* In the paper, we follow the most previous methods to use the [evaluation toolkit in ActivityNet Challenge 2018](https://github.com/ranjaykrishna/densevid_eval/tree/deba7d7e83012b218a4df888f6c971e21cfeea33). Note that the latest [evluation tookit](https://github.com/ranjaykrishna/densevid_eval/tree/9d4045aced3d827834a5d2da3c9f0692e3f33c1c) (METEOR2021) gives a higher METEOR score.
* Paragraph-level METEOR is evaluated on the ActivityNet Entity ae-val set, while others are on the standard  ActivityNet Captions validation set.

# TODO

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