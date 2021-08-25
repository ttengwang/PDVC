cd /apdcephfs/share_1367250/wybertwang/project/PDVC
#python train.py --cfg_path cfgs/anet_c3d_pdvcl.yml --disable_tqdm --debug

#python train.py --cfg_path cfgs/anet_c3d_pdvcl.yml --disable_tqdm

python train.py --cfg_path cfgs/anet_tsn_pdvcl.yml --disable_tqdm

#python train.py --cfg_path cfgs/anet_c3d_pdvc.yml --disable_tqdm

#python train.py --cfg_path cfgs/anet_tsn_pdvcl.yml --disable_tqdm

python train.py --cfg_path cfgs/anet_c3d_pdvc_gt.yml --disable_tqdm

python train.py --cfg_path cfgs/anet_c3d_pdvcl_gt.yml --disable_tqdm

python train.py --cfg_path cfgs/anet_c3d_pdvc_gt.yml --disable_tqdm

python train.py --cfg_path cfgs/anet_c3d_pdvc_gt.yml --disable_tqdm
