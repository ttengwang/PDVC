from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
import torch
import numpy as np
import time
from os.path import dirname, abspath

pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3/SODA'))
print(sys.path)

from eval_utils import evaluate
from models.pdvc import build
from misc.utils import create_logger
from data.video_dataset import PropSeqDataset, collate_fn
from torch.utils.data import DataLoader
# from aux.transfer import transfer

def main(opt):
    folder_path = os.path.join(opt.eval_save_dir, opt.eval_folder)
    infos_path = os.path.join(folder_path, 'info.json')
    logger = create_logger(folder_path, 'val.log')
    logger.info(vars(opt))

    with open(infos_path, 'rb') as f:
        logger.info('load info from {}'.format(infos_path))
        old_opt = json.load(f)['best']['opt']

    for k, v in old_opt.items():
        if k[:4] != 'eval':
            vars(opt).update({k: v})
    # opt.feature_dim = opt.raw_feature_dim
    opt.transformer_input_type = opt.eval_transformer_input_type
    opt.gt_proposal_sample_num = 30

    if not torch.cuda.is_available():
        opt.nthreads = 0
    # Create the Data Loader instance
    val_dataset = PropSeqDataset(opt.eval_caption_file,
                                 opt.visual_feature_folder,
                                 opt.dict_file, False, opt.eval_proposal_type,
                                 opt)
    loader = DataLoader(val_dataset, batch_size=opt.batch_size_for_eval,
                        shuffle=False, num_workers=opt.nthreads, collate_fn=collate_fn)


    model, criterion, postprocessors = build(opt)
    model.translator = val_dataset.translator

    if opt.eval_model_path:
        model_path = opt.eval_model_path
    else:
        model_path = os.path.join(folder_path, 'model-best.pth')

    while not os.path.exists(model_path):
        raise AssertionError('File {} does not exist'.format(model_path)) #TODO

    logger.debug('Loading model from {}'.format(model_path))
    loaded_pth = torch.load(model_path, map_location='cpu')
    epoch = loaded_pth['epoch']

    # loaded_pth = transfer(model, loaded_pth, model_path+'.transfer.pth')
    model.load_state_dict(loaded_pth['model'], strict=True)
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    out_json_path = os.path.join(folder_path, '{}_epoch{}_num{}_alpha{}.json'.format(
        time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime()) + str(opt.id), epoch, len(loader.dataset), opt.ec_alpha))
    logger.info('saving reults json to {}'.format(out_json_path))
    caption_scores,eval_loss = evaluate(model, criterion, postprocessors, loader, out_json_path,
                         logger, opt.eval_random, alpha=opt.ec_alpha, dvc_eval_version=opt.eval_tool_version)

    avg_eval_score = {key: np.array(value).mean() for key, value in caption_scores.items() if key !='tiou'}
    avg_eval_score2 = {key: np.array(value).mean() * 4917 / len(loader.dataset) for key, value in caption_scores.items() if key != 'tiou'}

    logger.info(
        '\nValidation result based on all 4917 val videos:\n {}\n avg_score:\n{}'.format(
                                                                                   caption_scores.items(),
                                                                                   avg_eval_score))

    logger.info(
            '\nValidation result based on {} available val videos:\n avg_score:\n{}'.format(len(loader.dataset),
                                                                                       avg_eval_score2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_save_dir', type=str, default='save')
    parser.add_argument('--eval_folder', type=str, required=True)
    parser.add_argument('--eval_model_path', type=str, default='')
    parser.add_argument('--eval_tool_version', type=str, default='2018', choices=['2018', '2021'])
    parser.add_argument('--eval_caption_file', type=str, default='data/anet/captiondata/val_1.json')
    parser.add_argument('--eval_proposal_type', type=str, default='gt')
    parser.add_argument('--eval_transformer_input_type', type=str, default='queries', choices=['gt_proposals', 'queries'])
    parser.add_argument('--eval_random', type=int, default=0)
    parser.add_argument('--gpu_id', type=str, nargs='+', default=['0'])
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    if True:
        torch.backends.cudnn.enabled = False
    main(opt)
