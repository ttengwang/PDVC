# coding:utf-8
# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
import glob
import shutil
import os
import colorlog
import random
import six
from six.moves import cPickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_parameter_group(model, opt):
    if opt.lr_proj:
        non_caption_param = [
            {
                "params":
                    [p for n, p in model.named_parameters()
                     if not match_name_keywords(n, opt.lr_backbone_names)
                     and not match_name_keywords(n,opt.lr_linear_proj_names)
                     and not match_name_keywords(n, opt.lr_captioner_names)
                     and p.requires_grad],
                "lr": opt.lr,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if match_name_keywords(n, opt.lr_backbone_names)
                           and p.requires_grad],
                "lr": opt.lr_backbone,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if not match_name_keywords(n, opt.lr_captioner_names)
                           and match_name_keywords(n,opt.lr_linear_proj_names)
                           and p.requires_grad],
                "lr": opt.lr * opt.lr_linear_proj_mult,
            }
        ]

        caption_param = [
            {
                "params":
                    [p for n, p in model.named_parameters()
                     if match_name_keywords(n, opt.lr_captioner_names)
                     and not match_name_keywords(n,opt.lr_linear_proj_names)
                     and p.requires_grad],
                "lr": opt.lr,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           match_name_keywords(n, opt.lr_captioner_names)
                           and match_name_keywords(n, opt.lr_linear_proj_names)
                           and p.requires_grad],
                "lr": opt.lr * opt.lr_linear_proj_mult,
            }
        ]
        for param in non_caption_param:
            param['lr'] = param['lr'] * opt.non_caption_lr_mult
    else:
        # param_dicts = model.parameters()
        caption_param = []
        for name, value in model.named_parameters():
            if match_name_keywords(name, opt.lr_captioner_names) and value.requires_grad:
                caption_param.append(value)
        non_caption_param = []
        for name, value in model.named_parameters():
            if (not match_name_keywords(name, opt.lr_captioner_names)) and value.requires_grad:
                non_caption_param.append(value)

    return non_caption_param, caption_param


def decide_two_stage(transformer_input_type, dt, criterion):
    if transformer_input_type == 'learnt_proposals':
        two_stage = True
        proposals = dt['lnt_boxes']
        proposals_mask = dt['lnt_boxes_mask']
        disable_iterative_refine = False
    elif transformer_input_type == 'learnt_proposals_no_bbox_refine':
        two_stage = True
        proposals = dt['lnt_boxes']
        proposals_mask = dt['lnt_boxes_mask']
        criterion.matcher.cost_caption = 0
        for q_k in ['loss_length', 'loss_ce']:
            for key in criterion.weight_dict.keys():
                if q_k in key:
                    criterion.weight_dict[key] = 0
        disable_iterative_refine = True
    elif transformer_input_type == 'gt_proposals':
        two_stage = True
        # assert len(dt['video_target'])==1 # batchsize = 1
        # proposals = dt['video_target'][0]['boxes'].unsqueeze(0)
        proposals = dt['gt_boxes']
        proposals_mask = dt['gt_boxes_mask']
        criterion.matcher.cost_caption = 0
        for q_k in ['loss_length', 'loss_ce', 'loss_bbox', 'loss_giou']:
            for key in criterion.weight_dict.keys():
                if q_k in key:
                    criterion.weight_dict[key] = 0
        disable_iterative_refine = True
    else:  # transformer_input_type='queries'
        two_stage = False
        proposals = None
        proposals_mask = None
        disable_iterative_refine = False
    return two_stage, disable_iterative_refine, proposals, proposals_mask


def pickle_load(f):
    """ Load a pickle.
    Parameters
    ----------
    f: file-like object
    """
    if six.PY3:
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)


def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: pickled object
    f: file-like object
    """
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if key not in dict_to.keys():
            raise AssertionError('key mismatching: {}'.format(key))
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]


def set_droprate(model, epoch, opt):
    frac = (epoch - opt.drop_rate_increase_start) // opt.drop_rate_increase_every
    decay_factor = opt.drop_rate_increase_value * frac
    r = min(opt.transformer_dropout_prob + decay_factor, opt.drop_rate_max)

    for layer in model.transformer.encoder.layers:
        layer.dropout1.p = r
        layer.dropout2.p = r
        layer.dropout3.p = r
    for layer in model.transformer.decoder.layers:
        layer.dropout4.p = r
        layer.dropout1.p = r
        layer.dropout2.p = r
        layer.dropout3.p = r
    print('drop_rate:{}'.format(r))


def print_opt(opt, model, logger):
    print_alert_message('All args:', logger)
    for key, item in opt._get_kwargs():
        logger.info('{} = {}'.format(key, item))
    print_alert_message('Model structure:', logger)
    logger.info(model)


def build_floder(opt):
    if opt.start_from:
        print('Start training from id:{}'.format(opt.start_from))
        save_folder = os.path.join(opt.save_dir, opt.start_from)
        assert os.path.exists(save_folder)
    else:
        if not os.path.exists(opt.save_dir):
            os.mkdir(opt.save_dir)
        save_folder = os.path.join(opt.save_dir, opt.id)
        if os.path.exists(save_folder):
            # wait_flag = input('Warning! ID {} already exists, rename it? (Y/N) : '.format(opt.id))
            wait_flag = 'Y'
            if wait_flag in ['Y', 'y']:
                opt.id = opt.id + '_v_{}'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
                save_folder = os.path.join(opt.save_dir, opt.id)
                print('Rename opt.id as "{}".'.format(opt.id))
            else:
                raise AssertionError('ID already exists, folder {} exists'.format(save_folder))
        print('Results folder "{}" does not exist, creating folder...'.format(save_folder))
        os.mkdir(save_folder)
        os.mkdir(os.path.join(save_folder, 'prediction'))
    return save_folder


def backup_envir(save_folder):
    backup_folders = ['cfgs', 'misc', 'models']
    # backup_folders = ['models']
    backup_files = glob.glob('./*.py')
    for folder in backup_folders:
        shutil.copytree(folder, os.path.join(save_folder, 'backup', folder))
    for file in backup_files:
        shutil.copyfile(file, os.path.join(save_folder, 'backup', file))


def create_logger(folder, filename):
    log_colors = {
        'DEBUG': 'blue',
        'INFO': 'white',
        'WARNING': 'green',
        'ERROR': 'red',
        'CRITICAL': 'yellow',
    }

    import logging
    logger = logging.getLogger('DVC')
    # %(filename)s$RESET:%(lineno)d
    # LOGFORMAT = "%(log_color)s%(asctime)s [%(log_color)s%(filename)s:%(lineno)d] | %(log_color)s%(message)s%(reset)s |"
    LOGFORMAT = ""
    LOG_LEVEL = logging.DEBUG
    logging.root.setLevel(LOG_LEVEL)
    stream = logging.StreamHandler()
    stream.setLevel(LOG_LEVEL)
    stream.setFormatter(colorlog.ColoredFormatter(LOGFORMAT, datefmt='%d %H:%M', log_colors=log_colors))

    # print to log file
    hdlr = logging.FileHandler(os.path.join(folder, filename))
    hdlr.setLevel(LOG_LEVEL)
    # hdlr.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    hdlr.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(hdlr)
    logger.addHandler(stream)
    return logger


def print_alert_message(str, logger=None):
    msg = '*' * 20 + ' ' + str + ' ' + '*' * (58 - len(str))
    if logger:
        logger.info('\n\n' + msg)
    else:
        print(msg)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for i, param in enumerate(group['params']):
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def plot_bbox_distribution(bboxes, save_path):
    # bboxes: [video_num, query_num, 4]
    N, M, C = bboxes.shape
    bboxes = bboxes.transpose(1, 0, 2)

    fig = plt.figure(figsize=(40, 20), dpi=300)
    for m in range(M):
        scores = []
        centers = []
        lengths = []
        n_row, n_col = int(np.ceil(np.sqrt(M))), int(2 * np.ceil(M / np.sqrt(M)))
        ax = fig.add_subplot(n_row, n_col, 2 * m + 1)
        ax2 = fig.add_subplot(n_row, n_col, 2 * m + 2)
        L = 100
        dstr = np.zeros(L)
        for n in range(N):
            c, l = bboxes[m, n, 0], bboxes[m, n, 1]
            centers.append(c * L)
            lengths.append(l * L)
            if bboxes.shape[2] > 5:
                score = bboxes[m, n, 4:-1].max() * L
            else:
                score = bboxes[m, n, 4:].max() * L
            scores.append(score)
            s, e = c - l / 2, c + l / 2
            s = int(max(0, min(s, 1)) * L)
            e = int(max(0, min(e, 1)) * L)
            dstr[s: e + 1] = dstr[s: e + 1] + 1
        dstr = dstr / N
        ax.plot(np.arange(L), dstr)
        scores, centers, lengths = map(np.array, [scores, centers, lengths])
        flierprops = dict(marker='.', markersize=1, markeredgecolor='r')
        ax2.boxplot([scores, centers, lengths], positions=[1, 2, 3], flierprops=flierprops, vert=False,
                    patch_artist=False, meanline=False, showmeans=True)
        ax.set_xlim([0, 100])
        ax2.set_xlim([0, 100])
    # pdb.set_trace()
    fig.savefig(save_path, bbox_inches='tight', dpi=fig.dpi, pad_inches=0.5)
    plt.close(fig)
    pass

if __name__ == '__main__':
    # import opts
    #
    # info = {'opt': vars(opts.parse_opts()),
    #         'loss': {'tap_loss': 0, 'tap_reg_loss': 0, 'tap_conf_loss': 0, 'lm_loss': 0}}
    # record_this_run_to_csv(info, 'save/results_all_runs.csv')

    logger = create_logger('./', 'mylogger.log')
    logger.info('debug')
    logger.info('test2')
