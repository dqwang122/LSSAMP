#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
import time
import yaml
from others.logging import init_logger
from models.tokenizer import Vocabulary, SSVocabulary
from train_lm import train_lm, test_lm, sample_lm, validate
from train_vae import train_vae, sample_vae, validate_vae, test_vae, predict_vae


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def loadConfig(args):
    with open(args.config, encoding='utf-8') as fin:
        opt = yaml.load(fin, Loader=yaml.FullLoader)
    args = vars(args)
    args.update(opt)
    return argparse.Namespace(**args)

def saveConfig(args, name):
    args_dict = vars(args)
    with open(name, 'w', encoding='utf-8') as fout:
        yaml.dump(args_dict, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", default='train.yaml', type=str)
    parser.add_argument("-task", default='LM', type=str, choices=['LM', 'AMP'])
    parser.add_argument("-arch", default='vae', type=str, choices=['vae', 'vqvae', 'cvqvae', 'crfvqvae'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test', 'sample'])
    parser.add_argument("-sample", default='top', type=str, choices=['top', 'random', 'beam'])
    parser.add_argument("-data_path", default='../data/')
    parser.add_argument("-vocab_path", default='../data/vocab')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/')

    parser.add_argument("-batch_size", default=1000, type=int, help='token level')
    parser.add_argument("-test_batch_size", default=140, type=int, help='token level')

    parser.add_argument("-vocab_size", default=0, type=int)
    # parser.add_argument("-do_lower_case", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-data_process", default='parallel')
    parser.add_argument("-seq_reverse", default=False, type=bool)
    parser.add_argument('-mask_ratio', default=0.2, type=float)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.1, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-use_enc", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-enc_hidden_size", default=768, type=int)
    parser.add_argument("-enc_heads", default=8, type=int)
    parser.add_argument("-enc_ff_size", default=2048, type=int)
    parser.add_argument("-enc_dropout", default=0.1, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)
    parser.add_argument("-conv_channel", default=32, type=int)

    parser.add_argument("-embed_size", default=768, type=int)
    parser.add_argument("-latent_size", default=200, type=int)
    parser.add_argument("-ss_size", default=8, type=int)
    parser.add_argument("-entry_num", default=8, type=int)
    parser.add_argument("-code_book", default=8, type=int)
    parser.add_argument("-ss_coef", default=1, type=float)
    parser.add_argument("-kl_coef", default=1, type=float)
    parser.add_argument("-vq_coef", default=0.2, type=float)
    parser.add_argument("-comit_coef", default=0.4, type=float)
    parser.add_argument("-decay", default=0.8, type=float)
    parser.add_argument("-loss_func", default='cvqvae', type=str)
    parser.add_argument("-sub_book", default=1, type=float)
    parser.add_argument("-vqvae_list", default=[1,2,4,8], type=list)
    parser.add_argument("-code_mode", default='cat', type=str)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=2, type=int)
    parser.add_argument("-max_length", default=512, type=int)
    parser.add_argument("-max_skip_length", default=1000, type=int)
    parser.add_argument("-max_tgt_len", default=512, type=int)
    parser.add_argument("-randomnum", default=1, type=int)


    parser.add_argument("-optim_init", type=str2bool, nargs='?',const=True, default=False)
    # parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-train_and_valid", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-valid_steps", default=10, type=int)
    parser.add_argument("-patience", default=3, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-save_index", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_dir', default='../logs/LM')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-train_from", default='')
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_name", default='test')
    parser.add_argument("-step", default='-1')

    parser.add_argument('-prefix_token', default='0')
    parser.add_argument("-filter_func", type=str, default='noss', choices=['noss', 'alpha', 'beta', 'coil'])
    parser.add_argument("-random_init", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=False)

    args = parser.parse_args()
    loadConfig(args)
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.gpu_num = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    log_file = os.path.join(args.log_dir, "{}_{}.log".format(args.task, time.strftime("%Y%m%d_%H%M", time.localtime())))
    args.log_file = log_file
    logger = init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if not os.path.exists('{}/{}'.format(args.log_dir, args.mode)):
        os.mkdir('{}/{}'.format(args.log_dir, args.mode))
    saveConfig(args, os.path.join('{}/{}'.format(args.log_dir, args.mode), 'run_args.yml'))
    logger.info(args)

    if (args.task == 'LM'):
        if (args.mode == 'train'):
            train_lm(args, device_id)
        elif (args.mode == 'validate'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                # step = 0
                step = -1
            validate(args, device_id, cp, step)
        elif (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                # step = 0
                step = args.step
            test_lm(args, device_id, cp, step)
        elif (args.mode == 'sample'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                # step = 0
                step = -1
            sample_lm(args, device_id, cp, step)
        else:
            raise NotImplementedError

    elif (args.task == 'AMP'):

        vocab = Vocabulary(args.vocab_path, args.vocab_size)
        args.vocab_size = len(vocab)
        if args.arch == 'crfvqvae':
            ssDict = SSVocabulary(args.ss_size, bos=True, eos=True)
        else:
            ssDict = SSVocabulary(args.ss_size)
        args.ss_type = args.ss_size
        args.ss_size = len(ssDict)
        args.ss_bos, args.ss_eos = ssDict.bos, ssDict.eos

        if (args.mode == 'train'):
            train_vae(args, device_id)
        elif (args.mode == 'validate'):
            validate_vae(args, device_id)
        elif (args.mode == 'test'):
            test_vae(args, device_id)
        elif (args.mode == 'predict'):
            predict_vae(args, device_id)
        elif (args.mode == 'sample'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                # step = 0
                step = -1
            sample_vae(args, device_id, cp, step)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
