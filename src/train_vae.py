#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time
import numpy as np
import re
import json
import logging

import torch

import distributed
from torch.serialization import validate_cuda_device
from models import data_loader, model_builder
from models.data_loader import load_dataset, load_raw_json
from models.loss import vae_loss
from models.model_builder import ProteinVAE, ProteinVQVAE, ProteinCVQVAE, ProteinCVQVAE2
from models.predictor import build_predictor
from models.trainer import build_trainer
from models.tokenizer import Vocabulary
from others.logging import logger, init_logger
from others.utils import nocheckSS, checkSS, checkSS_2, checkSS_3
from train_lm import ErrorHandler,str2bool

model_flags = ['embed_size', 'latent_size', 'kl_coef', 'vq_coef', 'comit_coef',
               'sub_book', 'code_book',
               'dec_layers', 'dec_hidden_size', 'dec_heads', 'dec_ff_size',
               'enc_layers','enc_hidden_size', 'enc_heads', 'enc_ff_size', 'use_enc']

PAD_TOKEN = 0
SENTENCE_START = 1
SENTENCE_END = 2
UNKNOWN_TOKEN = 3

MODELTYPE = {
    'vae': ProteinVAE ,
    'vqvae': ProteinVQVAE,
    'cvqvae': ProteinCVQVAE,
    'cvqvae2': ProteinCVQVAE2,
}

FilterFunc = {
    'noss': nocheckSS,
    'alpha': checkSS,
    'beta': checkSS_2,
    'coil': checkSS_3
}


def train_vae(args, device_id):
    if (args.gpu_num > 1):
        train_vae_multi(args)
    else:
        train_vae_single(args, device_id)

def train_vae_single(args, device_id):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True, reversed=args.seq_reverse), args.batch_size, device,
                                      shuffle=True, is_test=False)
    def valid_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=True), args.test_batch_size, device,
                                      shuffle=False, is_test=False)


    model = MODELTYPE[args.arch](args, device, checkpoint)
    optim = [model_builder.build_optim(args, model, checkpoint)]
    model.init_model(checkpoint)
    model.to(device)
    logger.info(model)

    # symbol for bos, eos, pad
    symbols = {'BOS': SENTENCE_START, 'EOS': SENTENCE_END, 'PAD': PAD_TOKEN}
    loss_conf = {
        'vae': {'label_smoothing': args.label_smoothing, 'kl_coef': args.kl_coef} ,
        'vqvae': {'label_smoothing': args.label_smoothing, 'vq_coef': args.vq_coef, 'comit_coef': args.comit_coef},
        'cvqvae': {'label_smoothing': args.label_smoothing, 'ss_size': args.ss_size, 'vq_coef': args.vq_coef, 'comit_coef': args.comit_coef, 'ss_coef': args.ss_coef},
    }
    kwargs = loss_conf[args.loss_func]
    if args.loss_func == 'cvqvae':
        kwargs['ss_generator'] = model.ss_generator
    elif args.loss_func == 'crfvqvae':
        kwargs['crf'] = model.crf
    train_loss = vae_loss(model.generator, symbols, model.vocab_size, device, args.loss_func, train=True, **kwargs)
    valid_loss = vae_loss(model.generator, symbols, model.vocab_size, device, args.loss_func, train=False, **kwargs)

    trainer = build_trainer(args, device_id, model, optim, train_loss, valid_loss=valid_loss)
    trainer.train(train_iter_fct, args.train_steps, valid_iter_fct=valid_iter_fct, valid_steps=args.valid_steps)


def train_vae_multi(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.gpu_num
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    def run(args, device_id, error_queue):
        """ run process """

        setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

        try:
            gpu_rank = distributed.multi_init(device_id, args.gpu_num, args.gpu_ranks)
            print('gpu_rank %d' % gpu_rank)
            if gpu_rank != args.gpu_ranks[device_id]:
                raise AssertionError("An error occurred in \
                    Distributed initialization")

            train_vae_single(args, device_id)
        except KeyboardInterrupt:
            pass  # killed by parent, do nothing
        except Exception:
            # propagate exception to parent process, keeping original traceback
            import traceback
            error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def validate_vae(args, device_id, pt='', step=-1):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = MODELTYPE[args.arch](args, device, checkpoint)
    model.init_model(checkpoint)
    model.to(device)
    model.eval()
    logger.info(model)

    symbols = {'BOS': SENTENCE_START, 'EOS': SENTENCE_END, 'PAD': PAD_TOKEN}
    loss_conf = {
        'vae': {'label_smoothing': args.label_smoothing, 'kl_coef': args.kl_coef} ,
        'vqvae': {'label_smoothing': args.label_smoothing, 'vq_coef': args.vq_coef, 'comit_coef': args.comit_coef},
        'cvqvae': {'label_smoothing': args.label_smoothing, 'ss_size': args.ss_size, 'ss_coef': args.ss_coef,'vq_coef': args.vq_coef, 'comit_coef': args.comit_coef},
    }
    kwargs = loss_conf[args.loss_func]
    if args.loss_func == 'cvqvae':
        kwargs['ss_generator'] = model.ss_generator
    elif args.loss_func == 'crfvqvae':
        kwargs['crf'] = model.crf

    part = 'valid' if args.mode == 'validate' else 'test'
    valid_iter = data_loader.Dataloader(args, load_dataset(args, part, shuffle=False),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False)

    valid_loss = vae_loss(model.generator, symbols, model.vocab_size, device, args.loss_func, train=False, **kwargs)
    trainer = build_trainer(args, device_id, model, None, valid_loss)
    stats = trainer.validate(valid_iter, step, is_test=(args.mode=='test'))
    return stats.xent()


def test_vae(args, device_id, pt='', step=-1):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = MODELTYPE[args.arch](args, device, checkpoint)
    model.init_model(checkpoint)
    model.to(device)
    model.eval()
    logger.info(model)

    symbols = {'BOS': SENTENCE_START, 'EOS': SENTENCE_END, 'PAD': PAD_TOKEN}
    loss_conf = {
        'vae': {'label_smoothing': args.label_smoothing, 'kl_coef': args.kl_coef} ,
        'vqvae': {'label_smoothing': args.label_smoothing, 'vq_coef': args.vq_coef, 'comit_coef': args.comit_coef},
        'cvqvae': {'label_smoothing': args.label_smoothing, 'ss_size': args.ss_size, 'ss_coef': args.ss_coef,'vq_coef': args.vq_coef, 'comit_coef': args.comit_coef},
    }
    kwargs = loss_conf[args.loss_func]
    if args.loss_func == 'cvqvae':
        kwargs['ss_generator'] = model.ss_generator
    elif args.loss_func == 'crfvqvae':
        kwargs['crf'] = model.crf

    log_dir = args.log_dir + '/test'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    vocab = Vocabulary(args.vocab_path, args.vocab_size)
    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False)

    def clean(x):
        x = re.sub(r'\<PAD\>|\<UNK\>|\<MASK\>|\<s\>','', x)
        x = x.split('</s>')[0]
        return x

    step = 0
    with open(os.path.join(log_dir, 'generate_{}_{}.txt'.format(args.sample, step)), 'w') as fout:
        for batch in test_iter:
                outputs, scores, ss_scores = model.generate(batch.src, batch.tgt, batch.mask_src, batch.mask_tgt)
                results = ["".join(vocab.ids_to_tokens(x.tolist())) for x in outputs]
                results = [clean(x) for x in results]
                golds = ["".join(vocab.ids_to_tokens(x.tolist())) for x in batch.src]
                golds = [clean(x) for x in golds]
                ss_golds = batch.ss.tolist()
                if ss_scores != None:
                    ss_pred = ss_scores.max(-1)[1].tolist()
                    for gold, gts ,res, ss in zip(golds, ss_golds, results, ss_pred):
                        ss = ss[:len(res)]
                        fout.write('[GOLD]\t{}\n[PRED]\t{}\n[GOLD_SS]\t{}\n[PRED_SS]\t{}\n'.format(gold, res, json.dumps(gts), json.dumps(ss)))
                else:
                    for gold,res in zip(golds, results):
                        fout.write('[GOLD]\t{}\n[PRED]\t{}\n'.format(gold, res))


def sample_vae(args, device_id, pt, step):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('[INFO] Loading checkpoint from %s' % test_from)

    vocab = Vocabulary(args.vocab_path, args.vocab_size)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = MODELTYPE[args.arch](args, device, checkpoint)
    model.init_model(checkpoint)
    model.to(device)
    model.eval()
    logger.info(model)

    filterSS = FilterFunc[args.filter_func]

    log_dir = args.log_dir + '/sample'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    part = 'all'
    def valid_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, part, shuffle=False),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False)

    save_file = os.path.join(log_dir, 'sample_{}_{}.txt'.format(args.sample, args.step))
    sample_num = args.test_batch_size
    valid_num = 0
    with open(save_file, 'w') as fout:
        while valid_num < sample_num:
            valid_iter = valid_iter_fct()
            for batch in valid_iter:
                src, tgt = batch.src, batch.tgt
                src = src.view(args.batch_size, args.sub_book, -1)
                tgt = tgt.view(args.batch_size, args.sub_book, -1)
                embed_id = getCandPool(src, tgt, randomnum=args.randomnum, ratio=args.mask_ratio)
                embed_id = embed_id.contiguous().permute(0, 2, 1)
                _, _, ss_scores = model.sample(embed_id, generation=False)
                ss_pred = ss_scores.max(-1)[1]

                # check SS
                new_embed_id = []
                for eid, ss in zip(embed_id, ss_pred):
                    if not filterSS(ss):
                        continue
                    else:
                        new_embed_id.append(eid)
                if len(new_embed_id) > 0:
                    new_embed_id = torch.stack(new_embed_id)
                else:
                    continue

                if args.arch == 'cvqvae2':
                    outputs, _, ss_scores = model.iterative_sample(new_embed_id, iteration=1)
                else:
                    outputs, _, ss_scores = model.sample(new_embed_id)

                ss_pred = ss_scores.max(-1)[1]
                results = [vocab.ids_to_tokens(x.tolist()) for x in outputs]
                # ss_pred = ss_pred.tolist()
                # print(results, ss_pred)
                # print(len(results[0]), len(ss_pred[0]))
                cnt = 0
                for res, ss in zip(results, ss_pred):
                    new_res,new_ss = "", torch.zeros(ss.size(), dtype=torch.int)
                    for i, (aa, s) in enumerate(zip(res, ss)):
                        if aa == "</s>":
                            break
                        if aa not in ['<PAD>','<UNK>','<MASK>','<s>','</s>']:
                            new_res += aa
                            new_ss[i] = s
                    new_ss = new_ss[:len(new_res)]
                    if filterSS(new_ss):
                        cnt += 1
                        fout.write('{}\n{}\n'.format(new_res, json.dumps(new_ss.tolist())))
                        fout.flush()
                valid_num += cnt
                if valid_num >= sample_num:
                    break


def predict_vae(args, device_id, pt=''):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('[INFO] Loading checkpoint from %s' % test_from)

    vocab = Vocabulary(args.vocab_path, args.vocab_size)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = MODELTYPE[args.arch](args, device, checkpoint)
    model.init_model(checkpoint)
    model.to(device)
    model.eval()
    logger.info(model)

    log_dir = args.log_dir + '/predict'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    filename = args.test_name.split('/')[-1].split('.')[0]
    valid_loader = data_loader.Dataloader(args, load_raw_json(args.test_name, vocab),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False)

    save_file = os.path.join(log_dir, 'predict_{}.txt'.format(filename))
    with open(save_file, 'w') as fout:
        for batch in valid_loader:
            src = batch.src
            tgt = batch.tgt
            mask_src = batch.mask_src
            mask_tgt = batch.mask_tgt

            outputs, _ = model(src, tgt, mask_src, mask_tgt)
            ss_feats = outputs[-1]
            ss_scores = model.ss_generator(ss_feats)
            ss_pred = ss_scores.max(-1)[1]
            results = [vocab.ids_to_tokens(x.tolist()) for x in src]
            ss_pred = ss_pred.tolist()
            for res, ss in zip(results, ss_pred):
                new_res,new_ss = "",[]
                for aa, s in zip(res, ss):
                    if aa == "</s>":
                        break
                    if aa not in ['<PAD>','<UNK>','<MASK>','<s>']:
                        new_res += aa
                        new_ss.append(s)
                fout.write('{}\n{}\n'.format(new_res, json.dumps(new_ss)))

def getCandPool(cand, gold, randomnum=1, ratio=1):
    """
        cand: [batch_size, sub_book, seq_len_1]
        gold: [batch_size, sub_book, seq_len_2]
    """
    if randomnum == 0:
        return gold
    bz, sub, _ = gold.size()
    # print(gold)
    seq_len = min(gold.size(-1), cand.size(-1))
    new_id = random.sample(range(sub), randomnum)
    if ratio == 1:
        gold = gold[:,:, :seq_len]
        gold[:, new_id, :] = cand[:, new_id, :seq_len]
    else:
        for line in new_id:
            prob = torch.rand(bz, seq_len)
            idx = (prob <= ratio)
            size, device = gold[:, line, :][idx].size(), gold.device
            gold[:, line, :][idx] = torch.randint(gold.min(), gold.max(), size).to(device)
    return gold
