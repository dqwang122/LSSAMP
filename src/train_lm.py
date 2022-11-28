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
import json

import torch

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset, load_raw_dataset
from models.loss import lm_loss
from models.model_builder import ProteinLM, ProteinRNN
from models.predictor import build_predictor
from models.trainer import build_trainer
from models.tokenizer import Vocabulary
from others.logging import logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']

PAD_TOKEN = 0
SENTENCE_START = 1
SENTENCE_END = 2
UNKNOWN_TOKEN = 3

MODELTYPE = {
    'tensorflow': ProteinLM ,
    'rnn': ProteinRNN, 
}


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def train_lm(args, device_id):
    if (args.gpu_num > 1):
        train_lm_multi(args)
    else:
        train_lm_single(args, device_id)


def train_lm_single(args, device_id):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
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
    # model = ProteinLM(args, device, checkpoint)
    optim = [model_builder.build_optim(args, model, checkpoint)]

    logger.info(model)

    # symbol for bos, eos, pad
    symbols = {'BOS': SENTENCE_START, 'EOS': SENTENCE_END, 'PAD': PAD_TOKEN}
    train_loss = lm_loss(model.generator, symbols, model.vocab_size, device, train=True,
                          label_smoothing=args.label_smoothing)

    trainer = build_trainer(args, device_id, model, optim, train_loss)

    trainer.train(train_iter_fct, args.train_steps, valid_iter_fct=valid_iter_fct, valid_steps=args.valid_steps)


def train_lm_multi(args):
    """ Spawns 1 process per GPU """

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

            train_lm_single(args, device_id)
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



def validate_lm(args, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            if (args.test_start_from != -1 and step < args.test_start_from):
                xent_lst.append((1e6, cp))
                continue
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if (i - max_step > 10):
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:5]
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            # test_abs(args, device_id, cp, step)
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args, device_id, cp, step)
                    # test_abs(args, device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def validate(args, device_id, pt='', step=-1):
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

    # model = ProteinLM(args, device, checkpoint)
    model = MODELTYPE[args.arch](args, device, checkpoint)
    model.eval()

    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False)

    symbols = {'BOS': SENTENCE_START, 'EOS': SENTENCE_END, 'PAD': PAD_TOKEN}
    valid_loss = lm_loss(model.generator, symbols, model.vocab_size, train=False, device=device)

    trainer = build_trainer(args, device_id, model, None, valid_loss)
    stats = trainer.validate(valid_iter, step)
    return stats.xent()


def test_lm(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    vocab = Vocabulary(args.vocab_path, args.vocab_size)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    # model = ProteinLM(args, device, checkpoint)
    model = MODELTYPE[args.arch](args, device, checkpoint)
    model.eval()

    def test_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, args.test_name, shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)
        
    # symbols = {'BOS': SENTENCE_START, 'EOS': SENTENCE_END, 'PAD': PAD_TOKEN}
    symbols = {'BOS': SENTENCE_START * 100, 'EOS': SENTENCE_END * 100, 'PAD': PAD_TOKEN * 100}
    predictor = build_predictor(args, vocab, symbols, model, logger)
    predictor.translate(test_iter_fct, step)

def sample_lm(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    vocab = Vocabulary(args.vocab_path, args.vocab_size)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    model = MODELTYPE[args.arch](args, device, checkpoint)
    model.eval()

    symbols = {'BOS': SENTENCE_START, 'EOS': SENTENCE_END, 'PAD': PAD_TOKEN}

    log_dir = args.log_dir + '/sample'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    save_file = os.path.join(log_dir, 'sample_{}_{}.txt'.format(args.sample, args.step))

    cnt = 0
    with open(save_file, 'w') as fout:
        while cnt < args.test_batch_size:
            sequences = model.sample(args.test_batch_size, symbols)         # [batch_size, seq_len]
            for seq in sequences:
                seq = [str(x) for x in seq.tolist() if x != 0]
                if len(seq) > 4:
                    cnt += 1
                    fout.write('{}\n'.format(' '.join(seq)))
                    fout.flush()



    

