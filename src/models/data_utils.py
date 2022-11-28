import bisect
import gc
import glob
from os import replace
import random
import os
import copy
import argparse

import numpy as np

from others.logging import logger
from models.tokenizer import Vocabulary


PAD_TOKEN = 0
SENTENCE_START = 1
SENTENCE_END = 2
UNKNOWN_TOKEN = 3

ss2id={'H':1, 'B':2, 'E':3, 'G':4, 'I':5, 'T': 6, 'S':7, '-':0}
id2ss={v:k for k,v in ss2id.items()}
id2ss[8] = '-'

def sample_index_2(args, ex, is_test):
    
    def remove_pad(x, pad_id=0):
        x = np.array(x)
        x = x[x!=pad_id]
        x = x - 1
        return list(x)

    def process(data):
        non_pad_src = [remove_pad(x) for x in data]               # [4, seq_len]
        min_len = min([len(x) for x in non_pad_src])
        index_seq = [x[:min_len] for x in non_pad_src]
        index_seq = [ins[:args.max_length] for ins in index_seq]
        return index_seq

    src = sum(process(ex['src']), [])
    tgt = sum(process(ex['tgt']), [])

    src_txt = src
    tgt_txt = tgt

    if(is_test):
        return src, tgt, src_txt, tgt_txt
    else:
        return src, tgt

def sample_index(args, ex, is_test):
    index_seq = ex['src']               # [4, seq_len]
    if len(index_seq[0]) > args.max_skip_length:
        return None
    src = [ins[:args.max_length] for ins in index_seq]
    src = sum(src, [])
    tgt = src

    src_txt = src
    tgt_txt = src

    if(is_test):
        return src, tgt, src_txt, tgt_txt
    else:
        return src, tgt

def parallel_index_all(args, ex, is_test):
    assert max(sum(ex['src'], [])) < args.vocab_size, "Can not add eos and bos to vocab"
    eos, bos = 1, 2
    index_seq = [[x + 3 for x in seq] for seq in ex['src'] ]
    if len(index_seq[0]) > args.max_skip_length:
        return None
    src = [ins[:args.max_length-2] for ins in index_seq]
    src = [[eos] + ins + [bos] for ins in src]
    src = sum(src, [])
    tgt = src

    src_txt = ' '.join([str(x) for x in src])
    tgt_txt = src_txt

    if(is_test):
        return src, tgt, src_txt, tgt_txt
    else:
        return src, tgt

def parallel_index(args, ex, is_test):
    index_seq = [x + 1 for x in ex['src'] ]
    if len(index_seq) > args.max_skip_length:
        return None
    src = index_seq[:args.max_length]
    tgt = index_seq[:args.max_length]

    src_txt = " ".join([str(x) for x in index_seq])
    tgt_txt = src_txt

    if(is_test):
        return src, tgt, src_txt, tgt_txt
    else:
        return src, tgt

def parallel_ss(args, ex, is_test):
    # src = ex['src'][:args.max_length-1] + [SENTENCE_END]
    if len(ex['src']) > args.max_skip_length or len(ex['tgt']) > args.max_skip_length:
        return None
    src = [SENTENCE_START] + ex['src'][:args.max_length-2] + [SENTENCE_END]
    tgt = [SENTENCE_START] + ex['tgt'][:args.max_tgt_len-2] + [SENTENCE_END]
    ss = ex['ss'][:args.max_length-2]

    def reduce_ss_1(ss):
        ss2id_sm = {'H':1, 'G':2, 'I':3, 'E':4, 'T':5, '-':6}
        ss_new = [ss2id_sm.get(id2ss[s],6) for s in ss]
        return ss_new

    def reduce_ss_2(ss):
        ss2id_sm = {'H':1, 'G':1, 'I':1, 'E':2, 'T':2, '-':3}
        ss_new = [ss2id_sm.get(id2ss[s],3) for s in ss]
        return ss_new

    if args.ss_type == 6:             # 6
        ss = reduce_ss_1(ss)
    elif args.ss_type == 3:           # 3
        ss = reduce_ss_2(ss)
    elif args.ss_type == 8:           # 8
        ss = [8 if s==0 else s for s in ss]

    src_txt = ex['src_txt']
    tgt_txt = ex['tgt_txt']

    if(is_test):
        return src, tgt, ss, src_txt, tgt_txt
    else:
        return src, tgt, ss

def parallel(args, ex, is_test):
    if len(ex['src']) > args.max_skip_length or len(ex['tgt']) > args.max_skip_length:
        return None
    src = [SENTENCE_START] + ex['src'][:args.max_length-2] + [SENTENCE_END]
    # tgt = ex['tgt'][:self.args.max_tgt_len-1] + [SENTENCE_END]
    tgt = [SENTENCE_START] + ex['tgt'][:args.max_tgt_len-2] + [SENTENCE_END]

    src_txt = ex['src_txt']
    tgt_txt = ex['tgt_txt']

    if(is_test):
        return src, tgt, src_txt, tgt_txt
    else:
        return src, tgt


def mask_lm(args, ex, is_test):
    MASK_TOKEN = args.vocab_size - 1

    src = ex['src']
    src_len = len(src)
    src_txt = copy.deepcopy(ex['src_txt'])
    tgt = copy.deepcopy(ex['src'])
    tgt = [SENTENCE_START] + tgt[:args.max_tgt_len-2] + [SENTENCE_END]

    mask_num = int(args.mask_ratio * src_len)
    if mask_num > 0:
        src = np.array(src)
        idx = np.random.choice(range(src_len), mask_num, replace=False)
        p = np.random.rand(mask_num)
    
        if len(idx[p <= 0.8]) > 0:                  # mask
            selected = idx[p <= 0.8]
            src[selected] = [MASK_TOKEN] * len(selected)
        if len(idx[p > 0.9]) > 0:                   # replace
            selected = idx[p > 0.9]
            src[selected] = np.random.randint(UNKNOWN_TOKEN+1, MASK_TOKEN, len(selected))
        if len(idx[(p > 0.8) & (p < 0.9)]) > 0:     # drop
            selected = idx[(p > 0.8) & (p < 0.9)]
            src = src[~selected]

        src = src.tolist()

    src = src[:args.max_length-1] + [SENTENCE_END]
    tgt_txt = ex['src_txt']

    # print(src, tgt)

    if(is_test):
        return src, tgt, src_txt, tgt_txt
    else:
        return src, tgt


def denoise_lm(args, ex, is_test):
    MASK_TOKEN = args.vocab_size - 1

    src = ex['src']
    src_len = len(src)
    tgt = [SENTENCE_START] + copy.deepcopy(ex['src']) + [SENTENCE_END]

    length = np.random.poisson(lam=3, size=src_len)
    idx = 0

    # xDAE

    mask_num = int(args.mask_ratio * src_len)
    if mask_num > 0:
        src = np.array(src)
        idx = np.random.choice(range(src_len),mask_num, replace=False)
        p = np.random.rand(mask_num)
    
        if len(idx[p <= 0.8]) > 0:
            src[idx[p <= 0.8]] = [MASK_TOKEN] * len(idx[p <= 0.8])
        if len(idx[p > 0.9]) > 0:
            src[idx[p > 0.9]] = random.sample(range(UNKNOWN_TOKEN+1, MASK_TOKEN), len(idx[p > 0.9]))

        src = src.tolist()

    src = src[:args.max_length-1] + [SENTENCE_END]

    src_txt = ex['src_txt']
    tgt_txt = ex['src_txt']

    # print(src, tgt)

    if(is_test):
        return src, tgt, src_txt, tgt_txt
    else:
        return src, tgt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-max_length", default=200, type=int)
    parser.add_argument("-max_tgt_len", default=200, type=int)
    parser.add_argument("-data_process", default='parallel')
    parser.add_argument('-vocab_size', default=27, type=int)
    parser.add_argument('-mask_ratio', default=0.2, type=float)
    parser.add_argument('-seed', default=888, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ex = {'src': [6,7,8,9,10]*4, 'tgt': [6,7,8,9,10]*4,
          'src_txt': 'RNDCQ'*4, 'tgt_txt': 'RNDCQ'*4}
    ret = mask_lm(args, ex, is_test=False)
    print(ret)
