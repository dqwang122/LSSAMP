import gc
import glob
import hashlib
import itertools
import json
import os
import random
from collections import Counter
from os.path import join as pjoin

import torch
from multiprocess import Pool

from others.logging import logger
from models.tokenizer import Vocabulary


PAD_TOKEN = 0
SENTENCE_START = 1
SENTENCE_END = 2
UNKNOWN_TOKEN = 3

ss2id={'H':1, 'B':2, 'E':3, 'G':4, 'I':5, 'T': 6, 'S':7, '-':0}
id2ss={v:k for k,v in ss2id.items()}
id2ss[8] = '-'

random.seed(666)

def readTxt(fname):
    data = []
    with open(fname) as fin:
        for line in fin:
            data.append(line.strip())
    logger.info("Loading {} from {}".format(len(data), fname))
    return data

def saveTxt(data, fname):
    with open(fname, 'w') as fout:
        for d in data:
            fout.write('{}\n'.format(d))
            fout.flush()
    logger.info("Saving {} to {}".format(len(data), fname))

def readJson(fname):
    data = []
    with open(fname) as fin:
        for line in fin:
            data.append(json.loads(line))
    logger.info("Reading {} example from {}".format(len(data), fname))
    return data

def saveJson(data, fname):
    with open(fname, 'w') as fout:
        for d in data:
            fout.write('{}\n'.format(json.dumps(d)))
            fout.flush()
    logger.info("Saving {} to {}".format(len(data), fname))


def uniqKey(data, key):
    if key not in data[0].keys():
        raise ValueError('{} is not the key of data'.format(key))
    ins = {d[key]:d for d in data}
    logger.info('Filter duplicate elements from {} to {}'.format(len(data), len(ins)))
    return list(ins.values())

def tokenizer(x, vocab):
    x = list(x)
    return vocab.tokens_to_ids(x)

def split_dataset_to_txt(args):
    if args.raw_path.endswith('.txt'):
        data = readTxt(args.raw_path)
    elif args.raw_path.endswith('.jsonl'):
        data = readJson(args.raw_path)
        data = [d['Sequence:'] for d in data]
    else:
        raise NotImplementedError
    random.shuffle(data)
    # valid_size, test_size = 3000, 3000
    # valid_size, test_size = 100, 100
    valid_size, test_size = int(len(data) * 0.05), int(len(data) * 0.05)
    train_size = len(data) - valid_size - test_size
    train, valid, test = data[:train_size], data[train_size:-test_size], data[-test_size:]
    saveTxt(train, f'{args.save_path}.train.txt')
    saveTxt(valid, f'{args.save_path}.valid.txt')
    saveTxt(test, f'{args.save_path}.test.txt')

def split_dataset_to_json(args):
    if args.raw_path.endswith('.txt'):
        data = readTxt(args.raw_path)
    elif args.raw_path.endswith('.jsonl'):
        data = readJson(args.raw_path)
    else:
        raise NotImplementedError
    data = uniqKey(data, 'seq')
    random.shuffle(data)
    # valid_size, test_size = 3000, 3000
    valid_size, test_size = 100, 100
    # valid_size, test_size = int(len(data) * 0.05), int(len(data) * 0.05)
    train_size = len(data) - valid_size - test_size
    train, valid, test = data[:train_size], data[train_size:-test_size], data[-test_size:]
    saveJson(train, f'{args.save_path}.train.jsonl')
    saveJson(valid, f'{args.save_path}.valid.jsonl')
    saveJson(test, f'{args.save_path}.test.jsonl')

def _format_to_json(args):
    dirpath = args.raw_path
    corpus_type = ['train', 'valid', 'test']
    suffix = args.suffix.split(',')
    
    vocab = Vocabulary(args.vocab_path, args.vocab_size)
    dataset = {}
    for t in corpus_type:
        data = {}
        for s in suffix:
            fname = f'{dirpath}.{t}.{s}'
            data[s] = readTxt(fname)
        
        dataset[t] = []
        if len(suffix) == 2:    # src, tgt            
            for src, tgt in zip(data['src'], data['tgt']):
                ex = {}
                ex['src_txt'], ex['tgt_txt'] = src, tgt
                ex['src'], ex['tgt'] = tokenizer(src, vocab), tokenizer(tgt, vocab)
                dataset[t].append(ex)
        else:
            for src in data[suffix[0]]:
                ex = {}
                ex['src_txt'] = src
                ex['src'] = tokenizer(src, vocab)
                ex['tgt_txt'], ex['tgt'] = ex['src_txt'], ex['src']
                dataset[t].append(ex)
            else:
                raise NotImplementedError

    return dataset


def _load_json(args):
    dirpath = args.raw_path
    corpus_type = ['train', 'valid', 'test']
    key = "seq"
    # key = "sequence"
    
    vocab = Vocabulary(args.vocab_path, args.vocab_size)
    dataset = {}
    for t in corpus_type:
        fname = f'{dirpath}.{t}.{args.suffix}'
        data = readJson(fname)
        
        dataset[t] = []
        for d in data:
            ex = {}
            ex['src_txt'] = d[key]
            ex['src'] = tokenizer(d[key], vocab)
            ex['tgt_txt'], ex['tgt'] = ex['src_txt'], ex['src']
            if 'ss_pred' not in d.keys():
                ex['ss'] = [ss2id[x] for x in d['ss']]
            else:
                ex['ss'], ex['ss_txt'] = d['ss_pred'], d['ss']
            dataset[t].append(ex)
    return dataset

def format_to_pt(args):
    if args.suffix == 'jsonl' or args.suffix == 'json':
        dataset = _load_json(args)
    else:
        dataset = _format_to_json(args)
    
    corpus_type = ['train', 'valid', 'test']
    for t in corpus_type:
        fname = f'{args.save_path}.{t}.pt'
        torch.save(dataset[t], fname)
        logger.info("Saving {} to {}".format(len(dataset[t]), fname))


def index_to_pt(args):
    """ python preprocess.py -mode index_to_pt -raw_path ../logs/cvqvae_ema_cls_4_ft/train/epoch_42_index.pt -save_path ../logs/cvqvae_ema_cls_4_ft/train/epoch_42_index
    """
    data = torch.load(args.raw_path)
    dsize, subnum = len(data), len(data[0])
    for i in range(subnum):
        subsets = [{'src':ins[i]} for ins in data]
        dataset = {}
        dataset['train'], dataset['valid'] = subsets[:int(dsize * 0.9)], subsets[int(dsize * 0.9):]
        torch.save(dataset['train'], f'{args.save_path}.{i}.train.pt')
        torch.save(dataset['valid'], f'{args.save_path}.{i}.valid.pt')
    dataset = [{'src':[list(x) for x in ins.values()]} for ins in data]
    print(dataset[0])
    torch.save(dataset, f'{args.save_path}.all.pt')
    torch.save(dataset[:dsize], f'{args.save_path}.all.train.pt')
    torch.save(dataset[dsize:], f'{args.save_path}.all.valid.pt')

def LM_to_pt(args):
    """ python preprocess.py -mode LM_to_pt -raw_path ../logs/ind_LM -save_path ../logs/ind_LM -step 1
    """
    corpus = {}
    for suffix in ['candidate', 'gold']:
        paths = [f'{args.raw_path}_{i}/generate/step_{args.step}.{suffix}' for i in range(4)]
        corpus[suffix] = []
        for p in paths:
            data = readTxt(p)
            data = [[int(x) for x in d.split()] for d in data]
            corpus[suffix].append(data)

    dataset = []
    parts, num = len(corpus['candidate']), len(corpus['candidate'][0])
    for i in range(num):
        cand, gold = [], []
        for j in range(parts):
            cand.append(corpus['candidate'][j][i])
            gold.append(corpus['gold'][j][i])
        dataset.append({'src':cand, 'tgt': gold})
    logger.info(f'Saving to {args.save_path}.step_{args.step}.all.pt')
    torch.save(dataset, f'{args.save_path}.step_{args.step}.all.pt')
    print(dataset[0])

def LM_to_pt_2(args):
    """ python preprocess.py -mode LM_to_pt_2 -raw_path ../logs/ind_LM -save_path ../logs/ind_LM -step 1
    """
    corpus = {}
    for suffix in ['candidate', 'gold']:
        path = f'{args.raw_path}/generate/step_{args.step}.{suffix}'
        data = readTxt(path)
        corpus[suffix] = data

    def split_book(ind):
        parts = ind.split('2 1')
        parts = [p.replace('1', '').replace('2','').replace('0','') for p in parts]
        ind_parts = [[int(x)-2 for x in p.split()] for p in parts]
        code_len = len(ind_parts[0])
        res_parts = sum(ind_parts[1:], [])
        min_len = min(code_len, int(len(res_parts) / 3))
        if min_len <= 0:
            return None
        code_ind = [ind_parts[0][:min_len]] + [res_parts[i:i+min_len] for i in range(0, len(res_parts), min_len)]
        if len(code_ind) >= 4:
            return code_ind[:4]
        else:
            return None


    dataset = []
    num = len(corpus['candidate'])
    for i in range(num):
        ex_cand, ex_gold = corpus['candidate'][i], corpus['gold'][i]
        # print(ex_cand)
        ex_cand_ind = split_book(ex_cand)
        ex_gold_ind = split_book(ex_gold)
        if ex_cand_ind != None and ex_gold_ind != None:
            dataset.append({'src':ex_cand_ind, 'tgt': ex_gold_ind})
    logger.info(f'Saving {len(dataset)} to {args.save_path}.step_{args.step}.all.pt')
    torch.save(dataset, f'{args.save_path}.step_{args.step}.all.pt')
    print(dataset[0])

    
def LM_to_pt_3(args):
    """ python preprocess.py -mode LM_to_pt_3 -raw_path ../logs/ind_LM -refer_path ../logs/cvqvae_ema_cls_13_ft_5/train/epoch_149_index -save_path ../logs/ind_LM -step 1
    """
    corpus = {}
    paths = [f'{args.raw_path}_{i}/sample/sample_top_{args.step}.txt' for i in range(4)]
    corpus['candidate'] = []
    for p in paths:
        data = readTxt(p)
        data = [[int(x) for x in d.split()] for d in data]
        corpus['candidate'].append(data)

    refers = [f'{args.refer_path}.{i}.train.pt' for i in range(4)]
    corpus['gold'] = []
    for p in refers:
        data = torch.load(p)
        data = [list(d['src']) for d in data]
        corpus['gold'].append(data)
    
    dataset = []
    parts = len(corpus['candidate'])
    num = min([len(corpus['candidate'][i]) for i in range(parts)])
    gold_size = min([len(corpus['gold'][i]) for i in range(parts)])
    for i in range(num):
        cand, gold = [], []
        for j in range(parts):
            cand.append(corpus['candidate'][j][i])
            gold.append(corpus['gold'][j][i % gold_size])
        dataset.append({'src':cand, 'tgt': gold})
    logger.info(f'Saving {num} to {args.save_path}.step_{args.step}.all.pt')
    torch.save(dataset, f'{args.save_path}.step_{args.step}.all.pt')
    print(dataset[0])