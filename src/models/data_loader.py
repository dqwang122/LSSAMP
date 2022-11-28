import bisect
import gc
import glob
import random
import os
import json
from random import shuffle

import torch

from others.logging import logger
from models.tokenizer import Vocabulary
from models.data_utils import *

PAD_TOKEN = 0
SENTENCE_START = 1
SENTENCE_END = 2
UNKNOWN_TOKEN = 3



class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_tgt = [x[1] for x in data]
        
            src = torch.tensor(self._pad(pre_src, PAD_TOKEN))
            tgt = torch.tensor(self._pad(pre_tgt, PAD_TOKEN))

            mask_src = ~(src == PAD_TOKEN)  # mask_src = 1 - (src == 0)
            mask_tgt = ~(tgt == PAD_TOKEN)  # mask_tgt = 1 - (tgt == 0)


            setattr(self, 'src', src.to(device))
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))

            if not is_test and len(data[0]) >= 3:
                pre_ss = [x[2] for x in data]
                ss = torch.tensor(self._pad(pre_ss, PAD_TOKEN))
                setattr(self, 'ss', ss.to(device))

            if (is_test):
                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size


def batch_size_fn(new, count):
    src, tgt = new[0], new[1]
    global max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_tokens=0
    max_size = max(max_size, len(tgt))
    src_elements = count * max_size
    if (count > 6):
        return src_elements + 1e3
    return src_elements



class Dataloader(object):
    def __init__(self, args, datasets,  batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        self.batch_size_fn = batch_size_fn
        self.process = args.data_process

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        return eval('{}(self.args, ex, is_test)'.format(self.process))

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):
            p_batch = sorted(buffer, key=lambda x: len(x[1]))
            p_batch = self.batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b)==0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return



def load_dataset(args, corpus_type, shuffle=False, reversed=False):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'dev'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    # assert corpus_type in ["train", "valid", "test"]

    fname = f'{args.data_path}.{corpus_type}.pt'
    dataset = torch.load(fname)
    if reversed:
        reverse_dataset = [reverse(d) for d in dataset]
        dataset.extend(reverse_dataset)
        random.shuffle(dataset)
    logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, fname, len(dataset)))
    yield dataset


def load_raw_dataset(data_path, vocab: Vocabulary):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        data_path: string, txt format,  one example each line
        vocab: Vocabulary
    Returns:
        dataset
    """
    def tokenizer(x, vocab):
        return vocab.tokens_to_ids(list(x))

    dataset = []
    with open(data_path) as fin:
        for line in fin:
            src = line.strip()
            ex = {}
            ex['src_txt'] = src
            ex['src'] = tokenizer(src, vocab)
            ex['tgt'], ex['tgt_txt'] = [SENTENCE_START], '<BOS>'
            dataset.append(ex)

    logger.info('Loading raw dataset from %s, number of examples: %d' %
                    (data_path, len(dataset)))
    yield dataset

def load_raw_json(data_path, vocab: Vocabulary):
    def tokenizer(x, vocab):
        return vocab.tokens_to_ids(list(x))

    dataset = []
    with open(data_path) as fin:
        for line in fin:
            data = json.loads(line)
            src = data['seq']
            ex = {}
            ex['src_txt'] = src
            ex['src'] = tokenizer(src, vocab)
            ex['tgt'], ex['tgt_txt'] = ex['src'], ex['src_txt']
            dataset.append(ex)

    logger.info('Loading raw dataset from %s, number of examples: %d' %
                    (data_path, len(dataset)))
    yield dataset



def reverse(ex):
    if isinstance(ex, dict):
        re_ex = {}
        for k, v in ex.items():
            re_ex[k] = v[::-1]
        return re_ex
    elif isinstance(ex, list):
        return ex[::-1]
    return None