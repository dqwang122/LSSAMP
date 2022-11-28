import bisect
import gc
import glob
import random

import torch

from others.logging import logger

PAD_TOKEN = '<PAD>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
MASK_TOKEN = '<MASK>'


class Vocabulary(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file, max_size):
        """
        Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.
        :param vocab_file: string; path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first.
        :param max_size: int; The maximum size of the resulting Vocabulary.
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 # keeps track of total number of words in the Vocab
        self._num = 0  # read number of words

        for w in [PAD_TOKEN, SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r', encoding='utf-8') as vocab_f: # New : add the utf8 encoding to prevent error
            cnt = 0
            for line in vocab_f:
                cnt += 1
                pieces = line.strip().split(" ")
                w = pieces[0]
                if w in self._word_to_id:
                    logger.error('Duplicated word in vocabulary file Line %d : %s' % (cnt, w))
                    cnt -= 1
                    continue
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size - 1:
                    logger.info("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break
            self._num = cnt
        logger.info("Finished constructing vocabulary of %i total words. Last word added: %s", self._count, self._id_to_word[self._count-1])
        
        self._word_to_id[MASK_TOKEN] = self._count
        self._id_to_word[self._count] = MASK_TOKEN
        self._count += 1
        logger.info("Add Mask token {} with id={} (vocab_size={})".format(MASK_TOKEN, self._word_to_id[MASK_TOKEN], self._count))

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def ids_to_tokens(self, ids):
        tokens = [self.id2word(x) for x in ids]
        return tokens

    def tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            tokens = tokens.split()
        ids = [self.word2id(x) for x in tokens]
        return ids

    def size(self):
        """Returns the word size of the vocabulary"""
        return self._num

    def word_list(self):
        """Return the word list of the vocabulary"""
        return self._word_to_id.keys()

    def __len__(self):
        return self._count



class SSVocabulary(object):
    """Vocabulary class for mapping between words and ids (integers)"""
    def __init__(self, vocab_size, bos=False, eos=False):
        self.vocab_size = vocab_size
        self._completed = {'H':1, 'B':2, 'E':3, 'G':4, 'I':5, 'T': 6, 'S':7, '-':8}
        sm1 = {'H':1, 'G':2, 'I':3, 'E':4, 'T':5, '-':6}
        sm2 = {'H':1, 'G':1, 'I':1, 'E':2, 'T':2, '-':3}
        self._truncated = sm1 if vocab_size == 6 else sm2

        self.ss2id = {'PAD_TOKEN': 0}
        if vocab_size == 8:
            self.ss2id.update(self._completed)
        elif vocab_size == 6 or vocab_size == 3:
            self.ss2id.update(self._truncated)
        else:
            raise NotImplementedError

        self.id2ss = {v:k for k, v in self.ss2id.items()}

        self.bos, self.eos = None, None
        if bos:
            self.ss2id['SENTENCE_START'] = len(self.id2ss)
            self.bos = self.ss2id['SENTENCE_START']
            self.id2ss[self.bos] = 'SENTENCE_START'
        if eos:
            self.ss2id['SENTENCE_END'] = len(self.id2ss) 
            self.eos = self.ss2id['SENTENCE_END']
            self.id2ss[self.eos] = 'SENTENCE_END'


        self._count = len(self.id2ss)
        logger.info("Finished constructing vocabulary of %i total words. The amid size is %i", self._count, self.vocab_size)

    def trunc_ss(self, ssidx):
        self._completed_rev = {v:k for k, v in self._completed.items()}
        ss_str = [self._completed_rev[sidx] for sidx in ssidx]
        ss_idx = [self._truncated.get(s, self._truncated['-']) for s in ss_str]
        return ss_idx

    def token2idx(self, ss):
        if ss not in self.ss2id.keys():
            return self.ss2id['-']
        return self.ss2id[ss]

    def idx2token(self, idx):
        return self.id2ss[idx]

    def __len__(self):
        return self._count
        