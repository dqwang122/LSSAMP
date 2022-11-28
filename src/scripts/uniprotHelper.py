
import bisect
import gc
import glob
from os import replace
import random
import os
import copy
import json
import argparse

import numpy as np

############################## Func ##############################

def readTxt(fname):
    data = []
    with open(fname, 'rb') as fin:
        for line in fin:
            data.append(line.decode('utf-8').strip())
    print("Reading {} example from {}".format(len(data), fname))
    return data

def readJson(fname):
    data = []
    with open(fname) as fin:
        for line in fin:
            data.append(json.loads(line))
    print("Reading {} example from {}".format(len(data), fname))
    return data

def saveJson(data, fname):
    with open(fname, 'w') as fout:
        for d in data:
            fout.write('{}\n'.format(json.dumps(d, ensure_ascii=False)))
    print('Save {} example to {}'.format(len(data), fname))

def saveTxt(data, fname):
    with open(fname, 'w') as fout:
        for d in data:
            fout.write('{}\n'.format(d))
    print('Save {} example to {}'.format(len(data), fname))


def parseFastaline(line):
    """
        example: 
            >sp|P0DPI4|TDB01_HUMAN T cell receptor beta diversity 1 OS=Homo sapiens OX=9606 GN=TRBD1 PE=4 SV=1
    """
    parts = line.split('|')
    assert len(parts) == 3
    code = parts[1]
    sub_parts = parts[-1].split()
    name = sub_parts[0]
    
    attr = {}
    others = []
    for s in sub_parts[1:]:
        if '=' in s:
            key, value = s.split('=')
            attr[key] = value
        else:
            others.append(s)
    attr['others'] = ' '.join(others)
    return code, name, attr

def parseClusterLine(line):
    """ 
        example: 
            1	11aa, >sp|B3A0K3|FAR9_PACBA... at 45.45%
    """
    parts = line.split('\t')
    assert len(parts) == 2
    idx, content = parts[0], parts[1]

    content = content.split(',')
    assert len(content) == 2
    head, attr  = content[0], content[1]

    attr = attr.split('|')
    assert len(attr) == 3
    code = attr[1]
    name = attr[2].split('...')[0]
    similar = attr[2].split(' ')[-1]

    return {'idx': int(idx), 'head': head, 
            'accession': code, 'entry':name, 'similarity': similar}


############################## Mode ##############################


def do_fasta(args):
    data = []
    ex = {}
    with open(args.i) as fin:
        for line in fin:
            line = line.strip()
            if line.startswith('>'):
                if ex != {}:
                    ex['sequence'] = ''.join(ex['sequence'])
                    data.append(ex)
                code, name, attr = parseFastaline(line)
                ex = {'accession':code, 'entry': name, 'attr': attr, 'sequence':[]}
            else:
                ex['sequence'].append(line)
    if ex != {}:
        ex['sequence'] = ''.join(ex['sequence'])
        data.append(ex)
    saveJson(data, args.o)
    return data

def do_cluster(args):
    cluster = []
    sequence = []
    with open(args.i) as fin:
        for line in fin:
            line = line.strip()
            if line.startswith('>'):
                if sequence != []:
                    cluster.append(sequence)
                    sequence = []
            else:
                seq = parseClusterLine(line)
                sequence.append(seq)
    if sequence != {}:
        cluster.append(sequence)
    saveJson(cluster, args.o)
    return cluster

def get_info(args):
    data = readJson(args.i)
    seq_len = [len(d['sequence']) for d in data]
    print('Sequence length: min {}, max {}, avg {:.3f}'.format(min(seq_len), max(seq_len), sum(seq_len)/len(seq_len)))
    

############################## Main ##############################


""" 
    Usage:
        python uniprotHelper.py -mode do_fasta -i ../raw_data/uniprot_len100/peptide-reviewed.fasta -o ../raw_data/uniprot_len100/peptide.fasta.jsonl
        python uniprotHelper.py -mode do_cluster -i ../raw_data/uniprot_len100/peptide-reviewed.clstr -o ../raw_data/uniprot_len100/peptide.clstr.jsonl
        python uniprotHelper.py -mode get_info -i ../raw_data/uniprot/uniprot_all.jsonl
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='peptide.fasta')
    parser.add_argument('-o', default='peptide.json')
    parser.add_argument("-mode", default='fasta')
    parser.add_argument('-seed', default=888, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    eval('{}(args)'.format(args.mode))