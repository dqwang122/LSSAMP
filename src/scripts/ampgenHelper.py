
import bisect
import gc
import glob
from os import replace
import random
import os
import re
import copy
import json
import argparse
import csv

import numpy as np


############################## Func ##############################

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

def prepare(path):
    with open(path) as f:
        data = json.load(f)
    print(f"Total number of entries: {len(data)}")

    print("Identifying poorly formatted entries...")
    errors = []
    for i, d in enumerate(data):
        try:
            d["peptideCard"]["complexity"]
        except KeyError:
            errors.append(i)
            print(
                f"\tEntry {i} is poorly formatted:\n{json.dumps(d, sort_keys=True, indent=4)}\n"
            )

    print("Filtering poorly formatted entries...")
    for error in errors:
        data = data[:error] + data[error + 1 :]
    print(f"Entries after filtering: {len(data)}")

    print("Filtering entries with missing values...")
    pre_val_filter = len(data)
    keys = ["id", "seq", "complexity", "targetGroups", "targets"]
    skip = False
    results = []
    for d in data:
        for key in keys:
            if key not in d["peptideCard"].keys():
                skip = True
        if skip:
            skip = False
            continue

        ins = {}
        for key in keys:
            ins[key] = d["peptideCard"][key]
        results.append(ins)

    post_val_filter = len(results)
    print(f"Removed {pre_val_filter - post_val_filter} rows with missing values.")
    return results



############################## Mode ##############################

def convertDbsaap(args):
    data = prepare(args.i)
    if args.mapping:
        mapping=json.load(open(args.mapping))
        for d in data:
            d['targets'] = [mapping.get(x, '') for x in d["targets"]]
            d['seq'] = d['seq'].upper()
    saveJson(data, args.o)

def convertAvpdb(args):
    head = None
    data = []
    with open(args.i) as fin:
        for line in fin:
            parts = line.strip().split('\t')
            if not head:
                head = [x.title() for x in parts]
                continue
            ins = {}
            for i, k in enumerate(head):
                ins[k] = parts[i]
            data.append(ins)

    if args.mapping:
        mapping=json.load(open(args.mapping))
        for d in data:
            if 'Target' in d.keys():
                d['targets'] = mapping.get(d["Target"], '')
            d['seq'] = d['Sequence'].upper()
    saveJson(data, args.o)

def convertCamp(args):
    data = []
    head = None
    csv_reader = csv.reader(open(args.i))
    for line in csv_reader:
        if not head:
            head = ['Id'] + line[1:]
            continue
        ins = {}
        for i, k in enumerate(head):
            ins[k] = line[i]
        ins['seq'] = ins['Sequence'].upper()
        data.append(ins)

    if args.mapping:
        mapping=json.load(open(args.mapping))
        for d in data:
            if 'Target' in d.keys():
                d['targets'] = mapping.get(d["Target"], '')
    saveJson(data, args.o)

def splitDbsaap(args):
    data = readJson(args.i)
    pos = [d for d in data if 'Gram+' in d['targetGroups'] and 'Gram-' not in d['targetGroups'] ]
    neg = [d for d in data if 'Gram-' in d['targetGroups'] and 'Gram+' not in d['targetGroups'] ]
    both = [d for d in data if 'Gram+' in d['targetGroups'] and 'Gram-' in d['targetGroups'] ]
    saveJson(pos, args.o.replace('json', 'pos.json'))
    saveJson(neg, args.o.replace('json', 'neg.json'))
    saveJson(both, args.o.replace('json', 'both.json'))

def AlignSS(args):
    data = readJson(args.i)
    refer = readJson('peptide/raw_data/library.ss.jsonl')
    refer_dict = dict([(r['seq'], r['ss']) for r in refer])
    for d in data:
        d['ss'] = refer_dict.get(d['seq'], [])
    hits = [d for d in data if d['ss'] != []]
    saveJson(hits, args.o)

def AlignAPD(args):
    data = readJson(args.i)
    refer = readJson('peptide/raw_data/APD.ss.jsonl')
    refer_dict = dict([(r['seq'], r['ss']) for r in refer])
    novel = []
    for d in data:
        if d['seq'] not in refer_dict.keys():
            novel.append(d)
    print(f"{len(data)} | Differ from APD ({len(refer_dict)}) with {len(novel)}")
    saveJson(novel, args.o)


############################## Main ##############################


"""
    Usage:
        python scripts/ampgenHelper.py -mode convertCamp -i dataset/Drug/amp-gan/data/campr3/consolidated.csv  \
                                        -o dataset/Drug/amp-gan/data/campr3/campr3.json
        python scripts/ampgenHelper.py -mode convertAvpdb -i dataset/Drug/amp-gan/data/avpdb/AVPdb_data.tsv  \
                                        -mapping dataset/Drug/amp-gan/data/avpdb/targets_mapping.json \
                                        -o dataset/Drug/amp-gan/data/avpdb/AVPdb_data.json
        python scripts/ampgenHelper.py -mode convertDbsaap -i dataset/Drug/amp-gan/data/dbaasp/raw.json \
                                        -mapping dataset/Drug/amp-gan/data/dbaasp/targets_mapping.json \
                                        -o dataset/Drug/amp-gan/data/dbaasp/dbaasp.json
        python scripts/ampgenHelper.py -mode splitDbsaap -i dataset/Drug/amp-gan/data/dbaasp/dssap.json  \
                                        -o dataset/Drug/amp-gan/data/dbaasp/dssap.json
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='raw.json')
    parser.add_argument('-o', default='peptide.json')
    parser.add_argument('-mapping', default=None)
    parser.add_argument("-mode", default='fasta')
    parser.add_argument('-seed', default=888, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    eval('{}(args)'.format(args.mode))
