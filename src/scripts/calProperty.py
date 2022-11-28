
import bisect
import gc
import glob
from os import replace
import random
import os
import copy
import json
import math
import argparse
import pickle as pkl
import pandas as pd

import numpy as np
from collections import Counter

from scipy.special import kl_div
from scipy.stats import entropy,norm

from others.utils import read_json, readLog, save_json
from modlamp.core import load_scale
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor

CThreMin=2
CThreMax=10
uHThre=1.75
HThre=0.25
DThre=2
SThre=0.3

AA = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
sstype=["*", "H", "B", "E", "G", "I", "T", "S", "-"]
ss2str = {k:v for k,v in enumerate(sstype)}
str2ss = {v:k for k, v in ss2str.items()}
str2ss['*'] = 8
    
############################## Func ##############################

def ss_code(ss):
    result=[]
    for t in ss:
        result.append(ss2str[t])
    return result

def net_charge(seq):
    """calculate net charge, pH=7.4"""
    netCharge=0.0
    charge={"X":0, "I":0, "V":0, "L":0, "F":0, "M":0, "A":0, "C":0, "W":0, "G":0, "Y":0, "P":0, "T":0, "S":0, "H":0, "Q":0, "N":0, "E":-1, "D":-1, "K":1, "R":1}
    for aa in seq:
        netCharge+=charge[aa]
    return netCharge

def ngram(seq, n):
    return [seq[i:i+n] for i in range(0, len(seq)-n+1)]

def calculate_moment(array, angle=100):
    if len(array) == 0:
        return 0
    sum_cos, sum_sin = 0.0, 0.0
    for i, hv in enumerate(array):
        rad_inc = ((i * angle) * math.pi) / 180.0
        sum_cos += hv * math.cos(rad_inc)
        sum_sin += hv * math.sin(rad_inc)
    return math.sqrt(sum_cos ** 2 + sum_sin ** 2) / len(array)

def calculate_hydrophobicity(seq):
    seq = seq.replace('X', '')
    d = PeptideDescriptor(seq)
    d.calculate_global()
    return float(d.descriptor[0])

def modified_moment(seq, ss):
    if ss == []:
        ss = [1] * len(seq)
    data = [(x, s) for x, s in zip(seq, ss) if x != 'X']
    seq, ss = list(zip(*data))
    scale = load_scale('eisenberg')[1]
    seq = np.array([scale[x][0] for x in seq])
    ss=np.array(ss_code(ss))
    seq_Helix = seq[(ss=="H")|(ss=="G")|(ss=="I")]
    A_mom = calculate_moment(seq_Helix, angle=100)
    seq_Beta = seq[(ss=="E")|(ss=="T")]
    B_mom = calculate_moment(seq_Beta, angle=180)
    mom = A_mom + B_mom
    return mom

def calcluate_charge(seq):
    desc = GlobalDescriptor(seq)
    desc.calculate_charge(ph=7.4, amide=True)
    return float(desc.descriptor[0][0])

def cal_characteristic(seq):
    glob = GlobalDescriptor(seq)
    glob.calculate_all()
    state = {k:v for k, v in zip(glob.featurenames, glob.descriptor[0])}
    desc = PeptideDescriptor(seq)
    desc.calculate_profile(prof_type='uH')
    state['uH'] = desc.descriptor[0]    # [slope, intercept]
    desc.calculate_profile(prof_type='H')
    state['H'] = desc.descriptor[0]     # [slope, intercept]
    return state


def calDist(refer, hypo, mode='distance'):
    import Levenshtein as L
    if mode == 'distance':
        dist_func = L.distance
    elif mode == 'hamming':
        dist_func = L.hamming
    else:
        raise NotImplementedError
    
    dist_list, close_list = [], []
    overlap = 0
    for h in hypo:
        mindist = math.inf
        close_str = ""
        for r in refer:
            if h == r:
                overlap += 1
                dist, close_str = 0, r
            else:
                dist = dist_func(r, h)
            if dist == 0:
                mindist = dist
                break
            elif dist < mindist:
                mindist = dist 
                close_str = r
        dist_list.append(mindist)
        close_list.append(close_str)
    
    print('Overlap: ', overlap)
    print('Average: ', sum(dist_list)/sum([len(x) for x in hypo]))
    return dist_list, close_list

def solubility(seq):
    trigram = ngram(seq, 3)
    strong = set(['W', 'F', 'Y'])
    for item in trigram:
        if len(set(item) - set(strong)) == 0:
            # print(set(item), strong)
            return False

    fifthgram = ngram(seq, 5)
    weak = set(['A', 'V', 'I', 'L', "M", 'P'])
    for item in fifthgram:
        if len(set(list(item)) - set(weak)) == 0:
            # print(set(item), weak)
            return False
    
    total = strong | weak
    cnt = 0
    for a in seq:
        # if a in total:
        if a in strong:
            cnt += 1
    ratio = cnt / len(seq)
    if ratio > SThre:
        return False
    return True

def FilterSet(results, strict=False):
    filters = {}
    filters['charge']=list(filter(lambda r: r['charge'] >= CThreMin and r['charge'] <= CThreMax, results))
    filters['H']=list(filter(lambda r: r['H'] >= HThre, results))
    # filters['uH']=list(filter(lambda r: (r['uH'] >= uHThre), results))
    filters['uH']=list(filter(lambda r: (r['uH'] >= uHThre or (r['uH'] >= 0.5 and r['uH'] <= 0.75)), results))
    if strict:
        filters['solubility']=list(filter(lambda r: solubility(r['seq']), results))
        filters['combine'] = [x for x in filters['charge'] if x in filters['H'] and x in filters['uH'] and x in filters['solubility']]
        l1, l2, l3, l4, l5 = len(filters['charge']), len(filters['H']), len(filters['uH']), len(filters['solubility']), len(filters['combine'])
        print(f'{len(results)} | net in [{CThreMin}, {CThreMax}] -> {l1}, H >= {HThre} -> {l2}, uH >= {uHThre} -> {l3}, solubility >= {SThre} -> {l4}, combine -> {l5}')
    else:
        filters['combine'] = [x for x in filters['charge'] if x in filters['H'] and x in filters['uH']]
        l1, l2, l3, l4 = len(filters['charge']), len(filters['H']), len(filters['uH']), len(filters['combine'])
        print(f'{len(results)} | net in [{CThreMin}, {CThreMax}] -> {l1}, H >= {HThre} -> {l2}, uH >= {uHThre} + [0.5, 0.75]-> {l3}, combine -> {l4}')
    return filters


def FilterbyDist(results):

    def dist_func(r, core=[3.3489, 0.08679, 0.5463]):
        return math.sqrt((r['charge'] - core[0])**2 + (r['H'] - core[1])**2 + (r['uH'] - core[2])**2)

    filters = {}
    dist_mean, dist_std = 2.5034, 2.2020
    min_dist_thre, max_dist_thre = dist_mean-dist_std, dist_mean+dist_std
    # min_dist_thre, max_dist_thre = 0.0, 4.7
    filters['distance'] = list(filter(lambda r: dist_func(r) >= min_dist_thre and dist_func(r) <= max_dist_thre, results))
    l1 = len(filters['distance'])
    print(f'{len(results)} | Distance in [{min_dist_thre:0.4}, {max_dist_thre:0.4}] -> {l1}({l1*100/len(results):0.4}%)')
    return filters['distance']

def Diversity(refer, hypo, mode='distance'):
    import Levenshtein as L
    if mode == 'distance':
        dist_func = L.distance
    elif mode == 'hamming':
        dist_func = L.hamming
    else:
        raise NotImplementedError
    
    dist_list = []
    min_list = []
    for h in hypo:
        dist_all = []
        mindist = math.inf
        for r in refer:
            if h == r:
                continue
            dist = dist_func(r, h)
            mindist = min(mindist, dist)
            dist_all.append(dist / len(h))
        min_list.append(mindist / len(h))
        dist_list.append(sum(dist_all) / len(dist_all))
    
    print('Sequences', len(dist_list))
    print('Average Similarity: ', sum(dist_list)/len(dist_list))
    print('Minimum Distance: ', sum(min_list)/len(min_list))
    return dist_list

############################## Module ##############################

def EvalSample(args):
    sequences, structures = readLog(args.i)
    
    results = []
    uniqSeq = []
    for seq, ss in zip(sequences, structures):
        seq = seq.upper()
        if len(seq) <= 1:
            continue
        if seq in uniqSeq:
            continue
        else:
            uniqSeq.append(seq)
        charge = net_charge(seq)
        # charge = calcluate_charge(seq)
        H= calculate_hydrophobicity(seq)
        uH = modified_moment(seq, ss)
        results.append({'charge':charge, 'uH':uH, 'seq': seq, 'ss': ss, 'H': H})
    print('{} | Uniq -> {}'.format(len(sequences), len(results)))
    FilterSet(results)
    save_json(results, args.o)

def EvalBaseline(args):
    data = read_json(args.i)
    results = []
    for d in data:
        if 'ss_pred' not in d.keys():
            d['ss_pred'] = [str2ss[s] for s in d['ss']]
        seq, ss = "", []
        for a, s in zip(d['seq'], d['ss_pred']):
            if a.upper() in AA:
                seq += a
                ss.append(s)
        if len(seq) <= 1:
            continue
        charge = net_charge(seq)
        H= calculate_hydrophobicity(seq)
        uH = modified_moment(seq, ss)
        results.append({'charge':charge, 'uH':uH, 'seq': seq, 'ss': ss, 'H': H})
    FilterSet(results, strict=True)
    FilterbyDist(results)
    save_json(results, args.o)

def EvalRandom(args):
    data = read_json(args.i)
    results = []
    while len(results) < args.num:
        for d in data:
            try:
                seq, ss = d['seq'], d['ss_pred']
            except:
                seq, ss = d['seq'], []
            N = len(seq)
            if len(seq) <= 1:
                continue
            idx = random.sample(range(N), int(N * args.ratio))
            nseq = ''.join([random.choice(AA) if i in idx else seq[i] for i in range(len(seq))])
            charge = net_charge(nseq)
            H= calculate_hydrophobicity(nseq)
            uH = modified_moment(nseq, ss)
            results.append({'charge':charge, 'uH':uH, 'seq': nseq, 'ss': ss, 'H': H})
            if len(results) >= args.num:
                break
    FilterSet(results)
    save_json(results, args.o)

def Align(args):
    parts = args.i.split(',')
    assert len(parts) == 2
    refer, hypo = read_json(parts[0]), read_json(parts[1])
    refer_seqs = [r['seq'].upper() for r in refer]
    hypo_seqs = [h['seq'].upper() for h in hypo]
    refer_info = {r['seq'].upper(): r for r in refer}
    dist_list, close_list = calDist(refer_seqs, hypo_seqs)
    fout = open(args.o, 'w')
    uniqfasta = open(args.o.replace('jsonl', 'fasta'), 'w')
    uniq=0
    uniq_hypo = []
    for h, d, s in zip(hypo, dist_list, close_list):
        if 'ss' in refer_info[s].keys():
            h['mindist'], h['align'], h['align_ss'] = d, s, refer_info[s]['ss']
        else:
            h['mindist'], h['align'], h['align_ss'] = d, s, ''
        fout.write(json.dumps(h) + '\n')
        if h['mindist'] != 0:
            uniq += 1
            uniq_hypo.append(h)
            uniqfasta.write('>sp|{}\n{}\n'.format(uniq, h['seq']))
    fout.close()
    uniqfasta.close()

    accepted_hypo = FilterbyDist(uniq_hypo)
    accepted_dist_list = [d['mindist'] for d in accepted_hypo if d['mindist'] != 0]
    print('{} | Average: {:0.4} '.format(len(accepted_dist_list), sum(accepted_dist_list)/sum([len(x['seq']) for x in accepted_hypo])))

    print('Saving aligned information to {}'.format(args.o))
    print('Saving {} uniq examples to {}'.format(uniq, args.o.replace('jsonl', 'fasta')))

def CalStat(args):
    sequences = []
    structures = []
    if args.i.endswith('jsonl'):
        data = read_json(args.i)
        for d in data:
            seq, ss = d['seq'], d['ss_pred']
            sequences.append(seq)
            structures.append(''.join(ss_code(ss)))
    elif args.i.endswith('txt') :
        sequences, structures = readLog(args.i)
        structures = [''.join(ss_code(ss)) for ss in structures]
    else:
        raise NotImplementedError

    sequences_str = ''.join(sequences)
    structures_str = ''.join(structures)
    seq_count = Counter(sequences_str)
    ss_count = Counter(structures_str)
    seq_ratio = {k:round(v/len(sequences_str)*100, 2) for k, v in seq_count.items()}
    ss_ratio = {k:round(v/len(sequences_str)*100, 2)  for k, v in ss_count.items()}
    print(seq_ratio)
    print(ss_ratio)


def ParseCAMP(args):
    results = {}
    with open(args.i) as fin:
        subset = []
        classifier = ''
        for line in fin:
            if line.startswith('Results with'):
                if subset != []:
                    results[classifier] = subset
                classifier = line.strip()[len('Results with '):][:-len(' classifier')]
                subset = []
            elif len(line.split('\t')) >= 2 and not line.startswith('Seq. ID.'):
                line = line.strip()
                parts = line.split('\t')
                assert len(parts) >= 2
                ex = {'id': parts[0], 'category': parts[1]}
                if len(parts) >= 3:
                    ex['prob'] = parts[2]
                subset.append(ex)
            else:
                continue
        results[classifier] = subset
    for key, value in results.items():
        cnt = len(value)
        AMP = [x for x in value if x['category']=='AMP']
        nAMP = [x for x in value if x['category']=='NAMP']
        print('{cls}\n\tAMP: {amp}/{cnt} ({ramp:.2f}%), nAMP: {namp}/{cnt} ({rnamp:.2f}%)'.format(
                    cls = key,
                    amp = len(AMP), namp=len(nAMP), cnt=cnt,
                    ramp = len(AMP)*100/cnt, rnamp = len(nAMP)*100/cnt))
    json.dump(results, open(args.o, 'w'))
    

def SelectCand(args):
    parts = args.i.split(',')
    if len(parts) >= 2:
        seq_file, pred_file = parts[0], parts[1]
    else:
        seq_file, pred_file = parts[0], None
    data = read_json(seq_file)
    data = [d for d in data if d['mindist'] != 0]

    if pred_file != None:
        pred = json.load(open(pred_file))
        vote = {}
        for key, value in pred.items():
            for v in value:
                id = v['id']
                if id not in vote:
                    vote[id] = {'classifier':[], 'AMP':0}
                vote[id]['classifier'].append(v['category'])
                if v['category'] == 'AMP':
                    vote[id]['AMP'] += 1

        for i, seq in enumerate(data):
            seq['CAMP'] = vote[str(i+1)]

    # filters
    # filters = FilterSet(data, strict=True)
    filters = FilterSet(data, strict=False)
    seq = filters['combine']
    if pred_file:
        seq = [s for s in seq if s['CAMP']['AMP'] >= 4]
        print(f'{len(data)} | CLS >= 4 -> {len(seq)}')
    # seq = [s for s in seq if s['mindist'] >= DThre]
    # print(f'{len(data)} | mindist >= {DThre} -> {len(seq)}')
    save_json(seq, args.o)
    

def SelectOnly(args):
    fasta_file, pred_file = args.i.split(',')
    data = [{'seq':line.strip()} for line in open(fasta_file) if not line.startswith('>')]
    pred = json.load(open(pred_file))
    vote = {}
    for key, value in pred.items():
        for v in value:
            id = v['id']
            if id not in vote:
                vote[id] = {'classifier':[], 'AMP':0}
            vote[id]['classifier'].append(v['category'])
            if v['category'] == 'AMP':
                vote[id]['AMP'] += 1

    for i, seq in enumerate(data):
        seq['CAMP'] = vote[str(i+1)]
        
    save_json(data, args.o)

def Log2Fasta(args):
    fout = open(args.o, 'w')
    sequences, structures = readLog(args.i)
    uniqSeq = []
    for seq, ss in zip(sequences, structures):
        if len(seq) <= 1:
            continue
        if seq in uniqSeq:
            continue
        else:
            uniqSeq.append(seq)
            fout.write('>sp|{}\n{}\n'.format(json.dumps(ss), seq))
    fout.close()
    print("Saving {} example from {}".format(len(uniqSeq), args.o))

def Json2Fasta(args):
    fout = open(args.o, 'w')
    data = read_json(args.i)
    uniqSeq = []
    for d in data:
        seq = d['seq'].replace('X', '')
        d.pop('ss_prob')
        if seq not in uniqSeq:
            fout.write('>sp|{}\n{}\n'.format(json.dumps(d), seq))
            uniqSeq.append(seq)
    fout.close()
    print("Saving {} example to {}".format(len(uniqSeq), args.o))

def CalKLDist(args):

    def fit_norm(v, box, x):
        counts, bins = np.histogram(v, range=box, bins=25)
        counts = counts / len(v1)
        mean, std = np.mean(v), np.std(v)
        y = norm(mean, std).pdf(x)
        return y

    f1, f2 = args.i.split(',')
    d1, d2 = read_json(f1), read_json(f2)
    d2 = [d for d in d2 if d['mindist'] != 0]
    metric = {}
    boxes = {'charge': [-5, 20], 'H':[-1, 1.0], 'uH': [0, 4]}
    for k in ['charge', 'H', 'uH']:
        v1 = np.array([d[k] for d in d1])
        v2 = np.array([d[k] for d in d2])
        x = np.random.uniform(boxes[k][0], boxes[k][1], 1000)
        y1 = fit_norm(v1, boxes[k], x)
        y2 = fit_norm(v2, boxes[k], x)
        y1[y1==0] = 1e-20
        y2[y2==0] = 1e-20
        metric[k] = np.mean(kl_div(y1, y2))
        print("{}:\t{:.4f}".format(k, metric[k]))
    

def AnalyseDiversity(args):
    parts = args.i.split(',')
    assert len(parts) == 2
    refer, hypo = read_json(parts[0]), read_json(parts[1])
    refer_seqs = [r['seq'].upper() for r in refer]
    hypo_seqs = [h['seq'].upper() for h in hypo]
    Diversity(hypo_seqs, hypo_seqs)
    Diversity(refer_seqs, hypo_seqs)


############################## Main ##############################


""" 
    Usage:
        python scripts/calProperty.py -m AnalyseDiversity -i ../raw_data/APD/APD.attr.jsonl,sample_top.align.jsonl
        python scripts/calProperty.py -m CalKLDist -i ../raw_data/APD/APD.attr.jsonl,sample_top.align.jsonl
        python scripts/calProperty.py -m Log2Fasta -i ../logs/cvqvae_ema_cls_8_ft/sample/sample_top_0.txt -o ../logs/cvqvae_ema_cls_8_ft/sample/sample_top_0.fasta
        python scripts/calProperty.py -m SelectCand -i ../logs/cvqvae_ema_cls_8_ft/sample/sample_top_0.align.jsonl,../logs/cvqvae_ema_cls_8_ft/sample/sample_top_0.align.camp.json
        python scripts/calProperty.py -m ParseCAMP -i ../logs/cvqvae_ema_cls_8_ft/sample/sample_top_0.camp -o ../logs/cvqvae_ema_cls_8_ft/sample/sample_top_0.camp.json
        python scripts/calProperty.py -m Align -i ../raw_data/APD.ss.jsonl,sample_top.jsonl -o sample_top.align.jsonl
        python scripts/calProperty.py -m EvalRandom -i ../raw_data/APD.ss.jsonl -o ../raw_data/random02.attr.jsonl
        python scripts/calProperty.py -m EvalSample -i ../logs/cvqvae5_ft/sample/sample_top_10000.txt -o ../logs/cvqvae5_ft/sample/sample_top_10000.jsonl
        python scripts/calProperty.py -m EvalBaseline -i ../raw_data/APD.ss.jsonl -o ../raw_data/APD.attr.jsonl
        python scripts/calProperty.py -m CalStat -i ../raw_data/APD.ss.jsonl
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='peptide.jsonl')
    parser.add_argument('-o', default='peptide.json')
    parser.add_argument("-mode", default='EvalSample')
    parser.add_argument('-seed', default=888, type=int)
    parser.add_argument('-num', default=5000, type=int)
    parser.add_argument('-ratio', default=0.2, type=float)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    eval('{}(args)'.format(args.mode))