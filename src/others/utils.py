import os
import re
import shutil
import time
import json
import csv

import pickle as pkl
from typing import Sequence

ss2id={'H':1, 'B':2, 'E':3, 'G':4, 'I':5, 'T': 6, 'S':7, '-':0}
id2ss={v:k for k,v in ss2id.items()}
id2ss[8] = '-'

def read_json(fName):
    data = []
    with open(fName) as fin:
        for line in fin:
            data.append(json.loads(line))
    print("Reading {} example from {}".format(len(data), fName))
    return data

    
def readLog(fName):
    sequences = []
    structures = []
    cnt = 0
    with open(fName) as fin:
        for line in fin:
            data = line.strip()
            if cnt % 2 == 0:
                sequences.append(data)
            else:
                if data != "":
                    structures.append(json.loads(data))
                else:
                    structures.append([])
            cnt += 1
    print("Load {} examples from {}".format(len(sequences), fName))
    return sequences, structures

def readFasta(fName):
    data = []
    info = None
    with open(fName) as fin:
        for line in fin:
            if line.startswith('>sp|'):
                info = json.loads(line.strip('>sp|'))
            else:
                seq = line.strip()
                info['seq'] = seq
                data.append(info)
                info = None
    return data
    

def save_json(data, fName):
    with open(fName, 'w') as fout:
        for d in data:
            fout.write('{}\n'.format(json.dumps(d)))
    print("Saving {} example from {}".format(len(data), fName))

def save_csv(data, head, fName):
    fout = open(fName, 'w', newline='', encoding='utf-8-sig')
    writer = csv.writer(fout)
    writer.writerow(head)
    for d in data:
        line = []
        for k in head:
            line.append(d[k])
        writer.writerow(line)
    print("Saving {} example from {}".format(len(data), fName))

def save_fasta(data, fName):
    num = 0
    with open(fName, 'w') as fout:
        for d in data:
            fout.write('>sp|{:04}\n{}\n'.format(num, d))
            num += 1
    print("Saving {} example from {}".format(len(data), fName))

def save(arr,fileName):
    """save in pickle format"""
    fileObject = open(fileName, 'wb')
    pkl.dump(arr, fileObject)
    fileObject.close()

def load(fileName):
    """load pickle format"""
    fileObject2 = open(fileName, 'rb')
    modelInput = pkl.load(fileObject2)
    fileObject2.close()
    return modelInput


def json2csv(fileName):
    save_name = fileName.replace('json', 'csv')
    data = read_json(fileName)
    no = 1
    for d in data:
        d['id'] = no
        no += 1
        ss = d['ss']
        ss_str = ''.join([id2ss[x] for x in ss])
        d['ss'] = ss_str
    keys = ['id', 'seq', 'ss', 'charge', 'H', 'uH', 'align']
    save_csv(data, keys, save_name)

         
def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

# filter function
def checkSS(ss_pred):
    if len(ss_pred) < 4:
        print(ss_pred)
        return False
    H = (ss_pred==1)
    if H.sum() < 4:
        return False
    # at least [1, 1, 1, 1]
    if (H[:-3] & H[1:-2] & H[2:-1] & H[3:]).sum() < 1:
        print(ss_pred)
        return False
    return True

def checkSS_2(ss_pred):
    if len(ss_pred) < 8:
        print(ss_pred)
        return False
    E = (ss_pred==3)
    T = (ss_pred==6)
    # at least 8 beta
    if E.sum() + T.sum() < 8:
        print(ss_pred)
        return False
    return True

def checkSS_3(ss_pred):
    if len(ss_pred) < 4:
        print(ss_pred)
        return False
    H = (ss_pred==1)
    if H.sum() != 0:
        if (H[:-3] & H[1:-2] & H[2:-1] & H[3:]).sum() < 1:
            print(ss_pred)
            return False
    N = (ss_pred==8) 
    if N.sum() >= int(0.3 * len(ss_pred)):
        print(ss_pred)
        return False
    return True

def nocheckSS(ss_pred):
    return True