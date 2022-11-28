import random
import os
import math
import json
import argparse


import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from others.utils import read_json
from calProperty import modified_moment

font = {'size': 16}

matplotlib.rc('font', **font)

############################## Func ##############################


def denplot(ax, x, y, xlabel="", ylabel="", title=""):
    sns.kdeplot(x, y, shade=True, cbar=True, ax=ax)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    # ax.legend(loc = 'best')

def violinplot(ax, data):
    """
        data = {'name':list}
    """
    x = []
    y = []
    for key, value in data.items():
        x.extend([key] * len(value))
        y.extend(value)
    sns.violinplot(x, y, ax=ax)

def tSNE(ax, data, color=None, label=''):
    """
        data: [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    """
    from MulticoreTSNE import MulticoreTSNE as TSNE
    data = np.array(data)
    print(data.shape)
    embeddings = TSNE(n_jobs=4).fit_transform(data)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    ax.scatter(vis_x, vis_y, c=color, label=label, alpha=.8, s=25, marker='.')
    ax.set_xticks([])
    ax.set_yticks([])
    return [vis_x, vis_y]



def plot_summary(data, libnames, filename=None, colors=None):
    """Method to generate a visual summary of different characteristics of the given library. The class methods
    are used with their standard options.

    :param data: {dict}, each value is a list
    :param libnames: {list} name of libraries
    :param filename: {str} path to save the generated plot to.
    :param colors: {str / list} color or list of colors to use for plotting. e.g. '#4E395D', 'red', 'k'
    :return: visual summary (plot) of the library characteristics 
    
    >>> g = GlobalAnalysis([seqs1, seqs2, seqs3])  # seqs being lists / arrays of sequences
    >>> g.plot_summary()
    
    .. image:: ../docs/static/summary.png
        :height: 600px
    """    
    from modlamp.core import count_aas

    try:
        length, aafreq, charge, H, uH = data['lengths'], data['aafreq'], data['charge'], data['H'], data['uH']
        shapes = [len(l) for l in data['lengths']]
    except:
        print("Missing key in data ({})".format(','.join(data.keys())))
        exit()

    # plot settings
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25, 15))
    ((ax2, ax5, ax1), (ax3, ax4, ax6)) = axes
    # plt.suptitle('Summary', fontweight='bold', fontsize=16.)
    labels = libnames
    if not colors:
        colors = ['#FA6900', '#69D2E7', '#542437', '#53777A', '#CCFC8E', '#9CC4E4']
    num = len(labels)
    
    for a in [ax1, ax2, ax3, ax4, ax5, ax6]:
        # only left and bottom axes, no box
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.xaxis.set_ticks_position('bottom')
        a.yaxis.set_ticks_position('left')
    
    # 1 length box plot
    box = ax1.boxplot(length, notch=1, vert=1, patch_artist=True)
    plt.setp(box['whiskers'], color='black')
    plt.setp(box['medians'], linestyle='-', linewidth=1.5, color='black')
    for p, patch in enumerate(box['boxes']):
        patch.set(facecolor=colors[p], edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Sequence Length', fontweight='bold', fontsize=18.)
    ax1.set_xticks([x + 1 for x in range(len(labels))])
    ax1.set_xticklabels(labels, rotation = 15, ha="right", fontweight='bold')
    
    # 2 AA bar plot
    d_aa = count_aas('')
    hands = [mpatches.Patch(label=labels[i], facecolor=colors[i], alpha=0.8) for i in range(len(labels))]
    w = .9 / num  # bar width
    offsets = np.arange(start=-w, step=w, stop=num * w)  # bar offsets if many libs
    for i, l in enumerate(aafreq):
        for a in range(20):
            ax2.bar(a - offsets[i], l[a], w, color=colors[i], alpha=0.8)
    ax2.set_xlim([-1., 20.])
    ax2.set_ylim([0, 1.05 * np.max(aafreq)])
    ax2.set_xticks(range(20))
    ax2.set_xticklabels(d_aa.keys(), fontweight='bold')
    ax2.set_ylabel('Fraction', fontweight='bold', fontsize=18.)
    ax2.set_xlabel('Amino Acids', fontweight='bold', fontsize=18.)
    ax2.legend(handles=hands, labels=labels)
    
    # 3 hydophobicity violin plot
    for i, l in enumerate(H):
        vplot = ax3.violinplot(l, positions=[i + 1], widths=0.5, showmeans=True, showmedians=False)
        # crappy adaptions of violin dictionary elements
        vplot['cbars'].set_edgecolor('black')
        vplot['cmins'].set_edgecolor('black')
        vplot['cmeans'].set_edgecolor('black')
        vplot['cmaxes'].set_edgecolor('black')
        vplot['cmeans'].set_linestyle('--')
        for pc in vplot['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.8)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
            pc.set_alpha(0.7)
            pc.set_label(labels[i])
    ax3.set_xticks([x + 1 for x in range(len(labels))])
    ax3.set_xticklabels(labels, rotation = 15, ha="right", fontweight='bold')
    ax3.set_ylabel('Global Hydrophobicity', fontweight='bold', fontsize=18.)
    
    # 4 hydrophobic moment violin plot
    for i, l in enumerate(uH):
        vplot = ax4.violinplot(l, positions=[i + 1], widths=0.5, showmeans=True, showmedians=False)
        # crappy adaptions of violin dictionary elements
        vplot['cbars'].set_edgecolor('black')
        vplot['cmins'].set_edgecolor('black')
        vplot['cmeans'].set_edgecolor('black')
        vplot['cmaxes'].set_edgecolor('black')
        vplot['cmeans'].set_linestyle('--')
        for pc in vplot['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.8)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
            pc.set_alpha(0.7)
            pc.set_label(labels[i])
    ax4.set_xticks([x + 1 for x in range(len(labels))])
    ax4.set_xticklabels(labels, rotation = 15, ha="right", fontweight='bold')
    ax4.set_ylabel('Global Hydrophobic Moment', fontweight='bold', fontsize=18.)

    # 3 hydophobicity violin plot
    # bwidth = 0.04
    # for i, x in enumerate(H):
    #     counts, bins = np.histogram(x, range=[-1, 1.0], bins=25)
    #     ax3.bar(bins[1:] + i * bwidth, counts / np.sum(counts), bwidth, color=colors[i], label=labels[i], alpha=0.8)
    # ax3.set_xlabel('Global Hydrophobic', fontweight='bold', fontsize=14.)
    # ax3.legend(loc='best')

    # # 4 hydrophobic moment violin plot
    # bwidth = 0.08
    # for i, x in enumerate(uH):
    #     counts, bins = np.histogram(x, range=[0, 4], bins=25)
    #     ax4.bar(bins[1:] + i * bwidth, counts / np.sum(counts), bwidth, color=colors[i], label=labels[i], alpha=0.8)
    # ax4.set_xlabel('Global Hydrophobic Moment', fontweight='bold', fontsize=14.)
    # ax4.legend(loc='best')
    
    # # 5 charge histogram
    bwidth = 1. / len(shapes)
    for i, c in enumerate(charge):
        counts, bins = np.histogram(c, range=[-5, 15], bins=25)
        # ax5.bar(bins[1:] + i * bwidth, counts / np.max(counts), bwidth, color=colors[i], label=labels[i], alpha=0.8)
        ax5.bar(bins[1:] + i * bwidth, counts / np.sum(counts), bwidth, color=colors[i], label=labels[i], alpha=0.8)
    ax5.set_xlabel('Global Charge', fontweight='bold', fontsize=18.)
    ax5.set_ylabel('Fraction', fontweight='bold', fontsize=18.)
    # ax5.title.set_text('pH: 7.4 ,  amide: true')
    ax5.legend(loc='best')

    # 6 3D plot
    # ax6.spines['left'].set_visible(False)
    # ax6.spines['bottom'].set_visible(False)
    # ax6.set_xticks([])
    # ax6.set_yticks([])
    # ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    # for i, l in enumerate(range(num)):
    #     xt = charge[l]  # find all values in x for the given target
    #     yt = H[l]  # find all values in y for the given target
    #     zt = uH[l]  # find all values in y for the given target
    #     ax6.scatter(xt, yt, zt, c=colors[l], alpha=.8, s=25, label=labels[i])
    
    # ax6.set_xlabel('Charge', fontweight='bold', fontsize=18.)
    # ax6.set_ylabel('H', fontweight='bold', fontsize=18.)
    # ax6.set_zlabel('uH', fontweight='bold', fontsize=18.)
    # data_c = [item for sublist in charge for item in sublist]  # flatten charge data into one list
    # data_H = [item for sublist in H for item in sublist]  # flatten H data into one list
    # data_uH = [item for sublist in uH for item in sublist]  # flatten uH data into one list
    # ax6.set_xlim([np.min(data_c), np.max(data_c)])
    # ax6.set_ylim([np.min(data_H), np.max(data_H)])
    # ax6.set_zlim([np.min(data_uH), np.max(data_uH)])
    # ax6.legend(loc='best')
    
    # 6 2D plot
    # ax6.spines['left'].set_visible(False)
    # ax6.spines['bottom'].set_visible(False)
    # ax6.set_xticks([])
    # ax6.set_yticks([])
    # ax6 = fig.add_subplot(2, 3, 6)
    # for i, l in enumerate(range(num)):
    #     xt = H[l]  # find all values in x for the given target
    #     yt = charge[l]  # find all values in y for the given target
    #     ax6.scatter(yt, xt, c=colors[l], alpha=.8, s=25, label=labels[i])
    
    # ax6.set_xlabel('Charge', fontweight='bold', fontsize=14.)
    # ax6.set_ylabel('H', fontweight='bold', fontsize=14.)
    # data_c = [item for sublist in charge for item in sublist]  # flatten charge data into one list
    # data_H = [item for sublist in H for item in sublist]  # flatten H data into one list
    # ax6.set_xlim([np.min(data_c), np.max(data_c)])
    # ax6.set_ylim([np.min(data_H), np.max(data_H)])
    # ax6.legend(loc='best')

    # 6 tSNE plot
    ax6.spines['left'].set_visible(False)
    ax6.spines['bottom'].set_visible(False)
    ax6.set_xticks([])
    ax6.set_yticks([])
    x_embed_all = []
    for i, l in enumerate(range(num)):
        data = [[x, y, z] for x, y, z in zip(charge[l], H[l], uH[l])]
        x_embed = tSNE(ax6, data, color=colors[i], label=labels[i])
        x_embed_all.append(x_embed)

    ax6.set_xlim([min([min(x[0]) for x in x_embed_all]), max([max(x[0]) for x in x_embed_all])])
    ax6.set_ylim([min([min(x[1]) for x in x_embed_all]), max([max(x[1]) for x in x_embed_all])])
    ax6.set_xlabel('tSNE for C,H,uH', fontsize=18.)
    ax6.legend(loc='best', prop={'size': 10})


    
    if filename:
        plt.savefig(filename, dpi=200)
    else:
        plt.show()


def plot_hist_summary(data, libnames, filename=None, colors=None):
    from modlamp.core import count_aas

    try:
        length, aafreq, charge, H, uH = data['lengths'], data['aafreq'], data['charge'], data['H'], data['uH']
        shapes = [len(l) for l in data['lengths']]
    except:
        print("Missing key in data ({})".format(','.join(data.keys())))
        exit()

    # plot settings
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))
    (ax1, ax2, ax3) = axes
    # plt.suptitle('Summary', fontweight='bold', fontsize=16.)
    labels = libnames
    if not colors:
        colors = ['#FA6900', '#69D2E7', '#542437', '#53777A', '#CCFC8E', '#9CC4E4']
    num = len(labels)
    
    for a in [ax1, ax2, ax3]:
        # only left and bottom axes, no box
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.xaxis.set_ticks_position('bottom')
        a.yaxis.set_ticks_position('left')


    
    # 1 charge histogram
    # bwidth = 1. / len(shapes)
    bwidth = 0.4
    for i, c in enumerate(charge):
        counts, bins = np.histogram(c, range=[-5, 15], bins=25)
        # ax5.bar(bins[1:] + i * bwidth, counts / np.max(counts), bwidth, color=colors[i], label=labels[i], alpha=0.8)
        ax1.bar(bins[1:] + i * bwidth, counts / np.sum(counts), bwidth, color=colors[i], label=labels[i], alpha=0.8)
    ax1.set_xlabel('Global Charge', fontweight='bold', fontsize=18.)
    ax1.set_ylabel('Fraction', fontweight='bold', fontsize=18.)
    # ax1.title.set_text('pH: 7.4 ,  amide: true')
    ax1.legend(loc='best')

    # 2 hydophobicity histogram
    bwidth = 0.04
    for i, x in enumerate(H):
        counts, bins = np.histogram(x, range=[-1, 1.0], bins=25)
        ax2.bar(bins[1:] + i * bwidth, counts / np.sum(counts), bwidth, color=colors[i], label=labels[i], alpha=0.8)
    ax2.set_xlabel('Global Hydrophobic', fontweight='bold', fontsize=18.)
    ax2.legend(loc='best')

    # 3 hydrophobic moment histogram
    bwidth = 0.08
    for i, x in enumerate(uH):
        counts, bins = np.histogram(x, range=[0, 3.5], bins=25)
        ax3.bar(bins[1:] + i * bwidth, counts / np.sum(counts), bwidth, color=colors[i], label=labels[i], alpha=0.8)
    ax3.set_xlabel('Global Hydrophobic Moment', fontweight='bold', fontsize=18.)
    ax3.legend(loc='best')

    
    if filename:
        plt.savefig(filename, dpi=200)
    else:
        plt.show()


############################## Mode ##############################

def DrawStat(args):
    data = read_json(args.i)
    stat = [(d['charge'], d['amphipathy']) for d in data]
    x, y = zip(*stat)
    _, ax = plt.subplots()

    # ax.set_xlim(-15, 20)
    # ax.set_ylim(-20, 40)

    denplot(ax, x, y, xlabel='charge', ylabel='amphipathy')
    plt.savefig(args.o, bbox_inches='tight')

def CompStat(args):
    files = args.i.split(',')
    names = [f.split('/')[-1].split('.')[0] for f in files]
    dataset = {}
    ymin, ymax = math.inf, -math.inf
    for name, f in zip(names, files):
        data = read_json(f)
        dataset[name] = [x[args.k] for x in data]
        ymin = min(min(dataset[name]), ymin)
        ymax = max(max(dataset[name]), ymax)

    _, ax = plt.subplots()

    # ax.set_ylim(ymin, ymax)

    violinplot(ax, dataset)
    plt.savefig(args.o)


def GetSumm(args):
    from modlamp.analysis import GlobalAnalysis

    def clean(x):
        seq = x['seq'].upper()
        ss = [s for idx,s in enumerate(x['ss']) if seq[idx] != 'X']
        seq = seq.replace('X', '')
        return seq, ss

    files = args.i.split(',')
    names = [f.split('/')[-1].split('.')[0] for f in files]
    data = [[clean(x) for x in read_json(f)] for f in files]
    data = [[x for x in lib if len(x[0]) <= 32] for lib in data]
    seq = [[x[0] for x in lib] for lib in data] 
    ss = [[x[1] for x in lib] for lib in data] 
    g = GlobalAnalysis(seq, names=names)
    g.plot_summary(args.o, plot=False)

    info = {'lengths': g.len, 'aafreq': g.aafreq, 'charge': g.charge, 'H': g.H}
    info['uH'] = [[modified_moment(x[0], x[1]) for x in lib] for lib in data]
    plot_summary(info, names, args.o)


def GetHistSumm(args):
    from modlamp.analysis import GlobalAnalysis

    def clean(x):
        seq = x['seq'].upper()
        ss = [s for idx,s in enumerate(x['ss']) if seq[idx] != 'X']
        seq = seq.replace('X', '')
        return seq, ss

    files = args.i.split(',')
    names = [f.split('/')[-1].split('.')[0] for f in files]
    data = [[clean(x) for x in read_json(f)] for f in files]
    data = [[x for x in lib if len(x[0]) <= 32] for lib in data]
    seq = [[x[0] for x in lib] for lib in data] 
    ss = [[x[1] for x in lib] for lib in data] 
    g = GlobalAnalysis(seq, names=names)
    g.plot_summary(args.o, plot=False)

    info = {'lengths': g.len, 'aafreq': g.aafreq, 'charge': g.charge, 'H': g.H}
    info['uH'] = [[modified_moment(x[0], x[1]) for x in lib] for lib in data]
    plot_hist_summary(info, names, args.o)

def GetAASumm(args):
    from collections import Counter

    AA = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']


    def clean(x):
        seq = x['seq'].upper()
        ss = [s for idx,s in enumerate(x['ss']) if seq[idx] != 'X']
        seq = seq.replace('X', '')
        return seq, ss

    files = args.i.split(',')
    names = [f.split('/')[-1].split('.')[0] for f in files]
    data = [[clean(x) for x in read_json(f)] for f in files]
    data = [[x for x in lib if len(x[0]) <= 32] for lib in data]
    seqs = [[x[0] for x in lib] for lib in data] 
    aafreq = [[[Counter(x)[a]/len(x) for a in AA] for x in lib] for lib in seqs]

    colors = ['#FA6900', '#69D2E7', '#542437', '#53777A', '#CCFC8E', '#9CC4E4']
    _, ax = plt.subplots()
    x_embed_all = []
    for i, lib in enumerate(aafreq):
        x_embed = tSNE(ax, lib, color=colors[i], label=names[i])
        x_embed_all.append(x_embed)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([min([min(x[0]) for x in x_embed_all]), max([max(x[0]) for x in x_embed_all])])
    ax.set_ylim([min([min(x[1]) for x in x_embed_all]), max([max(x[1]) for x in x_embed_all])])
    ax.set_xlabel('tSNE for Amino Acid', fontsize=14.)
    ax.legend(loc='best', prop={'size': 10})
    
    plt.savefig(args.o, dpi=200)


""" 
    Usage:
        python scripts/draw.py -m GetAASumm -i ../raw_data/APD.attr.jsonl,../raw_data/decoy.attr.jsonl -o test.jpg
        python scripts/draw.py -m GetSumm -i ../raw_data/APD.attr.jsonl,../raw_data/decoy.attr.jsonl -o test.jpg
        python scripts/draw.py -m CompStat -i ../raw_data/APD.attr.jsonl,../raw_data/decoy.attr.jsonl -o test.jpg -k charge
        python scripts/draw.py -m DrawStat -i ../raw_data/APD.attr.jsonl -o APD.jpg
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='peptide.jsonl')
    parser.add_argument('-o', default='peptide.json')
    parser.add_argument('-k', default='net')
    parser.add_argument("-mode", default='EvalSample')
    parser.add_argument('-seed', default=888, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    eval('{}(args)'.format(args.mode))