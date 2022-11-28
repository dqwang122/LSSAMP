#encoding=utf-8


import argparse
import time

from others.logging import init_logger, logger
from prepro import data_builder


def do_format_to_pt(args):
    start_time = time.time()
    data_builder.format_to_pt(args)
    print(time.time()-start_time)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


""" 
    Usage:
        python preprocess.py -mode split_dataset_to_txt -raw_path ../raw_data/uniprot_all.fasta -save_path ../data/uniprot_all
        python preprocess.py -mode do_format_to_pt -suffix txt -raw_path ../raw_data/uniprot_all -save_path ../data/uniprot_all -vocab_path ../data/vocab 
        
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='', type=str)
    parser.add_argument("-vocab_path", default='../../data/')
    parser.add_argument("-raw_path", default='../../line_data')
    parser.add_argument("-refer_path", default='../../line_data')
    parser.add_argument("-save_path", default='../../data/')
    parser.add_argument("-suffix", default='src,tgt')
    parser.add_argument("-step", default=0)

    parser.add_argument("-vocab_size", default=0, type=int)
    parser.add_argument('-log_file', default='../logs/prepare.log')
    parser.add_argument('-dataset', default='')
    parser.add_argument('-n_cpus', default=2, type=int)


    args = parser.parse_args()
    init_logger(args.log_file)
    logger.info(args)
    if args.mode.startswith('do'):
        eval(args.mode + '(args)')
    else:
        eval('data_builder.{}(args)'.format(args.mode))
