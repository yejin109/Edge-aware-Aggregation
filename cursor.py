import _init_env

import torch
import torch.nn as nn
import torch.optim as optim

import os
import time
import path
import shutil
import random
import datetime
import argparse
import numpy as np
import pandas as pd
from argparse import Namespace

# from HNHNII.model_ver1 import load_model
# from HNHNII.model_ver2 import load_model
from HNHNII.model_ver3 import load_model
from data import load_dataset
from functionals.utils import log_arguments, get_logger
import pickle

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--n_epoch', type=int, default=500, help='Number of epochs to train. 230')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--gamma', type=float, default=.51, help='Gamnma value for lr scheduler. 0.51, 5e-4')

parser.add_argument('--dataset', default='cora', help='dateset')
parser.add_argument('--data', type=str, default='cocitation', help='data name (coauthorship/cocitation)')
parser.add_argument('--label_percent', type=float, default=.052)
parser.add_argument("--alpha_e", default=0, type=float, help='alpha')
parser.add_argument("--alpha_n", default=0, type=float, help='alpha')
parser.add_argument("--use_exp_wt", dest='use_exp_wt', action='store_const', default=False, const=True, help='Fix seed for reproducibility and fair comparison.')


parser.add_argument('--n-runs', type=int, default=10, help='number of runs for repeated experiments')
parser.add_argument('--split', type=int, default=1,  help='choose which train/test split to use')

parser.add_argument('--out-dir', type=str, default='runs/test',  help='output dir')
parser.add_argument('--nostdout', action="store_true",  help='do not output logging to terminal')


# Model Structure
parser.add_argument('--n_layers', type=int, default=2, help='Number of layers.')
parser.add_argument('--n_hidden', type=int, default=128, help='hidden dimensions. For Pubmed, use 800. 400')
parser.add_argument('--final_edge_dim', type=int, default=64, help='Origianl : 100')

parser.add_argument('--mlp_layers', type=int, default=1)
parser.add_argument('--norm', type=str, default='ln')

parser.add_argument('--lamda', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.1)

parser.add_argument('--dropout_p', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--inp_dropout_p', type=float, default=0.2, help='Dropout rate (1 - keep probability).')

parser.add_argument('--activation', type=str, default='relu')

args = parser.parse_args()

with open('./args.pkl', 'wb') as f:
    pickle.dump(args, f)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

dataname = f'{args.data}_{args.dataset}'
dirname = f'{datetime.datetime.now()}'.replace(' ', '_').replace(':', '.')
out_dir = path.Path( f'{os.environ["ROOT_DIR"]}/{args.out_dir}/HNHNII-v2_{args.n_layers}_{dataname}/seed_{args.seed}' )


if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()


def set_args(namespace, src: dict, indices: list=[]):
    if len(indices)== 0:
        indices = src.keys()
    for index in indices:
        setattr(namespace, index, src[index])


if __name__=='__main__':    
    data_arg = Namespace()
    set_args(data_arg, vars(args), ['data', 'dataset', 'split', 'label_percent', 'use_exp_wt', 'alpha_e', 'alpha_n', 'n_hidden'])
    dataset = load_dataset(data_arg)
    log_arguments(data_arg, f'{os.environ["ROOT_DIR"]}/log/data.arguments')

    model_arg = Namespace()
    set_args(model_arg, vars(args), ['n_layers', 'n_hidden', 'final_edge_dim', 'dropout_p', 'lamda', 'alpha', 'inp_dropout_p', 'norm', 'mlp_layers', 'activation'])
    set_args(model_arg, vars(data_arg))
    model = load_model(model_arg)
    log_arguments(model_arg, F'{os.environ["ROOT_DIR"]}/log/model.arguments')

    print(model)