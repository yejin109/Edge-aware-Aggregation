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

from HNHNII.model_ver2 import load_model
from data import load_dataset
from functionals.utils import log_arguments, get_logger
import pickle
import hashlib
import datetime
import wandb

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--n_epoch', type=int, default=300, help='Number of epochs to train. 230')
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
exp_name = str(datetime.datetime.now()) + args.activation + args.norm
exp_name = hashlib.sha1(exp_name.encode()).hexdigest()[:4]

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

dataname = f'{args.data}_{args.dataset}'
dirname = f'{datetime.datetime.now()}'.replace(' ', '_').replace(':', '.')
out_dir = path.Path( f'{os.environ["ROOT_DIR"]}/{args.out_dir}/HNHNII-v2_{args.n_layers}_{dataname}/{exp_name}')

if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()

baselogger = get_logger('base logger', f'{out_dir}/logging.log', not args.nostdout)
resultlogger = get_logger('result logger', f'{out_dir}/result.log', not args.nostdout)
csvlogger = get_logger('csv logger', f'{out_dir}/per_epoch.csv', not args.nostdout)
baselogger.info(args)


def train(model: nn.Module, _args):
    v, e, label_idx, labels = _args.v, _args.e, _args.label_idx, _args.labels
    v_init = v
    e_init = e
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.all_params(), lr=_args.lr)
    milestones = [100*i for i in range(1, 4)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=_args.gamma)

    test_accs = []
    for epoch in range(_args.n_epoch):   
        start = time.time()
        v, e, pred_all = model(v_init, e_init)
        pred = pred_all[label_idx]            
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()            

        optimizer.step()
        scheduler.step()

        train_time = time.time()-start
        loss_tra = loss.item()

        acc_tra = eval(pred_all, _args.label_idx, _args)
        acc_tes = eval(pred_all, _args.test_idx, _args)
        test_accs.append(acc_tes)

        baselogger.info(f'epoch:{epoch} | loss:{loss_tra:.4f} | train acc:{acc_tra:.2f} | test acc:{acc_tes:.2f} | time:{train_time*1000:.1f}ms')
        logs.append(','.join([str(i) for i in [args.split, _args.run, epoch, loss_tra, acc_tra, acc_tes, train_time]]))
        wandb.log({
            'Train/Loss' : loss_tra,
            'Train/Accuracy': acc_tra,
            'Test/Accuracy': acc_tes,
            'Train/Time': train_time,
            "Train/L2 Gradient Norm": total_grad_norm(model.all_params())
        })

    return test_accs


def eval(pred_all, idx, args):
    pred = pred_all[idx]
    pred = torch.argmax(pred, -1)
    tgt = args.all_labels[idx]
    
    acc = torch.eq(pred, tgt).sum().item()/len(tgt) * 100
    return acc


def total_grad_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    return total_norm


def set_args(namespace, src: dict, indices: list=[]):
    if len(indices)== 0:
        indices = src.keys()
    for index in indices:
        setattr(namespace, index, src[index])


def write_cfg(arg, dir):
    for k, v in vars(arg).items():
        with open(f'{dir}/args.txt', 'w') as f:
            f.write(f"{k} : {v} \n")


if __name__=='__main__':    
    write_cfg(args, out_dir)
    logs = []
    for run in range(args.n_runs):
        wandb.init(project = "HNHNII", config = vars(args))
        wandb.run.name = '-'.join([str(i) for i in [args.dataset, args.n_layers, args.alpha, args.lamda, args.dropout_p, args.inp_dropout_p, args.activation, args.split, run]])

        data_arg = Namespace()
        set_args(data_arg, vars(args), ['data', 'dataset', 'split', 'label_percent', 'use_exp_wt', 'alpha_e', 'alpha_n', 'n_hidden'])
        dataset = load_dataset(data_arg)
        log_arguments(data_arg, f'{os.environ["ROOT_DIR"]}/log/data.arguments')

        model_arg = Namespace()
        set_args(model_arg, vars(args), ['n_layers', 'n_hidden', 'final_edge_dim', 'dropout_p', 'lamda', 'alpha', 'inp_dropout_p', 'norm', 'mlp_layers', 'activation'])
        set_args(model_arg, vars(data_arg))
        model = load_model(model_arg)
        log_arguments(model_arg, F'{os.environ["ROOT_DIR"]}/log/model.arguments')

        train_arg = Namespace()
        set_args(train_arg, vars(args), ['lr', 'gamma', 'n_epoch'])    
        set_args(train_arg, vars(data_arg), ['v', 'e', 'label_idx', 'all_labels', 'labels', 'test_idx'])        
        log_arguments(train_arg, f'{os.environ["ROOT_DIR"]}/log/train.argumens')

        run_start = time.time()
        setattr(train_arg, 'run', run)
        test_accuracy = train(model, train_arg)

        resultlogger.info(f"Average final test accuracy: {np.mean(test_accuracy)} ± {np.std(test_accuracy)}")
        resultlogger.info("Train cost: {:.4f}s".format(time.time() - run_start))
        wandb.log({
            'Metric/Test Acc mean' : np.mean(test_accuracy),
            'Metric/Test Acc std' : np.std(test_accuracy)
        })
        wandb.finish()
    a = pd.DataFrame([log.split(',') for log in logs])
    a.to_csv(f'{out_dir}/perf_split_{args.split}.csv')
