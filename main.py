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

from HNHNII.model_ver7 import load_model
from data import load_dataset
from functionals.utils import log_arguments, get_logger
import pickle

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--n_epoch', type=int, default=300, help='Number of epochs to train. 230, our 300')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--gamma', type=float, default=0.51, help='Gamnma value for lr scheduler. 0.51')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay. 0.01, 5e-4')

parser.add_argument('--dataset', default='cora', help='dateset')
parser.add_argument('--data', type=str, default='cocitation', help='data name (coauthorship/cocitation)')
parser.add_argument('--label_percent', type=float, default=.052)
parser.add_argument("--alpha_e", default=0, type=float, help='alpha')
parser.add_argument("--alpha_n", default=0, type=float, help='alpha')
parser.add_argument("--use_exp_wt", dest='use_exp_wt', action='store_const', default=False, const=True, help='Fix seed for reproducibility and fair comparison.')


parser.add_argument('--n-runs', type=int, default=10, help='number of runs for repeated experiments')
parser.add_argument('--split', type=int, default=1,  help='choose which train/test split to use')

parser.add_argument('--out-dir', type=str, default='runs/playground',  help='output dir, runs/test, runs/playground')
parser.add_argument('--nostdout', action="store_true",  help='do not output logging to terminal')


# Model Structure
parser.add_argument('--n_layers', type=int, default=2, help='Number of layers.')
parser.add_argument('--n_hidden', type=int, default=128, help='hidden dimensions. For Pubmed, use 800. 400, our 128, unigcnii 8, gcnii 64')
# parser.add_argument('--final_edge_dim', type=int, default=64, help='Origianl : 100')

parser.add_argument('--mlp_layers', type=int, default=1)
parser.add_argument('--norm', type=str, default='ln')

parser.add_argument('--lamda', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.1)

parser.add_argument('--dropout_p', type=float, default=0.6, help='Dropout rate (1 - keep probability). origina 0.6')
parser.add_argument('--inp_dropout_p', type=float, default=0.2, help='Dropout rate (1 - keep probability). original 0.2')

parser.add_argument('--activation', type=str, default='relu', help='original relu')

args = parser.parse_args()

with open('./args.pkl', 'wb') as f:
    pickle.dump(args, f)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

dataname = f'{args.data}_{args.dataset}'
dirname = f'{datetime.datetime.now()}'.replace(' ', '_').replace(':', '.')
out_dir = path.Path( f'{os.environ["ROOT_DIR"]}/{args.out_dir}/HNHNII-v7_{args.n_layers}_{dataname}/seed_{args.seed}' )


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
    # optimizer = optim.SGD(model.all_params(), lr=args.lr)
    optimizer = optim.Adam(model.all_params(), lr=_args.lr, weight_decay=args.wd)
    milestones = [100*i for i in range(1, 4)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=_args.gamma)

    test_accs = []
    for epoch in range(_args.n_epoch):   
        start = time.time()
        v, e, pred_all = model(v_init, e_init)
        # v, e, pred_all, pred_all_init = model(v_init, e_init)
        pred = pred_all[label_idx]     
        # pred_init = pred_all_init[label_idx]       
        # loss = loss_fn(pred, labels) + loss_fn(pred_init, labels)
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

    return test_accs



def embedding_distribution(model: nn.Module, _args):
    def plot(data, label, title):
        tsne_model = TSNE(n_components=2)
        coor = tsne_model.fit_transform(data)
        plt.figure()
        plt.scatter(coor[:, 0], coor[:, 1], c=label, s=5)
        plt.title(title)
        plt.savefig(f'{"/".join(out_dir.split("/")[:-1])}/Embedding_{args.dataset}_{title}.png')
        plt.close()
    v, e, label_idx, labels = _args.v, _args.e, _args.label_idx, _args.labels
    v_init = v
    e_init = e

    embs = model.emb_dist(v_init, e_init)
    embs = embs[-1]
    labels = _args.all_labels.detach().cpu().numpy()
    plot(embs[_args.label_idx], labels[_args.label_idx],'Train')
    plot(embs[_args.test_idx], labels[_args.test_idx],'Test')


def eval(pred_all, idx, args):
    pred = pred_all[idx]
    pred = torch.argmax(pred, -1)
    tgt = args.all_labels[idx]
    
    acc = torch.eq(pred, tgt).sum().item()/len(tgt) * 100
    return acc



def set_args(namespace, src: dict, indices: list=[]):
    if len(indices)== 0:
        indices = src.keys()
    for index in indices:
        setattr(namespace, index, src[index])


if __name__=='__main__':    
    torch.autograd.set_detect_anomaly(True)
    logs = []
    for run in range(args.n_runs):
        data_arg = Namespace()
        set_args(data_arg, vars(args), ['data', 'dataset', 'split', 'label_percent', 'use_exp_wt', 'alpha_e', 'alpha_n', 'n_hidden'])
        dataset = load_dataset(data_arg)
        log_arguments(data_arg, f'{os.environ["ROOT_DIR"]}/log/data.arguments')

        model_arg = Namespace()
        set_args(model_arg, vars(args), ['n_layers', 'n_hidden', 'dropout_p', 'lamda', 'alpha', 'inp_dropout_p', 'norm', 'mlp_layers', 'activation'])
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

        embedding_distribution(model, train_arg)
        resultlogger.info(f"Average final test accuracy: {np.mean(test_accuracy)} ± {np.std(test_accuracy)}")
        resultlogger.info("Train cost: {:.4f}s".format(time.time() - run_start))
    # a = pd.DataFrame([log.split(',') for log in logs])
    # a.to_csv(f'{"/".join(out_dir.split("/")[:-1])}/perf_split_{args.split}.csv')
