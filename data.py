import os
import math
import torch
import itertools
import numpy as np
from collections import defaultdict
from database import CiteseerDataSet, CoraDataSet, DBLPDataSet, PubmedDataSet, NTU2012DataSet, ModelNet40DataSet, ZooDataSet, NewsW100DataSet


device = os.environ['DEVICE']


def load_dataset(args):
    dset = _get_interface(args)
    train_idx, val_idx, test_idx = dset.load_splits(args.split)


    n_node = dset.num_nodes
    n_edge = dset.num_edges

    n_class = dset.num_classes
    
    node_feature = dset.features

    # n_labels = max(1, math.ceil(n_node*args.label_percent))


    node_edges = dset.hyperedge_index.T

    n_weight = dset.node_weight
    e_weight = dset.edge_weight

    n_weight = torch.Tensor([(1/w if w > 0 else 1) for w in n_weight]).unsqueeze(-1).to(device) 
    e_weight = torch.Tensor([(1/w if w > 0 else 1) for w in e_weight]).unsqueeze(-1).to(device) 

    #weights for regularization 
    paper2sum = defaultdict(list)
    author2sum = defaultdict(list)
    e_reg_weight = torch.zeros(node_edges.size(0)) ###
    n_reg_weight = torch.zeros(node_edges.size(0)) ###
    #a switch to determine whether to have wt in exponent or base
    use_exp_wt = args.use_exp_wt #True #False
    for i, (paper_idx, author_idx) in enumerate(node_edges.cpu().tolist()):
        e_wt = e_weight[author_idx]
        e_reg_wt = torch.exp(args.alpha_e*e_wt) if use_exp_wt else e_wt**args.alpha_e 
        e_reg_weight[i] = e_reg_wt
        paper2sum[paper_idx].append(e_reg_wt) ###
        
        n_wt = n_weight[paper_idx]
        n_reg_wt = torch.exp(args.alpha_n*n_wt) if use_exp_wt else n_wt**args.alpha_n
        n_reg_weight[i] = n_reg_wt
        author2sum[author_idx].append(n_reg_wt) ###        
    #'''
    n_reg_sum = torch.zeros(n_node) ###
    e_reg_sum = torch.zeros(n_edge) ###
    for paper_idx, wt_l in paper2sum.items():
        n_reg_sum[paper_idx] = sum(wt_l)
    for author_idx, wt_l in author2sum.items():
        e_reg_sum[author_idx] = sum(wt_l)

    e_reg_sum[e_reg_sum==0] = 1
    n_reg_sum[n_reg_sum==0] = 1

    # NOTE: argument updates
    args.n_reg_weight = torch.Tensor(n_reg_weight).unsqueeze(-1).to(device)
    args.n_reg_sum = torch.Tensor(n_reg_sum).unsqueeze(-1).to(device)
    args.e_reg_weight = torch.Tensor(e_reg_weight).unsqueeze(-1).to(device)
    args.e_reg_sum = torch.Tensor(e_reg_sum).unsqueeze(-1).to(device)

    args.e = torch.zeros(n_edge, args.n_hidden).to(device)
    args.v = node_feature
    args.input_dim = node_feature.shape[-1]

    args.vidx = node_edges[:, 0]
    args.eidx = node_edges[:, 1]
    args.ve_lists = node_edges

    args.n_weight = n_weight
    args.e_weight = e_weight

    args.ne = n_edge
    args.nv = n_node
    args.n_cls = n_class

    args.all_labels = dset.labels
    args.label_idx = torch.from_numpy(train_idx).to(torch.int64)    
    args.labels = args.all_labels[args.label_idx].to(device) 

    args.val_idx = val_idx
    args.test_idx = test_idx
    return args



def _get_interface(args):
    name = args.dataset
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        name = f'{args.data}/{name}'

    if args.dataset == 'cora':        
        dset = CoraDataSet(name)
    elif args.dataset == 'citeseer':
        dset = CiteseerDataSet(name)
    elif args.dataset == 'pubmed':
        dset = PubmedDataSet(name)
    elif args.dataset == 'NTU2012':
        dset = NTU2012DataSet(name)
    elif args.dataset == 'ModelNet40':
        dset = ModelNet40DataSet(name)
    elif args.dataset == 'Zoo':
        dset  = ZooDataSet(name)
    elif args.dataset == 'NewsW100':
        dset = NewsW100DataSet(name)    
    else:
        raise NameError(f'Currently, cannot support {name}')


    return dset




