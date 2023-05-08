import os
import math
import numpy as np

import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP


def load_model(args):
    return Hypergraph(args.vidx, args.eidx, args.nv, args.ne, args.n_weight, args.e_weight, args).to(os.environ['DEVICE'])


class HNHNIIConv(nn.Module):
    def __init__(self, in_features, out_features, vidx, eidx, args):
        super().__init__()
        self.n_weight = args.n_weight
        self.e_weight = args.e_weight
        self.vidx = vidx
        self.eidx = eidx

        self.node2msg = MLP(in_features, out_features, out_features, 2, dropout=.05, Normalization='ln', InputNorm=True)
        self.edge2msg = MLP(in_features+out_features, out_features, out_features, 2, dropout=.05, Normalization='ln', InputNorm=True)
        # self.edge2msg = MLP(out_features, out_features, out_features, 2, dropout=.05, Normalization='ln', InputNorm=True)
        self.node_agg = MLP(out_features, out_features, out_features, 2, dropout=.05, Normalization='ln', InputNorm=True)


        self.norm = nn.BatchNorm1d(out_features)
        self.act = nn.ReLU()

    def forward(self, v, e, v0, e0, alpha, beta):
        node_msg = self.node2msg(v*self.n_weight)
        node_msg = node_msg[self.vidx]

        # NOTE: h_e
        edge = torch_scatter.scatter_mean(node_msg, self.eidx, dim=0)
        # edge = (1-alpha) * ((1-beta) * edge + beta *e )+ alpha * e0
        edge = (1-alpha) * edge + alpha *e0

        edge_msg = edge[self.eidx]
        # edge_msg = self.edge2msg(torch.cat((v[self.vidx], edge_msg), -1))
        edge_msg = beta * self.edge2msg(torch.cat((v[self.vidx], edge_msg), -1)) + (1-beta) * e[self.eidx]

        # edge_msg = self.edge2msg(edge_msg)
        # edge_msg = beta * self.edge2msg(edge_msg) + (1-beta) * e[self.eidx]

        node = torch_scatter.scatter_mean(edge_msg, self.vidx, dim=0)
        
        node = (1-alpha) * node + alpha *v0
        node = beta * self.node_agg(node)  + (1-beta) * node
        # node = self.node_agg(node)

        return node, edge


        


class Hypergraph(nn.Module):
    '''
    Hypergraph class, uses weights for vertex-edge and edge-vertex incidence matrix.
    One large graph.
    '''
    def __init__(self, vidx, eidx, nv, ne, n_weight, e_weight, args, inp_dropout = 0.6, dropout=0.5):
        '''
        vidx: idx tensor of elements to select, shape (ne, max_n),
        shifted by 1 to account for 0th elem (which is 0)
        eidx has shape (nv, max n)..
        '''
        super(Hypergraph, self).__init__()
        self.args = args

        self.node_map = torch.nn.Linear(self.args.input_dim, self.args.n_hidden).to(os.environ['DEVICE'])
        self.edge_map = torch.nn.Embedding(ne, self.args.n_hidden, device=os.environ['DEVICE'])
        self.convs = []
        for _ in range(self.args.n_layers):
            self.convs.append(HNHNIIConv(args.n_hidden, args.n_hidden, vidx, eidx, args).to(os.environ['DEVICE']))
        
        # self.inp_dropout = nn.Dropout(inp_dropout)
        self.dropout = nn.Dropout(dropout)
        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu':nn.PReLU()}
        self.act = act['relu']
        self.cls = MLP(self.args.n_hidden, self.args.n_hidden//2, self.args.n_cls, 2, dropout=dropout, Normalization='ln', InputNorm=False)

    def forward(self, v, e):
        '''
        Take initial embeddings from the select labeled data.
        Return predicted cls.
        '''
        lamda, alpha = 0.5, 0.4
        # Initialize X_E \gets 0. Project X_V to hidden dimension
        v0 = F.relu(self.node_map(v)) 
        e0 = F.relu(self.edge_map(torch.arange(self.args.ne).long().to(os.environ['DEVICE'])))
                
        # For i=1 to n_layers do
        v, e = v0, e0
        for i, conv in enumerate(self.convs):
            beta = math.log(lamda/(i+1)+1)

            v = self.dropout(v)
            v, e = conv(v, e, v0, e0, alpha, beta)
            v = self.act(v)
            e = self.act(e)


        v = self.dropout(v)
        pred = self.cls(v)
        return v, e, pred
    
    def all_params(self):
        params = []
        for conv in self.convs:
            params.extend(conv.parameters())
        
        params.extend(self.parameters())
        return params