import os
import math
import numpy as np

import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP


def load_model(args):
    return Hypergraph(args.vidx, args.eidx, args.ne, args).to(os.environ['DEVICE'])


class HNHNIIConv(nn.Module):
    def __init__(self, in_features, out_features, vidx, eidx, dropout_p ,args):
        super().__init__()
        self.n_weight = args.n_weight
        self.e_weight = args.e_weight
        self.n_reg_weight = args.n_reg_weight
        self.e_reg_weight = args.e_reg_weight
        self.n_reg_sum = args.n_reg_sum
        self.e_reg_sum = args.e_reg_sum
        
        norm = args.norm
        layer = args.mlp_layers

        self.vidx = vidx
        self.eidx = eidx

        self.node2msg = MLP(in_features, out_features, out_features, layer, dropout=dropout_p, Normalization=norm, InputNorm=True)
        self.edge2msg = MLP(in_features+out_features, out_features, out_features, layer, dropout=dropout_p, Normalization=norm, InputNorm=True)
        self.node_agg = MLP(out_features, out_features, out_features, layer, dropout=dropout_p, Normalization=norm, InputNorm=True)


    def forward(self, v, e, v0, e0, alpha, beta):
        node_msg = self.node2msg(v * self.n_weight)
        node_msg = node_msg[self.vidx]
        node_msg *= self.n_reg_weight

        # NOTE: h_e
        edge = torch_scatter.scatter_mean(node_msg, self.eidx, dim=0) 
        edge = (1-alpha) * edge + alpha *e0
        edge /= self.e_reg_sum

        edge_msg = (edge * self.e_weight)[self.eidx]
        edge_msg = beta * self.edge2msg(torch.cat((v[self.vidx], edge_msg), -1)) + (1-beta) * (e)[self.eidx]
        edge_msg *= self.e_reg_weight

        node = torch_scatter.scatter_mean(edge_msg, self.vidx, dim=0)
        node /= self.n_reg_sum
        
        node = (1-alpha) * node + alpha *v0
        node = beta * self.node_agg(node)  + (1-beta) * node

        return node, edge


        


class Hypergraph(nn.Module):
    '''
    Hypergraph class, uses weights for vertex-edge and edge-vertex incidence matrix.
    One large graph.
    '''
    def __init__(self, vidx, eidx, ne, args):
        '''
        vidx: idx tensor of elements to select, shape (ne, max_n),
        shifted by 1 to account for 0th elem (which is 0)
        eidx has shape (nv, max n)..
        '''
        super(Hypergraph, self).__init__()
        self.args = args
        self.lamda = args.lamda
        self.alpha = args.alpha
        inp_dropout = args.inp_dropout_p
        dropout = args.dropout_p
        norm = args.norm
        layer = args.mlp_layers

        self.vidx = vidx
        self.eidx = eidx

        self.node_map = torch.nn.Linear(self.args.input_dim, self.args.n_hidden).to(os.environ['DEVICE'])
        self.edge_map = torch.nn.Linear(self.args.input_dim, self.args.n_hidden).to(os.environ['DEVICE'])

        self.convs = []
        for _ in range(self.args.n_layers):
            self.convs.append(HNHNIIConv(args.n_hidden, args.n_hidden, vidx, eidx, dropout, args).to(os.environ['DEVICE']))
        
        self.inp_dropout = nn.Dropout(inp_dropout)
        self.dropout = nn.Dropout(dropout)
        act = {'id': nn.Identity(), 'relu': nn.ReLU(), 'prelu':nn.PReLU()}
        self.act = act[args.activation]
        self.cls = MLP(self.args.n_hidden, self.args.n_hidden//2, self.args.n_cls, layer, dropout=dropout, Normalization=norm, InputNorm=False)

    def forward(self, v, e):
        '''
        Take initial embeddings from the select labeled data.
        Return predicted cls.
        '''
        # Initialize X_E \gets 0. Project X_V to hidden dimension
        e0 = torch_scatter.scatter_mean(v[self.vidx], self.eidx, dim=0)   
        e0 = self.inp_dropout(e0)   
        e0 = self.edge_map(e0)
        e0 = F.relu(e0)

        v = self.inp_dropout(v)
        v0 = self.node_map(v) 
        v0 = F.relu(v0)         
        

        # For i=1 to n_layers do
        v, e = v0, e0

        for i, conv in enumerate(self.convs):
            beta = math.log(self.lamda/(i+1)+1)

            v, e = conv(v, e, v0, e0, self.alpha, beta)
            v = self.act(v)
            e = self.act(e)

        pred = self.cls(v)
        return v, e, pred
    
    def all_params(self):
        params = []
        for conv in self.convs:
            params.extend(conv.parameters())
        
        params.extend(self.parameters())
        return params

