import os
import math
import numpy as np

import torch
import torch_scatter
from torch_scatter import scatter, scatter_add
from torch_geometric.utils import softmax
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP


def load_model(args):
    return Hypergraph(args.vidx, args.eidx, args.ne, args).to(os.environ['DEVICE'])


class HNHNIIConv(nn.Module):
    def __init__(self, in_features, out_features, vidx, eidx, dropout_p ,args):
        super().__init__()
        self.vidx = vidx
        self.eidx = eidx

        self.node_agg = nn.Linear(out_features, out_features, bias=False)
        self.edge_agg = nn.Linear(out_features, out_features, bias=False)

    def forward(self, v, e, v0, e0, alpha, beta):
        # Massage passing : Node -> edge
        node_msg = v[self.vidx]
        
        # Message aggregation
        edge= scatter(node_msg, self.eidx, 0, reduce= 'mean')

        # Edge update
        edge_alpha = alpha
        edge = (1-edge_alpha) * edge + edge_alpha *e0
        edge = beta * self.edge_agg(edge)  + (1-beta) * edge

        # Denoising Edge : Activation        
        edge = F.relu(edge)

        # Message passing : Edge -> node
        edge_msg = edge[self.eidx] 

        # Denoising Node : Message weighting
        sim = F.cosine_similarity(edge_msg, node_msg, dim=1).unsqueeze(-1)
        sim = torch.sigmoid(sim)
        
        # Message aggregation
        node = scatter(edge_msg * sim, self.vidx, dim=0, reduce='mean', dim_size=v.size(0))

        # Node update 
        node_alpha = alpha
        node = (1-node_alpha) * node + node_alpha *v0
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

    def orthogonal_reg(self):
        reg = self.convs[0].orthogonal_reg()
        for conv in self.convs[1:]:
            reg += conv.orthogonal_reg()
        return reg

    def emb_dist(self, v, e):
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
        embs = [v.detach().cpu().numpy()]
        for i, conv in enumerate(self.convs):
            beta = math.log(self.lamda/(i+1)+1)

            v, e = conv(v, e, v0, e0, self.alpha, beta)
            v = self.act(v)
            e = self.act(e)
            
            embs.append(v.detach().cpu().numpy())
        embs = np.array(embs)
        return embs