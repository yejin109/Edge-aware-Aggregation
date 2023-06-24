import os
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


def load_model(args):
    return Hypergraph(args.vidx, args.eidx, args.nv, args.ne, args.n_weight, args.e_weight, args).to(os.environ['DEVICE'])


class HyperMod(nn.Module):
    def __init__(self, input_dim, vidx, eidx, nv, ne, n_weight, e_weight, args, is_last=False, use_edge_lin=False):
        super(HyperMod, self).__init__()
        self.args = args

        # List of edge/node index for every (node index, edge index)
        self.eidx = eidx
        self.vidx = vidx

        # Arr[i] =  1 / (node weight or degree, the number of connected edges) or 1
        self.n_weight = n_weight

        # Arr[i] =  1 / (node weight or degree, the number of connected nodes) 
        self.e_weight = e_weight

        # The number of nodes/edges
        self.nv, self.ne = self.args.nv, self.args.ne
        
        # self.W_v2e = Parameter(torch.randn(self.args.n_hidden, self.args.n_hidden))
        # self.W_e2v = Parameter(torch.randn(self.args.n_hidden, self.args.n_hidden))
        # self.b_v = Parameter(torch.zeros(self.args.n_hidden))
        # self.b_e = Parameter(torch.zeros(self.args.n_hidden))
        self.W_v2e = nn.Linear(self.args.n_hidden, self.args.n_hidden)
        self.W_e2v = nn.Linear(self.args.n_hidden, self.args.n_hidden)

        self.is_last_mod = is_last
        self.use_edge_lin = use_edge_lin
        if is_last and self.use_edge_lin:
            self.edge_lin = torch.nn.Linear(self.args.n_hidden, self.args.final_edge_dim)
        
    def forward(self, v, e, v0=None, e0=None, alpha=None, beta=None):        
        # Normalize hypernodes
        if v0 is not None:            
            v = (1-alpha) * v + alpha * v0
            ve = F.relu((1-beta) * self.W_v2e(v) + beta * v)
        else:
            ve = F.relu(self.W_v2e(v))

        # ve = F.relu(self.W_v2e(v))

        v_fac = 1
        v = v*self.n_weight*v_fac 

        # Update hyperedges
        eidx = self.eidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        e = e.clone()
        # step 1
        ve = (ve*self.n_weight)[self.args.ve_lists[:, 0]]
        # step 2
        ve *= self.args.n_reg_weight
        # step 3
        e.scatter_add_(src=ve, index=eidx, dim=0)
        # step 4
        e /= self.args.e_reg_sum
        
        if e0 is not None:
            e = (1-alpha) * e + alpha * e0     
            ev = F.relu((1-beta) * self.W_e2v(e) + beta * e)
        else:
            ev = F.relu(self.W_e2v(e))
            
        # ev = F.relu(self.W_e2v(e))

        # Update hypernodes        
        vidx = self.vidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        # step 1
        ev_vtx = (ev*self.e_weight)[self.args.ve_lists[:, 1]]
        # step 2
        ev_vtx *= self.args.e_reg_weight
        # step 3
        v.scatter_add_(src=ev_vtx, index=vidx, dim=0)
        # step 4
        v /= self.args.n_reg_sum
        
        # NOTE: Initial Connection
        if v0 is not None:
            v = (1-alpha) * v + alpha * v0
        
        if not self.is_last_mod:
            v = F.dropout(v, self.args.dropout_p)
            # NOTE New
            e = F.dropout(e, self.args.dropout_p)


        if self.is_last_mod and self.use_edge_lin:
            ev_edge = (ev*torch.exp(self.e_weight)/np.exp(2))[self.args.ve_lists[:, 1]]
            v2 = torch.zeros_like(v)
            v2.scatter_add_(src=ev_edge, index=vidx, dim=0)
            v2 = self.edge_lin(v2)
            v = torch.cat([v, v2], -1)

        return v, e


class Hypergraph(nn.Module):
    '''
    Hypergraph class, uses weights for vertex-edge and edge-vertex incidence matrix.
    One large graph.
    '''
    def __init__(self, vidx, eidx, nv, ne, n_weight, e_weight, args):
        '''
        vidx: idx tensor of elements to select, shape (ne, max_n),
        shifted by 1 to account for 0th elem (which is 0)
        eidx has shape (nv, max n)..
        '''
        super(Hypergraph, self).__init__()
        self.args = args
        self.hypermods = []
        is_first = True
        for i in range(self.args.n_layers):
            is_last = True if i == self.args.n_layers-1 else False            
            self.hypermods.append(HyperMod(self.args.input_dim if is_first else self.args.n_hidden, vidx, eidx, nv, ne, n_weight, e_weight, self.args, is_last=is_last).to(os.environ['DEVICE']))
            is_first = False

        self.vtx_lin = torch.nn.Linear(self.args.input_dim, self.args.n_hidden)
        self.cls = nn.Linear(self.args.n_hidden, self.args.n_cls)

    def to_device(self, device):
        self.to(device)
        for mod in self.hypermods:
            mod.to('cuda')
        return self
        
    def all_params(self):
        params = []
        for mod in self.hypermods:
            params.extend(mod.parameters())
        return params
        
    def forward(self, v, e):
        '''
        Take initial embeddings from the select labeled data.
        Return predicted cls.
        '''
        lamda, alpha = 0.5, 0.4
        # Initialize X_E \gets 0. Project X_V to hidden dimension
        v = self.vtx_lin(v)

        v, e = self.hypermods[0](v, e)
        v0, e0 = v, e

        # For i=1 to n_layers do
        for i, mod in enumerate(self.hypermods[1:]):
            beta = math.log(lamda/(i+1)+1)
            v, e = mod(v, e, v0, e0, alpha, beta)

        # for i, mod in enumerate(self.hypermods):
        #     beta = math.log(lamda/(i+1)+1)
        #     v, e = mod(v, e)


        pred = self.cls(v)
        return v, e, pred
        