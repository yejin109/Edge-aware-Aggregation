import yaml
import scipy
import pickle
import itertools
import numpy as np
import pandas as pd
import os.path as osp
import networkx as nx

import torch
from torch_scatter import scatter_add

with open('/home/yjozn/pythonProject/database/cfg.yml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


class GraphDataSet:
    def __init__(self, name: str, data_dir: str) -> None:
        self.device = cfg['DEVICE']
        self.name = name
        
        self.dataset_dir = '/'.join([data_dir, name])
        self.split_dir = osp.join(self.dataset_dir, 'splits')

        self.features = None
        self.labels = None
        self.hypergraph = None

    def get_edge_list():
        """
        return list
        """
        raise NotImplementedError
    
    def get_nodes_list():
        """
        return list
        """
        raise NotImplementedError
    
    def get_node_attr():
        return None

    def get_edge_attr():
        return None

    def preprocess_graph_dataset(self):
        raise NotImplementedError

    def preprocess_hyperedge_dataset(self):
        edge_set = set(self.hypergraph.keys())
        edge_to_num = {}
        num_to_edge = {}
        num = 0
        for edge in edge_set:
            edge_to_num[edge] = num
            num_to_edge[num] = edge
            num += 1

        incidence_matrix = []
        processed_hypergraph = {}
        for edge in edge_set:
            nodes = self.hypergraph[edge]
            processed_hypergraph[edge_to_num[edge]] = nodes
            for node in nodes:
                incidence_matrix.append([node, edge_to_num[edge]])

        self.edges = sorted(list(edge_set))
        self.processed_hypergraph = processed_hypergraph
        self.features = torch.as_tensor(self.features.toarray())
        self.hyperedge_index = torch.LongTensor(incidence_matrix).T.contiguous()
        self.labels = torch.LongTensor(self.labels)
        self.num_nodes = int(self.hyperedge_index[0].max()) + 1
        self.num_edges = int(self.hyperedge_index[1].max()) + 1
        self.num_classes = int(self.labels.max()) + 1
        self.edge_to_num = edge_to_num
        self.num_to_edge = num_to_edge

        edge_weight = torch.zeros(self.num_edges)
        node_weight = torch.zeros(self.num_nodes)
        for edge in edge_set:
            nodes = self.processed_hypergraph[edge_to_num[edge]]
            edge_weight[edge_to_num[edge]] += 1
            for node in nodes:
                node_weight[node] += 1
        self.edge_weight = edge_weight
        self.node_weight = node_weight
        self.to()

    def clique_expansion(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.get_nodes_list())
        for hyperedge in self.hypergraph.values():
            for edge in itertools.combinations(hyperedge, 2):
                u, v = edge
                graph.add_edge(u, v)
        return graph
    
    def star_expansion(self):
        graph = nx.Graph()
        graph.add_nodes_from([f'n{i}' for i in self.get_nodes_list()])

        for edge_i, hyperedge in enumerate(self.hypergraph.values()):
            for node in hyperedge:
                graph.add_edge(f'n{node}', f'e{edge_i}')
        return graph

    def get_adj_matrix(self):
        """
        return scipy.sparse array
        """
        if cfg['EXPANSION'] == 'star':
            graph = self.star_expansion()
        else:
            graph = self.clique_expansion()

        return nx.adjacency_matrix(graph)

    def to(self):
        self.features = self.features.to(self.device)
        self.hyperedge_index = self.hyperedge_index.to(self.device)
        self.labels = self.labels.to(self.device)

    # Load data
    def load_dataset(self):
        """
        hypergraph = {int edge_i : Int[] node_i }
        features = (node_num, feature_dim)
        labels = (node_num, )
        """
        with open(osp.join('/home/yjozn/pythonProject', self.dataset_dir, 'features.pickle'), 'rb') as f:
            self.features = pickle.load(f)
        with open(osp.join('/home/yjozn/pythonProject', self.dataset_dir, 'hypergraph.pickle'), 'rb') as f:
            self.hypergraph = pickle.load(f)
        with open(osp.join('/home/yjozn/pythonProject', self.dataset_dir, 'labels.pickle'), 'rb') as f:
            self.labels = pickle.load(f)

    def load_splits(self, seed: int):
        """
        file : mask
        return index!
        """
        with open(osp.join('/home/yjozn/pythonProject', self.split_dir, f'{seed}.pickle'), 'rb') as f:
            splits = pickle.load(f)

        idx = np.array(self.get_nodes_list())
        train_idx = idx[splits['train_mask']]
        val_idx = idx[splits['val_mask']]
        test_idx = idx[splits['test_mask']]
        return train_idx, val_idx, test_idx
    
    # Logger
    def get_hgraph_topology(self):
        weight = torch.ones(self.num_edges)
        Dn = scatter_add(weight[self.hyperedge_index[1]], self.hyperedge_index[0], dim=0, dim_size=self.num_nodes)
        De = scatter_add(torch.ones(self.hyperedge_index.shape[1]), self.hyperedge_index[1], dim=0, dim_size=self.num_edges)

        print('=============== Dataset Stats ===============')
        print(f'dataset name: {self.name}')
        print(f'features size: [{self.features.shape[0]}, {self.features.shape[1]}]')
        print(f'num nodes: {self.num_nodes}')
        print(f'num edges: {self.num_edges}')
        print(f'num connections: {self.hyperedge_index.shape[1]}')
        print(f'num classes: {int(self.labels.max()) + 1}')
        print(f'avg hyperedge size: {torch.mean(De).item():.2f}+-{torch.std(De).item():.2f}')
        print(f'avg hypernode degree: {torch.mean(Dn).item():.2f}+-{torch.std(Dn).item():.2f}')
        print(f'max node size: {Dn.max().item()}')
        print(f'max edge size: {De.max().item()}')
        print('=============================================')

    def get_meta(self):
        for name, var in vars(self).items():
            if var is None:
                continue
            print("="*50)
            print(name)
            log(var)
            print("="*50)
    
    def get_dataset(self):
        return {'hypergraph': self.hypergraph, 'features': self.features, 'labels': self.labels, 'n': self.features.shape[0]}
    


def log(var):
    var_type = type(var)
    print(var_type)
    if isinstance(var, list):
        print('len : ', len(var))
    elif isinstance(var, dict):
        case = list(var.keys())[0]
        print(f'Key type : ', type(case))
        print('Key case : ', case)

        print(f'Value type : ', type(var[case]))
        print('Value case : ', var[case])
    elif isinstance(var, scipy.sparse._csr.csr_matrix):
        print(var.toarray().shape)
    elif isinstance(var, str) or isinstance(var, int):
        print(var)
    elif isinstance(var, pd.DataFrame) or isinstance(var, pd.Series):
        print(var.info())
    elif isinstance(var, torch.Tensor):
        print('Shape : ', var.numpy().shape)




