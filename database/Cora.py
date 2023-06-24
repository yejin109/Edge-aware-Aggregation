import yaml
import numpy as np
import os.path as osp

import database.utils as utils
from database.GraphDataSet import GraphDataSet

with open('/home/yjozn/pythonProject/database/cfg.yml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

if cfg['USE_SPLIT']: 
    data_dir = cfg['DATA_DIR_SPLIT']
else:
    data_dir = cfg['DATA_DIR']

words_len = 1433
links_len = 5429
nodes_len = 2708


class Cora(GraphDataSet):
    def __init__(self, device='cpu') -> None:
        super().__init__('cora', data_dir, device)
        self.cites = utils.load_file(self.dataset_dir, self.name, 'cites', sep="\t")
        self.cites = utils.file_to_df(self.cites, columns=None)

        self.content = utils.load_file(self.dataset_dir, self.name, 'content', sep="\t")
        self.content = utils.file_to_df(self.content, columns=["Index"]+[f'W{i}'for i in range(words_len)]+['Subject'])        
        
        self.nodes = self.content['Index'].astype(int)
        self.cites = self.cites.astype(int)

    def get_nodes(self):
        return self.nodes.to_list()
    
    def get_edge_list(self):
        return self.cites.to_numpy().tolist()

    def get_node_attr(self):
        return self.content.set_index('Index')['Subject'].reset_index().to_numpy()


class CoraSplit(GraphDataSet):
    def __init__(self, name: str) -> None:
        super().__init__(name, data_dir)

        self.load_dataset()
        self.preprocess_hyperedge_dataset()


    def get_nodes_list(self):
        return list(range(self.num_nodes))
    
    def get_edge_list(self):
        return list(self.hypergraph.keys())


if __name__ == '__main__':
    dset = Cora()
    print(dset.get_nodes())
    print(dset.get_edge_list())
    print(dset.get_node_attr())

