import yaml

import database.utils as utils
from database.GraphDataSet import GraphDataSet

with open('/home/yjozn/pythonProject/database/cfg.yml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

if cfg['USE_SPLIT']: 
    data_dir = cfg['DATA_DIR_SPLIT']
else:
    data_dir = cfg['DATA_DIR']


class ZooSplit(GraphDataSet):
    def __init__(self, name: str) -> None:
        super().__init__(name, data_dir)

        self.load_dataset()
        self.preprocess_hyperedge_dataset()


    def get_nodes_list(self):
        return list(range(self.num_nodes))
    
    def get_edge_list(self):
        return list(self.hypergraph.values())