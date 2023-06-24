import yaml

import database.utils as utils
from database.GraphDataSet import GraphDataSet

with open('/home/yjozn/pythonProject/database/cfg.yml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

if cfg['USE_SPLIT']: 
    data_dir = cfg['DATA_DIR_SPLIT']
else:
    data_dir = cfg['DATA_DIR']

words_len = 3703
links_len = 4732
nodes_len = 3312

class Citeseer(GraphDataSet):
    def __init__(self) -> None:
        self.dataset_name = "citeseer"
        self.cites = utils.load_file(f"{data_dir}/{self.dataset_name}", self.dataset_name, 'cites', sep="\t")
        self.cites = utils.file_to_df(self.cites, columns=None)

        self.content = utils.load_file(f"{data_dir}/{self.dataset_name}", self.dataset_name, 'content', sep="\t")
        self.content = utils.file_to_df(self.content, columns=["Index"]+[f'W{i}'for i in range(words_len)]+['Subject'])        
        
        self.nodes = self.content['Index']

    def get_nodes(self):
        return self.nodes.to_numpy()
    
    def get_edge_list(self):
        return self.cites.to_numpy()

    def get_node_attr(self):
        return self.content.set_index('Index')['Subject'].reset_index().to_numpy()


class CiteseerSplit(GraphDataSet):
    def __init__(self, name: str) -> None:
        super().__init__(name, data_dir)

        self.load_dataset()
        self.preprocess_hyperedge_dataset()


    def get_nodes_list(self):
        return list(range(self.num_nodes))
    
    def get_edge_list(self):
        return list(self.hypergraph.values())


if __name__ == '__main__':
    dset = Citeseer()
    # print(dset.get_nodes())
    # print(dset.get_edge_list())
    print(dset.get_node_attr())
