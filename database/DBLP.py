import json
import yaml
import pandas as pd

import database.utils as utils
from database.GraphDataSet import GraphDataSet

with open('/home/yjozn/pythonProject/database/cfg.yml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

data_dir = cfg['DATA_DIR']
ver = cfg['DBLP_ver']

_fname = "acm_output.txt"
if ver == 6:
    _fname = "acm_output.txt"

class DBLP(GraphDataSet):
    def __init__(self) -> None:
        super().__init__()
        self.dataset_name = "dblp"
        self.cites = utils.load_file(f"{data_dir}/{self.dataset_name}", self.dataset_name, 'cites', sep="\t")
        self.cites = utils.file_to_df(self.cites, columns=None)

        self.content = utils.load_file(f"{data_dir}/{self.dataset_name}", self.dataset_name, 'content', sep="\t")
        self.content = utils.file_to_df(self.content, columns=["Index", 'Title'])    
        
        self.nodes = self.content['Index']
        
    def get_nodes(self):
        return self.nodes.to_numpy()
    
    def get_edge_list(self):
        return self.cites.to_numpy()


class DBLPSplit(GraphDataSet):
    def __init__(self, name: str, device: str = 'cpu') -> None:
        super().__init__(name, data_dir, device)

        self.load_dataset()
        self.preprocess_hyperedge_dataset()


    def get_nodes_list(self):
        return list(range(self.num_nodes))
    
    def get_edge_list(self):
        return list(self.hypergraph.values())




def load_raw(version):    
    if version == 6:
        with open(f'./raw/dblp/{_fname}') as f:
            tmp = f.readlines()
        f.close()

        res = utils.parsing_dblp_v6(tmp)
        return res

if __name__ == "__main__":
    # NOTE: read raw file
    # raw = load_raw(ver)
    # raw.to_feather('/home/yjozn/pythonProject/root_data/raw/dblp/dblp_v6.feather')
    # raw.to_csv('/home/yjozn/pythonProject/root_data/raw/dblp/dblp_v6.csv')
    
    # NOTE: read feather file to generate cites and contents
    # raw = pd.read_feather('./raw/dblp/dblp_v6.feather')
    # print(raw)
    # paper_id = raw['index']
    # cite_id = raw['reference'].fillna(",").str.split(',').apply(lambda x: [i for i in x if i != ""])    
    # title_id = raw['title']

    # f_cite = open('./raw/dblp/dblp.cites', 'w')
    # f_cont = open('./raw/dblp/dblp.content', 'w')
    # for paper, cite, title in zip(paper_id, cite_id, title_id):
    #     f_cont.write(f"{paper}\t{title}\n")
    #     if len(cite) != 0:
    #         for cite_i in cite:
    #             f_cite.write(f"{paper}\t{cite_i}\n")

    dset = DBLP()
    print(dset.cites)
    print(dset.content)
    print(dset.get_nodes())
    print(dset.get_edge_list())
    