import yaml

with open('/home/yjozn/pythonProject/database/cfg.yml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

if cfg['USE_SPLIT']: 
    from database.Cora import CoraSplit as CoraDataSet
    from database.Pubmed import PubmedSplit as PubmedDataSet
    from database.Citeseer import CiteseerSplit as CiteseerDataSet
    from database.DBLP import DBLPSplit as DBLPDataSet
    from database.NTU2012 import NTU2012Split as NTU2012DataSet
    from database.ModelNet40 import ModelNet40Split as ModelNet40DataSet
    from database.Zoo import ZooSplit as ZooDataSet
    from database.NewsW100 import NewsW100Split as NewsW100DataSet

else:
    from database.Cora import Cora as CoraDataSet
    from database.Citeseer import Citeseer as CiteseerDataSet
    from database.DBLP import DBLP as DBLPDataSet