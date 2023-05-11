import os
import sys
import torch

os.environ["WANDB_API_KEY"] = "88d9728c967d7e5c3020c2cc2a93615bf870c5ee"
os.environ["WANDB_MODE"] = "dryrun"

os.environ['ROOT_DIR'] = '/home/yjozn/pythonProject/HNHNII'
os.environ['DEVICE'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sys.path.append('/home/yjozn/pythonProject')
