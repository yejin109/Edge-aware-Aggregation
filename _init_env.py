import os
import sys
import torch

os.environ['ROOT_DIR'] = '/home/yjozn/pythonProject/HNHNII'
os.environ['DEVICE'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sys.path.append('/home/yjozn/pythonProject')