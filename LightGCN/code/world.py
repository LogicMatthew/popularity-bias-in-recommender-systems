'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import os
from os.path import join
import torch
from parse import parse_args
import multiprocessing
import sys

args = parse_args()

ROOT_PATH  = os.path.dirname(os.path.dirname(__file__))
CODE_PATH  = join(ROOT_PATH, 'code')
DATA_PATH  = join(ROOT_PATH, 'data')
FILE_PATH  = join(CODE_PATH, 'checkpoints')
sys.path.append(join(CODE_PATH, 'sources'))

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

all_datasets = ['gowalla', 'yelp2018', 'amazon-book']
all_models  = ['lgcn', 'flgcn']
all_experiments = ['origin', 'data_disrupt', 'noise_disrupt', 'pin_sampling', 'fastgcn']
all_metrics = ['recall', 'ndcg', 'precision']
all_metrics_group_names = ['Bottom 20%', 'Middle 60%', 'Top 20%']

config = {}
config['experiment'] = args.experiment
config['early_stopping'] = args.early_stopping

# Disruption arguments
config['data_disrupt'] = args.data_disrupt
config['noise_disrupt'] = args.noise_disrupt

# PinSage sampling arguments
config['pin_sampling'] = args.pin_sampling
config['pin_n_hops'] = args.pin_n_hops
config['pin_n_traces'] = args.pin_n_traces
config['pin_top_k'] = args.pin_top_k

# FastGCN sampling arguments
config['unique'] = args.unique
config['layers_size'] = eval(args.layers_size)
config['sis_alpha'] = args.sis_alpha

# Smoothed BPR and IPS arguments
config['smoothed_bpr'] = args.smoothed_bpr
config['ips_clip'] = args.ips_clip

# Build-in arguments
config['model_name'] = args.model
config['dataset'] = args.dataset

config['epochs'] = args.epochs
config['n_layers'] = args.layer
config['latent_dim_rec'] = args.recdim
config['lr'] = args.lr
config['decay'] = args.decay
config['bpr_batch_size'] = args.bpr_batch
config['test_batch_size'] = args.test_batch
config['neg_k'] = args.neg_k
config['topKs'] = eval(args.topKs)

config['message_dropout'] = args.message_dropout
config['node_dropout'] = args.node_dropout

config['A_n_fold'] = args.a_fold
config['A_split'] = args.a_split
config['multicore'] = args.multicore
config['seed'] = args.seed

config['pretrain'] = args.pretrain
config['PATH'] = args.path

config['resume'] = args.resume
config['wandb'] = args.wandb

# Additional settings
config['GPU']     = torch.cuda.is_available()
config['device']  = torch.device('cuda' if config['GPU'] else "cpu")
config['CORES']   = multiprocessing.cpu_count() // 2

if config['experiment'] not in all_experiments:
    raise NotImplementedError(f"Haven't supported {config['experiment']} yet!, try {all_experiments}")
if config['dataset'] not in all_datasets:
    raise NotImplementedError(f"Haven't supported {config['dataset']} yet!, try {all_datasets}")
if config['model_name'] not in all_models:
    raise NotImplementedError(f"Haven't supported {config['model_name']} yet!, try {all_models}")

# Ignore warnings of Pandas future changes
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling