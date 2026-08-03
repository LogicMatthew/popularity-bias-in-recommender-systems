'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Parser for LightGCN and its variants + modifications")

    parser.add_argument('--experiment',     type=str, default='origin', help="Available experiments: [origin, data_disrupt, noise_disrupt, pin_sampling, fastgcn]")
    parser.add_argument('--early_stopping', type=int, default=0,        help="Enable early stopping patience (if 0, disable early stopping)")
    
    # Disruption arguments
    parser.add_argument('--data_disrupt',  type=int,   default=0,   help="The number of neigbours from which you can draw to disrupt the data")
    parser.add_argument('--noise_disrupt', type=float, default=0,   help="The standard deviation of the gaussian noise added to disrupt data")

    # PinSage sampling arguments
    parser.add_argument('--pin_sampling',  type=int,   default=0,   help='Enable PinSage sampling graph every X epochs')
    parser.add_argument('--pin_n_hops',    type=int,   default=4,   help='The length of random walk')
    parser.add_argument('--pin_n_traces',  type=int,   default=10,  help='The number of random walks per node')
    parser.add_argument('--pin_top_k',     type=int,   default=12,  help='Top-K neighbors to keep')
    
    # FastGCN sampling arguments
    parser.add_argument('--layers_size',  nargs='?',  default="[2000, 3000, 4000, 5000]", help="Layer sizes for FastGCN sampling")
    parser.add_argument('--unique',       type=int,   default=1,                          help="Enable unique nodes")
    parser.add_argument('--sis_alpha',    type=float, default=None,                       help="Enable Smoothed Importance Sampling with the given alpha")
    
    # Smoothed BPR and IPS arguments
    parser.add_argument('--smoothed_bpr', type=float, default=None,                       help="Enable Smoothed BPR sampling on negative items")
    parser.add_argument('--ips_clip',     type=float, default=0.0,                        help="Enable Inverse Propensity Scoring on BPR loss")

    # Build-in arguments 
    parser.add_argument('--model',        type=str,   default='lgcn',          help='Available models: [lgcn, flgcn]')
    parser.add_argument('--dataset',      type=str,   default='gowalla',       help="Available datasets: [gowalla, yelp2018, amazon-book]")
    parser.add_argument('--epochs',       type=int,   default=1000)
    parser.add_argument('--layer',        type=int,   default=3,               help="The number of layers")
    parser.add_argument('--recdim',       type=int,   default=64,              help="The embedding size")
    parser.add_argument('--lr',           type=float, default=1e-3,            help="The learning rate")
    parser.add_argument('--decay',        type=float, default=1e-4,            help="The weight decay for L2 normalizaton")
    parser.add_argument('--bpr_batch',    type=int,   default=2048,            help="The batch size for bpr loss training procedure")
    parser.add_argument('--test_batch',   type=int,   default=100,             help="The batch size of users for testing")
    parser.add_argument('--neg_k',        type=int,   default=1,               help="The number of negative samples per positive instance")
    parser.add_argument('--topKs',        nargs='?',  default="[20]",          help="@k test list")
    
    # Dropout arguments
    parser.add_argument('--message_dropout',  type=float, default=0,           help="The probability for dropping edges a.k.a. message dropout")
    parser.add_argument('--node_dropout',     type=float, default=0,           help="The probability for dropping nodes a.k.a. node dropout")
    
    # Adjacency matrix splitting arguments
    parser.add_argument('--a_fold',       type=int,   default=100,             help="The fold number used to split large adj matrix, like gowalla")
    parser.add_argument('--a_split',      type=int,   default=0,               help="Enable splitting of adjacency matrix for large datasets")
    
    # Multiprocessing and random seed arguments
    parser.add_argument('--multicore',    type=int,   default=1,               help="Enable multiprocessing")
    parser.add_argument('--seed',         type=int,   default=2020,            help="Random seed")
    
    # Checkpointing and logging arguments
    parser.add_argument('--pretrain',     type=int,   default=0,               help="Enable loading pre-trained weights")
    parser.add_argument('--path',         type=str,   default="./checkpoints", help="Path to save weights")
    parser.add_argument('--resume',       type=int,   default=0,               help="Resume from the latest checkpoint according to the paramters used")
    parser.add_argument('--wandb',        type=int,   default=1,               help="Weight and Biases logging")

    return parser.parse_args()