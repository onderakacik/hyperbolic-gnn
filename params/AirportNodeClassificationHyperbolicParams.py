import argparse
from utils import str2bool
import sys

def add_params(parser):
    # parser.add_argument('--dataset_str', type=str, 
    #                     default='cora', choices=['citeseer', 'cora', 'pubmed'])   
    parser.add_argument('--data_path', type=str, default='data/airport')
    parser.add_argument('--patience', type=int, default=100)    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_hyperbolic', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, 
                        default='adam', choices=['sgd', 'adam', 'amsgrad', 'noam']) 
    parser.add_argument('--lr_scheduler', type=str, 
                        default='exponential', choices=['exponential', 'cosine', 'cycle', 'none']) 
    parser.add_argument('--hyper_optimizer', type=str,
                        default='ramsgrad', 
                        choices=['rsgd', 'ramsgrad'])    
    parser.add_argument("--num_centroid", type=int, default=700)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--gnn_layer', type=int, default=2) 
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--activation', type=str, default='selu', choices=['relu', 'leaky_relu', 'rrelu', 'elu', 'prelu', 'selu'])
    parser.add_argument('--leaky_relu', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--add_neg_edge', type=str2bool, default='False')
    parser.add_argument('--lr_gamma', type=float, default=0.98)
    parser.add_argument('--proj_init', type=str, 
                        default='kaiming', 
                        choices=['xavier', 'orthogonal', 'kaiming', 'none'])
    parser.add_argument('--embed_size', type=int, default=600)   
    parser.add_argument('--apply_edge_type', type=str2bool, default="False") 
    parser.add_argument('--embed_manifold', type=str, default='euclidean', choices=['euclidean', 'hyperbolic']) 
    parser.add_argument('--tie_weight', type=str2bool, default="True") 
    parser.add_argument('--eucl_vars', type=list, default=[])    
    parser.add_argument('--hyp_vars', type=list, default=[]) 
