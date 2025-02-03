import argparse
from datetime import datetime
import random
import numpy as np
from tasks import *
import os
import time
from utils import *
from params import *
from manifolds import *
from models import RiemannianGNN, NodeClassification, GraphPrediction
import pytorch_lightning as pl  # Import PyTorch Lightning
import torch

def add_embed_size(args):
    # add 1 for Lorentz as the degree of freedom is d - 1 with d dimensions
    if args.select_manifold == 'lorentz':
        args.embed_size += 1

def parse_default_args():
    parser = argparse.ArgumentParser(description='RiemannianGNN')
    # First parse only the required arguments to determine the task
    parser.add_argument('--name', type=str, default='{:%Y_%m_%d_%H_%M_%S_%f}'.format(datetime.now()))
    parser.add_argument('--task', type=str, choices=['airport', 'qm8', 'qm9', 'zinc', 'ethereum', 'node_classification', 'synthetic', 'dd', 'enzymes', 'proteins', 'reddit', 'collab'])
    parser.add_argument('--select_manifold', type=str, default='lorentz', choices=['poincare', 'lorentz', 'euclidean'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true', help='Use 10% of the dataset for debugging')
    parser.add_argument('--use_tangent', action='store_true', help='Use tangent space instead of centroid-based approach')
    # Parse known args first to get the task
    args, remaining_argv = parser.parse_known_args()
    
    # Set up additional parameters based on the task
    if args.task == 'node_classification':
        NodeClassificationHyperbolicParams.add_params(parser)
    elif args.task == 'airport':
        AirportNodeClassificationHyperbolicParams.add_params(parser)
    elif args.task == 'zinc' and args.select_manifold == 'euclidean':
        ZINCEuclideanParams.add_params(parser)
    elif args.task == 'zinc' and args.select_manifold != 'euclidean':
        ZINCHyperbolicParams.add_params(parser)

    # Parse all arguments again with the task-specific parameters added
    args = parser.parse_args()
    add_embed_size(args)
    
    return args

def create_manifold(args, logger):
    if args.select_manifold == 'poincare':
        return PoincareManifold(args)
    elif args.select_manifold == 'lorentz':
        return LorentzManifold(args)
    elif args.select_manifold == 'euclidean':
        return EuclideanManifold(args)

if __name__ == '__main__':
    args = parse_default_args()
    pl.seed_everything(args.seed, workers=True)  # Use Lightning's seed_everything
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = create_logger('log/%s.log' % args.name)
    # logger.info("save debug info to log/%s.log" % args.name)
    logger.info(args)

    manifold = create_manifold(args, logger)
    rgnn = RiemannianGNN(args, manifold)

    if args.task == 'node_classification' or args.task == 'airport': # TODO: ADD AIRPORTS
        gnn_task = NodeClassificationTask(args, rgnn, manifold)
    else:
        gnn_task = GraphPredictionTask(args, rgnn, manifold)

    gnn_task.run()