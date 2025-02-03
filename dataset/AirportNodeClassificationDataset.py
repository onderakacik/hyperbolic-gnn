#!/usr/bin/env/python

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
import torch
from torch.utils.data import Dataset
from utils import normalize_weight, pad_sequence

class AirportNodeClassificationDataset(Dataset):
    """
    Dataset class for Airport Node Classification
    """
    def __init__(self, args):
        self.args = args
        self.load_data()

    def load_data(self):
        """
        Loads and preprocesses airport dataset
        """
        # Load raw data
        graph = pkl.load(open(os.path.join(self.args.data_path, 'airport.p'), 'rb'))
        adj = nx.adjacency_matrix(graph)
        features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
        
        # Extract labels and features
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        
        # Bin the labels into discrete classes
        labels = self.bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        
        # Convert labels to one-hot encoding
        num_classes = len(np.unique(labels))
        labels_one_hot = np.zeros((labels.shape[0], num_classes))
        labels_one_hot[np.arange(labels.shape[0]), labels] = 1

        # Create adjacency lists with self-connections
        adj_lists = [[i] for i in range(features.shape[0])]
        weights = [[1] for i in range(features.shape[0])]
        adj_matrix = adj.toarray()
        
        # Add neighbors and their weights
        for i in range(len(adj_matrix)):
            neighbors = np.where(adj_matrix[i] > 0)[0].tolist()
            adj_lists[i].extend(neighbors)
            weights[i].extend([1] * len(neighbors))

        # Create adjacency label matrix (for compatibility)
        adj_label = []
        for i in range(len(adj_lists)):
            for j in range(len(adj_lists)):
                if j in adj_lists[i]:
                    adj_label.append(1)
                else:
                    adj_label.append(0)

        # Normalize weights and pad sequences
        max_len = max([len(i) for i in adj_lists])
        normalize_weight(adj_lists, weights)
        adj_lists = pad_sequence(adj_lists, max_len)
        weights = pad_sequence(weights, max_len)

        # Create train/val/test splits
        num_nodes = adj_matrix.shape[0]
        indices = np.random.permutation(num_nodes)
        
        train_size = int(0.7 * num_nodes)
        val_size = int(0.15 * num_nodes)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        # Create masks
        train_mask = np.zeros(num_nodes, dtype=bool)  # Changed to bool type
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        
        train_mask[train_idx] = 1
        val_mask[val_idx] = 1
        test_mask[test_idx] = 1

        # Create masked labels
        y_train = np.zeros_like(labels_one_hot)
        y_val = np.zeros_like(labels_one_hot)
        y_test = np.zeros_like(labels_one_hot)
        
        y_train[train_mask] = labels_one_hot[train_mask]
        y_val[val_mask] = labels_one_hot[val_mask]
        y_test[test_mask] = labels_one_hot[test_mask]

        # Store processed data
        self.adj = np.array(adj_lists)
        self.weight = np.array(weights)
        self.features = self.preprocess_features(features)
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.train_mask = train_mask.astype(int)
        self.val_mask = val_mask.astype(int)
        self.test_mask = test_mask.astype(int)
        self.adj_label = np.array(adj_label)  # Added adj_label

        # Set up parameters for the model
        self.args.node_num = features.shape[0]
        self.args.input_dim = features.shape[1]
        self.args.num_class = num_classes

    def preprocess_features(self, features):
        """Row-normalize feature matrix"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def bin_feat(self, feat, bins):
        """Bin continuous features into discrete classes"""
        digitized = np.digitize(feat, bins)
        return digitized - digitized.min()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            'adj': self.adj,
            'weight': self.weight,
            'features': self.features,
            'y_train': self.y_train,
            'y_val': self.y_val,
            'y_test': self.y_test,
            'train_mask': self.train_mask,
            'val_mask': self.val_mask,
            'test_mask': self.test_mask,
            'adj_label': self.adj_label  # Added adj_label to return dict
        } 