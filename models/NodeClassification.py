import torch as th
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic_module.CentroidDistance import CentroidDistance
from utils import nn_init, get_activation

class NodeClassification(nn.Module):
    """
    NodeClassification model for node-level prediction tasks.

    This module first projects input features to a latent embedding space,
    performs message passing using the provided RiemannianGNN, then computes
    similarity scores between nodes and learnable centroids to obtain the final
    classification logits.
    """
    def __init__(self, args, rgnn, manifold):
        """
        Initialize the NodeClassification model.

        Args:
            args: Configuration object with parameters such as input_dim, embed_size,
                  node_num, num_centroid, num_class, proj_init, etc.
            rgnn: An instance of the RiemannianGNN module for message passing.
            manifold: Manifold object with methods like log_map_zero and exp_map_zero.
        """
        super(NodeClassification, self).__init__()
        self.args = args
        self.manifold = manifold

        # Linear transformation to embed node features.
        self.feature_linear = nn.Linear(self.args.input_dim, self.args.embed_size)
        nn_init(self.feature_linear, self.args.proj_init)
        self.args.eucl_vars.append(self.feature_linear)

        # Centroid distance module to compute node-to-centroid similarities.
        self.distance = CentroidDistance(args, manifold)

        # Riemannian GNN module that refines node representations.
        self.rgnn = rgnn

        # Final linear layer that produces classification logits.
        self.output_linear = nn.Linear(self.args.num_centroid, self.args.num_class)
        nn_init(self.output_linear, self.args.proj_init)
        self.args.eucl_vars.append(self.output_linear)

        # LogSoftmax for converting logits into log-probabilities.
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.activation = get_activation(self.args)

    def forward(self, adj, weight, features):
        """
        Forward pass for node classification.

        Args:
            adj: Tensor of shape [1, node_num, max_neighbor] with neighbor indices.
            weight: Tensor of shape [1, node_num, max_neighbor] with edge weights.
            features: Tensor of shape [1, node_num, input_dim] containing node features.

        Returns:
            Tensor of shape [node_num, num_class] with log-probabilities for each node.
        """
        # Ensure batch size is 1 and remove the batch dimension.
        assert adj.size(0) == 1, "Batch size must be 1 for NodeClassification."
        adj = adj.squeeze(0)
        weight = weight.squeeze(0)
        features = features.squeeze(0)

        # Project input features to latent space using a linear layer and non-linearity.
        node_repr = self.activation(self.feature_linear(features))
        
        # Construct a mask for valid nodes (here, all nodes are assumed valid).
        mask = th.ones((self.args.node_num, 1), device=features.device)
        
        # Update node representations via the RiemannianGNN.
        node_repr = self.rgnn(node_repr, adj, weight, mask)  # shape: [node_num, embed_size]
        
        # Compute node-to-centroid similarity using the centroid distance module.
        # The distance module returns a tuple; we use the second element (similarity scores).
        _, node_centroid_sim = self.distance(node_repr, mask)  # Expected shape: [1, node_num, num_centroid]
        
        # Remove the extra batch dimension if present and get final logits.
        class_logit = self.output_linear(node_centroid_sim.squeeze(0))
        # Return normalized log-probabilities for each node.
        return self.log_softmax(class_logit)