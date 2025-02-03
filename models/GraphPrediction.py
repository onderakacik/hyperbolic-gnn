import torch as th
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic_module.CentroidDistance import CentroidDistance
from utils import nn_init

class GraphPrediction(nn.Module):
    """
    GraphPrediction model for graph-level tasks.
    
    This module embeds raw node features (if applicable), applies a RiemannianGNN
    to update node representations, aggregates them using a centroid-based distance module,
    and produces the final graph prediction.
    """
    def __init__(self, args, rgnn, manifold):
        """
        Initialize the GraphPrediction model.
        
        Args:
            args: Configuration object with hyperparameters such as num_feature, embed_size,
                  remove_embed flag, embed_manifold setting, is_regression flag, etc.
            logger: Logger instance.
            rgnn: An instance of the RiemannianGNN module.
            manifold: The manifold object with methods like log_map_zero and exp_map_zero.
        """
        super(GraphPrediction, self).__init__()
        self.args = args
        # self.logger = logger
        self.manifold = manifold

        # Define an embedding layer if embedding is not removed.
        if not self.args.remove_embed:
            self.embedding = nn.Linear(args.num_feature, args.embed_size, bias=False)
            if self.args.embed_manifold == 'hyperbolic':
                # Initialize embedding for hyperbolic space.
                self.manifold.init_embed(self.embedding)
                self.args.hyp_vars.append(self.embedding)
            elif self.args.embed_manifold == 'euclidean':
                nn_init(self.embedding, self.args.proj_init)
                self.args.eucl_vars.append(self.embedding)
                print(f"from GraphPrediction: {len(self.args.eucl_vars)}")
        else:
            # Use identity if embedding should be removed.
            self.embedding = lambda x: x

        # Instantiate the centroid distance module for aggregating node representations.
        self.distance = CentroidDistance(args, manifold)
        
        # RiemannianGNN is provided externally.
        self.rgnn = rgnn

        # Set up the output layer based on task type.
        out_dim = 1 if self.args.is_regression else self.args.num_class
        self.output_linear = nn.Linear(self.args.num_centroid, out_dim)
        nn_init(self.output_linear, self.args.proj_init)
        self.args.eucl_vars.append(self.output_linear)
        print(f"from GraphPrediction after append: {len(self.args.eucl_vars)}")

    def forward(self, node, adj, weight, mask):
        """
        Forward pass through the model.
        
        Args:
            node: Tensor of shape [1, node_num, num_feature] representing node features.
            adj: Tensor of shape [1, node_num, max_neighbor] for the adjacency list.
            weight: Tensor of shape [1, node_num, max_neighbor] for edge weights.
            mask: Tensor (or scalar) indicating the number of valid nodes in the graph.
        
        Returns:
            Tensor of final predictions (logits for classification or regression outputs).
        """
        # Ensure batch size is 1.
        assert adj.size(0) == 1, "Batch size must be 1 for GraphPrediction."

        # Squeeze the batch dimension.
        node = node.squeeze(0)
        adj = adj.squeeze(0)
        weight = weight.squeeze(0)

        node_num = adj.size(0)
        max_neighbor = adj.size(1)
        
        # Create a mask for valid nodes. The mask indicates which nodes are real (non-padding).
        valid_mask = (th.arange(1, node_num + 1, device=node.device) <= mask.item()).view(-1, 1).float()

        # Apply node embedding.
        if self.args.embed_manifold == 'hyperbolic':
            # Map from hyperbolic space to the tangent space with log_map_zero.
            node_repr = self.manifold.log_map_zero(self.embedding(node)) * valid_mask
        elif self.args.embed_manifold == 'euclidean':
            # For Euclidean space, directly apply the embedding.
            node_repr = (self.embedding(node) if not self.args.remove_embed else node) * valid_mask

        # Update node representations via the provided RiemannianGNN.
        node_repr = self.rgnn(node_repr, adj, weight, valid_mask)  # Output shape: [node_num, embed_size]

        # Aggregate node representations; distance returns a tuple where the first element is the graph representation.
        graph_repr, _ = self.distance(node_repr, valid_mask)

        # Final prediction.
        return self.output_linear(graph_repr)