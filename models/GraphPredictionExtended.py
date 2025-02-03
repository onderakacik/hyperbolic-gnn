import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils import nn_init

class GraphPredictionExtended(nn.Module):
    """
    Extended version of GraphPrediction that uses tangent space for classification.
    """
    def __init__(self, args, rgnn, manifold):
        super(GraphPredictionExtended, self).__init__()
        self.args = args
        self.manifold = manifold

        # Define an embedding layer if embedding is not removed
        if not self.args.remove_embed:
            self.embedding = nn.Linear(args.num_feature, args.embed_size, bias=False)
            if self.args.embed_manifold == 'hyperbolic':
                self.manifold.init_embed(self.embedding)
                self.args.hyp_vars.append(self.embedding)
            elif self.args.embed_manifold == 'euclidean':
                nn_init(self.embedding, self.args.proj_init)
                self.args.eucl_vars.append(self.embedding)
        else:
            self.embedding = lambda x: x

        # RiemannianGNN module
        self.rgnn = rgnn

        # Projection layer for tangent space representations
        self.project_pool = nn.Linear(self.args.embed_size, self.args.embed_size)
        nn_init(self.project_pool, self.args.proj_init)
        self.args.eucl_vars.append(self.project_pool)

        # Output layer
        out_dim = 1 if self.args.is_regression else self.args.num_class
        self.output_linear = nn.Linear(self.args.embed_size, out_dim)
        nn_init(self.output_linear, self.args.proj_init)
        self.args.eucl_vars.append(self.output_linear)

    def forward(self, node, adj, weight, mask):
        """
        Forward pass using tangent space classification approach.
        """
        assert adj.size(0) == 1, "Batch size must be 1 for GraphPrediction."

        # Squeeze batch dimension
        node = node.squeeze(0)
        adj = adj.squeeze(0)
        weight = weight.squeeze(0)

        node_num = adj.size(0)
        valid_mask = (th.arange(1, node_num + 1, device=node.device) <= mask.item()).view(-1, 1).float()

        # Apply node embedding
        if self.args.embed_manifold == 'hyperbolic':
            node_repr = self.manifold.log_map_zero(self.embedding(node)) * valid_mask
        elif self.args.embed_manifold == 'euclidean':
            node_repr = (self.embedding(node) if not self.args.remove_embed else node) * valid_mask

        # Update node representations via RiemannianGNN
        node_repr = self.rgnn(node_repr, adj, weight, valid_mask)

        # Map to tangent space
        node_repr = self.manifold.log_map_zero(node_repr)

        # Masked mean pooling
        graph_repr = th.sum(node_repr * valid_mask, dim=0, keepdim=True) / (th.sum(valid_mask) + 1e-8)
        
        # Project and classify
        graph_repr = self.project_pool(graph_repr)
        return self.output_linear(graph_repr) 