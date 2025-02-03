import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils import nn_init, get_activation

class NodeClassificationExtended(nn.Module):
    """
    Extended NodeClassification model that uses tangent space for classification.
    """
    def __init__(self, args, rgnn, manifold):
        super(NodeClassificationExtended, self).__init__()
        self.args = args
        self.manifold = manifold

        # Linear transformation to embed node features
        self.feature_linear = nn.Linear(self.args.input_dim, self.args.embed_size)
        nn_init(self.feature_linear, self.args.proj_init)
        self.args.eucl_vars.append(self.feature_linear)

        # Projection layer for tangent space representations
        self.project_pool = nn.Linear(self.args.embed_size, self.args.embed_size)
        nn_init(self.project_pool, self.args.proj_init)
        self.args.eucl_vars.append(self.project_pool)

        # RiemannianGNN module
        self.rgnn = rgnn

        # Final classification layer
        self.output_linear = nn.Linear(self.args.embed_size, self.args.num_class)
        nn_init(self.output_linear, self.args.proj_init)
        self.args.eucl_vars.append(self.output_linear)

        # Activation and LogSoftmax
        self.activation = get_activation(self.args)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, adj, weight, features):
        # Ensure batch size is 1
        assert adj.size(0) == 1
        adj = adj.squeeze(0)
        weight = weight.squeeze(0)
        features = features.squeeze(0)

        # Project input features
        node_repr = self.activation(self.feature_linear(features))
        
        # Create mask for valid nodes
        mask = th.ones((self.args.node_num, 1), device=features.device)
        
        # Update node representations via RiemannianGNN
        node_repr = self.rgnn(node_repr, adj, weight, mask)

        # Map to tangent space
        node_repr = self.manifold.log_map_zero(node_repr)

        # Project and classify in tangent space
        node_repr = self.project_pool(node_repr)
        class_logit = self.output_linear(node_repr)
        
        return self.log_softmax(class_logit) 