import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils import get_activation, init_weight

class RiemannianGNN(nn.Module):
    """
    RiemannianGNN performs message passing on node representations on various manifolds.
    It supports diverse edge types (including negative edges) and multiple GNN layers,
    with weight tying if enabled.
    """
    def __init__(self, args, manifold):
        """
        Args:
            args: Configuration object containing hyperparameters (e.g., embed_size, gnn_layer,
                  dropout, proj_init, tie_weight, add_neg_edge, apply_edge_type, edge_type, etc.).
            manifold: Manifold object that provides functions like log_map_zero, exp_map_zero, 
                      from_lorentz_to_poincare, and from_poincare_to_lorentz.
        """
        super(RiemannianGNN, self).__init__()
        self.args = args
        self.manifold = manifold

        # Set up the activation function and dropout layer.
        self.activation = get_activation(self.args)
        self.dropout = nn.Dropout(self.args.dropout)

        # Instantiate message passing parameters for each message type.
        self.set_up_params()

    # def create_params(self):
    #     """
    #     Create a list of weight parameters (one per layer) for a specific message type.
    #     For Lorentz, the weight dimensions are adjusted by one.

    #     Returns:
    #         A nn.ParameterList containing a weight for each GNN layer (or one weight if tying).
    #     """
    #     message_weights = []
    #     num_layers = self.args.gnn_layer if not self.args.tie_weight else 1

    #     for _ in range(num_layers):
    #         if self.args.select_manifold in {"poincare", "euclidean"}:
    #             # Standard square weight matrix.
    #             M = th.zeros(self.args.embed_size, self.args.embed_size, requires_grad=True)
    #         elif self.args.select_manifold == "lorentz":
    #             # For Lorentz, one degree of freedom less.
    #             M = th.zeros(self.args.embed_size, self.args.embed_size - 1, requires_grad=True)
    #         else:
    #             raise ValueError("Unsupported manifold type: " + self.args.select_manifold)
    #         init_weight(M, self.args.proj_init)
    #         param = nn.Parameter(M)
    #         # Optionally, accumulate these parameters in a global list.
    #         self.args.eucl_vars.append(param)
    #         message_weights.append(param)
    #     return nn.ParameterList(message_weights)


    def create_params(self):
        """
        create the GNN params for a specific msg type
        """
        msg_weight = []
        layer = self.args.gnn_layer if not self.args.tie_weight else 1
        for _ in range(layer):
            # weight in euclidean space
            if self.args.select_manifold in {'poincare', 'euclidean'}:
                M = th.zeros([self.args.embed_size, self.args.embed_size], requires_grad=True)
            elif self.args.select_manifold == 'lorentz': # one degree of freedom less
                M = th.zeros([self.args.embed_size, self.args.embed_size - 1], requires_grad=True)
            init_weight(M, self.args.proj_init)
            M = nn.Parameter(M)
            self.args.eucl_vars.append(M)
            msg_weight.append(M)

        print(f"from RiemannianGNN: {len(self.args.eucl_vars)}")
        return nn.ParameterList(msg_weight)

    def set_up_params(self):
        """
        Set up message passing parameters based on edge-type configuration.
        Depending on whether negative edges or multiple edge types are used, we may have
        multiple parameter sets.
        """
        if not self.args.add_neg_edge and not self.args.apply_edge_type:
            self.type_of_msg = 1
        elif self.args.apply_edge_type and self.args.add_neg_edge:
            self.type_of_msg = self.args.edge_type
        elif self.args.add_neg_edge and not self.args.apply_edge_type:
            self.type_of_msg = 2
        else:
            raise Exception("Unsupported configuration for negative edges and edge type.")

        # Create a separate set of parameters for each message type.
        for i in range(self.type_of_msg):
            setattr(self, f"msg_{i}_weight", self.create_params())

    def retrieve_params(self, weight_list, step):
        """
        Retrieve the weight parameter for a given layer (or step).
        
        Args:
            weight_list: A list of weight parameters.
            step: The layer index (or 0 if weights are tied).

        Returns:
            The weight matrix adjusted for the manifold.
        """
        if self.args.select_manifold in {"poincare", "euclidean"}:
            layer_weight = weight_list[step]
        elif self.args.select_manifold == "lorentz":
            # For Lorentz, prepend a zero column to obtain a valid tangent matrix.
            zeros = th.zeros((self.args.embed_size, 1), device=weight_list[step].device)
            layer_weight = th.cat((zeros, weight_list[step]), dim=1)
        else:
            raise ValueError("Unsupported manifold type: " + self.args.select_manifold)
        return layer_weight

    def apply_activation(self, node_repr):
        """
        Apply the non-linearity on node representations based on the selected manifold.

        Args:
            node_repr: Tensor of shape [num_nodes, embed_size].

        Returns:
            Activated node representations.
        """
        if self.args.select_manifold in {"poincare", "euclidean"}:
            return self.activation(node_repr)
        elif self.args.select_manifold == "lorentz":
            # For Lorentz, map from Lorentz to Poincaré, apply activation, then map back.
            poincare_repr = self.manifold.from_lorentz_to_poincare(node_repr)
            activated = self.activation(poincare_repr)
            return self.manifold.from_poincare_to_lorentz(activated)
        else:
            raise ValueError("Unsupported manifold type: " + self.args.select_manifold)

    def split_graph_by_negative_edge(self, adj_mat, weight):
        """
        Split the graph into subgraphs based on positive and negative edges.

        Args:
            adj_mat: Adjacency matrix tensor.
            weight: Edge weight tensor.

        Returns:
            pos_adj_mat: Adjacency matrix for positive edges.
            pos_weight: Corresponding positive weights.
            neg_adj_mat: Adjacency matrix for negative edges.
            neg_weight: Corresponding (absolute) weights.
        """
        pos_mask = weight > 0
        neg_mask = weight < 0

        pos_adj_mat = adj_mat * pos_mask.long()
        neg_adj_mat = adj_mat * neg_mask.long()
        pos_weight = weight * pos_mask.float()
        neg_weight = -weight * neg_mask.float()
        return pos_adj_mat, pos_weight, neg_adj_mat, neg_weight

    def split_graph_by_type(self, adj_mat, weight):
        """
        Split the graph based on edge types for multi-relational datasets.
        
        Args:
            adj_mat: Adjacency matrix tensor.
            weight: Edge weight tensor.

        Returns:
            A tuple of (multi_relation_adj_mat, multi_relation_weight) lists.
        """
        multi_relation_adj_mat = []
        multi_relation_weight = []
        for relation in range(1, self.args.edge_type):
            type_mask = (weight.int() == relation)
            multi_relation_adj_mat.append(adj_mat * type_mask.long())
            multi_relation_weight.append(type_mask.float())
        return multi_relation_adj_mat, multi_relation_weight

    def split_input(self, adj_mat, weight):
        """
        Split the input adjacency and weight tensors based on edge type configurations.

        Args:
            adj_mat: Adjacency matrix tensor.
            weight: Edge weight tensor.

        Returns:
            Two lists: one for adjacency matrices and one for weight tensors.
        """
        if not self.args.add_neg_edge and not self.args.apply_edge_type:
            return [adj_mat], [weight]
        elif self.args.apply_edge_type and self.args.add_neg_edge:
            pos_adj, pos_weight, neg_adj, neg_weight = self.split_graph_by_negative_edge(adj_mat, weight)
            type_adj, type_weight = self.split_graph_by_type(pos_adj, pos_weight)
            type_adj.append(neg_adj)
            type_weight.append(neg_weight)
            return type_adj, type_weight
        elif self.args.add_neg_edge and not self.args.apply_edge_type:
            pos_adj, pos_weight, neg_adj, neg_weight = self.split_graph_by_negative_edge(adj_mat, weight)
            return [pos_adj, neg_adj], [pos_weight, neg_weight]
        else:
            raise Exception("Not implemented configuration for splitting input.")

    def aggregate_msg(self, node_repr, adj_mat, weight, layer_weight, mask):
        """
        Compute the message passing for a specific message type.
        
        Args:
            node_repr: Tensor [num_nodes, embed_size] containing node representations.
            adj_mat: Adjacency matrix for the current message type.
            weight: Corresponding edge weight tensor.
            layer_weight: The weight matrix for the current layer.
            mask: Mask tensor to filter valid nodes.
        
        Returns:
            Combined message tensor of shape [num_nodes, embed_size].
        """
        node_num, max_neighbor = adj_mat.size(0), adj_mat.size(1)
        # Linear transformation of node representations.
        msg = th.mm(node_repr, layer_weight) * mask
        # Gather neighbor representations via index selection.
        neighbors = th.index_select(msg, 0, adj_mat.view(-1))
        neighbors = neighbors.view(node_num, max_neighbor, -1)
        # Compute weighted sum over neighbors.
        neighbors = weight.unsqueeze(2) * neighbors
        combined_msg = th.sum(neighbors, dim=1)
        return combined_msg

    def get_combined_msg(self, step, node_repr, adj_list, weight, mask):
        """
        Combine messages from all edge types.

        Args:
            step: The current layer (or step) index.
            node_repr: Tensor of node representations.
            adj_list: List of adjacency matrices split by type.
            weight: List of weight tensors corresponding to each type.
            mask: Mask tensor for nodes.

        Returns:
            Aggregated message tensor.
        """
        layer_index = 0 if self.args.tie_weight else step
        combined_msg = None
        for relation in range(self.type_of_msg):
            weight_list = getattr(self, f"msg_{relation}_weight")
            layer_weight = self.retrieve_params(weight_list, layer_index)
            aggregated = self.aggregate_msg(node_repr, adj_list[relation], weight[relation], layer_weight, mask)
            combined_msg = aggregated if combined_msg is None else (combined_msg + aggregated)
        return combined_msg

    def forward(self, node_repr, adj_list, weight, mask):
        """
        Execute message passing through multiple GNN layers.

        Args:
            node_repr: Tensor [node_num, embed_size]. For hyperbolic embeddings, this is in Euclidean space,
                       and a log map is applied.
            adj_list: Adjacency list tensor [node_num, max_neighbor] to be split into multiple types.
            weight: Edge weight tensor corresponding to adj_list.
            mask: Tensor [node_num, 1] indicating valid nodes (1 for valid, 0 for padded nodes).

        Returns:
            Updated node representations in the proper manifold space.
        """
        # Split the input based on edge or sign properties.
        adj_list, weight = self.split_input(adj_list, weight)

        # Iterate through each message passing layer.
        for step in range(self.args.gnn_layer):
            # Map node representations to the tangent space on all but the first layer.
            node_repr = (self.manifold.log_map_zero(node_repr) * mask) if step > 0 else (node_repr * mask)
            combined_msg = self.get_combined_msg(step, node_repr, adj_list, weight, mask)
            combined_msg = self.dropout(combined_msg) * mask
            # Map back to the manifold and apply activation.
            node_repr = self.manifold.exp_map_zero(combined_msg) * mask
            node_repr = self.apply_activation(node_repr) * mask
        return node_repr