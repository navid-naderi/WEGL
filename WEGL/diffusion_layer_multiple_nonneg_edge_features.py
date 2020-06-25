import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class Diffusion_layer(MessagePassing):
    """
    Generalized diffusion layer to incorporate multiple non-negative edge features
    """
    def __init__(self, aggr="add"):
        super(Diffusion_layer, self).__init__()
        
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr, self_loop_feature=1):
        
        num_nodes = x.size(0)
        
        # add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        norm = 0
        num_edge_labels = edge_attr.size(1)
        
        for i in range(num_edge_labels):
            
            # add unity features corresponding to self-loop edges
            self_loop_attr = self_loop_feature * torch.ones(num_nodes)
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr_i = torch.cat((edge_attr[:, i], self_loop_attr), dim = 0)

            # calculate node degrees corresponding to the current feature
            row, col = edge_index

            degrees = torch.zeros((num_nodes), dtype=x.dtype, device=row.device)
            degrees = degrees.scatter_add_(0, row, edge_attr_i * degrees.new_ones((row.size(0))))
            
            deg_inv_sqrt = degrees.pow(-0.5)
            norm += deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_attr_i
        
        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    def update(self, aggr_out):
        return aggr_out
    