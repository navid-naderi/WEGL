import torch
from .diffusion_layer_multiple_nonneg_edge_features import Diffusion_layer

class Diffusion(torch.nn.Module):
    """
    The diffusion class, which is a forward pass of the node and edge features through the diffusion process
    """
    
    def __init__(self, num_hidden_layers, final_node_embedding='final'):
        super(Diffusion, self).__init__()
            
        # create the hidden layers
        self.layers = torch.nn.ModuleList([Diffusion_layer() for _ in range(num_hidden_layers)])
        
        self.final_node_embedding = final_node_embedding
            
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        out_concat = [x]
        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index, edge_attr)
            out_concat.append(x)

        if self.final_node_embedding == 'final':
            return x
        elif self.final_node_embedding == 'avg':
            return torch.mean(torch.stack(out_concat), dim=0)
        elif self.final_node_embedding == 'concat':
            return torch.cat(out_concat, dim=1)
        else:
            raise NotImplementedError()