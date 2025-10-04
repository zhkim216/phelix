import torch

#### PROTEIN-MPNN UTILS ####

# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # In 3D case, gather_dim = 1,
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    
    # In 4D case, gather_dim = 2, 
    # Features [B,N,M,C] at neighbor indices [B,N,M,K] => [B,N,M,K,C]
    # Flatten and expand indices per batch [B,N,M,K] => [B,N,M*K] => [B,N,M*K,C]
    
    len_shape = len(nodes.shape)
    if len_shape == 3:
        gather_dim = 1
        neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1)) # [B,N*K]
        neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(-1)) # [B,N*K,C]
    elif len_shape == 4:
        gather_dim = 2
        neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], neighbor_idx.shape[1], -1)) # [B,N,M*K]
        neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, -1, nodes.size(-1)) # [B,N,M*K,C]
    else:
        raise ValueError(f"Nodes must be 3D or 4D, but got {len_shape}D")
        
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, gather_dim, neighbors_flat) # [B,N*K,C] or [B,N,M*K,C]
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:gather_dim+2] + [-1]) # [B,N,K,C] or [B,N,M,K,C]
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1) 
    #! (JH) h_neighbors should have the same dimension at -2 as h_nodes
    #! (JH) as this is for adding the neighbor node to the edge between the center node and the neighbor node
    return h_nn
