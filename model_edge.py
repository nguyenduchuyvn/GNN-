import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, add_remaining_self_loops

class Edge_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.edge_dim = edge_dim  # new

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_index, _ = add_remaining_self_loops(edge_index)   # 2 x (E+N), [E+N]

        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)  
        # self_loop_edges has N x edge_dim
        edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)   # 
        # edge_attr has (E+N) x edge_dim = E x edge_dim + N x edge_dim 
        # print('edge_attr', edge_attr)
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)   
                             # 2 x (E+N), N x emb(out), [E+N], [E+N]
        # self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, edge_attr, norm):
        # x_j has shape [E, out_channels]
        # print('x_j', x_j)
        # print(f'Norm*x_j: {norm.view(-1, 1) * x_j}')
        # print('edge_attr', edge_attr)
        x_j = torch.cat([x_j, edge_attr], dim=-1)
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j




edge_index = torch.tensor([[1, 2, 3],[0, 0, 0]], dtype=torch.long)   # 2 x E
# print(edge_index)
x = torch.tensor([[1], [1], [1], [1]], dtype=torch.float)   # N x emb(in)
edge_attr = torch.tensor([[10], [20], [30]], dtype=torch.float)   # E x edge_dim


edge_index, _ = add_remaining_self_loops(edge_index)   # 2 x (E+N), [E+N]
# edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
# print(edge_index)

# # Add node's self infomation (value=0) to edge_attr
# self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)  
# # N x edge_dim
# edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)   
# # (E+N) x edge_dim = E x edge_dim + N x edge_dim 
# print(edge_attr)

# from 1 dim -> 2 dim
emb_in, emb_out, edge_dim = 1 , 2,  1
conv = Edge_GCNConv(emb_in, emb_out, edge_dim)  # emb(in), emb(out), edge_dim

# forward
x1 = conv(x, edge_index, edge_attr)
print('x',x1)

# # Edge classification

# mlp = torch.nn.Linear(2*(emb_out +1), 2)
# # mlp = MLP(2 * hidden_channels, num_classes, num_layers)

# h = conv(x, edge_index, edge_attr)
# print('h', h.shape)
# h_src = h[edge_index[0]]
# h_dst = h[edge_index[1]]
# print('sum h_src, h_dst',(h_src*h_dst).sum(dim = -1))
# y = torch.cat([h_src, h_dst], dim=-1)

# y= mlp(y)
# print(y.shape)

#============or=============
# Using a hidden embedding for each node 
# and then take the dot product between nodes connected by edges.
# (The edge-wise dot product, not the GNNs obviously.)
# src, dst = edge_index
# score = (h[src] * h[dst]).sum(dim=-1)
# print("score", score)
# loss = F.cross_entropy(score, data.edge_label)  # Classification
# loss = F.mse_loss(score, data.edge_value)  # Regression