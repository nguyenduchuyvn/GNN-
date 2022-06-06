import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp



class GCN(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(GCN, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        dense_neurons = model_params["model_dense_neurons"]
        self.conv1 = GCNConv(feature_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size*2)

        # Linear layers
        self.linear1 = Linear(embedding_size*4, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons/2))  
        self.linear3 = Linear(int(dense_neurons/2), 1)  

    def forward(self, x, edge_attr, edge_index, batch_index):
        x = self.conv1(x, edge_index)
        
        x = F.relu(x)
        # x = F.dropout(x, p= )
        x = self.conv2(x, edge_index)
        
        # Global Pooling (stack different aggregations)
        x = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)
        
        x = self.linear1(x)
        
        x = self.linear2(x)
        
        x = self.linear3(x)
        


        return x

model_params = dict()
model_params["model_embedding_size"] = 32
model_params["model_dense_neurons"] = 16
feature_size = 9
print(model_params)
model = GCN(feature_size, model_params)
print(model)
