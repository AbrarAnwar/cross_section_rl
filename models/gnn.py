import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

class SpaceTimeElementEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden=16):
        super(SpaceTimeElementEncoder, self).__init__()
        self.output_dim = output_dim
        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GCNConv(hidden, output_dim)

    def forward(self, data, training):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # edge_index = batch.edge_index
        # x, edge_index = data.x, data.edge_index

        batch_num = torch.max(batch).item() + 1

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=training)
        x = self.conv2(x, edge_index)

        graph_embeds = []
        for i in range(batch_num):
            z = x[torch.where(batch == i)]
            z = torch.sum(z, dim=0)
            graph_embeds.append(z)
        graph_embeds = torch.stack(graph_embeds)

        return graph_embeds


# x = torch.tensor([[2,1], [5,6], [3,7], [12,0], [1,4]], dtype=torch.float)
# edge_index = torch.tensor([[0, 1, 2, 0, 3, 1], [1, 0, 1, 3, 2, 4]], dtype=torch.long)
# data1 = Data(x=x, edge_index=edge_index)

# x = torch.tensor([[2,31], [3,9], [3,7], [2,60]], dtype=torch.float)
# edge_index = torch.tensor([[0, 1, 2, 0, 3], [3, 0, 1, 1, 2]], dtype=torch.long)
# data2 = Data(x=x, edge_index=edge_index)
# print(x.shape)
# print(edge_index.shape)

# data_list = [data1, data2]
# print(data_list)
# batch = Batch.from_data_list(data_list)

# print(batch.num_graphs)
# print(batch)

# print(batch.x)
# print(batch.edge_index)
# print(batch.batch)


# model = SpaceTimeElementEncoder(2, 8)

# print(model(batch, False))
# print(model(batch, False).shape)