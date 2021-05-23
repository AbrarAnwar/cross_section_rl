"""
model by dhiraj inspried from Charles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCAutoEncoder(nn.Module):
    """ Autoencoder for Point Cloud 
    Input: 
    Output: 
    """
    def __init__(self, point_dim, num_points, out_dim):
        super(PCAutoEncoder, self).__init__()
        self.out_dim = out_dim

        self.conv1 = nn.Conv1d(in_channels=point_dim, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=self.out_dim, kernel_size=1)

        self.fc1 = nn.Linear(in_features=self.out_dim, out_features=self.out_dim)
        self.fc2 = nn.Linear(in_features=self.out_dim, out_features=self.out_dim)
        self.fc3 = nn.Linear(in_features=self.out_dim, out_features=num_points*3)

        #batch norm
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.out_dim)
    
    def forward(self, x):

        batch_size = x.shape[0]
        point_dim = x.shape[1]
        num_points = x.shape[2]

        #encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.bn3(self.conv5(x)))

        # do max pooling 
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_dim)
        # get the global embedding
        global_feat = x

        #decoder
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))
        reconstructed_points = self.fc3(x)

        #do reshaping
        reconstructed_points = reconstructed_points.reshape(batch_size, point_dim, num_points)

        return reconstructed_points , global_feat

# if __name__ == "__main__":
#     model = PCAutoEncoder(3, 500)
#     data = torch.rand(size=(2,3,500))
#     res = model(data)
#     print(res[0].shape)
#     print(res[1].shape)