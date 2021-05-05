# from https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263

# An important point here is initialisation of the output matrix. We want it to be identity by default to start
#   training with no transformations at all. So, we just add an identity matrix to the output:
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import optim

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# from models.point_net import *
from models.point_net_ae import *

if torch.cuda.is_available():
    from losses.chamfer_distance_gpu import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance
else:
    from losses.chamfer_distance_cpu import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance

# get cross sections, stack them as points, then reconstruct them in an auto-encoder setup. The middle spot is the latent space
def main():
    input_file = '/home/abrar/thesis/cross_sections_rl/data/cross_section_data/sphere_resampled.npz'
    data = np.load(input_file, allow_pickle=True)
    M = data['cross_sections']
    sample_spacing = data['step']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    autoencoder = PCAutoEncoder(3, len(state_neighborhood(M, 1, sample_spacing)))
    autoencoder.to(device)

    states = []
    for i in range(len(M)):
        s = state_neighborhood(M, i, sample_spacing)
        states.append(s)
    print(len(states))
    train(autoencoder, states, sample_spacing, epochs=1000, save=True, name='sphere')

def state_neighborhood(M, step_i, sample_spacing, k=5, same_obs_size=True):
    neighborhood = []
    
    for it in range(k, -1 , -1):
      idx = step_i - it - 1
      if idx < 0: # to ensure equal sizes, we will just add the first one
        if same_obs_size:
          neighborhood.append(M[0])
          idx = 0
        else:
          continue
      else:
        neighborhood.append(M[idx])
      # print('mhat add', idx)
      

    # iterate from i+1 to next_mesh_neighborhood inclusive
    for it in range(1, k + 1):
      idx = step_i + it - 1
      if idx > len(M)-1:  # to ensure equal sizes, we will just add the last one
        if same_obs_size:
          neighborhood.append(M[len(M)-1])
          idx = len(M)-1
        else:
          continue
      else:
        neighborhood.append(M[idx])
      # print('m add', idx)
    # print(np.array(neighborhood, dtype=object).shape)
    # print(np.array(neighborhood).shape)
    neighborhood = np.array(neighborhood)
    # print(neighborhood.shape)

    pts = np.empty(shape = (0,3))
    for i,m in enumerate(neighborhood):
        col_to_add = np.ones(len(m))*i*sample_spacing
        res = np.hstack([m, np.atleast_2d(col_to_add).T])
        pts = np.concatenate([pts, res])

    return torch.tensor(np.array(pts))

def iterate_minibatches(inputs, batchsize, shuffle=True):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield [inputs[i] for i in excerpt]
        # yield inputs[excerpt]

def train(model, states, sample_spacing, epochs=15, save=True, name='train'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    chamfer_dist = ChamferDistance()
    for epoch in range(epochs): 
        model.train()
        latent_vector_all = torch.Tensor().to(device)

        running_loss = 0.0
        for i, data in enumerate(iterate_minibatches(states, 5)):
            data = np.stack(data)
            points = torch.tensor(data).to(device).float()
            points = points.transpose(2, 1)

            optimizer.zero_grad()

            reconstructed_points, latent_vector = model(points)
            latent_vector_all = torch.cat((latent_vector_all, latent_vector), 0) 

            points = points.transpose(1,2)
            reconstructed_points = reconstructed_points.transpose(1,2)
            dist1, dist2 = chamfer_dist(points, reconstructed_points)   # calculate loss
            train_loss = (torch.mean(dist1)) + (torch.mean(dist2))

            # print(f"Epoch: {epoch}, Iteration#: {i}, Train Loss: {train_loss}")
            
            train_loss.backward() # Calculate the gradients using Back Propogation
            optimizer.step()

            # print statistics
            running_loss += train_loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(states), running_loss / 10))
                    running_loss = 0.0

        # pointnet.eval()
        # correct = total = 0

        # # validation
        # if val_loader:
        #     with torch.no_grad():
        #         for data in val_loader:
        #             inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
        #             outputs, __, __ = pointnet(inputs.transpose(1,2))
        #             _, predicted = torch.max(outputs.data, 1)
        #             total += labels.size(0)
        #             correct += (predicted == labels).sum().item()
        #     val_acc = 100. * correct / total
        #     print('Valid accuracy: %d %%' % val_acc)

        # save the model
        if save and (epoch % 100 == 0):
            os.makedirs('saved_models/'+name, exist_ok=True)
            torch.save(model.state_dict(), 'saved_models/'+name+"/save_"+str(epoch)+".pth")
    if save:
        os.makedirs('saved_models/'+name, exist_ok=True)
        torch.save(model.state_dict(), 'saved_models/'+name+"/save_"+str(epoch)+".pth")

if __name__ == "__main__":
    main()