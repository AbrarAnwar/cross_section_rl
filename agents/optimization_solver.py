import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

import trimesh
from scipy.spatial import Delaunay
import pyrender

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import utils
from envs.simple_cross_section_env import SimpleCrossSectionEnv

class Function(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros(size))
        # self.W.requires_grad = True
    def forward(self, input):
        return input + self.W
    def loss(self, pts, faces):
        return torch.Tensor([0.0])

class OptimizationAgent:
    def __init__(self, env):
        self.env = env
        self.step_t = 0
    def act(self, observation):

        if self.step_t < self.env.k + 1:
            idx = self.step_t % (self.env.k + 1)
        else:
            idx = self.env.k + 1

        # triangulate
        pts, tri_face, tetra_face = utils.triangulate_list(observation, self.env.sample_spacing)
        f = Function(observation[idx].shape)
        opt = torch.optim.SGD(f.parameters(), lr=.001)

        
        main = observation[idx]

        main_tensor = torch.from_numpy(observation[idx])

        pred = None
        for i in range(1):
            pred = f(main_tensor)
            observation[idx] = pred.detach().cpu().numpy()
            pts, tri_face, tetra_face = utils.triangulate_list(observation, self.env.sample_spacing)

            loss = f.loss(pts, tri_face)

            # backprop
            opt.zero_grad()
            # loss.backward()
            opt.step()
        diff = pred.detach().cpu().numpy() - main 
        self.step_t += 1
        return diff

input_file = '/home/abrar/thesis/cross_sections_rl/data/cross_section_data/sphere.npz'
env = SimpleCrossSectionEnv(input_file)

agent = OptimizationAgent(env)
obs = env.reset()

for i in range(50):
  action = agent.act(obs)
  obs, _, _, _ = env.step(action)
#   print(len(obs))
#   env.render()
