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

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import utils

import time

# os.environ['PYOPENGL_PLATFORM'] = 'egl'



class SimpleCrossSectionEnv(gym.Env):

  def __init__(self, input_file, k_state_neighborhood=5, previous_mesh_neighborhood=5, next_mesh_neighborhood=5, same_obs_size=False):
    data = np.load(input_file, allow_pickle=True)
    self.M = data['cross_sections']
    self.sample_spacing = data['step']

    self.same_obs_size = same_obs_size

    longest = 0
    for x in self.M:
      if len(x) > longest:
        longest = len(x)
    shape = (longest,2)
    self.longest = longest
    
    self.min_action = -.1
    self.max_action = .1

    #self.action_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=shape)

    default = np.ones(shape=(longest*2))
    weight_clip = np.ones(shape=longest)

    
    self.action_space = spaces.Box(low=np.concatenate([default*self.min_action, -weight_clip]), high=np.concatenate([default*self.max_action, weight_clip]), dtype=np.float32, shape=(longest*3,))
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=shape)

    # for x in self.M:
    #   print(x.shape)

    self.Mhat = []
    self.weights = []

    self.step_i = 0
    self.step_t = 0
    self.k = k_state_neighborhood
    self.previous_mesh_neighborhood = previous_mesh_neighborhood
    self.next_mesh_neighborhood = next_mesh_neighborhood

    self.curr_pts = None
    self.curr_tri_face = None
    self.curr_tetra_face = None

    self.first_rendering = True
    self.render_ax = None
    self.renderer = pyrender.OffscreenRenderer(512, 512)

    self.state = None

    self.pts = None

  def step(self, action):


    # print('action, min, max, mean', min(action), max(action), np.mean(action))
    #action = min(max(action, self.min_action), self.max_action)

    # expand for easier indexing if necessary
    action = action.reshape(-1, 1)

    # print('step_i: {} \t step_t: {}'.format(self.step_i, self.step_t))
    # print(action.shape)

    if self.step_i == len(self.M):
      # we are done. let's reconstruct the entire mesh and calculate the metrics as a reward
      pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(self.Mhat, self.sample_spacing, weights=self.weights)
      print('final reward', reward)
      return np.array(self.state_neighborhood()), reward, True, {}

    weights = action[self.longest*2:]
    action = action[:self.longest*2]
    
    self.weights.append(weights)

    action = np.clip(action, self.min_action, self.max_action)


    if type(action) != int:
      action = action.reshape(-1, 2)

    # take given mesh and add action to it
    Mhat_cur = self.M[self.step_i] + action
    self.Mhat.append(Mhat_cur)
    info = {}


    to_reconstruct = []
    to_reconstruct_weights = []
    # iterate from i to prev_mesh_neighborhood inclusive. in reverse order too
    for it in range(self.previous_mesh_neighborhood, -1 , -1):
      idx = self.step_t - it
      if idx < 0:
        continue
      # print('mhat add', idx)
      to_reconstruct.append(self.Mhat[idx])
      to_reconstruct_weights.append(self.weights[idx])

    # # iterate from i+1 to next_mesh_neighborhood inclusive
    # for it in range(1, self.next_mesh_neighborhood + 1):
    #   idx = self.step_i + it
    #   if idx > len(self.M)-1:
    #     continue
    #   # print('m add', idx)
    #   to_reconstruct.append(self.M[idx])
    to_reconstruct = np.array(to_reconstruct)
    to_reconstruct_weights = np.array(to_reconstruct_weights)

    self.to_recon = to_reconstruct
    self.to_recon = to_reconstruct_weights

    # if it's 1, just return
    if(len(to_reconstruct) == 1):
      # for simple case, step_t follows the size of self.Mhat
      self.step_t += 1
      self.step_i += 1
      # print('reward', reward)
      return self.state_neighborhood(), 0, False, info



    pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(to_reconstruct, self.sample_spacing, weights=to_reconstruct_weights)
    self.pts = pts
    self.tri_face = tri_face
    self.tetra_face = tetra_face # this is None because pyvista representation does treats everything as a triangle


    # done if at last time step
    done = False
    if self.step_i == len(self.M):
      done = True
      return None, reward, done, info

    # for simple case, step_t follows the size of self.Mhat
    self.step_t += 1
    self.step_i += 1
    # print('reward', reward)
    return self.state_neighborhood(), reward, done, info

  # returns a neighborhood around the mesh of the size defined during initialization
  def state_neighborhood(self):
    neighborhood = []
    
    for it in range(self.k, -1 , -1):
      idx = self.step_t - it - 1
      if idx < 0: # to ensure equal sizes, we will just add the first one
        if self.same_obs_size:
          neighborhood.append(self.Mhat[0])
          idx = 0
        else:
          continue
      else:
        neighborhood.append(self.Mhat[idx])
      # print('mhat add', idx)
      

    # iterate from i+1 to next_mesh_neighborhood inclusive
    for it in range(1, self.k + 1):
      idx = self.step_i + it - 1
      if idx > len(self.M)-1:  # to ensure equal sizes, we will just add the last one
        if self.same_obs_size:
          neighborhood.append(self.M[len(self.M)-1])
          idx = len(self.M)-1
        else:
          continue
      else:
        neighborhood.append(self.M[idx])
      # print('m add', idx)
    # print(np.array(neighborhood, dtype=object).shape)
    # print(np.array(neighborhood).shape)
    return np.array(neighborhood)


  def reset(self):

    if not os.path.exists('saved'):
        os.makedirs('saved')

    if len(self.Mhat) != 0:
        pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(self.Mhat, self.sample_spacing, weights=self.weights)
        img = utils.draw(pts, tri_face, self.renderer)
        t = time.time()
        cv2.imwrite('{}_{}_{:.4f}.png'.format('saved/sphere', t, reward), img)

        mesh = trimesh.Trimesh(vertices=pts, faces=tri_face)
        mesh.export(file_obj='{}_{}_{:.4f}.stl'.format('saved/sphere', t, reward))


    # else:
    #     pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(self.M, self.sample_spacing)
    #     img = utils.draw(pts, tri_face, self.renderer)
    #     t = time.time()
    #     cv2.imwrite('{}_{}_{:.4f}.png'.format('saved/sphere', t, reward), img)

    #     mesh = trimesh.Trimesh(vertices=pts, faces=tetra_face)
    #     mesh.export(file_obj='{}_{}_{:.4f}.stl'.format('saved/sphere', t, reward))



    self.Mhat = []
    self.weights = []
    self.step_i = 0
    self.step_t = 0

    neighborhood = []
    if self.same_obs_size:
      for it in range(self.k, -1 , -1):
        neighborhood.append(self.M[0])
      

    # iterate from i+1 to next_mesh_neighborhood inclusive
    for it in range(1, self.k + 1):
      idx = self.step_i + it - 1
      if idx > len(self.M)-1:  # to ensure equal sizes, we will just add the last one
        if self.same_obs_size:
          neighborhood.append(self.M[len(self.M)-1])
          idx = len(self.M)-1
        else:
          continue
      else:
        neighborhood.append(self.M[idx])
    
    return np.array(neighborhood)



  def render(self, filename=None):
    if self.pts is None:
      return


    img = utils.draw(self.pts, self.tri_face, self.renderer)

    if self.first_rendering:
      self.first_rendering = False
      self.render_ax = plt.imshow(img)
    else:
      self.render_ax.set_data(img)

    if filename != None:
      # utils.save_3d_surface_wiremesh(self.pts, self.tri_face, '{}_{}.png'.format(filename, self.step_t))
      cv2.imwrite('{}_{}.png'.format(filename, self.step_t), img)
    plt.pause(.001)





  def close(self):
    return
  
  def calculate_M_reward(self):
    pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(self.M, self.sample_spacing)
    return reward


# input_file = '/home/abrar/cross_section_rl/data/cross_section_data/sphere_resampled.npz'
# env = SimpleCrossSectionEnv(input_file, same_obs_size=True)

# # print(env.observation_space)
# # env.reset()

# # for i in range(len(env.M)):
# done = False
# i = 0
# while not done:
#   obs, reward, done, _ = env.step(np.random.rand(300,1))
#   print(i, obs.shape, reward, done)
#   i+=1
#   env.render()

# env.reset()
# test1 = np.array([[1,2], [3,4], [5,6]])
# test2 = np.array([[1,32], [3,64]])
# triangulate_list([test1, test2], 1)

