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

import copy

# os.environ['PYOPENGL_PLATFORM'] = 'egl'



class SimpleCrossSectionEnv(gym.Env):

  def __init__(self, input_file, k_state_neighborhood=5, previous_mesh_neighborhood=5, next_mesh_neighborhood=5, same_obs_size=False, trials=200):
    self.trials = trials
    self.input_file = input_file
    data = np.load(input_file, allow_pickle=True)
    self.M = data['cross_sections']
    self.sample_spacing = data['step']
    self.spacing_multiplier = 2
    self.folder_name = time.time()

    self._max_episode_steps = len(self.M) + 1
    for x in self.M:
      x += np.random.normal(size=(x.shape), loc=0, scale=(.05))

    self.same_obs_size = same_obs_size

    self.num_points = len(self.M[0])

    longest = 0
    for x in self.M:
      if len(x) > longest:
        longest = len(x)
    shape = (longest,2)
    self.longest = longest
    
    self.min_action = -.05
    self.max_action = .05

    #self.action_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=shape)
    self.action_space = spaces.Box(low=self.min_action, high=self.max_action, dtype=np.float32, shape=(longest*2,))
    
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=shape)

    # for x in self.M:
    #   print(x.shape)

    self.Mhat = []
    self.local_faces = []

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
    self.renderer = pyrender.OffscreenRenderer(256, 256)

    self.state = None

    self.pts = None
    self.qualities = []

    for t, x in enumerate(self.M):
      self.Mhat.append(x)
      # should only be the current and previous mhat
      to_reconstruct = []
      # iterate from i to prev_mesh_neighborhood inclusive. in reverse order too
      for it in range(1, -1 , -1):
        idx = t - it
        if idx < 0:
          continue
        # print('mhat add', idx)
        to_reconstruct.append(self.Mhat[idx])


      to_reconstruct = np.array(to_reconstruct)

      # if it's 1, just return
      if(len(to_reconstruct) == 1):
        continue

      # triangulate only the previous and current steps

      pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(to_reconstruct, self.sample_spacing*self.spacing_multiplier, weights=None)
      self.pts = pts
      self.tri_face = tri_face
      self.tetra_face = tetra_face 

      # store the faces that we get. These faces are only between two cross-sections, so they are going to be between 0 and 200
      shifted_faces = tri_face + (t - 1)*self.num_points # this 100 is the number of points!!!
      
      self.local_faces.append(shifted_faces)
      self.qualities.append(reward)

      # self.curr_position = np.argmin(self.qualities) + 1
      self.curr_position = 0

    self.ground_truth = copy.deepcopy(data['cross_sections'])
    self.initial_qualities = copy.deepcopy(self.qualities)


  def step(self, action):

    action = action.reshape(-1, 1)

    

    # action = np.clip(action, self.min_action, self.max_action)
    action = action/20.0

    if type(action) != int:
      action = action.reshape(-1, 2)

    self.Mhat[self.curr_position] = self.Mhat[self.curr_position] + action 
    info = {}


    # should only be the current and previous mhat
    to_reconstruct = []
    # iterate from i to prev_mesh_neighborhood inclusive. in reverse order too
    for it in range(1, -1 , -1):
      idx = self.curr_position - it
      if idx < 0:
        continue
      # print('mhat add', idx)
      to_reconstruct.append(self.Mhat[idx])


    to_reconstruct = np.array(to_reconstruct)


    # if it's 1, just return
    if(len(to_reconstruct) == 1):
      # for simple case, step_t follows the size of self.Mhat
      self.step_t += 1
      self.step_i += 1

      # self.curr_position = np.argmin(self.qualities) + 1
      self.curr_position += 1

      # print('going to {} next because it has lowest quality of {}'.format(self.curr_position, np.min(self.qualities)))
      # print('reward', reward)
      return self.state_neighborhood(), 0, False, info

    # triangulate only the previous and current steps

    pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(to_reconstruct, self.sample_spacing*self.spacing_multiplier, weights=None)
    self.pts = pts
    self.tri_face = tri_face
    self.tetra_face = tetra_face 

    # store the faces that we get. These faces are only between two cross-sections, so they are going to be between 0 and 200
    shifted_faces = tri_face + (self.curr_position - 1)*self.num_points # this 100 is the number of points!!!

    old_reward = self.qualities[self.curr_position-1]
    self.local_faces[self.curr_position-1] = shifted_faces


    # reward = (reward - old_reward)/self.initial_qualities[self.curr_position-1]
    if len(shifted_faces) == 0:
      print("oof, new faces are empty")
      return self.state_neighborhood(), -100, False, info
    old_err = (np.linalg.norm(self.ground_truth[self.curr_position] - self.M[self.curr_position]))
    new_err = (np.linalg.norm(self.ground_truth[self.curr_position] - self.Mhat[self.curr_position]))

    err_reward = old_err - new_err

    # reward += err_reward

    self.qualities[self.curr_position-1] = reward

    # reward += err_reward*10
    # if old_reward > reward:
    #   reward = -10


    # done if at last time step
    done = False
    if self.step_t == self.trials:
    # if self.curr_position+1 == len(self.Mhat):
      print(self.curr_position)
      done = True

      # do i want the final reward to be based on the entire thing?
      pts = np.empty(shape = (0,3))
      for i, m in enumerate(self.Mhat):
        col_to_add = np.ones(len(m))*i*self.sample_spacing
        res = np.hstack([m, np.atleast_2d(col_to_add).T])
        pts = np.concatenate([pts, res])

      tri_face = np.empty(shape = (0,3), dtype=np.int64)
      for i, m in enumerate(self.local_faces):
        tri_face = np.concatenate([tri_face, m])
      total_reward = utils.pyvista_to_reward(pts, tri_face, face_size=3)

      # Massive negative reward if we didn't succeed. Expected future reward should be lower
      if total_reward < 0:
          total_reward = -100
      return self.state_neighborhood(), total_reward, done, info

      # return self.state_neighborhood(), reward, done, info

    # for simple case, step_t follows the size of self.Mhat
    self.step_t += 1
    self.step_i += 1

    # self.curr_position = np.argmin(self.qualities) + 1
    self.curr_position += 1
    if self.curr_position == len(self.Mhat):
      self.curr_position = 0

    # print('going to {} next because it has lowest quality of {}'.format(self.curr_position, np.min(self.qualities)))

    # if self.step_t % 50 == 0:

    # DELETE LATER
    pts = np.empty(shape = (0,3))
    for i, m in enumerate(self.Mhat):
      col_to_add = np.ones(len(m))*i*self.sample_spacing
      res = np.hstack([m, np.atleast_2d(col_to_add).T])
      pts = np.concatenate([pts, res])

    tri_face = np.empty(shape = (0,3), dtype=np.int64)
    for i, m in enumerate(self.local_faces):
      tri_face = np.concatenate([tri_face, m])
    total_reward = utils.pyvista_to_reward(pts, tri_face, face_size=3)
    if self.step_t % 50 == 0:

      print('total mesh reward, local reward, error from orig', total_reward, reward, err_reward, self.step_t, self.curr_position)
    return self.state_neighborhood(), reward, done, info

  # returns a neighborhood around the mesh of the size defined during initialization
  def state_neighborhood(self):
    neighborhood = []
    
    for it in range(self.k, -1 , -1):
      idx = self.curr_position - it - 1
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
      idx = self.curr_position + it - 1
      if idx > len(self.M)-1:  # to ensure equal sizes, we will just add the last one
        if self.same_obs_size:
          neighborhood.append(self.Mhat[len(self.M)-1])
          idx = len(self.Mhat)-1
        else:
          continue
      else:
        neighborhood.append(self.Mhat[idx])
      # print('m add', idx)

    return np.array(neighborhood)


  def reset(self, is_eval=False):
    if not os.path.exists('saved'):
      os.makedirs('saved')

    if not os.path.exists('saved/local_jumpy_{}'.format(self.folder_name)):
      os.makedirs('saved/local_jumpy_{}'.format(self.folder_name))


    if len(self.Mhat) != 0:
      # local changes

      pts = np.empty(shape = (0,3))
      for i, m in enumerate(self.Mhat):
        col_to_add = np.ones(len(m))*i*self.sample_spacing
        res = np.hstack([m, np.atleast_2d(col_to_add).T])
        pts = np.concatenate([pts, res])

      tri_face = np.empty(shape = (0,3), dtype=np.int64)
      for i, m in enumerate(self.local_faces):
        tri_face = np.concatenate([tri_face, m])
      print('gonna calc reward')
      reward = utils.pyvista_to_reward(pts, tri_face, face_size=3)
      print('whole mesh quality: ', reward)
      img = utils.draw(pts, tri_face, self.renderer)

      if is_eval:
        reward = reward
        t = time.time()
      else:
        reward = ""
        t = ""

      cv2.imwrite('saved/local_jumpy_{}/sphere_{}_{}.png'.format(self.folder_name, t, reward), img)

      mesh = trimesh.Trimesh(vertices=pts, faces=tri_face)
      mesh.export(file_obj='saved/local_jumpy_{}/sphere_{}_{}.stl'.format(self.folder_name, t, reward))


    else:
      pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(self.M, self.sample_spacing)
      img = utils.draw(pts, tri_face, self.renderer)

      if is_eval:
        reward = reward
        t = time.time()
      else:
        reward = ""
        t = ""


      cv2.imwrite('saved/local_jumpy_{}/sphere_no_weights_no_local{}_{}.png'.format(self.folder_name, t, reward), img)

      mesh = trimesh.Trimesh(vertices=pts, faces=tri_face)
      mesh.export(file_obj='saved/local_jumpy_{}/sphere_no_weights_no_local{}_{}.stl'.format(self.folder_name, t, reward))


    self.Mhat = []
    self.step_i = 0
    self.step_t = 0
    self.local_faces = []
    self.qualities = []

    for t, x in enumerate(self.M):
      self.Mhat.append(x)
      # should only be the current and previous mhat
      to_reconstruct = []
      # iterate from i to prev_mesh_neighborhood inclusive. in reverse order too
      for it in range(1, -1 , -1):
        idx = t - it
        if idx < 0:
          continue
        # print('mhat add', idx)
        to_reconstruct.append(self.Mhat[idx])


      to_reconstruct = np.array(to_reconstruct)

      # if it's 1, just return
      if(len(to_reconstruct) == 1):
        continue

      # triangulate only the previous and current steps

      pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(to_reconstruct, self.sample_spacing*self.spacing_multiplier, weights=None)
      self.pts = pts
      self.tri_face = tri_face
      self.tetra_face = tetra_face 

      # store the faces that we get. These faces are only between two cross-sections, so they are going to be between 0 and 200
      shifted_faces = tri_face + (t - 1)*self.num_points # this 100 is the number of points!!!
      
      self.local_faces.append(shifted_faces)
      self.qualities.append(reward)

      self.curr_position = np.argmin(self.qualities) + 1






    # reset and add noise to Mhat
    #data = np.load(self.input_file, allow_pickle=True)
    #self.M = data['cross_sections']

    #for x in self.M:
      #x += np.random.normal(size=(x.shape), loc=0, scale=(.05))

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


    # img = utils.draw(self.pts, self.tri_face, self.renderer)

    # these lines are a progressive rendering of the actions being taken. above is only the local changes
    # get all previous points
    pts = np.empty(shape = (0,3))
    for i, m in enumerate(self.Mhat):
      col_to_add = np.ones(len(m))*i*self.sample_spacing
      res = np.hstack([m, np.atleast_2d(col_to_add).T])
      pts = np.concatenate([pts, res])

    faces = np.empty(shape = (0,4), dtype=np.int64)
    for i, m in enumerate(self.local_faces):
      faces = np.concatenate([faces, m])
    img = utils.draw(pts, faces, self.renderer)

    if self.first_rendering:
      self.first_rendering = False
      self.render_ax = plt.imshow(img)
    else:
      self.render_ax.set_data(img)

    if filename != None:
      # utils.save_3d_surface_wiremesh(self.pts, self.tri_face, '{}_{}.png'.format(filename, self.step_t))
      cv2.imwrite('{}_{}.png'.format(filename, self.step_t), img)
    plt.pause(.001)


  def calculate_M_reward(self):
    pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(self.M, self.sample_spacing)
    return reward


  def close(self):
    return
  
  def __getstate__(self):
    state = self.__dict__.copy()
    # Don't pickle renderer since it has pointers
    del state["renderer"]
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    # Add renderer back since it doesn't exist in the pickle
    self.renderer = pyrender.OffscreenRenderer(256, 256)

# input_file = '/home/abrar/cross_section_rl/data/cross_section_data/sphere_resampled.npz'
# env = SimpleCrossSectionEnv(input_file, same_obs_size=True)

# print(env.observation_space)
# env.reset()

# for i in range(len(env.M)):
# done = False
# i = 0
# while not done:
#   obs, reward, done, _ = env.step(np.random.rand(200,1))
#   # obs, reward, done, _ = env.step(np.zeros((200,1)))

#   print(i, obs.shape, reward, done)
#   i+=1
#   env.render()

# env.reset()
# test1 = np.array([[1,2], [3,4], [5,6]])
# test2 = np.array([[1,32], [3,64]])
# triangulate_list([test1, test2], 1)
