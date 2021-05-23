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

import random

class SimpleCrossSectionEnv(gym.Env):

  def __init__(self, input_files, k_state_neighborhood=5, previous_mesh_neighborhood=5, next_mesh_neighborhood=5, same_obs_size=False, trials=300, reward_type="HOT", noise=.05):
    self.trials = trials
    self.input_files = input_files
    self.reward_type = reward_type


    self.noise = noise

    # from the list, choose a random one
    f_idx = random.randrange(len(self.input_files))
    data = np.load(self.input_files[f_idx], allow_pickle=True)
    self.M = data['cross_sections']
    self.sample_spacing = data['step']
    self.file_name = self.input_files[f_idx].split(os.sep)[-1][:-4]

    self.spacing_multiplier = 1
    self.folder_name = str(time.time()) + "_" + reward_type

    self._max_episode_steps = len(self.M) + 1

    # add noise
    noise_scale = self.noise
    for x in self.M:
        n = np.random.normal(loc=0, scale=x[:,0].std(), size=x[:,0].shape) * noise_scale
        x[:,0] += n
        n = np.random.normal(loc=0, scale=x[:,1].std(), size=x[:,1].shape) * noise_scale
        x[:,1] += n

    self.same_obs_size = same_obs_size

    self.num_points = len(self.M[0])

    longest = 0
    for x in self.M:
      if len(x) > longest:
        longest = len(x)
    shape = (longest,2)
    self.longest = longest

    self.episode_number = 0

    # bounds of the action
    self.min_action = -.1
    self.max_action = .1

    #self.action_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=shape)

    default = np.ones(shape=(longest*2))
    weight_clip = np.ones(shape=longest)

    
    self.action_space = spaces.Box(low=np.concatenate([default*self.min_action, -weight_clip]), \
            high=np.concatenate([default*self.max_action, weight_clip]), dtype=np.float32, shape=(longest*3,))
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=shape)

    # for x in self.M:
    #   print(x.shape)

    self.Mhat = []
    self.weights = []


    # this is a mesh refinement process, so let the initial be the a non-weighted delaunay triangulation
    self.local_faces = self.fill_local_faces(self.M, self.sample_spacing, self.spacing_multiplier)

    for x in self.M:
      self.weights.append(np.zeros(shape=(len(x), 1)))

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


    # expand for easier indexing if necessary
    action = action.reshape(-1, 1)

    # print('step_i: {} \t step_t: {}'.format(self.step_i, self.step_t))

    weights = action[self.longest*2:]
    action = action[:self.longest*2] 

    # scale action down
    action = action*self.max_action

    # apply weights' displacement
    self.weights[self.step_i] + weights


    if type(action) != int:
      action = action.reshape(-1, 2)

    # take given mesh and add action to it
    Mhat_cur = self.M[self.step_i] + action
    self.Mhat.append(Mhat_cur)
    info = {}


    # should only be the current and previous mhat
    to_reconstruct = []
    to_reconstruct_weights = []
    # iterate from i to prev_mesh_neighborhood inclusive. in reverse order too
    for it in range(1, -1 , -1):
      idx = self.step_i - it
      if idx < 0:
        continue
      to_reconstruct.append(self.Mhat[idx])
      to_reconstruct_weights.append(self.weights[idx])

    to_reconstruct = np.array(to_reconstruct)
    to_reconstruct_weights = np.array(to_reconstruct_weights)


    # if it's 1, just return
    if(len(to_reconstruct) == 1):
      # for simple case, step_t follows the size of self.Mhat
      self.step_t += 1
      self.step_i += 1
      # print('reward', reward)
      return self.state_neighborhood(), 0, False, info

    # triangulate only the previous and current steps
    pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(to_reconstruct, self.sample_spacing*self.spacing_multiplier, weights=to_reconstruct_weights, reward_type=self.reward_type, twodWDT=False)
    # for rendering
    self.pts = pts
    self.tri_face = tri_face
    self.tetra_face = tetra_face 

    # store the faces that we get. These faces are only between two cross-sections, so they are going to be between 0 and 200
    shifted_faces = tri_face + (self.step_i - 1)*self.num_points # this 100 is the number of points!!!

    self.local_faces[self.step_i - 1] = shifted_faces


    # done if at last time step
    done = False
    if self.step_i == len(self.M) - 1:
      done = True
      # NOTE: this is for if we want to have the last step have a higher reward
      # do i want the final reward to be based on the entire thing?
      # pts = np.empty(shape = (0,3))
      # for i, m in enumerate(self.Mhat):
      #   col_to_add = np.ones(len(m))*i*self.sample_spacing
      #   res = np.hstack([m, np.atleast_2d(col_to_add).T])
      #   pts = np.concatenate([pts, res])

      # tri_face = np.empty(shape = (0,3), dtype=np.int64)
      # for i, m in enumerate(self.local_faces):
      #   tri_face = np.concatenate([tri_face, m])
      # total_reward = utils.pyvista_to_reward(pts, tri_face, weights=self.weights, reward_type=self.reward_type, step=self.sample_spacing*self.spacing_multiplier, face_size=3)

      # Massive negative reward if we didn't succeed. Expected future reward should be lower
      # if total_reward < 0:
      #     total_reward = -100

      # from instability of HOT metric we clip the reward
      reward = np.clip(reward, -100, 100)
      return self.state_neighborhood(), reward, done, info


    reward = np.clip(reward, -100, 100)

    # for simple case, step_t follows the size of self.Mhat
    self.step_t += 1
    self.step_i += 1
    print('reward', reward)
    return self.state_neighborhood(), reward, done, info

  # returns a neighborhood around the mesh of the size defined during initialization
  def state_neighborhood(self):
    neighborhood = []
    for it in range(self.k, 0 , -1):
      idx = self.step_i - it 
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
    for it in range(0, self.k + 1):
      idx = self.step_i + it
      if idx > len(self.M)-1:  # to ensure equal sizes, we will just add the last one
        if self.same_obs_size:
          neighborhood.append(self.M[len(self.M)-1])
          idx = len(self.M)-1
        else:
          continue
      else:
        neighborhood.append(self.M[idx])
      # print('m add', idx)

    # local faces
    # now i want a window of the same size for this
    windowed_faces = []
    for it in range(-self.k, self.k, 1):
      idx = self.step_i + it
      if idx < 0: # to ensure equal sizes, we will just add the first one
        continue
      if idx > len(self.local_faces)-1:
        continue
      else:
        faces = self.local_faces[idx]
        windowed_faces.extend(faces)
      # print('local faces intersection add', idx)
    
    idx = self.step_i - self.k
    if idx < 0: # to ensure equal sizes, we will just add the first one
      idx = 0
    if idx > len(self.local_faces)-1:
      idx = len(self.local_faces) - 1

    # get the indices of the window to be correct by shifting it
    windowed_faces = np.array(windowed_faces)  - idx*self.num_points

    # now turn these into undirected edges
    edges = trimesh.graph.shared_edges(windowed_faces, windowed_faces)
    swapped_edges = copy.deepcopy(edges)
    swapped_edges[:, [1, 0]] = edges[:, [0, 1]]
    edges = np.concatenate([edges, swapped_edges])

    state = (np.array(neighborhood), edges)
    return state

  def reset(self, is_eval=False):
    # open file to record results:

    if not os.path.exists('saved'):
      os.makedirs('saved')

    if not os.path.exists('saved/weights_local_{}'.format(self.folder_name)):
      os.makedirs('saved/weights_local_{}'.format(self.folder_name))

    if not os.path.exists('saved/weights_local_{}/{}'.format(self.folder_name, self.file_name)):
      os.makedirs('saved/weights_local_{}/{}'.format(self.folder_name, self.file_name))


    qualities_file = open('saved/weights_local_{}/{}_qualities.txt'.format(self.folder_name, self.file_name), "a+")

    # if it's not the first time, we aggreate results
    if len(self.Mhat) != 0:
      pts = np.empty(shape = (0,3))
      for i, m in enumerate(self.Mhat):
        col_to_add = np.ones(len(m))*i*self.sample_spacing
        res = np.hstack([m, np.atleast_2d(col_to_add).T])
        pts = np.concatenate([pts, res])

      tri_face = np.empty(shape = (0,3), dtype=np.int64)
      for i, m in enumerate(self.local_faces):
        tri_face = np.concatenate([tri_face, m])
      reward = utils.pyvista_to_reward(pts, tri_face, weights=self.weights, step=self.sample_spacing*self.spacing_multiplier, reward_type=self.reward_type, face_size=3)
      print('whole mesh quality: ', reward)
      img = utils.draw(pts, tri_face, self.renderer)

      if is_eval:
        reward = reward
        t = time.time()
      else:
        reward = ""
        t = ""


      mesh = trimesh.Trimesh(vertices=pts, faces=tri_face)

      eval_tag = ""
      if is_eval:
        eval_tag = "eval_"

      cv2.imwrite('saved/weights_local_{}/{}/{}episode{}.png'.format(self.folder_name,  self.file_name, eval_tag, self.episode_number), img)
      mesh.export(file_obj='saved/weights_local_{}/{}/{}episode_{}.stl'.format(self.folder_name,  self.file_name, eval_tag, self.episode_number))
      # exit()

      HOT_qual = utils.pyvista_to_reward(pts, tri_face, weights=self.weights, step=self.sample_spacing, reward_type='HOT', face_size=3)
      rewards_dict =  utils.record_pyvista_rewards(pts, tri_face, step=self.sample_spacing, face_size=3)
      rewards_dict["HOT"] = HOT_qual
      if is_eval == True:
        qualities_file.write("Eval Episode {} ".format(self.episode_number) + str(rewards_dict) + '\n')
      else:
       qualities_file.write("Episode {} ".format(self.episode_number) + str(rewards_dict) + '\n')

    # otherwise, if we haven't refined anything yet, output what it looks like
    else:

      pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(self.M, self.sample_spacing, weights=self.weights, reward_type=self.reward_type)
      img = utils.draw(pts, tri_face, self.renderer)

      if is_eval:
        reward = reward
        t = time.time()
      else:
        reward = ""
        t = ""

      cv2.imwrite('saved/weights_local_{}/{}_no_weights_no_local{}_{}.png'.format(self.folder_name, self.file_name, t, reward), img)

      mesh = trimesh.Trimesh(vertices=pts, faces=tri_face)
      mesh.export(file_obj='saved/weights_local_{}/{}_no_weights_no_local{}_{}.stl'.format(self.folder_name, self.file_name, t, reward))

      HOT_qual = utils.pyvista_to_reward(pts, tri_face, weights=self.weights, step=self.sample_spacing, reward_type=self.reward_type, face_size=3)
      rewards_dict =  utils.record_pyvista_rewards(pts, tri_face, step=self.sample_spacing, face_size=3)
      rewards_dict["HOT"] = HOT_qual
      qualities_file.write("Episode {} ".format(self.episode_number) + str(rewards_dict) + '\n')

    qualities_file.close()

    # properly reset now
    self.episode_number += 1


    self.Mhat = []
    self.step_i = 0
    self.step_t = 0
    self.weights = []


    for x in self.M:
      self.weights.append(np.zeros(shape=(len(x), 1)))


    # from the list, choose a random one
    f_idx = random.randrange(len(self.input_files))
    data = np.load(self.input_files[f_idx], allow_pickle=True)
    self.M = data['cross_sections']
    self.sample_spacing = data['step']
    self.file_name = self.input_files[f_idx].split(os.sep)[-1][:-4]
    # add noise
    noise_scale = self.noise
    for x in self.M:
        n = np.random.normal(loc=0, scale=x[:,0].std(), size=x[:,0].shape) * noise_scale
        x[:,0] += n
        n = np.random.normal(loc=0, scale=x[:,1].std(), size=x[:,1].shape) * noise_scale
        x[:,1] += n


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

    self.local_faces = self.fill_local_faces(self.M, self.sample_spacing, self.spacing_multiplier)

    windowed_faces = []
    for it in range(-self.k, self.k, 1):
      idx = self.step_i + it
      if idx < 0: # to ensure equal sizes, we will just add the first one
        continue
      if idx > len(self.local_faces)-1:
        continue
      else:
        faces = self.local_faces[idx]
        windowed_faces.extend(faces)
    
    idx = self.step_i - self.k - 1
    if idx < 0: # to ensure equal sizes, we will just add the first one
      idx = 0
    if idx > len(self.local_faces)-1:
      idx = len(self.local_faces) - 1

    windowed_faces = np.array(windowed_faces)  - idx*self.num_points


    # now turn these into undirected edges
    edges = trimesh.graph.shared_edges(windowed_faces, windowed_faces)
    swapped_edges = copy.deepcopy(edges)
    swapped_edges[:, [1, 0]] = edges[:, [0, 1]]
    edges = np.concatenate([edges, swapped_edges])


    # we know the size of k, so we just do this here. TODO: Make more general
    return np.array([self.M[0], self.M[1], self.M[2]]), edges
    # return np.array(neighborhood), edges


  # fills a list of faces for a neighborhood of cross-sections
  def fill_local_faces(self, M, sample_spacing, spacing_multiplier):
  
    local_faces = []
    for i in range(1, len(M)):
      to_reconstruct = np.array([M[i-1], M[i]])
      _, tri_face, _, _ = utils.triangulate_list_and_reward(to_reconstruct, \
          sample_spacing*spacing_multiplier, weights=None, reward_type="scaled_jacobian")
      # store the faces that we get. These faces are only between two cross-sections, so they are going to be between 0 and 200

      shifted_faces = tri_face + (i-1)*self.num_points # this 100 is the number of points!!!
      local_faces.append(shifted_faces)
    return local_faces




  def pre_initialize_experiences(self, num_times=10):
    experiences = []
    for _ in range(num_times):
      for f_idx in range(len(self.input_files)):
        data = np.load(self.input_files[f_idx], allow_pickle=True)
        Mhat = data['cross_sections'] # correct original sequence is mesh

        sample_spacing = data['step']
        M = copy.deepcopy(data['cross_sections'])
        prev_state = None

        noise_scale = self.noise
        for x in M:
            n = np.random.normal(loc=0, scale=x[:,0].std(), size=x[:,0].shape) * noise_scale
            x[:,0] += n
            n = np.random.normal(loc=0, scale=x[:,1].std(), size=x[:,1].shape) * noise_scale
            x[:,1] += n
        # iterate through to build experiences

        # build local faces
        local_faces = self.fill_local_faces(M, sample_spacing, self.spacing_multiplier)


        for step_i in range(0, len(Mhat)):
          # print('at step', step_i)
          # Goal: I want to iteratively build space-time elements that match the proper actions in the order they need to be
          
          state = []
          # first let's get the current state:
          for it in range(self.k, 0, -1):
            idx = step_i - it
            if idx >= 0:
              state.append(Mhat[idx])
              # print('mhat add', idx)
          for it in range(0, self.k + 1):
            idx = step_i + it
            if idx < len(M):
              state.append(M[idx])
              # print('m add', idx)

          # local faces
          # now i want a window of the same size for this
          windowed_faces = []
          for it in range(-self.k, self.k, 1):
            idx = step_i + it
            if idx < 0: # to ensure equal sizes, we will just add the first one
              continue
            if idx > len(local_faces)-1:
              continue
            else:
              faces = local_faces[idx]
              windowed_faces.extend(faces)
            # print('local faces intersection add', idx)
          
          idx = step_i - self.k
          if idx < 0: # to ensure equal sizes, we will just add the first one
            idx = 0
          if idx > len(local_faces)-1:
            idx = len(local_faces) - 1

          windowed_faces = np.array(windowed_faces)  - idx*self.num_points
          print(windowed_faces.shape)
          # now turn these into undirected edges
          edges = trimesh.graph.shared_edges(windowed_faces, windowed_faces)
          swapped_edges = copy.deepcopy(edges)
          swapped_edges[:, [1, 0]] = edges[:, [0, 1]]
          edges = np.concatenate([edges, swapped_edges])

          state = (state, edges)

          if step_i == 0:
            prev_state = state
            continue

          # now calculate the action

          idx = self.k
          if (step_i-1) < self.k:
            idx = step_i-1

          # print('state', idx, len(prev_state[0]))
          M_i = prev_state[0][idx]

          idx = self.k
          if (step_i-1) < self.k:
            idx = step_i
          # print(M_i)
          # print(M[step_i])

          # print('next', idx-1, len(state[0]))
          Mhat_i = state[0][idx-1]
          # print(Mhat_i)
          # print(Mhat[step_i-1])

          action = (Mhat_i - M_i).reshape(-1)

          reward = 0
          to_reconstruct = np.array([Mhat[step_i-2], M[step_i-1]])
          weights = None
          if self.reward_type != "HOT":
            pts, tri_face, _, reward = utils.triangulate_list_and_reward(to_reconstruct, \
              sample_spacing*self.spacing_multiplier, weights=None, reward_type=self.reward_type, twodWDT=False)
          else:
            weights =  np.zeros((2, self.num_points))
            pts, tri_face, _, reward = utils.triangulate_list_and_reward(to_reconstruct, \
              sample_spacing*self.spacing_multiplier, weights=weights, reward_type=self.reward_type, twodWDT=False)
          
          action = np.concatenate([action, np.zeros((self.num_points))])

          # img = utils.draw(pts, tri_face, self.renderer)
          # if self.first_rendering:
          #   self.first_rendering = False
          #   self.render_ax = plt.imshow(img)
          # else:
          #   self.render_ax.set_data(img)
          # plt.imshow(img)
          # plt.pause(.01)

          print('reward', reward)
          done = (len(Mhat)-1) == step_i
          experiences.append( (prev_state, action, reward, state, done) )
          print('-'*10, 'done', '-'*10)

          

          prev_state = state

    return experiences


  def list_to_pts(self, M_list):
    pts = np.empty(shape = (0,3))
    for i, m in enumerate(M_list):
      col_to_add = np.ones(len(m))*i*self.sample_spacing
      res = np.hstack([m, np.atleast_2d(col_to_add).T])
      pts = np.concatenate([pts, res])
    return pts

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

    faces = np.empty(shape = (0,3), dtype=np.int64)
    for i, m in enumerate(self.local_faces):
      faces = np.concatenate([faces, m])
      if (len(self.Mhat)-1) == (i+1):
        break
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
    pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(self.M, self.sample_spacing, weights=self.weights, reward_type=self.reward_type)
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
    self.renderer = pyrender.OffscreenRenderer(512, 512)


# input_file = ['/home/abrar/cross_section_rl/data/cross_section_data/sphere_10verts_10sections.npz']
# env = SimpleCrossSectionEnv(input_file, same_obs_size=False, \
#      k_state_neighborhood=2, previous_mesh_neighborhood=2, next_mesh_neighborhood=2)

# # experiences = env.pre_initialize_experiences(num_times=1)
# # np.savez('exp', experiences)


# # print(env.observation_space)
# env.reset()

# done = False
# i = 0
# while not done:
#   print('taking step', i)
#   obs, reward, done, _ = env.step(np.random.rand(20+10,1))
#   print(i, len(obs), reward, done)
#   i+=1
#   env.render()

# env.reset()
# test1 = np.array([[1,2], [3,4], [5,6]])
# test2 = np.array([[1,32], [3,64]])
# triangulate_list([test1, test2], 1)

