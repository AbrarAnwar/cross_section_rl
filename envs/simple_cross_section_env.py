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

os.environ['PYOPENGL_PLATFORM'] = 'egl'



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
    
    self.min_action = -.1
    self.max_action = .1

    #self.action_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=shape)
    self.action_space = spaces.Box(low=self.min_action, high=self.max_action, dtype=np.float32, shape=(longest*2,))
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=shape)

    # for x in self.M:
    #   print(x.shape)

    self.Mhat = []
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


    print('action, min, max, mean', min(action), max(action), np.mean(action))
    #action = min(max(action, self.min_action), self.max_action)
    action = np.clip(action, self.min_action, self.max_action)

    # print('step_i: {} \t step_t: {}'.format(self.step_i, self.step_t))
    if self.step_i == len(self.M):
      # we are done. let's reconstruct the entire mesh and calculate the metrics as a reward
      pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(self.Mhat, self.sample_spacing)
      print('final reward', reward)
      return np.array(self.state_neighborhood()), reward, True, {}


    if type(action) != int:
      action = action.reshape(-1, 2)

    # take given mesh and add action to it
    Mhat_cur = self.M[self.step_i] + action
    self.Mhat.append(Mhat_cur)
    info = {}


    # use a classical triangulation algorithm between the neighborhood of timesteps
    # FIGURE THIS OUT
    to_reconstruct = []
    # iterate from i to prev_mesh_neighborhood inclusive. in reverse order too
    for it in range(self.previous_mesh_neighborhood, -1 , -1):
      idx = self.step_t - it
      if idx < 0:
        continue
      # print('mhat add', idx)
      to_reconstruct.append(self.Mhat[idx])

    # iterate from i+1 to next_mesh_neighborhood inclusive
    for it in range(1, self.next_mesh_neighborhood + 1):
      idx = self.step_i + it
      if idx > len(self.M)-1:
        continue
      # print('m add', idx)
      to_reconstruct.append(self.M[idx])
    self.to_recon = to_reconstruct

    # for easy case, sample spacing can be consistent. for subsampling, we might need to employ some tricks.

    # pts, tri_face, tetra_face = utils.triangulate_list(to_reconstruct, self.sample_spacing)
    # self.pts = pts
    # self.tri_face = tri_face
    # self.tetra_face = tetra_face

    pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(to_reconstruct, self.sample_spacing)
    self.pts = pts
    self.tri_face = tri_face
    self.tetra_face = tetra_face

    # calculate the reward of the triangulation

    # reward = self.reward_function(['aspect_ratio'])



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
        pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(self.Mhat, self.sample_spacing)
        img = self.draw(pts, tri_face)
        t = time.time()
        cv2.imwrite('{}_{}_{:.4f}.png'.format('saved/sphere', t, reward), img)

        mesh = trimesh.Trimesh(vertices=pts, faces=tetra_face)
        mesh.export(file_obj='{}_{}_{:.4f}.stl'.format('saved/sphere', t, reward))


    else:
        pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(self.M, self.sample_spacing)
        img = self.draw(pts, tri_face)
        t = time.time()
        cv2.imwrite('{}_{}_{:.4f}.png'.format('saved/sphere', t, reward), img)

        mesh = trimesh.Trimesh(vertices=pts, faces=tetra_face)
        mesh.export(file_obj='{}_{}_{:.4f}.stl'.format('saved/sphere', t, reward))



    self.Mhat = []
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

    print('render')

    img = self.draw(self.pts, self.tri_face)

    if self.first_rendering:
      self.first_rendering = False
      self.render_ax = plt.imshow(img)
    else:
      self.render_ax.set_data(img)

    if filename != None:
      # utils.save_3d_surface_wiremesh(self.pts, self.tri_face, '{}_{}.png'.format(filename, self.step_t))
      cv2.imwrite('{}_{}.png'.format(filename, self.step_t), img)
    print('done1')
    plt.pause(.001)

    print('done2')

  def draw(self, pts, tri_face):
    mesh = trimesh.Trimesh(pts, tri_face)
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False, wireframe=False)

    scene = pyrender.Scene()
    camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)

    scene.add(mesh, pose=  np.eye(4))
    scene.add(light, pose=  np.eye(4))

    scene.add(camera, pose=[
                            [ 0,  0,  1,  1],
                            [ 1,  0,  0,  -.5],
                            [ 0,  1,  0,  .5],
                            [ 0,  0,  0,  1]
                            ])
    # scene.add(camera, pose=[
    #                         [ 1,  0,  0,  1],
    #                         [ 0,  1,  0,  0],
    #                         [ 0,  0,  1,  2],
    #                         [ 0,  0,  0,  1]
    #                         ])
    img, _ = self.renderer.render(scene)
    return img


  def close(self):
    return
  
  def calculate_M_reward(self):
    pts, tri_face, tetra_face, reward = utils.triangulate_list_and_reward(self.M, self.sample_spacing)
    return reward


#input_file = '/home1/07435/aanwar/cross_section_rl/data/cross_section_data/sphere_resampled.npz'
#env = SimpleCrossSectionEnv(input_file, same_obs_size=True)

#print(env.observation_space)
#env.reset()

# for i in range(len(env.M)):
#done = False
#while not done:
  #obs, reward, done, _ = env.step(0)
  #print(obs.shape, reward, done)
#   env.render()


# test1 = np.array([[1,2], [3,4], [5,6]])
# test2 = np.array([[1,32], [3,64]])
# triangulate_list([test1, test2], 1)

