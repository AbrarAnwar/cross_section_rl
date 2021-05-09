import numpy as np
from scipy.spatial import Delaunay
# import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pyvista as pv

import trimesh
from scipy.spatial import ConvexHull
import pyrender
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'


def triangulate_list(subsequence, step):
  # list is of size (N, 2), since we have a sequence of 2D cross sections
  pts = np.empty(shape = (0,3))
  # put them all into a single point list separated by step distance away
  for i,m in enumerate(subsequence):
    col_to_add = np.ones(len(m))*i*step
    res = np.hstack([m, np.atleast_2d(col_to_add).T])
    pts = np.concatenate([pts, res])

  # now let's triangulate this list
  tri_face, tetra_face = delaunay_triangulation(pts)

  return pts, tri_face, tetra_face

# From https://github.com/kiranvad/WeightedDelaunay/blob/master/weighted_delaunay.ipynb
def WeightedDelaunay(points,weights):

    num, dim = np.shape(points)

    lifted = np.zeros((num,dim+1))

    for i in range(num):
        p = points[i,:]
        lifted[i,:] = np.append(p,np.sum(p**2) - weights[i]**2)
    pinf = np.append(np.zeros((1,dim)),1e10);    
    lifted = np.vstack((lifted, pinf))
    hull = ConvexHull(lifted)
    delaunay = []
    for simplex in hull.simplices:
        if num not in simplex:
            delaunay.append(simplex.tolist())
            
    return delaunay

def pyvistaToTrimeshFaces(cells):
    faces = []
    idx = 0
    # for i in range(nCells):
    while idx < len(cells):
      curr_cell_count = cells[idx]
      curr_faces = cells[idx+1:idx+curr_cell_count]
      faces.append(curr_faces)
      idx += curr_cell_count+1

    return np.array(faces)


# main difference here is it uses pyvista
def triangulate_list_and_reward(subsequence, step, reward_type='scaled_jacobian', weights=None):
  # stack by heights
  pts = np.empty(shape = (0,3))
  for i,m in enumerate(subsequence):
    col_to_add = np.ones(len(m))*i*step
    res = np.hstack([m, np.atleast_2d(col_to_add).T])
    pts = np.concatenate([pts, res])

  # use pyvista to turn this into a PolyData
  
  # print(poly)

  if(weights is not None):
    print(weights.shape)
    print(pts.shape)
    tetra_face = np.array(WeightedDelaunay(pts, weights))

    # make it compatible with pyvista
    fours = np.ones(len(tetra_face))*4
    tetra_face = np.insert(tetra_face, 0, fours, axis=1)
    mesh = pv.PolyData(pts, tetra_face)

    # pyvista's internal representation is int riangles
    tri_face =  pyvistaToTrimeshFaces(np.array(mesh.faces))

  else: 
    poly = pv.PolyData(pts)
    mesh = poly.delaunay_3d()
    # mesh = poly.delaunay_2d()
    # mesh = mesh.triangulate()
    # print("is all triangles", mesh.is_all_triangles())

    if((type(mesh) is pv.PolyData)):
      cells = mesh.faces
      # faces = mesh.faces.reshape(-1,4)
      # tri_face = faces[:, 1:4]
    else: 
      cells = np.array(mesh.cells)
      
    tri_face = pyvistaToTrimeshFaces(cells)


  # tri_face = np.array(list(get_surface_tris_from_tet(tetra_face)))
  tetra_face = None

  # REWARD. Need the mesh to be in a triangulated PolyData
  if reward_type == 'scaled_jacobian':
    qual = mesh.compute_cell_quality(quality_measure=reward_type)
    
  # print(qual)
  quality = np.array(qual.cell_arrays['CellQuality'])

  # qual.plot()

  
  # exit()
  return pts, tri_face, tetra_face, np.mean(quality)*100

  

def delaunay_triangulation(pts):
  tris = Delaunay(pts)
  tetra_face = tris.simplices
  tri_face = np.array(list(get_surface_tris_from_tet(tetra_face)))
  return tri_face, tetra_face


# from https://stackoverflow.com/questions/66607716/how-to-extract-surface-triangles-from-a-tetrahedral-mesh
def get_surface_tris_from_tet(tetraedrons):
  envelope = set()
  for tet in tetraedrons:
      for face in ( (tet[0], tet[1], tet[2]), 
                    (tet[0], tet[2], tet[3]), 
                    (tet[0], tet[3], tet[2]),
                    (tet[1], tet[3], tet[2]) ):
          # if face has already been encountered, then it's not on the envelope
          # the magic of hashsets makes that check O(1) (eg. extremely fast)
          if face in envelope:    envelope.remove(face)
          # if not encoutered yet, add it flipped
          else:                   envelope.add((face[2], face[1], face[0]))

  # there is now only faces encountered once (or an odd number of times for paradoxical meshes)

  return envelope

# def save_3d_surface_wiremesh(pts, faces, filename, size=(1024,1024)):
#     mlab.options.offscreen = True # render off screen so it doesn't show up.
#     mlab.figure(size=size)
#     # mlab.clf()
#     mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], faces, representation='wireframe', color=(1,1,1))
#     mlab.savefig(filename)
#     # mlab.show()

def draw(pts, tri_face, renderer):
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

  img, _ = renderer.render(scene)
  return img



# TESTING CODE
# input_file = '/home/abrar/thesis/cross_sections_rl/data/cross_section_data/sphere.npz'
# data = np.load(input_file, allow_pickle=True)
# Mhat = data['cross_sections']
# sample_spacing = data['step']
# pts, tri_face, tetra_face = triangulate_list(Mhat, sample_spacing)
# save_3d_surface_wiremesh(pts, tri_face, 'sphere.png')

# renderer = pyrender.OffscreenRenderer(512, 512)
# input_file = '/home/abrar/cross_section_rl/data/cross_section_data/sphere_resampled.npz'
# data = np.load(input_file, allow_pickle=True)
# Mhat = data['cross_sections']
# sample_spacing = data['step']

# weights = np.random.rand((len(Mhat)*100))

# print(Mhat.shape)

# pts, tri_face, tetra_face, reward = triangulate_list_and_reward(Mhat, sample_spacing, weights=weights)
# mesh = trimesh.Trimesh(vertices=pts, faces=tri_face)
# mesh.export(file_obj='{}_{:.4f}.stl'.format('saved/sphere', reward))