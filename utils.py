import numpy as np
from scipy.spatial import Delaunay
# import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pyvista as pv

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

# main difference here is it uses pyvista
def triangulate_list_and_reward(subsequence, step, reward_type='scaled_jacobian'):
  # stack by heights
  pts = np.empty(shape = (0,3))
  for i,m in enumerate(subsequence):
    col_to_add = np.ones(len(m))*i*step
    res = np.hstack([m, np.atleast_2d(col_to_add).T])
    pts = np.concatenate([pts, res])

  # use pyvista to turn this into a PolyData
  poly = pv.PolyData(pts)
  # print(poly)


  mesh = poly.delaunay_2d()
  # print(mesh.faces.reshape(-1,4).shape)

  if reward_type == 'scaled_jacobian':
    qual = mesh.compute_cell_quality(quality_measure='scaled_jacobian')
  # print(qual)
  quality = np.array(qual.cell_arrays['CellQuality'])

  tetra_face = mesh.faces.reshape(-1,4)

  tri_face = np.array(list(get_surface_tris_from_tet(tetra_face)))
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

def bezier_interpolate_list(seq):
  # get largest number of verts in one
  longest = max(len(l) for l in seq)

  new_cross_sections = []

  # put them all into a single point list separated by step distance away
  for i,m in enumerate(subsequence):
    col_to_add = np.ones(len(m))*i*step
    res = np.hstack([m, np.atleast_2d(col_to_add).T])
    pts = np.concatenate([pts, res])



# TESTING CODE
# input_file = '/home/abrar/thesis/cross_sections_rl/data/cross_section_data/sphere.npz'
# data = np.load(input_file, allow_pickle=True)
# Mhat = data['cross_sections']
# sample_spacing = data['step']
# pts, tri_face, tetra_face = triangulate_list(Mhat, sample_spacing)
# save_3d_surface_wiremesh(pts, tri_face, 'sphere.png')
