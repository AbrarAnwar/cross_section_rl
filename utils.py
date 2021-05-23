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

REWARD_TYPES = ['scaled_jacobian', 'max_angle', 'shape', 'shape_and_size', 'shear', 'skew', 'stretch', 'volume', 'radius_ratio', 'aspect_ratio', 'area', 'distortion']

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



def pyvistaToTrimeshFaces(cells):
    faces = []
    idx = 0
    while idx < len(cells):
      curr_cell_count = cells[idx]
      curr_faces = cells[idx+1:idx+curr_cell_count+1]
      faces.append(curr_faces)
      idx += curr_cell_count+1
    return np.array(faces)


import utils2

def main():
  fs = ['/home/abrar/cross_section_rl/data/cross_section_data/sphere_10verts_10sections.npz']
  fs.append('/home/abrar/cross_section_rl/data/cross_section_data/12_sided_pyramid_short_10verts_10sections.npz')
  fs.append('/home/abrar/cross_section_rl/data/cross_section_data/Twisted_Vase_Basic_10verts_10sections.npz')

  f_idx = 2
  data = np.load(fs[f_idx], allow_pickle=True)
  M = data['cross_sections']
  sample_spacing = data['step']


  renderer = pyrender.OffscreenRenderer(512, 512)
  pts, tri_face, tetra_face, reward = triangulate_list_and_reward(M, sample_spacing, weights=None, reward_type='scaled_jacobian')
  mesh = trimesh.Trimesh(pts, tri_face)
  mesh.export('test.stl')


# main difference here is it uses pyvista
def triangulate_list_and_reward(subsequence, step, reward_type='scaled_jacobian', weights=None, twodWDT=False):
  # stack by heights
  tetra_face = None

  pts = np.empty(shape = (0,3))
  wts = np.empty(shape = (0,1))
  if weights is not None:
    for i, (m,w) in enumerate(zip(subsequence, weights)):
      col_to_add = np.ones(len(m))*i*step
      res = np.hstack([m, np.atleast_2d(col_to_add).T])
      pts = np.concatenate([pts, res])
      wts = np.append(wts, np.array(w))
    weights = wts

  else: 
    for i, m in enumerate(subsequence):
      col_to_add = np.ones(len(m))*i*step
      res = np.hstack([m, np.atleast_2d(col_to_add).T])
      pts = np.concatenate([pts, res])
  # use pyvista to turn this into a PolyData
  
  # print(poly)

  if(weights is not None):
    if not twodWDT:
      tetra_face = np.array(WeightedDelaunay(pts, weights))
      # make it compatible with pyvista
      fours = np.ones(len(tetra_face))*4
      tetra_face = np.insert(tetra_face, 0, fours, axis=1)
      mesh = pv.PolyData(pts, tetra_face)
      tetra_face =  pyvistaToTrimeshFaces(np.array(mesh.faces))
      tri_face = np.array(list(get_surface_tris_from_tet(tetra_face)))
    else:
      # 2d delaunay triangulation then lift back up
      zs = pts[:, 2]
      pts2d = pts[:, :2]
      tri_face = np.array(WeightedDelaunay(pts2d, weights))
      # now lift it triangulates the 3D surface
      tetra_face = None



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
      
    if(cells[0] == 4):
      tetra_face = pyvistaToTrimeshFaces(cells)
      tri_face = np.array(list(get_surface_tris_from_tet(tetra_face)))
    else:
      tri_face = pyvistaToTrimeshFaces(cells)

  # surf = mesh.extract_surface().triangulate()
  # surf.plot()
  # print(surf.faces)
  # tri_face = tetra_face
  # tetra_face = None
  # print(tri_face)

  # tri_face = pyvistaToTrimeshFaces(surf.faces)
  # tri_face = tetra_face


  reward = pyvista_to_reward(pts, tri_face, face_size=3, reward_type=reward_type, weights=weights)

  return pts, tri_face, tetra_face, reward

def record_pyvista_rewards(pts, pyvista_face, step=None, face_size=3):
  idxs = np.ones(len(pyvista_face))*face_size
  face = np.insert(pyvista_face, 0, idxs, axis=1)
  mesh = pv.PolyData(pts, face)

  rewards_dict = {}
  for reward_type in REWARD_TYPES:
    qual = mesh.compute_cell_quality(quality_measure=reward_type)
    qual_arr = np.array(qual.cell_arrays['CellQuality'])
    quality = np.mean(qual_arr)
    rewards_dict[reward_type] = quality
  # calculate number of degenerate triangles
  qual = mesh.compute_cell_quality(quality_measure=reward_type)
  qual_arr = np.array(qual.cell_arrays['CellQuality'])  
  qual = np.sum(qual_arr <= 0)
  rewards_dict['scaled_jacobian_degen_tri'] = str(int(qual)) + "/" + str(len(qual_arr))
  return rewards_dict

def pyvista_to_reward(pts, pyvista_face, weights=None, step=None, reward_type='scaled_jacobian', face_size=3):
  # print('in reward')
  if reward_type in REWARD_TYPES:
    # this means that we need to become a pyvista mesh to get the quality
    idxs = np.ones(len(pyvista_face))*face_size
    face = np.insert(pyvista_face, 0, idxs, axis=1)
    mesh = pv.PolyData(pts, face)
    qual = mesh.compute_cell_quality(quality_measure=reward_type)
    quality = np.array(qual.cell_arrays['CellQuality'])
    return np.mean(quality)*100


  if reward_type == "HOT":
    assert weights is not None

    wts = np.empty(shape = (0,1))
    for i, w in enumerate(weights):
      wts = np.append(wts, np.array(w))

    if(face_size == 4):
      HOT = utils2.compute_HOT_with_tetra(pts, wts, pyvista_face, degree3=[1,0,0,0])
    if(face_size == 3):
      HOT = utils2.compute_HOT_with_tri(pts, wts, pyvista_face, degree2=[1,0,0])
    return HOT
    # return HOT/len(pyvista_face)
    


def delaunay_triangulation(pts):
  tris = Delaunay(pts)
  tetra_face = tris.simplices
  tri_face = np.array(list(get_surface_tris_from_tet(tetra_face)))
  return tri_face, tetra_face




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
  light = pyrender.PointLight(color=[1,1,1], intensity=2e3)

  scene.add(mesh, pose=  np.eye(4))
  cam_pose = [            [ 0,  0,  1,  2],
                          [ 1,  0,  0,  .5],
                          [ 0,  1,  0,  .5],
                          [ 0,  0,  0,  1]
                          ]
  scene.add(light, pose=  cam_pose)

  scene.add(camera, pose=[
                          [ 0,  0,  1,  2],
                          [ 1,  0,  0,  .5],
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
if __name__ == "__main__":
    main()