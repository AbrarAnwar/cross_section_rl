import numpy as np
from numpy.linalg import norm, det
from math import acos, tan
import itertools

import pyvista as pv
from pyvista import examples
import trimesh

# import utils
from utils import WeightedDelaunay, get_surface_tris_from_tet
######## ROUTINES ##########
def cot1(x, y):
    return 1.0/tan(acos(np.dot(x, y)/(norm(x)*norm(y))))

def area_vec(x, y):
    return norm(np.cross(x, y))/2

def check_collinear(x, y):
    return abs(abs(np.dot(x, y)) - norm(x)*norm(y)) < 1e-9

def intersection(p1, d1, p2, d2):
    p3 = p2 - p1
    ratio = norm(np.cross(p3, d2))/norm(np.cross(d1, d2))
    c1 = p1 + ratio*d1
    c2 = p1 - ratio*d1
    #print(check_collinear(c1-p2, d2) or check_collinear(c2-p2, d2))
    if check_collinear(c1-p2, d2): 
        return c1
    
    return c2

###################### COMPUTE HOT_2, 2 functionals of a triangle #############################
# v: list of vertices. v[i] is 3D-vector coordinate of ith vertex
# w: list of weights. w[i] is the weight of ith vertex 
# degree: a combination of weights indicating how one sums HOT functionals of different degrees.
# For example, degree = [0.5, 0, 0.5] means we take 0.5 * 0-HOT_2,2 + 0.5 * 2-HOT_2, 2
###############################################################################################

def compute_HOT_tri(v, w, degree=[1, 0, 0]):
    # area: area of the triangle
    area = area_vec(v[0]-v[1], v[0]-v[2])
    ind = np.array([0, 1, 2], dtype=int)
    d = np.zeros((3, 3)) # d_ij
    e = np.zeros((3, 3)) # e_ij
    h = np.zeros(3) # h_k
    # Compute e_ij and d_ij
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            e[i, j] = norm(v[j]-v[i])
            d[i, j] = (e[i, j]**2 + w[i]-w[j])/(2*e[i, j])
    # Compute tangent of beta[k]
    beta = np.zeros(3)
    for k in range(3):
        v1 = np.delete(v, k, axis=0)
        beta[i] = cot1(v[k]-v1[0], v[k]-v1[1])
    # Compute h_k
    for k in range(3):
        ind1 = np.delete(ind, k)
        i, j = ind1[0], ind1[1]
        h[k] = beta[k]*e[i, j]/2 + (beta[i]*w[j] + beta[j]*w[i])/(2*e[i, j]) \
                - w[k]*e[i, j]/(4*area)
    # Go over all permuations and compute H_{2, 2} on a single triangle
    perms = list(itertools.permutations([0, 1, 2]))
    ret = 0
    # Default: (degree = [1, 0, 0])
    factor = 1/(np.array([[4, 12], [6, 6], [12, 4]]).T)
    coef = np.matmul(factor, degree)
    for p in perms:
        ret += (d[p[0], p[1]]**3)*h[p[2]]*coef[0] + d[p[0],p[1]]*(h[p[2]]**3)*coef[1]
    return ret

def circumcenter_line(v, w):
    ind1 = np.array([0, 1, 2], dtype=int)
    cij = np.zeros((3, 3))
    for k in range(3):
        ind = np.delete(ind1, k)
        i, j = ind[0], ind[1]
        cij[k] = v[i] + 0.5*(norm(v[i]-v[j])**2 + w[i]-w[j])*(v[j]-v[i])/(norm(v[i]-v[j])**2)
    return cij

def circumcenter_tri(v, w):
    normal_v = np.cross(v[0]-v[1], v[0]-v[2])
    # First compute circumcenters of segment or 1-simplex. Hence the name c1
    c1 = circumcenter_line(v, w)
    # Compute circumcenter of triangle or 2-simplex by orthogonal lines
    d = np.zeros((2, 3)) # two orthgonal directions
    ind1 = np.array([0, 1, 2], dtype=int)
    for k in range(2):
        ind = np.delete(ind1, k)
        d[k] = np.cross(v[ind[0]]-v[ind[1]], normal_v)

    return intersection(c1[0], d[0], c1[1], d[1])
       
def circumcenter_tetra(v, w):
    # circumcenters of triangle or 2-simplex. Hence the name c2
    c2 = np.zeros((4, 3))
    for k in range(4):
        v1 = np.delete(v, k, axis=0)
        w1 = np.delete(w, k)
        # If the tetrahedron is ijlk then this c2[k] is weighted circumcenter of ijl
        c2[k] = circumcenter_tri(v1, w1)
    # Again compute circumcenter of tetrahedron or 3-simplex by orthogonal lines
    normal_ijl = np.zeros((2, 3)) # orthogonal directions
    ind1 = np.array([0, 1, 2, 3], dtype=int)
    for l in range(2):
        ind = np.delete(ind1, l)
        i, j, k = ind[0], ind[1], ind[2]
        normal_ijl[l] = np.cross(v[i]-v[j], v[i]-v[k])

    return c2, intersection(c2[0], normal_ijl[0], c2[1], normal_ijl[1])

def signed_dist_c23(v, w):
    H = np.zeros(4)
    c2, c3 = circumcenter_tetra(v, w)
    for k in range(4):
        v1 = np.delete(v, k, axis=0)
        H[k] = norm(c2[k]-c3)
        s1 = np.concatenate((v1[1:], [c3]), axis=0) - v1[0]
        s2 = np.concatenate((v1[1:], [v[k]]), axis=0) - v1[0]
        if det(s1) != det(s2):
            H[k] *= -1
    return H 

###################### COMPUTE HOT_2, 2 functionals of a tetrahedron ##########################
# v: list of vertices. v[i] is 3D-vector coordinate of ith vertex
# w: list of weights. w[i] is the weight of ith vertex 
# degree: a combination of weights indicating how one sums HOT functionals of different degrees.
# For example, degree = [0.5, 0, 0, 0.5] means we take 0.5 * 0-HOT_2,2 + 0.5 * 3-HOT_2, 2
###############################################################################################

def compute_HOT_tetra(v, w, degree=[1, 0, 0, 0]):
    ind = np.array([0, 1, 2, 3], dtype=int)
    # area: list of area of 2-simplices or triangles of tetrahedron. area[l] = area of lth triangle
    area = np.zeros(4)
    d = np.zeros((4, 4)) # d_ij
    e = np.zeros((4, 4)) # e_ij
    h = np.zeros((4, 4)) # h(l, k) is the h_k in triangle/face ijk with the opposite vertex l
    # Compute e_ij and d_ij
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            e[i, j] = norm(v[j]-v[i])
            d[i, j] = (e[i, j]**2 + w[i]-w[j])/(2*e[i, j])
    # Compute tangent of beta[k]
    beta = np.zeros((4, 4))
    ind = np.array([0, 1, 2, 3], dtype=int)
    # Compute h(l, k)
    for l in range(4):
        v1 = np.delete(v, l, axis=0); ind1 = np.delete(ind, l)
        area[l] = area_vec(v1[0]-v1[1], v1[0]-v1[2])
        for k in ind1:
            v2 = np.delete(v, [k, l], axis=0)
            beta[l, k] = cot1(v[k]-v2[0], v[k]-v2[1])
        
        for k in ind1:
            ind2 = np.delete(ind, [k, l])
            i, j = ind2[0], ind2[1]
            h[l, k] = beta[l, k]*e[i, j]/2 + (beta[l, i]*w[j] \
                    + beta[l, j]*w[i])/(2*e[i, j]) - w[k]*e[i, j]/(4*area[l])
    # Compute H(l)
    H = signed_dist_c23(v, w)
    # Go over all permuations and compute H_{2, 2} on a single triangle
    perms = list(itertools.permutations([0, 1, 2, 3])); ret = 0
    # Default: (degree = [1, 0, 0, 0]-only 0-HOT_2,2 is computed on this tetrahedron)
    factor = 1/(np.array([[12*5, 4*5, 2*5], [12*3, 4*3, 6*3], [6*3, 4*3, 12*3], [2*5, 4*5, 12*5]]).T)
    coef = np.matmul(factor, degree)
    for p in perms:
        l, k, i, j = p[0], p[1], p[2], p[3]
        ret += (H[l]**3)*h[l, k]*d[i, j]*coef[0] + H[l]*(h[l, k]**3)*d[i, j]*coef[1] \
            + H[l]*h[l, k]*(d[i, j]**3)*coef[2]
    return ret

##################################################################################################
########## HOT functional for (Delaunay triangulation) from set of points and weights ############
##################################################################################################
# vertices: set of vertex points
# weights: set of corresponding weights
# degree2 = weight combination for HOT functional on 2-simplices (triangle)
# degree3 = weight combination for HOT functional on 3-simplices (tetrahedron)

def compute_HOT(vertices, weights, degree2=[0, 1, 2], degree3=[1, 0, 0, 0]):
    tetra_faces = WeightedDelaunay(vertices, weights)
    tri_faces = get_surface_tris_from_tet(tetra_faces)
    HOT = 0
    # Go over each triangle and tetrahedron and sum up all HOT functionals (with correponding degrees)
    for tri in tri_faces:
        v = np.array([vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]])
        w = np.array([weights[tri[0]], weights[tri[1]], weights[tri[2]]])
        HOT += compute_HOT_tri(v, w, degree2)
    for tet in tetra_faces:
        v = np.array([vertices[tet[0]], vertices[tet[1]], vertices[tet[2]], vertices[tet[3]]])
        w = np.array([weights[tet[0]], weights[tet[1]], weights[tet[2]], weights[tet[3]]])
        HOT += compute_HOT_tetra(v, w, degree3)
        
    return HOT

def compute_HOT_with_tetra(vertices, weights, tetra_faces, degree3=[1, 0, 0, 0]):
    HOT = 0
    print(vertices.shape)
    print(weights.shape)
    print(tetra_faces.shape)
    for tet in tetra_faces:
        v = np.array([vertices[tet[0]], vertices[tet[1]], vertices[tet[2]], vertices[tet[3]]])
        w = np.array([weights[tet[0]], weights[tet[1]], weights[tet[2]], weights[tet[3]]])
        HOT += compute_HOT_tetra(v, w, degree3)
        
    return HOT

def compute_HOT_with_tri(vertices, weights, tri_faces, degree2=[0, 1, 2]):
    HOT = 0

    # v = np.array([vertices[tri_faces[:,0]], vertices[tri_faces[:,1]], vertices[tri_faces[:,2]] ])
    # w = np.array([weights[tri_faces[:,0]], weights[tri_faces[:,1]], weights[tri_faces[:,2]] ])

    # for i in range(len(tri_faces)):
    #     HOT += compute_HOT_tri(v[:, i], w[:, i], degree2)

    for tri in tri_faces:
        v = np.array([vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]])
        w = np.array([weights[tri[0]], weights[tri[1]], weights[tri[2]]])
        HOT += compute_HOT_tri(v, w, degree2)
        
    return HOT
    
# EXAMPLES of using compute_HOT
vertices = np.array([[0, 0, 0], [1, 0, 0], [.5, 2, 0], [.5, .5, 1]], dtype=float)



# num_points = len(vertices)
# w = .5*np.random.rand(num_points)
# # w = np.zeros(num_points)
# verts_no_zs = vertices[:, :-1]
# print(verts_no_zs.shape)

# # remove last dimension temporarily

# tetra_faces = WeightedDelaunay(vertices, w)
# print(tetra_faces)
# tri_faces = get_surface_tris_from_tet(tetra_faces)

# for tri in tri_faces:
#     v = np.array([vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]])
#     print('v[tri[]] shape', v)
# tri_faces = np.array(list(tri_faces))

# print('new')

# print(tri_faces)

# compute_HOT_with_tri(vertices, w, tri_faces)

# v = np.array([vertices[tri_faces[:,0]], vertices[tri_faces[:,1]], vertices[tri_faces[:,2]] ])
# weights = np.array([w[tri_faces[:,0]], w[tri_faces[:,1]], w[tri_faces[:,2]] ])

# compute_HOT_tri(v, weights, [1,0,0])

# mesh = trimesh.Trimesh(vertices, tetra_faces)
# mesh.export(file_obj='test.stl')

# degree3=[1, 0, 0, 0]
# HOT = compute_HOT(vertices, w, degree3=degree3)
# print(f'The (combined) hot functional is {HOT}')

'''
UNIT_TESTS:
# Test circumcenter_tetra
v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]], dtype=float)
w = np.zeros(4)
print(circumcenter_tetra(v, w))
# Test intersection
p1 = np.array([0, 1, 2], dtype=float)
p2 = np.array([0, 1, 4], dtype=float)
d1 = np.array([0, 3, 2], dtype=float)
d2 = np.array([0, 22, 2], dtype=float)
print(intersection(p1, d1, p2, d2))
'''

''' Unweighted circumcenters
def circumcenter_tetra(a, b, c, d):   
    return a + (norm(d-a)**2) * np.cross(b-a, c-a) \
            + (norm(c-a)**2) * np.cross(d-a, b-a) \
            + (norm(b-a)**2) * np.cross(c-a, d-a)
            
def circumcenter_tri(a, b, c):
    return a + ((norm(c-a)**2) * np.cross(np.cross(b-a, c-a), b-a) \
        + (norm(b-a)**2) * np.cross(c-a, np.cross(b-a, c-a)))\
            /(2*norm(np.cross(b-a, c-a))**2)
'''
