import os
import numpy as np
import numpy.linalg as la
import itertools
import matplotlib.pyplot as plt

import trimesh
from shapely.geometry import LineString


def simplify_spline(path, smooth=None, verbose=False):
    """
    Replace discrete curves with b-spline or Arc and
    return the result as a new Path2D object.
    Parameters
    ------------
    path : trimesh.path.Path2D
      Input geometry
    smooth : float
      Distance to smooth
    Returns
    ------------
    simplified : Path2D
      Consists of Arc and BSpline entities
    """

    new_vertices = []
    new_entities = []
    scale = path.scale

    for discrete in path.discrete:
        # entities for this path
        entity, vertices = trimesh.path.simplify.points_to_spline_entity(discrete, smooth=smooth)
        # reindex returned control points
        entity.points += len(new_vertices)
        # save entity and vertices
        new_vertices.extend(vertices)
        new_entities.append(entity)

    # create the Path2D object for the result
    simplified = type(path)(entities=new_entities,
                            vertices=new_vertices)

    return simplified

if __name__ == '__main__':

    # file_name = 'Twisted_Vase_Basic.stl'
    file_name = 'sphere.ply'
    # file_name = '12_sided_pyramid_short.stl'

    example_fname = os.path.join('data/shape_data', file_name)
    mesh = trimesh.load_mesh(example_fname)

    num_sections = 10
    num_verts = 10

    verts = mesh.vertices

    max_v = np.max(verts[:,1])
    min_v = np.min(verts[:,1])

    # normalize between 0 and 1
    mesh.vertices = (mesh.vertices-min_v)/(max_v-min_v)


    verts = mesh.vertices



    # get new mesh vertices
    max_v = np.max(verts[:,2]) - 0.05
    min_v = np.min(verts[:,2]) + 0.05



    z_levels, step_size = np.linspace(min_v, max_v, num=num_sections, retstep=True)


    sections = mesh.section_multiplane(plane_origin=(0,0,0), 
                                   plane_normal=[0,0,1], 
                                   heights=z_levels)


    print(sections)
    for x in sections:
      # if x == None:
      #   continue
      x = x.vertices
      plt.scatter(x[:,0], x[:,1])
    plt.show()
    simplified = []
    for x in sections:
        # x = x.simplify_spline(smooth=.0001)
        x = simplify_spline(x,smooth=.01)
        simplified.append(x)
    sections = simplified

    cross_sections = []
    for x in sections:
        # x.plot_entities()
        # print(x.entities[0].points.shape, x.entities[0].points)
        print(x.entities)
        test = np.array(x.entities[0].discrete(x.vertices, count=num_verts + 1))
        test = test[:-1]
        cross_sections.append(test)
        # print(x.vertices.shape, test.shape)
        # plt.plot(*test.T)




    
    # x = trimesh.path.simplify.resample_spline(sections[0].vertices, count = 150)

    # print('single shape', x.shape, type(x))
    # cross_sections = [np.array(sect.vertices) for sect in sections]
    # cross_sections = [trimesh.path.simplify.resample_spline(sect.vertices, count = 150, degree=1) for sect in sections]

    # SIMPLE, and only works for cases where there's one topology I think

    cross_sections = np.array(cross_sections)
    print(cross_sections.shape)
    print(cross_sections[0].shape)

    for x in cross_sections:
        plt.scatter(x[:,0], x[:,1])
        # break
    plt.show()
    name = file_name[:-4]
    out_file = os.path.join(os.path.realpath('.'), 'data/cross_section_data', '{}_{}verts_{}sections'.format(name, num_verts, num_sections))
    print(cross_sections)
    np.savez(out_file, cross_sections=cross_sections, step=step_size, start=min_v, end=max_v)
    print('saved')

