import scipy as sp
import numpy as np
import pandas as pd
import os 
import sys
import importlib
import MovieData_methods as MM #Joris' code
import geometry_creation_methods as geometry_methods #Joris' code
#from functions import *
import re
import glob
import warnings
warnings.filterwarnings('ignore')
import os.path
import math
import networkx as nx
import csv
import pickle
import shutil
import random
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from matplotlib import collections  as mc
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from scipy.spatial import Delaunay
#modules for 2D curvature calculations
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from scipy.interpolate import splev
from scipy.interpolate import BSpline
from scipy.interpolate import splrep

################

def update_springs(springs,ball_positions,compute_lo=False):
    springs_ball1s=ball_positions.loc[pd.Series.tolist(springs.ball1)]
    springs_ball1s.columns=['x1','y1','z1']
    springs_ball1s.reset_index(drop=True, inplace=True)
    springs.loc[:,['x1','y1','z1']]=springs_ball1s

    springs_ball2s=ball_positions.loc[pd.Series.tolist(springs.ball2)]
    springs_ball2s.columns=['x2','y2','z2']
    springs_ball2s.reset_index(drop=True, inplace=True)
    springs.loc[:,['x2','y2','z2']]=springs_ball2s

    #change the l1 and dls for the springs
    springs_ball1s.columns=['x','y','z']
    springs_ball2s.columns=['x','y','z']
    disp=[springs_ball2s.x-springs_ball1s.x,springs_ball2s.y-springs_ball1s.y,springs_ball2s.z-springs_ball1s.z]
    length=disp[0]**2+disp[1]**2+disp[2]**2
    length=length.apply(lambda row: math.sqrt(row))
    springs.l1=length
    
    if compute_lo:
        springs.l0=springs.l1
    springs.dl=springs.l1-springs.l0
    
    
    return(springs)

def dftoGraph(balls,springs):
    import networkx as nx
    
    springs_attributes=list(springs.columns)
    G = nx.from_pandas_edgelist(springs, source = 'ball1', target = 'ball2', edge_attr = springs_attributes).to_undirected()
    #add node attributes
    balls_attributes=list(balls.columns)
    for attribute in balls_attributes:
        nx.set_node_attributes(G, pd.Series(list(balls[attribute]), index=balls['ID']).to_dict(), attribute)
        
    return(G)

    
def get_polygons(G=None, balls=None, springs=None, solid = True, stack = None, compute_attributes=True, pattern='trianglemesh', *argv, **kwarg):
    
    if G is None:
        #if G is None then I assume that balls and springs has been sent as argument
        G = dftoGraph(balls,springs)
        
    polygon_size=3
        
    all_cliques= nx.enumerate_all_cliques(G)
    triad_cliques=[x for x in all_cliques if len(x)==polygon_size ]
    
    polygons=pd.DataFrame({
                'vertices':triad_cliques,
                'Nbvertices':[polygon_size]*len(triad_cliques)
                })
    
    #get all node attributes
    node_attributes = list(set([k for n in G.nodes for k in G.nodes[n].keys()]))

    if 'stack' in node_attributes:

        stack_0 = set([n for n,d in G.nodes(data=True) if d['stack']==0])
        stack_1 = set([n for n,d in G.nodes(data=True) if d['stack']==1])
        
        def get_polygon_stack(vertices):
            if vertices.issubset(stack_0):
                return(0)
            elif vertices.issubset(stack_1):
                return(1)
            else:
                return(-99999)
        
        polygons['stack']=polygons.apply(lambda row: get_polygon_stack(set(row['vertices'])), axis=1)
    
        if not solid:
            polygons=polygons.loc[polygons['stack']>=0]
            
        if stack is not None:
            polygons=polygons.loc[polygons['stack']==stack]
        
    if compute_attributes:
        polygons = update_triangles(polygons=polygons, G=G)
        
    return(polygons)



def add_triangle_prop(text_vtk, prop_data, prop_name='prop', first=False):
    if first == True:
        nb_triangles=len(prop_data)
        text_vtk=text_vtk+"CELL_DATA "+str(nb_triangles)+'\n'
        
    text_vtk=text_vtk+"SCALARS "+prop_name+" float 1\nLOOKUP_TABLE default\n"
    text_vtk=text_vtk+'\n'.join([str(v) for v in prop_data])+'\n'
    
    
    return(text_vtk)


def measure_integrated_curvature(balls, springs, triangles = None, filename = None, debug = False, nonboundary_indices = None, write_vtk = True,z_offset = 0):
   
    if triangles is None:
        triangles = get_oriented_triangles(balls, springs)
        
    vertices = np.sort(balls.ID.values)


    #measuring curvature
    #this is Joris Paijman's code
    #vertices, triangles = geometry_methods.load_ply_file(os.path.abspath('test.ply'))
    #vertices, triangles = load_ply_file_ectopic(os.path.abspath('test.ply'))
    
    #getting dataframes that are compatible with Joris' code
    vertices = balls[['x', 'y', 'z']]
    vertices.columns = ['x_pos', 'y_pos', 'z_pos']
    triangles = pd.DataFrame(triangles, columns=['vertex_id_1', 'vertex_id_2', 'vertex_id_3'])
    #reorient triangles so all normals face the same way
    #reduntant now since I have oriented the triangles correctly
    #print('debug : ', triangles['vertex_id_1'].values)
    center = (vertices.sum()/len(vertices)).values
    triangles = geometry_methods.order_face_vertices_by_left_hand_rule(vertices, triangles, center)

    dcel_table = geometry_methods.create_DCEL_from_polygonal_mesh(triangles[['vertex_id_1', 'vertex_id_2', 'vertex_id_3']].values, triangle_mesh=True)

    #Calculate the normal vector on each triangle.
    triangles = MM.calculate_triangle_area_and_unit_normal_vector(vertices, triangles)
    #Calculate the angular defect on each vertex of the mesh.
    vertices = MM.calculate_angular_defect_per_vertex(dcel_table, vertices, triangles)
    #Calculate the angle and integrated mean curvature on each bond of the mesh.
    bond_geometry = MM.calculate_mean_curvature_on_dbonds(dcel_table, vertices, triangles)
    #Calcualte the mean and Gaussian curvature on each triangle of the mesh
    triangles = MM.calculate_triangle_mean_and_gaussian_curvature(dcel_table, vertices, triangles, bond_geometry)
    #Calculate integrated mean curvatures
    #integrated_GC = ((triangles['gaussian_curvature']*triangles['area'])).sum()/(4*np.pi)
    integrated_MC = ((triangles['mean_curvature']*triangles['area'])).sum()
    
    #claculate integrated gaussian curvature
    vertices['in_plane_radius'] = np.sqrt(vertices['x_pos']**2 + vertices['y_pos']**2)
    integrated_GC = vertices.loc[nonboundary_indices, 'angular_defect'].sum()
    
    #Output a vtk file with curvature values stored
    #vtk file should not have lines in it
    if write_vtk:
        #with open(filename, 'r') as myfile:
        #    text_vtk = myfile.read()
    
        balls['z'] = balls['z'] + z_offset
        
        text_vtk = dfToVtk(balls,springs,filename = filename, lines = False, add_polygons=True, return_text=True)
        mod_text_vtk=text_vtk
        mod_text_vtk=add_triangle_prop(mod_text_vtk, list(triangles['gaussian_curvature']), prop_name="gaussian_curvature", first=True)
        mod_text_vtk=add_triangle_prop(mod_text_vtk, list(triangles['mean_curvature']), prop_name="mean_curvature")
        #save vtk modified file
        with open(filename, "w") as file:
                file.write(mod_text_vtk)

    return([integrated_GC, integrated_MC, triangles, vertices])



def dfToVtk(balls, springs, only_top_surface = False, only_bottom_surface = False,
            filename='trianglemesh.vtk', pattern='trianglemesh',
            lines=True,add_polygons=True, add_lines_properties = False, add_polygon_properties = False,
            return_text = False, **kwargs):
    
    #if add_lines_properties and add_polygons both are true then Paraview does not show the properties
    #we give preference to showing lines properties hence we put add_polygons False if add_lines_properties is True
    if add_lines_properties:
        add_polygons = False
    if add_polygon_properties:
        lines = False
    
    if only_top_surface:
        #balls_orig = balls.copy(deep = True)
        #springs_orig = springs.copy(deep = True)
        springs = springs[(springs['ball1'] >= len(balls)/2) & (springs['ball2'] >= len(balls)/2)]
        balls = balls[balls['ID'] >= len(balls)/2]
        
    if only_bottom_surface:
        #balls_orig = balls.copy(deep = True)
        #springs_orig = springs.copy(deep = True)
        springs = springs[(springs['ball1'] < len(balls)/2) & (springs['ball2'] < len(balls)/2)]
        balls = balls[balls['ID'] < len(balls)/2]
        
    
    #Removing extra small values because they give errors
    balls.loc[abs(balls.x)<1e-10,'x']=0
    balls.loc[abs(balls.y)<1e-10,'y']=0
    balls.loc[abs(balls.z)<1e-10,'z']=0

    if 'ID' not in list(balls.columns): #fixes some contradiction on calling get_polygons
        balls['ID'] = list(balls.index)
    
    #fixing indexing issues
    #map id to index
    keys=list(balls.index)
    values=list(range(len(balls)))
    map_dict=dict(zip(keys,values))
    
    balls=balls.reset_index(drop=True)
    springs=springs.reset_index(drop=True)
    springs['ball1']=springs.apply(lambda row: map_dict[row['ball1']],axis=1)
    springs['ball2']=springs.apply(lambda row: map_dict[row['ball2']],axis=1)
    
    
    ##########
    # header #
    ##########
    
    text='# vtk DataFile Version 1.0\nTissue data\nASCII\n\nDATASET POLYDATA\nPOINTS '
    
    ##########
    # points #
    ##########
    text=text+str(len(balls))+' float\n'
    for i in range(len(balls)):
        #you can sort the dataframe by the ID however for now the ID is i
        text=text+str(balls.loc[i,'x'])+' '+str(balls.loc[i,'y'])+' '+str(balls.loc[i,'z'])+'\n'
    text=text+'\n'
        
    ############
    # polygons #
    ############

    if add_polygons:

        if 'polygons' in kwargs.keys():
            #you can explicitly define polygons
            polygons = kwargs['polygons']
        elif 'graph' in kwargs.keys():
            G = kwargs['graph']
            tri=nx.triangles(G)
            all_cliques= nx.enumerate_all_cliques(G)
            triad_cliques=[x for x in all_cliques if len(x)==3 ]

            #preparing the polygons dataframe
            polygons=pd.DataFrame({
                    'vertices':triad_cliques,
                    'Nbvertices':[3]*len(triad_cliques)
                    })
        else:
            polygons=get_polygons(balls=balls[['ID','x','y','z']], springs=springs,compute_attributes = False)

        text=text+'POLYGONS '+str(len(polygons))+' '+str(len(polygons)+polygons['Nbvertices'].sum())+'\n'
        for i in range(len(polygons)):
            text=text+str(polygons.loc[i,'Nbvertices'])
            ar=polygons.loc[i,'vertices']
            for x in ar:
                text=text+' '+str(x)
            text=text+'\n'
        text=text+'\n'
    
    #########
    # lines #
    #########
    
    if lines:
        text=text+'LINES '+str(len(springs))+' '+str(len(springs)*3)+'\n'
        for i in range(len(springs)):
            text=text+str(2)+' '+str(springs.loc[i,'ball1'])+' '+str(springs.loc[i,'ball2'])+'\n'
            #we can also get the lines from the edges column of the polygons
        text=text+'\n'
        
        
    ####################
    # lines properties #
    ####################
    
    if add_lines_properties:
    
        first = True
        data_length = len(springs)
        col_names = springs.columns
        props_to_avoid = ['ID', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'k','ball1', 'ball2', 'type', 'viscoelastic_coeff']

        for col_name in col_names:

            if col_name in props_to_avoid:
                continue

            #naming property
            prop_name = col_name
            #prop_name = prop_name.replace('l0', 'PreferredLength')
            prop_name = prop_name.replace('l1', 'l')

            #getting array of values
            prop_data = springs[col_name].values

            if first == True:
                text=text+"\nCELL_DATA "+str(data_length)+'\n'
                first = False

            text=text+"SCALARS "+prop_name+" float 1\nLOOKUP_TABLE default\n"
            text=text+'\n'.join([str(v) for v in prop_data])+'\n'
    
    ###############
    # Saving file #
    ###############
    
    with open(filename, "w") as file:
        file.write(text)
        
    ##########
    # Return #
    ##########

    if return_text:
        return(text)

    
def get_oriented_triangles(balls, springs, debug = False, returnType = "ndArray"):
    
    vertices = np.sort(balls.ID.values)

    triangles = np.ndarray(shape=(0,3), dtype=int)
    
    polygons = get_polygons(balls = balls[['ID','x','y','z']], springs=springs, compute_attributes=False)
    
    for index, row in polygons.iterrows():
        ar = row['vertices']
        [v1,v2,v3] = ar
        
        e12 = [balls.x[v1] - balls.x[v2], balls.y[v1] - balls.y[v2], 0]
        e23 = [balls.x[v2] - balls.x[v3], balls.y[v2] - balls.y[v3], 0]
        #cross product of e12 X e23
        area_normal = np.cross(e12, e23)[2]
        #we want all area_normal to be >0
        if area_normal<0:
            vtemp = v2
            v2 = v3
            v3 = vtemp

        triangle = [v1, v2, v3]

        polygons.at[index, "vertices"] = [v1, v2, v3]

        #add vertices to triangles
        triangles = np.vstack((triangles,triangle))
        
    if returnType == "ndArray":
        return(triangles)
    else:
        return(polygons)

    
    
def reindex_balls_springs(balls, springs, reindex_neighbours = False):
    
    keys=list(balls.ID)
    values=list(range(len(balls)))
    map_dict=dict(zip(keys,values))
    
    balls=balls.reset_index(drop=True)
    balls['ID'] = balls.apply(lambda row: map_dict[row['ID']], axis = 1)
    
    springs=springs.reset_index(drop=True)
    springs['ball1']=springs.apply(lambda row: map_dict[row['ball1']],axis=1)
    springs['ball2']=springs.apply(lambda row: map_dict[row['ball2']],axis=1)
    
    
    #reindexing neighbours
    if reindex_neighbours:
        for index, row in balls.iterrows():
            #print(index, ': ', row['neighbours'])
            #if len(row['neighbours']) == 0:
            #    print(index, ': ', row['neighbours'])
            ar = []
            for x in row['neighbours']:
                if not(x in map_dict.keys()):
                    continue
                ar.append(map_dict[x])

            balls['neighbours'][index] = ar
        
    return([balls, springs])

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(hxy, z)
    az = np.arctan2(y, x)
    return r, el, az

    
