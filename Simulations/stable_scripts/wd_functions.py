#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:40:53 2021

@author: krishna
"""

from functions import *
import glob
import re
import os
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from matplotlib import collections  as mc
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
#modules for 2D curvature calculations
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from scipy.interpolate import splev
from scipy.interpolate import BSpline
from scipy.interpolate import splrep

import meshzoo
import optimesh

import warnings
warnings.filterwarnings('ignore')

def get_meshzoo_icosa_sphere(refine_factor = 20):
    
    #points, cells = meshzoo.tetra_sphere(20)
    points, cells = meshzoo.icosa_sphere(refine_factor)
    class Sphere:
        def f(self, x):
            return 1.0 - (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        def grad(self, x):
            return -2 * x
    # You can use all methods in optimesh:
    points, cells = optimesh.optimize_points_cells(
        points,
        cells,
        "CVT (full)",
        1.0e-2,
        100,
        verbose=False,
        implicit_surface=Sphere(),
        # step_filename_format="out{:03d}.vtk"
    )
    

    a1 = cells[:,[0,1]]
    a2 = cells[:,[1,2]]
    a3 = cells[:,[2,0]]
    redundant_edges_df = pd.DataFrame(np.vstack([a1,a2,a3]), columns=["v1", "v2"])
    G = nx.from_pandas_edgelist(redundant_edges_df, source = 'v1', target = 'v2').to_undirected()
    edges_df = nx.to_pandas_edgelist(G, source='v1', target='v2')
    
    cells_df = pd.DataFrame(cells, columns=["V0", "V1", "V2"])
    balls_df = pd.DataFrame(points, columns=["x", "y", "z"])

    return([balls_df,edges_df,cells_df])

def add_thickness_to_mesh(v_df=None, edges_df=None, cells_df = None, mesh_file = 'gmsh_sphere_lattice_r_20.pickle',theta_max = None,R = 1, nstack = 2,
             thickness = 0.5, thickness_polynomial_obj = None,theta_ref = None,
             spring_const = 1, viscoelastic_coeff = 1,
             output_pickle = 'mesh_thick.pickle', output_vtk = 'foo_thick.vtk',
             crop_mesh_bool = False, rotate_mesh_bool = False, angle_of_rotation = 0.1
            ):
    
    if edges_df is None:
        #if edges_df is None then edges need to be extracted from cells_df
        cells = cells_df.values
        a1 = cells[:,[0,1]]
        a2 = cells[:,[1,2]]
        a3 = cells[:,[2,0]]
        redundant_edges_df = pd.DataFrame(np.vstack([a1,a2,a3]), columns=["v1", "v2"])
        G = nx.from_pandas_edgelist(redundant_edges_df, source = 'v1', target = 'v2').to_undirected()
        edges_df = nx.to_pandas_edgelist(G, source='v1', target='v2')
        

    if thickness_polynomial_obj is None:
        thickness_polynomial_obj = np.poly1d([thickness])

    balls_colnames=['ID', 'x', 'y', 'z', 'neighbours', 'spring1', 'spring2','stack']
    balls=pd.DataFrame(0, index=range(0), columns=range(len(balls_colnames)))
    balls.columns=balls_colnames
    balls.index = range(balls.shape[0])

    #dataframe of springs
    #ID, X1, Y1, Z1, X2, Y2, Z2, k, Natural length, Extension in length, Ball1, Ball2
    springs_colnames=['ID', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2',
                      'k', 'l0', 'l1','dl',
                      'ball1', 'ball2','type','viscoelastic_coeff']
    springs=pd.DataFrame(0, index=range(0), columns=range(len(springs_colnames)))
    springs.columns=springs_colnames

    nballs_per_stack = len(v_df)

    ###########
    # Rotating vertices
    ###########
    
    if rotate_mesh_bool:

        from numpy import cross, eye, dot
        from scipy.linalg import expm, norm

        def M(axis, theta):
            return expm(cross(eye(3), axis/norm(axis)*theta))

        axis, theta = [4,4,1], angle_of_rotation
        M0 = M(axis, theta)

        for index, row in v_df.iterrows():
            v = [row['x'],row['y'],row['z']]
            #print(v)
            v_ = dot(M0,v)
            v_df.loc[index, ['x', 'y', 'z']] = v_

    ############
    # Cropping mesh above a threshold
    ###########
    
    v_df['r'] = np.sqrt(v_df['x']**2 + v_df['y']**2 + v_df['z']**2)
    v_df['theta'] = np.arccos(v_df['z']/v_df['r'])
    
    if crop_mesh_bool:

        v_df['r'] = np.sqrt(v_df['x']**2 + v_df['y']**2 + v_df['z']**2)
        v_df['theta'] = np.arccos(v_df['z']/v_df['r'])

        #
        if theta_max is None:
            theta_max = np.pi/4
        if theta_ref is None:
            theta_ref = theta_max
        v_df = v_df[v_df['theta']<=theta_max]

        v = v_df[['x', 'y', 'z']].values
        v = v.T
        nballs_per_stack = v.shape[1]

        ##########
        # Reindexing
        ##########

        vertex_IDs = np.array(v_df.index)
        v_df = v_df.reset_index(drop=True)

        keys_list = vertex_IDs
        values_list = np.arange(len(keys_list))
        zip_iterator = zip(keys_list, values_list)
        map_dictionary = dict(zip_iterator)

        edges_df_orig = edges_df.copy()
        edges_df = edges_df[(edges_df['v1'].isin(vertex_IDs)) & (edges_df['v2'].isin(vertex_IDs))]
        edges_df = edges_df.reset_index(drop = True)

        for index,row in edges_df.iterrows():

            edges_df.loc[index, 'v1'] = map_dictionary[row['v1']]
            edges_df.loc[index, 'v2'] = map_dictionary[row['v2']]

    if theta_ref is None:
        if theta_max is None:
            theta_max = max(v_df['theta'])
            print(theta_max)
        theta_ref = theta_max
        

    ############
    # Converting to df
    ###########
    


    nstack = 2
    #thickness = 0.05
    #R = 1 - thickness
    #spring_const = 1
    #viscoelastic_coeff = 1

    ball_id = 0
    for stack in range(nstack):

        for i in range(nballs_per_stack):

            [x,y,z] = [v_df.iloc[i]['x'], v_df.iloc[i]['y'], v_df.iloc[i]['z'],]
            #r = np.linalg.norm([x,y,z])
            [r,theta,phi] = cart2sph(x,y,z)
            p = theta/theta_ref

            #if point is on pole then we should offset the point a little so that lambda is not undefined there
            #if x == 0 and y == 0:
            #    if shift_pole:
            #        [x,y,z] = [r*np.sin(pole_offset_theta)*np.cos(pole_offset_phi), r*np.sin(pole_offset_theta)*np.sin(pole_offset_phi), r*np.cos(pole_offset_theta)]

            #[x,y,z] = [(R + thickness*stack)*x/r, (R + thickness*stack)*y/r, (R + thickness*stack)*z/r]

            thickness_temp = (1-stack)*thickness_polynomial_obj(p)
            [x,y,z] = [(R - thickness_temp)*x/r, (R - thickness_temp)*y/r, (R - thickness_temp)*z/r]

            row = pd.DataFrame([[ball_id,x,y,z, [], [], [], stack]], columns=balls_colnames)
            #print(row)
            balls = pd.concat([balls, row] , ignore_index = True)

            ball_id += 1

    #adding connections
    #for each point
    #    find particles which are neighbours according to edges_df
    #    To that list, add the same neighbours in the next stack
    #    Also add the same point in the next stack
    #    From this list select only particles which have id greater than the point
    #    get the coordinates of point as x1, y1, z1
    #    for each neighbour
    #        define spring id
    #        add id as spring1 for the point
    #        add id as spring2 for the neighbour
    #        add neighbour to list of neighbours of point
    #        add point to list of neighbours of this neighbour
    #        get the coordinates of neighbour as x2, y2, z2
    #        get the length of the spring
    #        check the type of spring (if the radius from center is same then it face otherwise it is edge)
    #        ball1 = point and ball2 = neighbour
    #        Add spring to the dataframe

    for i in range(len(balls)):

        point = i #id of ball - get id if the ids are not consecutive integers

        #[x1, y1, z1] = [ balls.loc[point,'x'] , balls.loc[point,'y'] , balls.loc[point,'z']]

        #get the list of neighbours according
        neighbours1 = np.array(edges_df.loc[(edges_df['v1'] == point%nballs_per_stack),'v2'].values) #here we take modulo of point because v and edges contain info about only one stack
        neighbours2 = np.array(edges_df.loc[(edges_df['v2'] == point%nballs_per_stack),'v1'].values) #here we take modulo of point because v and edges contain info about only one stack
        neighbours = np.concatenate([neighbours1, neighbours2, neighbours1 + nballs_per_stack, neighbours2 + nballs_per_stack, [(point%nballs_per_stack)+nballs_per_stack]])
        neighbours = np.unique(neighbours)
        neighbours = neighbours[(neighbours>point) & (neighbours<len(balls))]


        for neighbour in neighbours:


            if neighbour == point:
                #no self loop
                continue
            if( balls.loc[point, 'neighbours'].count(neighbour) > 0):
                #then this neighbour has already been connected
                continue

            balls.loc[point, 'neighbours'].append(neighbour)
            balls.loc[neighbour, 'neighbours'].append(point)


            spring_id = springs.shape[0]

            balls.loc[point, 'spring1'].append(spring_id)
            balls.loc[neighbour, 'spring2'].append(spring_id)


            length = np.sqrt(sum(np.square(balls.loc[point, ['x', 'y', 'z']].values -
                                           balls.loc[neighbour, ['x', 'y', 'z']].values)))

            if balls.loc[point,'stack'] == balls.loc[neighbour,'stack']:
                spring_type = 'inplane' #springs on the apical surface or basal surface
            else:
                spring_type = 'face' #pillars connecting the apical and basal

            row=pd.DataFrame([[spring_id] +
                              list(balls.loc[point, ['x', 'y', 'z']].values) +
                              list(balls.loc[neighbour, ['x', 'y', 'z']].values) +
                             [spring_const, length, length, 0, point, neighbour, spring_type, viscoelastic_coeff]])

            row.columns=springs.columns
            springs=pd.concat([springs,row])
            springs.index = range(springs.shape[0])



    dfToVtk(balls, springs, filename=output_vtk)

    pickle.dump([balls, springs], open(output_pickle, 'wb'))
    
    return([balls, springs])

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(hxy, z)
    az = np.arctan2(y, x)
    return r, el, az

def sph2cart(r, el, az):
    rsin_theta = r * np.sin(el)
    x = rsin_theta * np.cos(az)
    y = rsin_theta * np.sin(az)
    z = r * np.cos(el)
    return x, y, z

def change_radius_of_cap_mesh(balls_df, springs_df, R_new = 10, 
    #R_old = None, 
    thickness = 0.1):

    rs, thetas, phis = cart2sph(balls_df['x'].values, balls_df['y'].values, balls_df['z'].values)

    balls_df['R'] = rs
    balls_df['theta'] = thetas
    balls_df['phi'] = phis
    #collapsing everything on the same stack
    R_old = max(balls_df['R'])
    #calculating new value of theta
    balls_df['theta_new'] = R_old*balls_df['theta']/R_new
    #setting new radius
    balls_df['R_new'] = (R_new - thickness) + balls_df['stack']*thickness

    xs, ys, zs = sph2cart(balls_df['R_new'].values, balls_df['theta_new'].values, balls_df['phi'].values )

    balls_df['x'] = xs
    balls_df['y'] = ys
    balls_df['z'] = zs

    springs_df = update_springs(springs_df, balls_df[['x', 'y', 'z']])

    return([balls_df, springs_df])

def initialize_cpp_simulation(balls_df, springs_df, dt = 0.1,  csv_t_save = 1,
                              tol = 1e-5, path = '/'):



    #path = '/Users/szalapak/openfpm/springlatticemodel/SpringLatticeModel_multicore/'
    #path = '/Users/krishna/PhD/codes/openfpm/SpringLatticeModel/springlatticemodel/SpringLatticeModel_multicore/'

    balls_df['active'] = True
    springs_df['active'] = True
    balls_df['active'] = balls_df['active'].astype(int)

    print('Implementing external forces')
    balls_df['ext_fx'] = 0
    balls_df['ext_fy'] = 0
    balls_df['ext_fz'] = 0

    save_files_for_cpp(balls_df, springs_df, path + 'runfiles/' , spring_columns = ['l0', 'k', 'viscoelastic_coeff'], part_columns = ['active', 'ext_fx', 'ext_fy', 'ext_fz'])

    dim_max = np.max(balls_df[['x', 'y', 'z']].values.flatten()) + 3 # I don't understand this right now : why +3?

    sim_params = np.array([dt, csv_t_save, dim_max, tol])
    np.savetxt(path + "runfiles/sim_params.csv", sim_params, delimiter=" ", newline=" ")

    with open(path + "runfiles/sim_params.csv", "a") as csvfile:
        csvfile.write('\ndt' + ' csv_t_save' + ' dim_max' + 'tol')


    return ([balls_df, springs_df])

def plot_shell(balls_df, springs_df, x = 'x', y = 'z', filename = None, title = '', line_color_values = None,
               color_min = None, color_max = None, cbar_labels = None, cbar_ticks = None, cmap = 'viridis',
               cbar_name = None,
               linewidth = 1,
               xlim_min = None, xlim_max = None, ylim_min = None, ylim_max = None,
               plot_only_top = False,
              ):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm

    #x = 'x'
    #y = 'z'

    #springs_df_stack = springs_df
    if plot_only_top:
        springs_df_stack = springs_df[(springs_df['ball1'] >= len(balls_df)/2) & (springs_df['ball2'] >= len(balls_df)/2)]
    else:
        springs_df_stack = springs_df


    fig,ax = plt.subplots()
    #points_demo = np.array(balls_final[['x', 'y']]).reshape(-1, 1, 2)

    #plot the original network too
    #springs = update_springs(springs_final, balls_unstretched[['x', 'y', 'z']])
    points_1 = np.array(springs_df_stack[[x + '1', y + '1']]).reshape(-1, 1, 2)
    points_2 = np.array(springs_df_stack[[x + '2', y + '2']]).reshape(-1, 1, 2)

    #create a collection of lines
    segments_demo = np.concatenate([points_1, points_2], axis = 1)

    #value by which to color lines
    #if relative_to == 'initial':
    #    springs_final['l1_l0'] = springs_final['l1']/springs_final['l0']
    #elif relative_to == 'absolute':
    #    springs_final['l1_l0'] = springs_final['l1']

    if line_color_values is None:
        line_color_values = np.array(springs_df_stack['l0_target']/springs_df_stack['l1_initial'])
    #dydx = np.array(springs_final['l1_l0']) #just given a name from tutorial, dydx does not mean anything in my case

    # Create a continuous norm to map from data points to colors

    #color_min = 1.5
    #color_max = 1

    if color_min is None:
        color_min = line_color_values.min()
    if color_max is None:
        color_max = line_color_values.max()

    norm = plt.Normalize(color_min, color_max)
    lc = LineCollection(segments_demo, cmap=cmap, norm=norm)

    # Set the values used for colormapping
    lc.set_array(line_color_values)
    lc.set_linewidth(linewidth)
    line = ax.add_collection(lc)
    #fig.colorbar(line, ax=ax
    #            )

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    if cbar_ticks is None:
        cbar_ticks = [color_min,
                      #(color_max + color_min)/2,
                      color_max]
    if cbar_labels is None:
        cbar_labels = [str(round(tick,2)) for tick in cbar_ticks]

    cbar = fig.colorbar(line, ax = ax, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(labels = cbar_labels, fontsize = 15)  # vertically oriented colorbar
    if not(cbar_name is None):
        cbar.ax.set_ylabel(cbar_name, rotation=0, fontsize = 30)

    if xlim_min is None:
        [xlim_min,xlim_max] = [min(balls_df[x]), max(balls_df[x])]
        [ylim_min,ylim_max] = [min(balls_df[y]), max(balls_df[y])]

    plt.ylim(ylim_min, ylim_max)
    plt.xlim(xlim_min, xlim_max)
    plt.axis('equal')

    plt.title(title, fontsize = 20)

    if not(filename is None):
        plt.savefig(filename, bbox_inches = 'tight')

def get_lambda(row, DV_present = True,
                   lambda_anisotropic_obj = None, lambda_isotropic_obj = None,
                   inDV_lambda_anisotropic_obj = None, inDV_lambda_isotropic_obj = None,
                   #lambda_height_obj = None, inDV_lambda_height_obj = None,
                   ):
    

    pos_vector = row[["x","y","z"]].values
    #get in surface radial coordinate (scaled from 0 to 1)
    p = row["pathlength_scaled"]

    if DV_present and row["DV_bool"] == 1 :

        #get basis vectors
        e_h = row[["e_h_x", "e_h_y", "e_h_z"]].values
        e_R = row[["e_DV_R_x","e_DV_R_y","e_DV_R_z"]].values
        e_phi = row[["e_DV_phi_x","e_DV_phi_y","e_DV_phi_z"]].values
        #get lambda coefficients
        lambda_anisotropic = inDV_lambda_anisotropic_obj(p) 
        lambda_isotropic = inDV_lambda_isotropic_obj(p) 

    else:

        #get basis vectors
        e_h = row[["e_h_x", "e_h_y", "e_h_z"]].values
        e_R = row[["e_R_x","e_R_y","e_R_z"]].values
        e_phi = row[["e_phi_x","e_phi_y","e_phi_z"]].values
        #get lambda coefficients
        lambda_anisotropic = lambda_anisotropic_obj(p) 
        lambda_isotropic = lambda_isotropic_obj(p) 

    lambda_RR = lambda_isotropic*lambda_anisotropic
    lambda_phiphi = lambda_isotropic*(lambda_anisotropic**(-1))
    lambda_hh = 1

    lambda_alpha = lambda_RR*np.tensordot(e_R, e_R, axes = 0) + lambda_phiphi*np.tensordot(e_phi, e_phi, axes = 0) + lambda_hh*np.tensordot(e_h, e_h, axes = 0)

    return(lambda_alpha)

def plot_shell_on_given_ax(balls_df, springs_df, x = 'x', y = 'z', filename = None, title = '', line_color_values = None,
               color_min = None, color_max = None, cbar_labels = None, cbar_ticks = None, cmap = 'viridis',
               cbar_name = None,
               linewidth = 2, 
               xlim_min = None, xlim_max = None, ylim_min = None, ylim_max = None,
               plot_only_top = False,
               ax = None, fig = None,
               show_cbar = True,
               line_color = "gray",

              ):

    #import numpy as np
    #import matplotlib.pyplot as plt
    #from matplotlib.collections import LineCollection
    #from matplotlib.colors import ListedColormap, BoundaryNorm

    #x = 'x'
    #y = 'z'

    #springs_df_stack = springs_df
    if plot_only_top:
        springs_df_stack = springs_df[(springs_df['ball1'] >= len(balls_df)/2) & (springs_df['ball2'] >= len(balls_df)/2)]
    else:
        springs_df_stack = springs_df


    #fig,ax = plt.subplots()
    #points_demo = np.array(balls_final[['x', 'y']]).reshape(-1, 1, 2)

    #plot the original network too
    #springs = update_springs(springs_final, balls_unstretched[['x', 'y', 'z']])
    points_1 = np.array(springs_df_stack[[x + '1', y + '1']]).reshape(-1, 1, 2)
    points_2 = np.array(springs_df_stack[[x + '2', y + '2']]).reshape(-1, 1, 2)

    #create a collection of lines
    segments_demo = np.concatenate([points_1, points_2], axis = 1)

    #value by which to color lines
    #if relative_to == 'initial':
    #    springs_final['l1_l0'] = springs_final['l1']/springs_final['l0']
    #elif relative_to == 'absolute':
    #    springs_final['l1_l0'] = springs_final['l1']
    
    if line_color_values == 'final_vs_initial':
        line_color_values = np.array(springs_df['l1']/springs_df['l1_initial'])
        
    
    if line_color_values is None:
        line_color_values = np.array(springs_df_stack['l0_target']/springs_df_stack['l1_initial'])
    #dydx = np.array(springs_final['l1_l0']) #just given a name from tutorial, dydx does not mean anything in my case

    # Create a continuous norm to map from data points to colors

    #color_min = 1.5
    #color_max = 1

    if color_min is None:
        color_min = line_color_values.min()
    if color_max is None:
        color_max = line_color_values.max()

    if show_cbar:
        norm = plt.Normalize(color_min, color_max)
        lc = LineCollection(segments_demo, cmap=cmap, norm=norm)

        # Set the values used for colormapping
        lc.set_array(line_color_values)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)
        #fig.colorbar(line, ax=ax
        #            )
        
        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        if cbar_ticks is None:
            cbar_ticks = [color_min, 
                          #(color_max + color_min)/2, 
                          color_max]
        if cbar_labels is None:
            cbar_labels = [str(round(tick,2)) for tick in cbar_ticks]
        
        cbar = fig.colorbar(line, ax = ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(labels = cbar_labels, fontsize = 30)  # vertically oriented colorbar
        if not(cbar_name is None):
            cbar.ax.set_ylabel(cbar_name, rotation=0, fontsize = 40)

    else:
        lc = LineCollection(segments_demo, color = line_color)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)
        
    if xlim_min is None:
        [xlim_min,xlim_max] = [min(balls_df[x]), max(balls_df[x])]
        [ylim_min,ylim_max] = [min(balls_df[y]), max(balls_df[y])]

    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xlim(xlim_min, xlim_max)
    ax.axis('equal')

    #plt.title(title, fontsize = 20)
    
    #if not(filename is None):
    #    plt.savefig(filename, bbox_inches = 'tight')

def get_2D_curve_from_simulation(balls, springs, projection_x = 'x', projection_y = 'z',):
    
    #defining x and y values
    y = 'y' if projection_x == 'x' else 'x'
    [x, x1, x2, y, y1, y2, z, z1, z2 ] = [projection_x, projection_x + str(1), projection_x + str(2), #either x or y
                                      y, y + str(1), y + str(2), #either x or y
                                      projection_y, projection_y + str(1), projection_y + str(2)] #this is probably always z
    
    #get the top surface
    top_springs = springs[(springs['ball1'] >= len(balls)/2) & (springs['ball2'] >= len(balls)/2)]
    top_balls = balls[balls['ID'] >= len(balls)/2]
    
    #get springs that cross the xz plane
    #such springs will have the y values of their end points of different signs
    xz_plane_springs = top_springs[top_springs[y1]*top_springs[y2]<0]
    indices = xz_plane_springs.index
    
    #computing the point of intersection of the springs with the xz plane
    xz_plane_springs[x] = xz_plane_springs[x1] - xz_plane_springs[y1] * ((xz_plane_springs[x1] - xz_plane_springs[x2])/(xz_plane_springs[y1] - xz_plane_springs[y2]))
    xz_plane_springs[y] = 0
    xz_plane_springs[z] = xz_plane_springs[z1] - xz_plane_springs[y1] * ((xz_plane_springs[z1] - xz_plane_springs[z2])/(xz_plane_springs[y1] - xz_plane_springs[y2]))
    
    #making a df of just the intersection points
    xz_curve = xz_plane_springs[[x, z]] #.reset_index(drop=True)
    xz_curve['theta_xz'] = np.arctan(xz_curve[x]/xz_curve[z])
    xz_curve['norm_angle_xz'] = (xz_curve['theta_xz'] - min(xz_curve['theta_xz']))/(max(xz_curve['theta_xz']) - min(xz_curve['theta_xz']))
    for prop in ["DV_bool", "pathlength_scaled",]:
        if prop in top_balls.columns:
            #here we will take a mean of the prop between the two end points
            xz_curve[prop] = 0.5*(top_balls.loc[xz_plane_springs['ball1'].values, prop].values+top_balls.loc[xz_plane_springs['ball2'].values, prop].values)
    
    #sort the points by x
    xz_curve = xz_curve.sort_values(by = x, axis=0, ascending=False).reset_index(drop = True)
    
    return(xz_curve)

def compute_2D_curve_curvature(curve):
    
    #curve.columns = ['x', 'y']

    #curve = xz_curve.copy()

    for i in range(curve.index[1], curve.index[-1]):

        xi = curve.loc[i,'x']
        yi = curve.loc[i, 'y']
        xi0 = curve.loc[i-1,'x']
        yi0 = curve.loc[i-1, 'y']
        xi1 = curve.loc[i+1,'x']
        yi1 = curve.loc[i+1, 'y']

        #calculating. length. of curve
        if i == curve.index[1]:
            length = np.sqrt( (xi - xi0)**2 + (yi - yi0)**2 )
            curve.loc[i-1, 'arclength'] = 0

        length = length + np.sqrt( (xi1 - xi)**2 + (yi1- yi)**2 )
        curve.loc[i, 'arclength'] = curve.loc[i-1, 'arclength'] + np.sqrt( (xi - xi0)**2 + (yi - yi0)**2 )

        if i == curve.index[-2]:
            curve.loc[i + 1, 'arclength'] = curve.loc[i, 'arclength'] + np.sqrt( (xi1 - xi)**2 + (yi1 - yi)**2 )

    #setting arclength zero
    if "pathlength_scaled" in curve.columns:
        #choose the points with the minimum pathlength_scaled
        #take the average arclength of these points
        arclength_DV = curve[curve["pathlength_scaled"] == curve["pathlength_scaled"].min()]["arclength"].mean()
    else:
        #now we just take the mid point along the curve
        arclength_DV =  curve.loc[len(curve)-1, 'arclength']/2 #(curve_ventral.iloc[-1]['arclength'] + curve_dorsal.iloc[0]['arclength'])/2
    curve['arclength_offset'] = curve['arclength'] - arclength_DV

    #smooth cubic spline
    #s = 10
    degree = 5
    #bspline_x = Bspline(curve['arclength_offset'], curve['x'], k = degree)
    #bspline_y = Bspline(curve['arclength_offset'], curve['y'], k = degree)



    #t_x = [np.array(curve['arclength_offset'])[0]]*(int((len(curve) + degree + 1)/2)) + [np.array(curve['arclength_offset'])[-1]]*(int((len(curve) + degree + 1)/2))
    #t_y = t_x
    #break

    # t -> knots 
    # the function splrep will add k+1 knots as the initial point and k+1 knots as the last point
    # the size of t has to be n + k + 1 where n is the number of points
    # we have to add n + k + 1 - 2(k + 1)  = n - k - 1 knots ourselves
    # we can do this by dropping the first (k + 1)/2 points and the last (k+1)/2 points

    #t = np.array(curve['arclength_offset'])[int((degree + 1)/2):-int((degree + 1)/2)]

    spl_x = splrep(curve['arclength_offset'], curve['x'], k = degree, w = [1]*len(curve),
                   #task = -1, # to find the least square solution to the B spline
                   t = [3*min(curve['arclength_offset'])/4, min(curve['arclength_offset'])/2,min(curve['arclength_offset'])/4,0,max(curve['arclength_offset'])/4,max(curve['arclength_offset'])/2,3*max(curve['arclength_offset'])/4,]
                   #t = t
                  )
    spl_y = splrep(curve['arclength_offset'], curve['y'], k = degree, w = [1]*len(curve),
                   #task = -1, # to find the least square solution to the B spline
                   t = [3*min(curve['arclength_offset'])/4, min(curve['arclength_offset'])/2,min(curve['arclength_offset'])/4,0,max(curve['arclength_offset'])/4,max(curve['arclength_offset'])/2,3*max(curve['arclength_offset'])/4,]
                   #t = t
                  )

    arclengths = np.linspace(min(curve['arclength_offset']), max(curve['arclength_offset']), 100)

    #arclengths = np.linspace(max_negative_s, min_positive_s, 100)

    x = splev(arclengths, spl_x)
    y = splev(arclengths, spl_y)

    x_1 = splev(arclengths, spl_x, der = 1)
    x_2 = splev(arclengths, spl_x, der = 2)
    y_1 = splev(arclengths, spl_y, der = 1)
    y_2 = splev(arclengths, spl_y, der = 2)
    curvatures = (y_2*x_1 - x_2*y_1)/(x_1**2 + y_1**2)**(3/2)


    df = pd.DataFrame({
        #'disc':[disc_name]*len(arclengths),
        'arclength':arclengths,
        'x':x,
        'y':y,
        'curvature':curvatures
                      })

    #df_all = pd.concat([df_all,df], ignore_index = True)
    #df['curvatures_100'] = df.curvatures.rolling(100).mean()

    #np.allclose(spl(x), y)


    return(df)

def get_final_dataframe(path = '', balls_initial = None, springs_initial = None,):
    
    #get balls_initial
    if balls_initial is None:
        print('balls_initial is not known')
        return
    
    #get springs initial
    if springs_initial is None:
        print('springs_initial is not known')
        return
    
    #path = path + 'growth_sim_DV_growth_1.5_thickness_0.1_theta_DV_0.15/'

    files = glob.glob(path + "growth_*/")

    N_iters_str = [re.search(path + 'growth_(.*)/',x).group(1) for x in files]
    N_iters = np.sort([int(N_iter) for N_iter in N_iters_str])
    
    N_iter_max = max(N_iters)
    
    folder = path + "growth_" + str(N_iter_max) + "/"
    
    timepoint_pos = pd.read_csv(folder + "final_0_0.csv")
    
    balls_timepoint = balls_initial.copy(deep = True)
    springs_timepoint = springs_initial.copy(deep = True)
    
    balls_timepoint[['x', 'y', 'z']] =  timepoint_pos[['x[0]', 'x[1]', 'x[2]']]
    
    springs_timepoint = update_springs(springs_timepoint, balls_timepoint[['x', 'y', 'z']])
    
    return([balls_timepoint, springs_timepoint])








