#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:40:53 2021

@author: krishna
"""

from functions import *
import glob
import re

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

import warnings
warnings.filterwarnings('ignore')

def init_semisphere_mesh(R = 1, nlat = 10, nlong = 10, nstack = 2, thickness = 0.5, spring_const = 1, viscoelastic_coeff = 1,):

    balls_colnames=['ID', 'x', 'y', 'z', 'neighbours', 'spring1', 'spring2','lat','long','stack']
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


    ball_id = 0
    for stack in range(nstack):

        for lat in range(nlat+1):

            for long in range(nlong):

                #add vertex
                if lat == 0:
                    #row = pd.DataFrame([[ball_id, 0, 0, R + thickness*stack, [], [], [], lat, lat, stack]], columns=balls_colnames)
                    #balls = pd.concat([balls, row], ignore_index = True)
                    #ball_id = ball_id + 1
                    break


                theta = lat * np.pi / (2*nlat)
                phi = long * 2*np.pi / (nlong)
                x = (R + thickness*stack) * np.sin(theta) * np.cos(phi)
                y = (R + thickness*stack) * np.sin(theta) * np.sin(phi)
                z = (R + thickness*stack) * np.cos(theta)

                row = pd.DataFrame([[ball_id,x,y,z, [], [], [], lat, long, stack]], columns=balls_colnames)
                #print(row)
                balls = pd.concat([balls, row] , ignore_index = True)

                ball_id = ball_id + 1

    #adding connections
    #for each point,
    #    find particles which have +-1 lat and long
    #    select only the particles which have id greater than the point and call them neighbours
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
        lat = balls.loc[point, 'lat']
        long = balls.loc[point, 'long']
        #[x1, y1, z1] = [ balls.loc[point,'x'] , balls.loc[point,'y'] , balls.loc[point,'z']]

        neighbours_df = balls[(balls['lat'].isin([lat-1, lat, lat+1])) & (balls['long'].isin([(long-1)%(nlong), long, (long+1)%(nlong)]))]

        if lat == 0: #The pole
            neighbours_df = balls[(balls['lat'].isin([0,1]))]

        #neighbours = neighbours_df[neighbours_df['ID']>point]['ID'].values
        neighbours = neighbours_df['ID'].values


        for neighbour in neighbours:

            #print(neighbour)

            if neighbour == point:
                continue
            if( balls.loc[point, 'neighbours'].count(neighbour) > 0):
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



    #update spring types as either face (default) or inplane
    #if z1 == z2
    #springs['type'] = 'face'
    #springs.loc[(springs.z1 == springs.z2), 'type'] = 'inplane'


        # works till here

    return(balls, springs)

def init_gmsh_hemisphere_mesh(R = 1, nstack = 2, thickness = 0.5, spring_const = 1, viscoelastic_coeff = 1, shift_pole = True):

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

    #created using gmsh
    [v, edges] = pickle.load( open("wd_3d/gmsh_hemispherical_lattice_r_20.pickle", "rb") )

    nballs_per_stack = v.shape[1]
    edges_df = pd.DataFrame({'v1':edges[0].data, 'v2':edges[1].data})

    #pole_offset_theta = np.pi/(200) #fixed offset for pole
    #pole_offset_phi = 1 #fixed offset for pole


    ball_id = 0
    for stack in range(nstack):

        for i in range(v.shape[1]):

            [x,y,z] = [v[0][i], v[1][i], v[2][i]]
            r = np.linalg.norm([x,y,z])

            #if point is on pole then we should offset the point a little so that lambda is not undefined there
            #if x == 0 and y == 0:
            #    if shift_pole:
            #        [x,y,z] = [r*np.sin(pole_offset_theta)*np.cos(pole_offset_phi), r*np.sin(pole_offset_theta)*np.sin(pole_offset_phi), r*np.cos(pole_offset_theta)]

            [x,y,z] = [(R + thickness*stack)*x/r, (R + thickness*stack)*y/r, (R + thickness*stack)*z/r]

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



    #update spring types as either face (default) or inplane
    #if z1 == z2
    #springs['type'] = 'face'
    #springs.loc[(springs.z1 == springs.z2), 'type'] = 'inplane'


        # works till here

    return(balls, springs)

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(hxy, z)
    az = np.arctan2(y, x)
    return r, el, az

def init_gmsh_spherical_cap_mesh(mesh_file = 'gmsh_sphere_lattice_r_20.pickle',theta_max = None,R = 1, nstack = 2,
             thickness = 0.5, thickness_polynomial_obj = None,
              spring_const = 1, viscoelastic_coeff = 1,
             output_pickle = 'mesh.pickle', output_vtk = 'mesh.vtk'
            ):

    import pickle

    #this funtion reads a spherical mesh generated before hand
    #then rotates it by some angle about some angle
    #then crops the mesh above a threshold theta

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

    #created using gmsh
    [v, edges] = pickle.load( open(mesh_file, "rb") )


    nballs_per_stack = v.shape[1]
    edges_df = pd.DataFrame({'v1':edges[0].data, 'v2':edges[1].data})

    v_df = pd.DataFrame(v.T, columns=['x', 'y', 'z'])

    ###########
    # Rotating vertices
    ###########

    from numpy import cross, eye, dot
    from scipy.linalg import expm, norm

    def M(axis, theta):
        return expm(cross(eye(3), axis/norm(axis)*theta))

    axis, theta = [4,4,1], 0.001*np.pi/180
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

    #
    if theta_max is None:
        theta_max = np.pi/4
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

        for i in range(v.shape[1]):

            [x,y,z] = [v[0][i], v[1][i], v[2][i]]
            #r = np.linalg.norm([x,y,z])
            [r,theta,phi] = cart2sph(x,y,z)
            p = theta/theta_max

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

def change_radius_of_cap_mesh(balls_df, springs_df, R_new = 10, 
    #R_old = None, 
    thickness = 0.1):

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




def get_lambda(pos_vector):

    #pos_vector = np.array([[row['x1'], row['y1'], row['z1']]]).T #position vector as a column vector
    pos_vector = np.array(pos_vector) #position vector as a row vector

    r = np.linalg.norm(pos_vector)
    theta = np.arccos(pos_vector[2]/r)
    if theta == 0:
        print('theta is zero')
    phi = np.arcsin(pos_vector[1]/(r*np.sin(theta)))

    e_r = pos_vector/r
    e_theta = np.array([pos_vector[0], pos_vector[1], pos_vector[2] - r/np.cos(theta)]) #not normalized
    e_theta = e_theta/np.linalg.norm(e_theta)
    e_phi = np.cross(e_r, e_theta) #not normalized
    e_phi = e_phi/np.linalg.norm(e_phi)

    lambda_rr = 1 #might depend on r theta phi
    m = 2/np.pi
    #lambda_thetaTheta = 1.1 #might depend on r theta phi
    m_1 = 1/30
    m_2 = 3
    C_1 = np.pi/(m_2*30*30)
    #rel_areas = 2 - m_1*np.sqrt(2*thetas/(m_2*C_1))
    #lambdas = np.sqrt(rel_areas)
    #lambda_thetaTheta = np.sqrt(2 - theta/(np.pi/2))
    lambda_thetaTheta = np.sqrt(2 - m_1*np.sqrt(2*theta/(m_2*C_1)))
    #lambda_phiPhi = 1.1 #might depend on r theta phi
    lambda_phiPhi = np.sqrt(2 - m_1*np.sqrt(2*theta/(m_2*C_1)))

    #lambda_alpha is the value of lambda tensor on vertex alpha
    lambda_alpha = lambda_rr*np.tensordot(e_r, e_r, axes = 0) + lambda_thetaTheta*np.tensordot(e_theta, e_theta, axes = 0) + lambda_phiPhi*np.tensordot(e_phi, e_phi, axes = 0)

    return(lambda_alpha)


def implement_growth_with_DV(balls_df, springs_df, growth_DV = 1, width = 0.25, DV_indices = None, Radius = 1, thickness = 0.2):

    if DV_indices is None:
        #find particles on apical surface that lie within DV boundary region
        DV_indices = np.array(balls_df[(np.abs(balls_df['y']) < 0.25*max(balls_df['y'])) & (balls_df['x']**2 + balls_df['y']**2 + balls_df['z']**2 > (Radius + 0.1*thickness))].index)
        #get indices of same particles on basal surface
        DV_indices_apical = DV_indices
        DV_indices = np.concatenate([DV_indices, [x-len(balls_df)/2 for x in DV_indices]])

    #get springs which have both their end balls in this list
    DV_springs_indices = np.array(springs_df[springs_df['ball1'].isin(DV_indices) & springs_df['ball2'].isin(DV_indices)].index)
    springs_df.loc[DV_springs_indices, 'l0'] = springs_df.loc[DV_springs_indices, 'l1'] * growth_DV

    #change the dl of these springs
    #implement growth

    return(springs_df)


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

def get_fit_lambda(pos_vector, DV_present = True, volume_conservation = True,
                   theta_DV = 0, theta_max = np.pi/2,
                   lambda_anisotropic_obj = None, lambda_isotropic_obj = None,
                   inDV_lambda_anisotropic_obj = None, inDV_lambda_isotropic_obj = None,
                   lambda_height_obj = None,
                  ):
    
    #theta_DV : the angle where the DV boundary ends. It should be 0 if DV is not present
    #volume_conservation : along thickness, poisson effect is applied if this is True

    [lambda_rr, lambda_thetaTheta, lambda_phiPhi] = [1,1,1]
    
    if not(DV_present):
        theta_DV = 0
    if volume_conservation:
        nu = 1 

    pos_vector = np.array(pos_vector)
    r = np.linalg.norm(pos_vector)

    if DV_present and np.abs(pos_vector[0]) <= r*np.sin(theta_DV/2) :

        #position is inside the DV boundary
        #this section of if loop is messy but it works as of now
        
        pos_vector = np.array(pos_vector) #position vector as a row vector
        pos_vector_orig = [pos_vector[0],pos_vector[1],pos_vector[2]]

        #rotating the coordinate axis
        pos_vector[0] = pos_vector_orig[2]
        pos_vector[1] = pos_vector_orig[1]
        pos_vector[2] = -pos_vector_orig[0]

        pos_vector = np.array(pos_vector) #position vector as a row vector

        #get coordinates
        r = np.linalg.norm(pos_vector)
        theta = np.arccos(pos_vector[2]/r)
        theta = round(theta, 4)
        phi = np.arcsin(pos_vector[1]/(r*np.sin(theta)))
        abs_phi = np.abs(phi)

        #rotating back the coordinate axis axis
        pos_vector[0] = pos_vector_orig[0]
        pos_vector[1] = pos_vector_orig[1]
        pos_vector[2] = pos_vector_orig[2]

        #get basis vectors
        e_r = pos_vector/r
        #we compute e_theta in the rotated frame
        #this is the direction running ACROSS DV boundary
        e_theta = np.array([pos_vector[0] + r/np.cos(theta), pos_vector[1], pos_vector[2] ]) #this is for the theta computed in the rotated frame
        e_theta = e_theta/np.linalg.norm(e_theta)
        #this is the radial direction along DV
        e_phi = np.cross(e_r, e_theta) #not normalized
        e_phi = e_phi/np.linalg.norm(e_phi)

        #get lambda coefficients
        lambda_anisotropic = inDV_lambda_anisotropic_obj(abs_phi/theta_max) 
        lambda_isotropic = inDV_lambda_isotropic_obj(abs_phi/theta_max) 
        
        lambda_thetaTheta = lambda_isotropic*(lambda_anisotropic**(-1))
        lambda_phiPhi = lambda_isotropic*lambda_anisotropic
        if (lambda_height_obj is None) and (volume_conservation):
            lambda_rr = 1/((lambda_isotropic)**(2*nu))
        else:
            lambda_height = lambda_height_obj(abs_phi/theta_max)
            lambda_rr = lambda_height
        #get lambda tensor
        lambda_alpha = lambda_rr*np.tensordot(e_r, e_r, axes = 0) + lambda_thetaTheta*np.tensordot(e_theta, e_theta, axes = 0) + lambda_phiPhi*np.tensordot(e_phi, e_phi, axes = 0)

    else:

        #position is outside DV boundary

        #get coordinates
        pos_vector = np.array(pos_vector)

        #get basis vectors
        r = np.linalg.norm(pos_vector)
        theta = np.arccos(pos_vector[2]/r)
        phi = np.arcsin(pos_vector[1]/(r*np.sin(theta)))
        
        #get basis vectors
        e_r = pos_vector/r
        e_theta = np.array([pos_vector[0], pos_vector[1], pos_vector[2] - r/np.cos(theta)]) #not normalized
        e_theta = e_theta/np.linalg.norm(e_theta)
        e_phi = np.cross(e_r, e_theta) #not normalized
        e_phi = e_phi/np.linalg.norm(e_phi)
        
        #get lambda coefficients
        lambda_anisotropic = lambda_anisotropic_obj((theta - theta_DV/2)/(theta_max - theta_DV/2)) #in-surface distance = r*theta
        lambda_isotropic = lambda_isotropic_obj((theta - theta_DV/2)/(theta_max - theta_DV/2))
        lambda_thetaTheta = lambda_isotropic*lambda_anisotropic
        lambda_phiPhi = lambda_isotropic*(lambda_anisotropic**(-1))
        if (lambda_height_obj is None) and (volume_conservation):
            lambda_rr = 1/((lambda_isotropic)**(2*nu))
        else:
            lambda_height = lambda_height_obj(abs_phi/theta_max)
            lambda_rr = lambda_height
        
        #get lambda tensor 
        lambda_alpha = lambda_rr*np.tensordot(e_r, e_r, axes = 0) + lambda_thetaTheta*np.tensordot(e_theta, e_theta, axes = 0) + lambda_phiPhi*np.tensordot(e_phi, e_phi, axes = 0)
        
        
    return(lambda_alpha)

####
# Old functions to get lambda

def get_lambda_with_DV(pos_vector, DV_present = True, DV_growth = 1, theta_DV = 1, theta_max = np.pi/2, passive_boundary_present = False, volume_conservation = False, outDV_gradient = True, inDV_gradient = True, slope_DV = 0, inDV_isotropic = True, 
    lambda_anisotropic = 1, lambda_anisotropic_inDV = 1,
    lambda_anisotropic_slope = 0,
    ):

    #pos_vector = np.array([[row['x1'], row['y1'], row['z1']]]).T #position vector as a column vector

    lambda_rr = 1 #might depend on r theta phi
    lambda_thetaTheta = 1
    lambda_phiPhi = 1

    pos_vector = np.array(pos_vector) #position vector as a row vector

    r = np.linalg.norm(pos_vector)
    theta = np.arccos(pos_vector[2]/r)
    if theta == 0:
        print('theta is zero')
    phi = np.arcsin(pos_vector[1]/(r*np.sin(theta)))

    #get basis vectors
    e_r = pos_vector/r
    e_theta = np.array([pos_vector[0], pos_vector[1], pos_vector[2] - r/np.cos(theta)]) #not normalized
    e_theta = e_theta/np.linalg.norm(e_theta)
    e_phi = np.cross(e_r, e_theta) #not normalized
    e_phi = e_phi/np.linalg.norm(e_phi)

    lambda_anisotropic = lambda_anisotropic + lambda_anisotropic_slope*theta

    #print('theta', theta)
    #print('theta_max', theta_max)

    if not(DV_present):
        theta_DV = 0

    #for everywhere
    if theta<= theta_max:
        lambda_rr = 1 #might depend on r theta phi
        if outDV_gradient:
            lambda_thetaTheta = np.sqrt(2 - (theta - theta_DV/2)/(theta_max - theta_DV/2))
            lambda_phiPhi = np.sqrt(2 - (theta - theta_DV/2)/(theta_max - theta_DV/2))
        else:
            lambda_thetaTheta = 1
            lambda_phiPhi = 1
        lambda_thetaTheta = lambda_anisotropic*lambda_thetaTheta
        lambda_phiPhi = (lambda_anisotropic**(-1))*lambda_phiPhi


    #if DV boundary implemented
    if DV_present and not(inDV_gradient):
        # find if position is inside DV boundary
        if np.abs(pos_vector[0]) <= r*np.sin(theta_DV/2) :

            lambda_rr = 1 #might depend on r theta phi
            lambda_thetaTheta = np.sqrt(DV_growth)
            lambda_phiPhi = np.sqrt(DV_growth)

    #if passive material present
    if passive_boundary_present:
        if theta>theta_max:
            lambda_rr = 1 #might depend on r theta phi
            lambda_thetaTheta = 1
            lambda_phiPhi = 1

    if volume_conservation:

        lambda_rr = 1/(lambda_thetaTheta * lambda_phiPhi)

    #lambda_alpha is the value of lambda tensor on vertex alpha
    lambda_alpha = lambda_rr*np.tensordot(e_r, e_r, axes = 0) + lambda_thetaTheta*np.tensordot(e_theta, e_theta, axes = 0) + lambda_phiPhi*np.tensordot(e_phi, e_phi, axes = 0)

    if DV_present and inDV_gradient:

        if np.abs(pos_vector[0]) <= r*np.sin(theta_DV/2) :

            #lambda_alpha = get_DV_boundary_lambda(pos_vector, slope_DV, inDV_isotropic)
            lambda_alpha = get_DV_boundary_lambda(pos_vector, slope_DV, inDV_isotropic, theta_max, mean_DV_growth = DV_growth, volume_conservation = volume_conservation, lambda_anisotropic_inDV = lambda_anisotropic_inDV)

    return(lambda_alpha)

#get k_DV lambda

def get_lambda_k_DV_with_DV(pos_vector, DV_present = True, DV_growth = 1, theta_DV = 0, theta_min = np.pi/4, theta_max = 3*np.pi/4, passive_boundary_present = False, volume_conservation = False, outDV_gradient = True, inDV_gradient = True, slope_DV = 0, inDV_isotropic = True):

    #pos_vector = np.array([[row['x1'], row['y1'], row['z1']]]).T #position vector as a column vector
    pos_vector = np.array(pos_vector) #position vector as a row vector

    lambda_rr = 1 #might depend on r theta phi
    lambda_thetaTheta = 1
    lambda_phiPhi = 1

    pos_vector_orig = [pos_vector[0],pos_vector[1],pos_vector[2]]

    #rotating the coordinate axis
    pos_vector[0] = pos_vector_orig[2]
    pos_vector[1] = pos_vector_orig[1]
    pos_vector[2] = -pos_vector_orig[0]

    pos_vector = np.array(pos_vector) #position vector as a row vector

    r = np.linalg.norm(pos_vector)

    theta = np.arccos(pos_vector[2]/r)
    theta = round(theta, 4)
    theta_min = round(theta_min, 4)
    theta_max = round(theta_max, 4)

    if theta == 0:
        print('theta is zero')
    phi = np.arcsin(pos_vector[1]/(r*np.sin(theta)))

    #rotating the coordinate axis axis
    pos_vector[0] = pos_vector_orig[0]
    pos_vector[1] = pos_vector_orig[1]
    pos_vector[2] = pos_vector_orig[2]


    e_r = pos_vector/r
    e_theta = np.array([pos_vector[0] + r/np.cos(theta), pos_vector[1], pos_vector[2] ]) #this is for the theta computed in the rotated frame
    e_theta = e_theta/np.linalg.norm(e_theta)
    e_phi = np.cross(e_r, e_theta) #not normalized
    e_phi = e_phi/np.linalg.norm(e_phi)




    flag = 0
    #for everywhere

    if not(DV_present):
        theta_DV = 0

    if (theta>=theta_min) and (theta<=(np.pi/2 - theta_DV/2)):
        #if (theta<= theta_max) and (theta>= theta_min):
        flag = 1
        lambda_rr = 1 #might depend on r theta phi

        if outDV_gradient:
            lambda_thetaTheta = np.sqrt(2 - (theta - (np.pi/2 - theta_DV/2))/(theta_min - (np.pi/2 - theta_DV/2)))
            lambda_phiPhi = np.sqrt(2 - (theta - (np.pi/2 - theta_DV/2))/(theta_min - (np.pi/2 - theta_DV/2)))
        else:
            lambda_thetaTheta = 1
            lambda_phiPhi = 1

    if (theta<=theta_max) and (theta>=(np.pi/2 + theta_DV/2)):
        flag = 1
        lambda_rr = 1 #might depend on r theta phi
        if outDV_gradient:
            lambda_thetaTheta = np.sqrt(2 - (theta - (np.pi/2 + theta_DV/2))/(theta_max - (np.pi/2 + theta_DV/2)))
            lambda_phiPhi = np.sqrt(2 - (theta - (np.pi/2 + theta_DV/2))/(theta_max - (np.pi/2 + theta_DV/2)))
        else:
            lambda_thetaTheta = 1
            lambda_phiPhi = 1

    #if DV boundary implemented
    if DV_present and not(inDV_gradient):
        # find if position is inside DV boundary
        #if np.abs(pos_vector[1]) <= r*np.sin(theta_DV) :
        if np.abs(theta - np.pi/2) <= theta_DV/2:
            flag = 1
            lambda_rr = 1 #might depend on r theta phi
            lambda_thetaTheta = np.sqrt(DV_growth)
            lambda_phiPhi = np.sqrt(DV_growth)


    #if passive material present
    if passive_boundary_present:

        #compute theta in correct frame

        theta = np.arccos(pos_vector[2]/r)
        theta = round(theta, 4)
        theta_boundary = np.pi/4

        if theta>theta_boundary:
            flag = 1
            lambda_rr = 1 #might depend on r theta phi
            lambda_thetaTheta = 1
            lambda_phiPhi = 1

    if volume_conservation:

        lambda_rr = 1/(lambda_thetaTheta * lambda_phiPhi)

    #lambda_alpha is the value of lambda tensor on vertex alpha
    lambda_alpha = lambda_rr*np.tensordot(e_r, e_r, axes = 0) + lambda_thetaTheta*np.tensordot(e_theta, e_theta, axes = 0) + lambda_phiPhi*np.tensordot(e_phi, e_phi, axes = 0)

    if DV_present and inDV_gradient:

        if np.abs(theta - np.pi/2) <= theta_DV/2:
            flag = 1
            #get lambda from another function
            #in theta_max, we pass the angle of the spherical cap
            #in k_DV, this angle corresponds to
            lambda_alpha = get_DV_boundary_lambda(pos_vector, slope_DV, inDV_isotropic, theta_max = (theta_max - theta_min)/2, mean_DV_growth = DV_growth, volume_conservation = volume_conservation)
    if flag == 0:
        #print if no lambda is calculated because position did not fall in any category
        print('theta', theta)
        print('theta_min', theta_min)
        print('theta_max', theta_max)

    return(lambda_alpha)

def get_DV_boundary_lambda(pos_vector, slope_DV = 0, inDV_isotropic = True, theta_max = 0.1, mean_DV_growth = 2, volume_conservation = False, lambda_anisotropic_inDV = 1):

    #In this function, we find the theta, phi direction in the 90 deg rotated DataFrame
    #At the distal tip we have some amount of deformation in the phi and theta directions
    #This deformation decreases as we move away from the distal tip along the DV boundary
    #For finding the theta,phi,r directions in rotated frame we will use the same
    #   procedure that we use when calculating lamnbda for k_DV

    pos_vector = np.array(pos_vector) #position vector as a row vector

    lambda_rr = 1 #might depend on r theta phi
    lambda_thetaTheta = 1
    lambda_phiPhi = 1

    pos_vector_orig = [pos_vector[0],pos_vector[1],pos_vector[2]]

    #rotating the coordinate axis
    pos_vector[0] = pos_vector_orig[2]
    pos_vector[1] = pos_vector_orig[1]
    pos_vector[2] = -pos_vector_orig[0]

    pos_vector = np.array(pos_vector) #position vector as a row vector

    r = np.linalg.norm(pos_vector)

    theta = np.arccos(pos_vector[2]/r)
    theta = round(theta, 4)
    #theta_min = round(theta_min, 4)
    #theta_max = round(theta_max, 4)

    if theta == 0:
        print('theta is zero')
    phi = np.arcsin(pos_vector[1]/(r*np.sin(theta)))
    abs_phi = np.abs(phi)

    #rotating the coordinate axis axis
    pos_vector[0] = pos_vector_orig[0]
    pos_vector[1] = pos_vector_orig[1]
    pos_vector[2] = pos_vector_orig[2]


    e_r = pos_vector/r
    e_theta = np.array([pos_vector[0] + r/np.cos(theta), pos_vector[1], pos_vector[2] ]) #this is for the theta computed in the rotated frame
    e_theta = e_theta/np.linalg.norm(e_theta)
    e_phi = np.cross(e_r, e_theta) #not normalized
    e_phi = e_phi/np.linalg.norm(e_phi)

    #if inDV_isotropic:
    #    #lambda_thetaTheta = np.sqrt(2 - (theta - (np.pi/2 - theta_DV/2))/(theta_min - (np.pi/2 - theta_DV/2)))
    #    #lambda_phiPhi = np.sqrt(2 - (theta - (np.pi/2 - theta_DV/2))/(theta_min - (np.pi/2 - theta_DV/2)))
    #    lambda_thetaTheta = np.sqrt(slope_DV*(abs_phi - theta_max/2) + mean_DV_growth) #this is a straight line with the lambda at mid point is mean_lambda
    #    lambda_phiPhi = np.sqrt(slope_DV*(abs_phi - theta_max/2) + mean_DV_growth) #this is a straight line with the lambda at mid point is mean_lambda
    #    lambda_rr = 1

    #if not(inDV_isotropic):
    #    lambda_phiPhi = slope_DV*(abs_phi - theta_max/2) + mean_DV_growth
    #    lambda_thetaTheta = 1
    #    lambda_rr = 1

    lambda_thetaTheta = (lambda_anisotropic_inDV**(-1))*np.sqrt(slope_DV*(abs_phi - theta_max/2) + mean_DV_growth) #this is a straight line with the lambda at mid point is mean_lambda
    lambda_phiPhi = lambda_anisotropic_inDV*np.sqrt(slope_DV*(abs_phi - theta_max/2) + mean_DV_growth) #this is a straight line with the lambda at mid point is mean_lambda
    lambda_rr = 1

    if volume_conservation:

        lambda_rr = 1/(lambda_thetaTheta * lambda_phiPhi)

    lambda_alpha = lambda_rr*np.tensordot(e_r, e_r, axes = 0) + lambda_thetaTheta*np.tensordot(e_theta, e_theta, axes = 0) + lambda_phiPhi*np.tensordot(e_phi, e_phi, axes = 0)

    return(lambda_alpha)

def get_lambda_with_DV_flat(pos_vector, DV_present = True, volume_conservation = False, 
                            outDV_gradient = True, inDV_gradient = True, slope_DV = 0, inDV_isotropic = True, 
                            lambda_anisotropic_for_sphere = False, R_sphere = 1,
                            lambda_isotropic = 1, lambda_anisotropic = 1, DV_growth = 1, lambda_anisotropic_inDV = 1,
                            lambda_isotropic_slope = 0, lambda_anisotropic_slope = 0, DV_growth_slope = 0, lambda_anisotropic_inDV_slope = 0,
                            R_flat = 1, width_DV = 15, Pathlength_max = 85.64, 
                           ):

    #all values of lambdas are the values at r = 0
    #slope is of lambdas are kept 0 as default

    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    
    lambda_rr = 1
    lambda_thetaTheta = 1
    lambda_phiPhi = 1
    
    [x,y,z] = pos_vector
    pos_vector = np.array(pos_vector)    
    
    (r, theta) = cart2pol(x,y)
    
    e_r = np.array([np.cos(theta), np.sin(theta), 0])
    e_theta = np.array([-np.sin(theta), np.cos(theta), 0])
    e_z = np.array([0, 0, 1])
    e_x = np.array([1, 0, 0])
    e_y = np.array([0, 1, 0])
    
    #write the definition of lambda (isotropic deformation)
    #lambda_isotropic = 1 #lambda isotropic is the value at the r = R_flat/2
    lambda_isotropic = r*lambda_isotropic_slope + lambda_isotropic
    lambda_anisotropic = r*lambda_anisotropic_slope + lambda_anisotropic
    DV_growth = r*DV_growth_slope + DV_growth #this is lambda_isotropic_inDV
    lambda_anisotropic_inDV = r*lambda_anisotropic_inDV_slope + lambda_anisotropic_inDV

    if lambda_anisotropic_for_sphere:
        lambda_anisotropic = 1/(np.sqrt(1 - (r**2)/(4*(R_sphere**2))))
    
    #determine if position of vertex is inside DV or outside DV
    #definition of DV
    DV_x_margin = R_flat*(width_DV/(width_DV + 2*Pathlength_max))
    
    if DV_present and np.abs(x) <= DV_x_margin:
        
        #vertex lies inside DV boundary
        lambda_xx = DV_growth*(lambda_anisotropic_inDV**(-1))
        lambda_yy = DV_growth*(lambda_anisotropic_inDV)
        lambda_zz = 1
        if volume_conservation:
            lambda_zz = 1/(lambda_xx*lambda_yy)
            
        #lambda_alpha is the value of lambda tensor on vertex alpha
        lambda_alpha = lambda_zz*np.tensordot(e_z, e_z, axes = 0) + lambda_yy*np.tensordot(e_y, e_y, axes = 0) + lambda_xx*np.tensordot(e_x, e_x, axes = 0)
        
    else:
        
        lambda_rr = lambda_isotropic*lambda_anisotropic
        lambda_thetaTheta = lambda_isotropic*(lambda_anisotropic**(-1))
        lambda_zz = 1
        if volume_conservation:
            lambda_zz = 1/(lambda_rr*lambda_thetaTheta)
            
        #lambda_alpha is the value of lambda tensor on vertex alpha
        lambda_alpha = lambda_zz*np.tensordot(e_z, e_z, axes = 0) + lambda_rr*np.tensordot(e_r, e_r, axes = 0) + lambda_thetaTheta*np.tensordot(e_theta, e_theta, axes = 0)
        
    return(lambda_alpha)

def plot_shell_on_given_ax(balls_df, springs_df, x = 'x', y = 'z', filename = None, title = '', line_color_values = None,
               color_min = None, color_max = None, cbar_labels = None, cbar_ticks = None, cmap = 'viridis',
               cbar_name = None,
               linewidth = 2, 
               xlim_min = None, xlim_max = None, ylim_min = None, ylim_max = None,
               plot_only_top = False,
               ax = None, fig = None,
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
    
    if xlim_min is None:
        [xlim_min,xlim_max] = [min(balls_df[x]), max(balls_df[x])]
        [ylim_min,ylim_max] = [min(balls_df[y]), max(balls_df[y])]

    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xlim(xlim_min, xlim_max)
    ax.axis('equal')

    #plt.title(title, fontsize = 20)
    
    #if not(filename is None):
    #    plt.savefig(filename, bbox_inches = 'tight')

def get_2D_curve_from_simulation(springs, projection_x = 'x', projection_y = 'z', indices = None, 
                                 balls_initial = None, springs_initial = None,flat = False):
    
    if projection_x == 'x':
        y = 'y'
    else:
        y = 'x'
        
    [x, x1, x2, y, y1, y2, z, z1, z2 ] = [projection_x, projection_x + str(1), projection_x + str(2), #either x or y
                                          y, y + str(1), y + str(2), #either x or y
                                          projection_y, projection_y + str(1), projection_y + str(2)] #this is probably always z
    
    #first we get the indices of the springs which intersec with the xz plane
    if indices is None:
        
        if balls_initial is None:
            #uploading the initial state of the simulation
            #change the path if the initial stage is stored somewhere else
            path = '../openfpm/SpringLatticeModel/springlatticemodel/SpringLatticeModel_multicore/wd_3d/gmsh_r_20_rotated_spherical_cap_theta_32_thickness_0.1.pickle'
            [balls_initial, springs_initial] = pickle.load( open(path, 'rb') )
            springs_initial['l1_initial'] = springs_df['l1']
        balls_initial['r'] = np.sqrt(balls_initial['x']**2 + balls_initial['y']**2 + balls_initial['z']**2)

        #getting just the top surface of the shell
        if not(flat):
            #print('code should not be here!')
            rmax = max(balls_initial['r'])
            rmin = min(balls_initial['r'])
            #top_balls = balls_initial[balls_initial['r'] > 0.95]
            top_balls = balls_initial[balls_initial['r']>(rmax+rmin)/2]
        else:
            mid_z = (max(balls_initial['z']) + min(balls_initial['z']))/2
            #print('\n mid_z = '+ str(mid_z) + '\n')
            top_balls = balls_initial[balls_initial['z']>mid_z]
        #print('length of top balls: ' + str(len(top_balls)))
        top_springs = springs_initial[springs_initial['ball1'].isin(top_balls['ID']) & springs_initial['ball2'].isin(top_balls['ID'])]
        #springs which cross the xz plane will have different signs of y1 and y2
        xz_plane_springs = top_springs[top_springs[y1]*top_springs[y2]<0]
        indices = xz_plane_springs.index
    
    #knowing the indices of the springs, we extract those springs from the current state of the shell
    #print('indices')
    xz_plane_springs = springs.loc[indices]
    
    #computing the point of intersection of the springs with the xz plane
    xz_plane_springs[x] = xz_plane_springs[x1] - xz_plane_springs[y1] * ((xz_plane_springs[x1] - xz_plane_springs[x2])/(xz_plane_springs[y1] - xz_plane_springs[y2]))
    xz_plane_springs[y] = 0
    xz_plane_springs[z] = xz_plane_springs[z1] - xz_plane_springs[y1] * ((xz_plane_springs[z1] - xz_plane_springs[z2])/(xz_plane_springs[y1] - xz_plane_springs[y2]))

    #xz_curve = xz_plane_springs[['x', 'y', 'z']]
    xz_curve = xz_plane_springs[[x, y, z]]

    #to decide the order of points, we compute their polar angle coordinate and sort the points based on the angle
    xz_curve['theta_xz'] = np.arctan(xz_curve[x]/xz_curve[z])
    xz_curve = xz_curve.sort_values(by = 'theta_xz', axis=0, ascending=False, 
                                    #inplace=False, kind='quicksort', na_position='last'
                                   ).reset_index(drop = True)
    xz_curve['norm_angle_xz'] = (xz_curve['theta_xz'] - min(xz_curve['theta_xz']))/(max(xz_curve['theta_xz']) - min(xz_curve['theta_xz']))

    return(xz_curve)
    

def compute_2D_curve_curvature(curve):
    
    curve.columns = ['x', 'y']

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







