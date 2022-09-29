
import pandas as pd
import numpy as np
import math
import sys
#from mpl_toolkits import mplot3d

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import matplotlib.pyplot as plt
import random
from scipy.integrate import solve_ivp
import os
import shutil
import networkx as nx
import pickle

import csv

import math
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.colors as mcolors

from matplotlib import collections  as mc
import matplotlib.cm as cm
from scipy.spatial import Delaunay

"""Version: 2.0.2"""


def save_files_for_cpp(balls_df, springs_df, path, spring_columns, part_columns):
    
    """The function generates a ball.csv file with particle ID, 
    position (x,y,z), number of neighbours + attributes from part_columns
    
    
    as well as files containing attributes of springs - 
    a separate .csv file for each of the chosen spring_columns
    
    For ith row in the .csv, the values give info on selected attribute 
    of all neighbours of the ith particle 
    (eg in neighbours.csv, jth value in ith row corresponds 
    to the ID of the jth neighbour of ith particle)
    the attributes are saved in the same order as neighbours 
    IDs are saved in the list in the 'neighbours' column of balls_df dataframe
    This is then imported into cpp for each simulation
    """
    
    #make neighbour_lengths dataframe and populate the 'neighbours' column with a list of neighbours for each ball
    neigh_lengths = pd.DataFrame(index=balls_df['ID'], columns=['neighbours'] + spring_columns)
    neigh_lengths['neighbours'] = balls_df['neighbours']

    for ball in balls_df['ID']:  
        #for each ball, grab all the springs that have it as ball 1 or 2
        cp = springs_df.loc[(springs_df['ball1']==ball) | (springs_df['ball2']==ball)].copy()
        cp['neighbour'] = np.where(cp['ball1'] != ball, cp['ball1'], cp['ball2'])

        for column in spring_columns:
            neigh_lengths[column][ball] = cp.set_index('neighbour').loc[balls_df['neighbours'][ball]][column].values
        
    for column in ['neighbours'] + spring_columns:
        neighb_list = neigh_lengths[column].values.tolist()
        
        if column != 'neighbours': name = 'neigh_' + column
        else: name = column
        
        with open(path + name + '.csv', "w", newline="") as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerows(neighb_list)
            
    
    
    balls_df['n_neigh'] = balls_df['neighbours'].str.len()
    
    #balls.csv file contains particle attributes - global ID, position, number of neighbours and
    #the indication whether the particle is active or not
    balls=balls_df[['ID','x','y','z', 'n_neigh']+part_columns].values
    np.savetxt(path + "balls.csv", balls, delimiter=" ")
    
    

def poisson_disc_sampling(r, dim): 
    """code from: https://github.com/scipython/scipython-maths/tree/master/poisson_disc_sampled_noise"""
    
    # Choose up to k points around each reference point as candidates for a new
    # sample point
    k = 30

    #r - Minimum distance between sample

    (width, height) = (dim)

    # Cell side length
    a = r/np.sqrt(2)
    # Number of cells in the x- and y-directions of the grid
    nx, ny = int(width / a) + 1, int(height / a) + 1

    # A list of coordinates in the grid of cells
    coords_list = [(ix, iy) for ix in range(nx) for iy in range(ny)]
    # Initilalize the dictionary of cells: each key is a cell's coordinates, the
    # corresponding value is the index of that cell's point's coordinates in the
    # samples list (or None if the cell is empty).
    cells = {coords: None for coords in coords_list}

    def get_cell_coords(pt):
        """Get the coordinates of the cell that pt = (x,y) falls in."""

        return int(pt[0] // a), int(pt[1] // a)

    def get_neighbours(coords):
        """Return the indexes of points in cells neighbouring cell at coords.

        For the cell at coords = (x,y), return the indexes of points in the cells
        with neighbouring coordinates illustrated below: ie those cells that could 
        contain points closer than r.

                                         ooo
                                        ooooo
                                        ooXoo
                                        ooooo
                                         ooo

        """

        dxdy = [(-1,-2),(0,-2),(1,-2),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),
                (-2,0),(-1,0),(1,0),(2,0),(-2,1),(-1,1),(0,1),(1,1),(2,1),
                (-1,2),(0,2),(1,2),(0,0)]
        neighbours = []
        for dx, dy in dxdy:
            neighbour_coords = coords[0] + dx, coords[1] + dy
            if not (0 <= neighbour_coords[0] < nx and
                    0 <= neighbour_coords[1] < ny):
                # We're off the grid: no neighbours here.
                continue
            neighbour_cell = cells[neighbour_coords]
            if neighbour_cell is not None:
                # This cell is occupied: store this index of the contained point.
                neighbours.append(neighbour_cell)
        return neighbours

    def point_valid(pt):
        """Is pt a valid point to emit as a sample?

        It must be no closer than r from any other point: check the cells in its
        immediate neighbourhood.

        """

        cell_coords = get_cell_coords(pt)
        for idx in get_neighbours(cell_coords):
            nearby_pt = samples[idx]
            # Squared distance between or candidate point, pt, and this nearby_pt.
            distance2 = (nearby_pt[0]-pt[0])**2 + (nearby_pt[1]-pt[1])**2
            if distance2 < r**2:
                # The points are too close, so pt is not a candidate.
                return False
        # All points tested: if we're here, pt is valid
        return True

    def get_point(k, refpt):
        """Try to find a candidate point relative to refpt to emit in the sample.

        We draw up to k points from the annulus of inner radius r, outer radius 2r
        around the reference point, refpt. If none of them are suitable (because
        they're too close to existing points in the sample), return False.
        Otherwise, return the pt.

        """
        i = 0
        while i < k:
            rho, theta = np.random.uniform(r, 2*r), np.random.uniform(0, 2*np.pi)
            pt = refpt[0] + rho*np.cos(theta), refpt[1] + rho*np.sin(theta)
            if not (0 <= pt[0] < width and 0 <= pt[1] < height):
                # This point falls outside the domain, so try again.
                continue
            if point_valid(pt):
                return pt
            i += 1
        # We failed to find a suitable point in the vicinity of refpt.
        return False

    # Pick a random point to start with.
    pt = (np.random.uniform(0, width), np.random.uniform(0, height))
    samples = [pt]
    # Our first sample is indexed at 0 in the samples list...
    cells[get_cell_coords(pt)] = 0
    # ... and it is active, in the sense that we're going to look for more points
    # in its neighbourhood.
    active = [0]

    nsamples = 1
    # As long as there are points in the active list, keep trying to find samples.
    while active:
        # choose a random "reference" point from the active list.
        idx = np.random.choice(active)
        refpt = samples[idx]
        # Try to pick a new point relative to the reference point.
        pt = get_point(k, refpt)
        if pt:
            # Point pt is valid: add it to the samples list and mark it as active
            samples.append(pt)
            nsamples += 1
            active.append(len(samples)-1)
            cells[get_cell_coords(pt)] = len(samples) - 1
        else:
            # We had to give up looking for valid points near refpt, so remove it
            # from the list of "active" points.
            active.remove(idx)
            
    samples = np.array(samples)
    samples[:,0] -= width/2
    samples[:,1] -= height/2
    
            
    return(samples)

def get_oct_vertices(r, alpha):
    
    """generates octagon from r 
    (distance from the center of octagon to the center of its top vertex) 
    and alpha (octagon is semi-regular, has 4 alpha and 4 beta angles, where beta = 270-alpha)"""
    
    #calculate vertices of the oxtagon
    #in our case, octagon is a square with isosceles triangles at each of the sides
    #length of side of square is 2*r
    #A is the total height of the octagon
    A = 2*r 

    #sum of angles in an octagon is 1080
    #in our case, this is 4*(alpha + beta) = 1080
    #alpha + beta = 270
    #alpha is the angle at the 'top' of the traingle
    #beta is the angle by square vertices
    #gamma is the acute angle in the triangle
    beta = 270 - alpha
    
    print('alpha: ' + str(alpha))
    print('beta: ' + str(beta))
    
    gamma = (beta-90)/2
    
    #convert to radians since I only think in degrees lol
    alpha_rad = math.radians(alpha)
    beta_rad = math.radians(beta)
    gamma_rad = math.radians(gamma)
    
    #tan(gamma) = h/(a/2)
    #where a is the side of the square
    #A = a + 2h
    a = A/(math.tan(gamma_rad) + 1)
    #sq_c are the coordinates of square vertices
    sq_c = a/2

    h = (A-a)/2
    
    oct_vert = [[0, sq_c+h],
             [sq_c, sq_c],
             [sq_c+h, 0],
             [sq_c, -sq_c],
             [0, -(sq_c+h)],
             [-sq_c, -sq_c],
             [-(sq_c+h), 0],
            [-sq_c, sq_c]]
    
    oct_vert = np.asarray(oct_vert)
    
    return(oct_vert)

def make_poiss_layer(dim, r, shape, border_bool, alpha):

    if shape == 'square':
        
        if border_bool: 
            x_min = -dim[0]/2
            x_max = dim[0]/2
            y_min = -dim[1]/2
            y_max = dim[1]/2

            vert_border_y = np.arange(y_min, y_max, r)
            hor_border_x = np.arange(x_min, x_max, r)

            v_xmin = np.column_stack((np.repeat(x_min, len(vert_border_y)).T, vert_border_y.T))
            v_xmax = np.column_stack((np.repeat(x_max, len(vert_border_y)).T, -1*vert_border_y.T,))
            h_ymin = np.column_stack((-1*hor_border_x.T, np.repeat(y_min, len(hor_border_x)).T))
            h_ymax = np.column_stack((hor_border_x.T, np.repeat(y_max, len(hor_border_x)).T))

            border = np.vstack((v_xmin, v_xmax, h_ymin, h_ymax))

        #generate points in first layer
        samples = poisson_disc_sampling(r, (dim[0] - r, dim[1] - r))
    
    elif(shape == 'octagon'):
        #generate border points
        oct_vert = get_oct_vertices(dim[0], alpha)
        
        if border_bool: 

            oct_x = []
            oct_y = []

            for i in range(len(oct_vert)):
                if (i == 7): j = 0
                else: j = i+1

                dist = np.sqrt((oct_vert[i,0] - oct_vert[j,0])**2 + (oct_vert[i,1] - oct_vert[j,1])**2)

                xs = np.linspace(oct_vert[i,0], oct_vert[j,0], int(dist/(1.2*r)))
                ys = np.linspace(oct_vert[i,1], oct_vert[j,1], int(dist/(1.2*r)))

                oct_x.extend(xs[:-1])
                oct_y.extend(ys[:-1])

            border = np.column_stack((oct_x, oct_y))

        #generate points inside octagon
        limit_oct_vert = get_oct_vertices(dim[0] - r/2.5, alpha)
        oct_path = mpath.Path(np.append(limit_oct_vert, [limit_oct_vert[0]], axis = 0), closed = True)
        #print(dim[0])
        samples = poisson_disc_sampling(r, (dim[0]*2, dim[1]*2))
        #print(len(samples))
        samples = samples[oct_path.contains_points(samples)]
        
    elif(shape == 'circle'):
        #generate border points
        if border_bool: 

            angle = r/dim[0]

            angles = np.linspace(0, 2*np.pi, int(np.round(2*np.pi/angle)))

            circle_x = dim[0]*np.sin(angles)
            circle_y = dim[0]*np.cos(angles)

            border = np.column_stack((circle_x, circle_y))

        #generate points inside octagon
        samples = poisson_disc_sampling(r, (dim[0]*2 - r/1.5, dim[1]*2 - r/1.5))
        #print(len(samples))
        samples = samples[np.sqrt(samples[:,0]**2 + samples[:,1]**2) < (dim[0] - r/3)]
        #print(len(samples))


    if border_bool: 
        samples = np.vstack((samples, border))


    balls_colnames=['ID', 'x', 'y', 'z', 'ax', 'ay', 'az', 'neighbours', 'spring1', 'spring2','row','column','stack', 'vx', 'vy', 'vz', 'mass']
    balls=pd.DataFrame(0, index=range(0), columns=range(len(balls_colnames)))
    balls.columns=balls_colnames
    balls.index = range(balls.shape[0])

    balls.x = samples[:,0]
    balls.y = samples[:,1]
    balls.z = 0 
        
    return(balls)

def poisson_mesh(dim, r, layers, thickness, k, ns, mass = 1, shape = 'square', border_bool = True, alpha = 180):    
    
    """Poisson mesh generated using the poisson disc sampling algorithm 
    (poisson_disc_sampling function above)
    
    The function takes dimensions (total size of mesh), 
    r_poiss (determines density of points), number of layers, layer thickness, 
    and other particle attributes (k, ns, mass)
    """
    
    """square mesh is default, regular border generated
    
    in the future, should implement alternative borders as well!
    """

    balls = make_poiss_layer(dim, r, shape, border_bool, alpha)

    #deal with additional layers
    if layers > 1:
        for i in range(layers)[1:]:
            new_balls = make_poiss_layer(dim, r, shape, border_bool, alpha)

            new_balls.z = thickness*i
            new_balls.index = range(balls.shape[0], balls.shape[0]+new_balls.shape[0])
            balls=pd.concat([balls,new_balls])

    balls[['vx', 'vy', 'vz']] = 0
    balls.mass = mass

    balls.ID = balls.index

    #empty lists for neighbours and springs, will be extended in loop below
    balls.neighbours = [[] for _ in range(len(balls))]
    balls.spring1 = [[] for _ in range(len(balls))]
    balls.spring2 = [[] for _ in range(len(balls))]



    #dataframe of springs
    #ID, X1, Y1, Z1, X2, Y2, Z2, k, Natural length, Extension in length, Ball1, Ball2
    springs_colnames=['ID', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 
                      'k', 'l0', 'l1','dl', 
                      'ball1', 'ball2','type','viscoelastic_coeff']
    springs=pd.DataFrame(0, index=range(0), columns=range(len(springs_colnames)))
    springs.columns=springs_colnames

    #delunay triangulation used to connect the balls into triangles
    tri = Delaunay(balls[['x', 'y', 'z']].values)

    #iterate through simplexes and save connections as springs 
    #could potentially be done without the horrifying nested loops but not sure how atm
    for simplex in tri.simplices:
        for point in simplex:
            for neighbour in simplex[simplex != point]:
                #check whether the connection has already been saved, 
                #if not, add it as a spring
                if( balls.loc[point, 'neighbours'].count(neighbour) == 0):
                    balls.loc[point, 'neighbours'].append(neighbour)
                    balls.loc[neighbour, 'neighbours'].append(point)


                    spring_id = springs.shape[0]

                    balls.loc[point, 'spring1'].append(spring_id)
                    balls.loc[neighbour, 'spring2'].append(spring_id)


                    length = np.sqrt(sum(np.square(balls.loc[point, ['x', 'y', 'z']].values - 
                                                   balls.loc[neighbour, ['x', 'y', 'z']].values)))

                    row=pd.DataFrame([[spring_id] + 
                                      list(balls.loc[point, ['x', 'y', 'z']].values) + 
                                      list(balls.loc[neighbour, ['x', 'y', 'z']].values) + 
                                     [k, length, length, 0, point, neighbour, 0, ns]])

                    row.columns=springs.columns
                    springs=pd.concat([springs,row])
                    springs.index = range(springs.shape[0])

    #update spring types as either face (default) or inplane
    #if z1 == z2
    springs['type'] = 'face'
    springs.loc[(springs.z1 == springs.z2), 'type'] = 'inplane'
    
    return(balls, springs)

"""Footprints"""

def apply_footprint(balls_df, springs_df, footprint_shape, active_r, drop_footprint_bool, alpha=180):
    
    """Basically hiding away all the if statements"""
    
    #octogonal footprint returns vertices of the octagon, would be nice to keep this, 
    #for all other footprints None will be returned
    oct_vert = None
    
    if (footprint_shape == 'circle'):
        #active material only within the circle with radius = r
        balls_df['active'] = np.where((balls_df.x**2+balls_df.y**2)**0.5 <= active_r, True, False)
        
    elif (footprint_shape == 'square'): 
        #active material only where particle x and y <= active_r
        balls_df['active'] = np.where((abs(balls_df.x) <= active_r) & (abs(balls_df.y) <= active_r), True, False)
        
    elif (footprint_shape == 'octagon'):
        balls_df, oct_vert = octogonal_footprint(balls_df, r = active_r, alpha = alpha)
        
    #spring is active only if both particles it connects are active
    active_spring_bool = (balls_df.iloc[springs_df.ball1].active == True).values & (balls_df.iloc[springs_df.ball2].active == True).values
    springs_df['active'] = np.where(active_spring_bool, True, False)
    

    #if drop_footprint_bool == True
    #remove all the passive particles and springs
    if drop_footprint_bool:
        balls_df, springs_df = drop_footprint(balls_df, springs_df)
        
    return(balls_df, springs_df, oct_vert)

def octogonal_footprint(balls_df, r, alpha):
    
    """generates octagon from r 
    (distance from the center of octagon to the center of its top vertex) 
    and alpha (octagon is semi-regular, has 4 alpha and 4 beta angles, where beta = 270-alpha)"""
    
    #calculate vertices of the oxtagon
    #in our case, octagon is a square with isosceles triangles at each of the sides
    #length of side of square is 2*r
    #A is the total height of the octagon
    A = 2*r 

    #sum of angles in an octagon is 1080
    #in our case, this is 4*(alpha + beta) = 1080
    #alpha + beta = 270
    #alpha is the angle at the 'top' of the traingle
    #beta is the angle by square vertices
    #gamma is the acute angle in the triangle
    beta = 270 - alpha
    
    print('alpha: ' + str(alpha))
    print('beta: ' + str(beta))
    
    gamma = (beta-90)/2
    
    #convert to radians since I only think in degrees lol
    alpha_rad = math.radians(alpha)
    beta_rad = math.radians(beta)
    gamma_rad = math.radians(gamma)
    
    #tan(gamma) = h/(a/2)
    #where a is the side of the square
    #A = a + 2h
    a = A/(math.tan(gamma_rad) + 1)
    #sq_c are the coordinates of square vertices
    sq_c = a/2

    h = (A-a)/2
    
    oct_vert = [[0, sq_c+h],
             [sq_c, sq_c],
             [sq_c+h, 0],
             [sq_c, -sq_c],
             [0, -(sq_c+h)],
             [-sq_c, -sq_c],
             [-(sq_c+h), 0],
            [-sq_c, sq_c]]
    
    oct_vert = np.asarray(oct_vert)
    
    oct_path = mpath.Path(np.append(oct_vert, [oct_vert[0]], axis = 0), closed = True)
    
    balls_df['active'] = oct_path.contains_points(balls_df[['x', 'y']])
    
    return(balls_df, oct_vert)


def drop_footprint(balls_df, springs_df):
    
    """removes all the passive particles and springs
    and updates IDs
    
    lots of the code copied from Abhijeet's functions 
    (init_trianglemesh_circularboundary)
    """
    
    #springs_df.active = (springs_df.active).astype(bool)
    #balls_df.active = (balls_df).active.astype(bool)
    
    springs_keep = (springs_df.active | springs_df.frame).astype(bool)
    balls_keep = (balls_df.active | balls_df.frame).astype(bool)
    
    ## grab indices of balls and springs to be dropped
    ball_indices=balls_df.index[~balls_keep]
    ball_ids=balls_df.ID[ball_indices]
    balls_df=balls_df.drop(ball_indices)

    spring_indices = springs_df.index[~springs_keep]
    spring_ids=springs_df.ID[spring_indices]
    springs_df=springs_df.drop(spring_indices)

    balls_df.spring1=balls_df.apply(lambda row : [item for item in row['spring1'] if item not in spring_ids], axis=1)
    balls_df.spring2=balls_df.apply(lambda row : [item for item in row['spring2'] if item not in spring_ids], axis=1)
    #in the balls dataframe, remove those neighbours whare present in ball_ids
    balls_df.neighbours=balls_df.apply(lambda row : [item for item in row['neighbours'] if not(item in ball_ids)], axis=1)

    balls_df.reset_index(drop=True, inplace=True)
    key=balls_df.ID
    value=balls_df.index
    ball_dict=dict(zip(key, value))
    #spring_dict
    springs_df.reset_index(drop=True, inplace=True)
    key=springs_df.ID
    value=springs_df.index
    spring_dict=dict(zip(key, value))

    #replacing old ids with new ids in balls_df
    balls_df.ID=balls_df.apply(lambda row : ball_dict[row['ID']], axis=1)
    balls_df.neighbours=balls_df.apply(lambda row : [ball_dict[item] for item in row['neighbours']], axis=1)
    balls_df.spring1=balls_df.apply(lambda row : [spring_dict[item] for item in row['spring1']], axis=1)
    balls_df.spring2=balls_df.apply(lambda row : [spring_dict[item] for item in row['spring2']], axis=1)


    #replacing old ids with new ids in springs_df
    springs_df.ID=springs_df.apply(lambda row : spring_dict[row['ID']], axis=1)
    springs_df.ball1=springs_df.apply(lambda row : ball_dict[row['ball1']], axis=1)
    springs_df.ball2=springs_df.apply(lambda row : ball_dict[row['ball2']], axis=1)
    
    return(balls_df, springs_df)

#Differenct mesh generators

def init_cubicmesh(nrow=3,ncol=5,spring_const=1,znoise=False,mu=0,sigma=0.01,thickness=2, base='flat', thin=False,viscoelastic_coeff=0.5):

    #we can make it multilayer but when I add a number to stack then it does not update and shows Nan
    #stack does not update
    balls_colnames=['ID', 'x', 'y', 'z', 'ax', 'ay', 'az', 'neighbours', 'spring1', 'spring2','row','column','stack', 'vx', 'vy', 'vz', 'mass']
    balls=pd.DataFrame(0, index=range(0), columns=range(len(balls_colnames)))
    balls.columns=balls_colnames
    balls.index = range(balls.shape[0])

    #nrow = 4
    #ncol = 4

    dim = [ncol-1,nrow-1]
    #r = 1
    #thickness = 1
    mass = 1
    layers = 2
    #k = 1
    #ns = 1

    x_min = -dim[0]/2
    x_max = dim[0]/2
    y_min = -dim[1]/2
    y_max = dim[1]/2

    #making an array with different x values
    xs = np.linspace(-dim[0]/2, dim[0]/2, ncol)
    #repeating x values to make a lattice -> [-1, 0, 1, -1, 0, 1, -1, 0, 1]
    xs = np.tile(xs,nrow)
    #print(xs)
    #print("xs")
    #making an array with different y values
    ys = np.linspace(-dim[1]/2, dim[1]/2, nrow)
    #repeating y values to make a lattice -> [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    ys = np.repeat(ys, ncol)
    #print(ys)
    #print("ys")
    #giving rows and column numbers based on x and y pos
    rows = np.repeat(np.arange(0,nrow,1),ncol)
    #print(rows)
    #print("rows")
    cols = np.tile(np.arange(0,ncol,1), nrow)
    #print(cols)
    #print("cols")

    balls.x = xs
    balls.y = ys
    balls.z = 0
    balls.row = rows
    balls.column = cols
    balls.stack = 0 #this line deos not work


    #deal with additional layers
    if layers > 1:
        balls_single_layer = balls.copy(deep=True)
        stack = 0 
        for i in range(layers)[1:]:
            
            #copy the single layer dataframe
            new_balls = balls_single_layer.copy(deep=True)
            new_balls.z = thickness*i
            new_balls.stack = stack+i
            new_balls.index = range(balls.shape[0], balls.shape[0]+balls_single_layer.shape[0])
            balls=pd.concat([balls,new_balls])


    #import matplotlib.pyplot as plt
    #plt.scatter(balls.x, balls.y)


    balls[['vx', 'vy', 'vz']] = 0
    balls.mass = mass

    balls.ID = balls.index

    #empty lists for neighbours and springs, will be extended in loop below
    balls.neighbours = [[] for _ in range(len(balls))]
    balls.spring1 = [[] for _ in range(len(balls))]
    balls.spring2 = [[] for _ in range(len(balls))]

    #dataframe of springs
    #ID, X1, Y1, Z1, X2, Y2, Z2, k, Natural length, Extension in length, Ball1, Ball2
    springs_colnames=['ID', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 
                      'k', 'l0', 'l1','dl', 
                      'ball1', 'ball2','type','viscoelastic_coeff']
    springs=pd.DataFrame(0, index=range(0), columns=range(len(springs_colnames)))
    springs.columns=springs_colnames



    for point in balls.ID:
        
        #get an array of all neighbours
        #find all balls for which the difference in row is less than 2 AND difference in column is less than 2 AND difference in stack is less than 2
        [row,col,stack] = [balls.loc[point,'row'], balls.loc[point,'column'], balls.loc[point,'stack']]
        
        neighbours = balls.loc[(np.abs(balls.row - row) < 2) & (np.abs(balls.column - col) < 2),'ID'].values
        #print(point)
        #print(neighbours)
        
        #when stack starts working then uncomment this, it should work and make sure to make nodes from only adjacent stacks neighbours :
        #neighbours = balls.loc[((np.abs(balls.row - row) < 2) & (np.abs(balls.column - col) < 2)) & (np.abs(balls.stack - stack) < 2),'ID'].values
        for neighbour in neighbours:
            #check whether the connection has already been saved, 
            #if not, add it as a spring
            if neighbour == point:
                continue
            if( balls.loc[point, 'neighbours'].count(neighbour) == 0):
                balls.loc[point, 'neighbours'].append(neighbour)
                balls.loc[neighbour, 'neighbours'].append(point)


                spring_id = springs.shape[0]

                balls.loc[point, 'spring1'].append(spring_id)
                balls.loc[neighbour, 'spring2'].append(spring_id)


                length = np.sqrt(sum(np.square(balls.loc[point, ['x', 'y', 'z']].values - 
                                               balls.loc[neighbour, ['x', 'y', 'z']].values)))

                row=pd.DataFrame([[spring_id] + 
                                  list(balls.loc[point, ['x', 'y', 'z']].values) + 
                                  list(balls.loc[neighbour, ['x', 'y', 'z']].values) + 
                                 [spring_const, length, length, 0, point, neighbour, 0, viscoelastic_coeff]])

                row.columns=springs.columns
                springs=pd.concat([springs,row])
                springs.index = range(springs.shape[0])



    #update spring types as either face (default) or inplane
    #if z1 == z2
    springs['type'] = 'face'
    springs.loc[(springs.z1 == springs.z2), 'type'] = 'inplane'

    return([balls, springs])

def init_squaremesh(nrow=3,ncol=3,spring_const=1,znoise=False,mu=0,sigma=0.01):
    #dataframe of balls
    #ID, X, Y, Z, vel, Y_vel, Z_vel, acc, Y_acc, Z_acc, Neighbours, SpringsAttached.
    
    #produces ID for a particle based on row and column number
    def IDbyPosition(row,column,nrow=nrow,ncol=ncol,pattern='squaremesh'): #this currently works for only squaremesh with no 
        if row<0 or column<0:
            return(None)
        if row>=nrow or column>=ncol:
            return(None)
        if pattern=='squaremesh':
            ID=row*ncol+column

        return(int(ID))
    
    balls_colnames=['ID', 'x', 'y', 'z', 'ax', 'ay', 'az', 'neighbours', 'spring1', 'spring2','row','column']
    balls=pd.DataFrame(0, index=range(0), columns=range(len(balls_colnames)))
    balls.columns=balls_colnames
    balls.index = range(balls.shape[0])


    #dataframe of springs
    #ID, X1, Y1, Z1, X2, Y2, Z2, k, Natural length, Extension in length, Ball1, Ball2
    springs_colnames=['ID', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'k', 'l0', 'l1','dl', 'ball1', 'ball2']
    springs=pd.DataFrame(0, index=range(0), columns=range(len(springs_colnames)))
    springs.columns=springs_colnames
    
    k=1
    
    #knots help to set coordinates of balls
    knots_x=pd.DataFrame([pd.Series(np.linspace(0,ncol-1, ncol), dtype=float)]*nrow)

    #knots_y : y coordinates
    knots_y=pd.DataFrame([pd.Series(np.linspace(0,nrow-1, nrow), dtype=float)]*ncol)
    knots_y=knots_y.T

    #knots_z : z coordinates
    if znoise == True:
        knots_z=pd.DataFrame(np.random.normal(mu,sigma,size=(nrow,ncol)))
    else:
        knots_z=pd.DataFrame(0.0, index=range(nrow), columns=range(ncol)) #you can add z noise to this 
    
    #make balls dataframe and fill with ID, x, y z and neighbours (leave spaces for acc and springs)
    for i in range(nrow):
        for j in range(ncol):
            #balls_colnames=['ID', 'x', 'y', 'z', 'x_acc', 'y_acc', 'z_acc', 'neighbours', 'spring1', 'spring2']
            neighbours=[IDbyPosition(i-1,j),IDbyPosition(i+1,j),IDbyPosition(i,j-1),IDbyPosition(i,j+1)]
            neighbours=list(filter(None.__ne__, neighbours))
            row=pd.DataFrame([[IDbyPosition(i,j), knots_x.loc[i][j], knots_y.loc[i][j], knots_z.loc[i][j]]+[0.0]*3+[neighbours]+['']*2+[i,j]])
            row.columns=balls.columns
            balls=pd.concat([balls,row])
            balls.index = range(balls.shape[0])


    #now we define the springs dataframe
    #first for horizontal springs
    for i in range(nrow):
        for j in range(ncol-1):
            #springs_colnames=['ID', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'k', 'lo', 'l1','dl', 'ball1', 'ball2']
            ID=i*(ncol-1)+j
            #get ID of ball1 from i,j. From that get the x,y,z values
            ball1_ID=IDbyPosition(i,j)
            ball2_ID=IDbyPosition(i,j+1)
            row=pd.DataFrame([[ID, balls.iloc[ball1_ID]['x'], balls.iloc[ball1_ID]['y'], balls.iloc[ball1_ID]['z'], balls.iloc[ball2_ID]['x'], balls.iloc[ball2_ID]['y'], balls.iloc[ball2_ID]['z'], spring_const, 1.0, 1.0, 0.0, ball1_ID, ball2_ID]])
            #in the above assignment of lo and l1, we can start using exact computed lengths instead of putting 1
            row.columns=springs.columns
            springs=pd.concat([springs,row])
            springs.index = range(springs.shape[0])

    #next for vertical springs
    for i in range(nrow-1):
        for j in range(ncol):
            #springs_colnames=['ID', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'k', 'lo', 'l1','dl', 'ball1', 'ball2']
            ID=(ncol-1)*nrow+i*ncol+j
            #get ID of ball1 from i,j. From that get the x,y,z values
            ball1_ID=IDbyPosition(i,j)
            ball2_ID=IDbyPosition(i+1,j)
            row=pd.DataFrame([[ID, balls.iloc[ball1_ID]['x'], balls.iloc[ball1_ID]['y'], balls.iloc[ball1_ID]['z'], balls.iloc[ball2_ID]['x'], balls.iloc[ball2_ID]['y'], balls.iloc[ball2_ID]['z'], spring_const, 1.0, 1.0, 0.0, ball1_ID, ball2_ID]])
            #in the above assignment of lo and l1, we can start using exact computed lengths instead of putting 1
            row.columns=springs.columns
            springs=pd.concat([springs,row])
            springs.index = range(springs.shape[0])

    #For completion, we now want to add the connected springs for each ball in the balls dataframe
    #Note that we can very easily write a code which is specific to this arrangement
    #However, we want to do this in a generalized manner, that is this part can be used irrespective of the setting of the springs and balls

    for i in range(balls.shape[0]):
        ID=balls.loc[i,'ID'] #i and ID are the same here anyway #start using better indexing
        #find the row numbers for which the springs have ball1 as ID
        balls.at[i,'spring1']=list(springs.loc[springs.ball1==ID,'ID']) #since we want to add an array to the cell, it is better to use .at than .loc
        #find the row numbers for which the springs have ball2 as ID
        balls.at[i,'spring2']=list(springs.loc[springs.ball2==ID,'ID']) #since we want to add an array to the cell, it is better to use .at than .loc

    centre=[int(nrow/2),int(ncol/2)] #check if there is a ball at that place
    centre_ball=balls[(balls.row==centre[0]) & (balls.column==centre[1])]
    [centre_ballID]=centre_ball.ID.values
    balls.loc[:,['x','y','z']]=balls.loc[:,['x','y','z']]-balls.loc[centre_ballID,['x','y','z']]

    return([balls,springs])

def init_trianglemesh(nrow=3,ncol=5,spring_const=1,znoise=False,mu=0,sigma=0.01,thickness=2, base='flat', thin=False,viscoelastic_coeff=0.5):
    
    if thin:
        stacks=[0]
    else:
        stacks=[0,1]

    def idbp(i,j,k,nrow=nrow,ncol=ncol): #IDbyPosition
        nballs=int(nrow*ncol/2) #number of balls on one sheet
        if i>=nrow or j>=ncol or i<0 or j<0 or k<0 or k>1:
            return(None)
        if thin and k!=0: #If thin is true then k can only be 0
            return(None)
        if ncol%2==1:
            return(int((i*ncol+j-1)/2)+k*nballs)  # think about this
        if ncol%2==0:
            return(int((i*ncol+j)/2)+k*nballs)
            
    
    balls_colnames=['ID', 'x', 'y', 'z', 'ax', 'ay', 'az', 'neighbours', 'spring1', 'spring2','row','column','stack', 'vx', 'vy', 'vz', 'mass']
    balls=pd.DataFrame(0, index=range(0), columns=range(len(balls_colnames)))
    balls.columns=balls_colnames
    balls.index = range(balls.shape[0])


    #dataframe of springs
    #ID, X1, Y1, Z1, X2, Y2, Z2, k, Natural length, Extension in length, Ball1, Ball2
    springs_colnames=['ID', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'k', 'l0', 'l1','dl', 'ball1', 'ball2','type','viscoelastic_coeff']
    springs=pd.DataFrame(0, index=range(0), columns=range(len(springs_colnames)))
    springs.columns=springs_colnames
    
    
    pattern=np.zeros((nrow,ncol))
    
    # this whole odd even thing can be made very simple, in two lines, 10 lines of code not needed for this
    even=[x for x in range(nrow) if x%2==0]
    odd=[x for x in range(ncol) if x%2==1]
    even1=[]
    for i in range(len(even)):
        even1+=[even[i]]*len(odd)
    odd1=odd*len(even)
    #for nrow, ncol = 5, even1 - > [0, 0, 2, 2, 4, 4], odd1 -> [1, 3, 1, 3, 1, 3]

    even=[x for x in range(ncol) if x%2==0]
    odd=[x for x in range(nrow) if x%2==1]
    even2=[]
    for i in range(len(even)):
        even2+=[even[i]]*len(odd)
    odd2=odd*len(even)
    
    even3=even1+odd2
    odd3=odd1+even2
    pattern[even3,odd3]=1
    #pattern has ones in places where nodes are
    ######################

    spring_ID=-1
    for k in stacks: #if thin then stacks = [0] else stacks = [0,1]
        for i in range(nrow):
            for j in range(ncol):
                if pattern[i,j]==0:
                    continue
                ID=idbp(i,j,k)
                neighbours=[idbp(i-1,j-1,k), idbp(i-1,j+1,k), idbp(i-1,j-1,k-1), idbp(i-1,j+1,k-1), idbp(i-1,j-1,k+1), idbp(i-1,j+1,k+1),
                            idbp(i,j-2,k), idbp(i,j+2,k), idbp(i,j-2,k-1), idbp(i,j+2,k-1), idbp(i,j-2,k+1), idbp(i,j+2,k+1),
                            idbp(i+1,j-1,k), idbp(i+1,j+1,k), idbp(i+1,j-1,k-1), idbp(i+1,j+1,k-1), idbp(i+1,j-1,k+1), idbp(i+1,j+1,k+1), 
                            idbp(i,j,k-1), idbp(i,j,k+1)]
                neighbours=list(filter(None.__ne__, neighbours))

                #balls_colnames=['ID', 'x', 'y', 'z', 'ax', 'ay', 'az', 'neighbours', 'spring1', 'spring2','row','column','stack', 'vx', 'vy', 'vz', 'mass']
                row=pd.DataFrame([[ID]+[j,i*math.sqrt(3),k*thickness]+[0]*3+[neighbours]+['']*2+[i,j,k]+[0]*3+[1]])
                row.columns=balls.columns
                balls=pd.concat([balls,row])
                balls.index = range(balls.shape[0])
                
                #we add springs to only some of the neighbours to avoid adding the same spring again from the other end
                neighbour1s=[idbp(i,j+2,k),idbp(i+1,j-1,k),idbp(i+1,j+1,k),idbp(i,j,k+1), 
                             idbp(i,j+2,k+1),idbp(i+1,j-1,k+1),idbp(i+1,j+1,k+1), 
                             idbp(i,j+2,k-1),idbp(i+1,j-1,k-1),idbp(i+1,j+1,k-1)]
                types=['inplane','inplane','inplane','edge',
                      'face','face','face',
                      'face','face','face']
                indices=[x for x in range(len(neighbour1s)) if not(neighbour1s[x]==None)]
                
                neighbour1s=[neighbour1s[x] for x in indices]
                type1s=[types[x] for x in indices]
                for x in range(len(neighbour1s)):
                    #add spring
                    #springs_colnames=['ID', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'k', 'lo', 'l1','dl', 'ball1', 'ball2','type']
                    spring_ID=spring_ID+1
                    row=pd.DataFrame([[spring_ID]+[0]*6+[spring_const]+[0]*3+[ID,neighbour1s[x]]+[type1s[x]]+[viscoelastic_coeff]])
                    row.columns=springs.columns
                    springs=pd.concat([springs,row])
                    springs.index = range(springs.shape[0])
    
    #centre=[int(nrow/2),int(ncol/2)] #check if there is a ball at that place
    #flag=1 #indicates that centre does not correspond to a ball
    #centre_ballID=-999
    #while flag:
    #    centre_ball=balls.query('row=='+str(centre[0])+' and column==' +str(centre[1]) + ' and stack==0')
    #    if len(centre_ball)==0:
    #        centre[1]+=1 #shift to the next column
    #    else:
    #        [centre_ballID]=centre_ball.ID.values #now it will give us two values which we don't want so we should do something about it.
    #        flag=0
    #        
    #balls.loc[:,['x','y','z']]=balls.loc[:,['x','y','z']]-balls.loc[centre_ballID,['x','y','z']]

    origin_x=(max(balls['x'])+min(balls['x']))/2
    origin_y=(max(balls['y'])+min(balls['y']))/2
    
    #subtract from each ball the coordinates of center
    balls.loc[:,['x']]=balls.loc[:,['x']]-origin_x
    balls.loc[:,['y']]=balls.loc[:,['y']]-origin_y
    
    springs=update_springs(springs,balls.loc[:,['x','y','z']],compute_lo=True)
    
    for i in range(balls.shape[0]): #optimize this
        ID=balls.loc[i,'ID'] #i and ID are the same here anyway #start using better indexing
        #find the row numbers for which the springs have ball1 as ID
        balls.at[i,'spring1']=list(springs.loc[springs.ball1==ID,'ID']) #since we want to add an array to the cell, it is better to use .at than .loc
        #find the row numbers for which the springs have ball2 as ID
        balls.at[i,'spring2']=list(springs.loc[springs.ball2==ID,'ID']) #since we want to add an array to the cell, it is better to use .at than .loc

    if znoise == True:
        noise=pd.DataFrame(abs(np.random.normal(mu,sigma,size=(len(balls),1))),columns=['z'],index=range(len(balls)))
        balls['z']=balls['z']+noise['z']
    return([balls,springs])

def init_trianglemesh_circularboundary(r_thres=None,nrow=3,ncol=None,spring_const=1,znoise=False,mu=0,sigma=0.01,thickness=2, base='soft', thin=False,viscoelastic_coeff=0.5):
    
    if ncol==None: # if not ncol not input, we try to make a square sheet first and then draw a circle inside it.
        ncol=int(nrow*1.73) #distance between rows is 1.73 (root of 3) units and distance between rows is 1 unit
        ncol= ncol if ncol%2!=0 else ncol+1 #just making ncol odd otherwise some errors arise (not yet debugged)

    [balls_df,springs_df]=init_trianglemesh(nrow=nrow,ncol=ncol,spring_const=spring_const,znoise=znoise,mu=mu,sigma=sigma,thickness=thickness,base=base,thin=thin,viscoelastic_coeff=viscoelastic_coeff)
    balls_df['r']=(balls_df.x**2+balls_df.y**2)**0.5
    if r_thres==None:
        r_thres=nrow*np.sqrt(3)/2 - 0.005
        
    #identifying balls outside the circle
    #can do this since coordinates of the balls are given relative to the middle of the framne (point 0,0)
    ball_indices=balls_df.index[balls_df.r>r_thres]
    ball_ids=balls_df.ID[ball_indices]
    #now delete the balls which have ids as ball_ids
    balls_df=balls_df.drop(ball_indices)

    #now delete those rows in springs which have ids in ball_1 or ball_2
    #get the ids of these springs as well
    spring_indices=springs_df.index[(springs_df.ball1.isin(ball_ids)) | (springs_df.ball2.isin(ball_ids))]
    spring_ids=springs_df.ID[spring_indices]
    springs_df=springs_df.drop(spring_indices)

    #in the balls dataframe, remove these springs if they are present in spring1 or spring2 for any ball
    balls_df.spring1=balls_df.apply(lambda row : [item for item in row['spring1'] if item not in spring_ids], axis=1)
    balls_df.spring2=balls_df.apply(lambda row : [item for item in row['spring2'] if item not in spring_ids], axis=1)
    #in the balls dataframe, remove those neighbours whare present in ball_ids
    balls_df.neighbours=balls_df.apply(lambda row : [item for item in row['neighbours'] if not(item in ball_ids)], axis=1)
    
    #we will have to change the ids of the balls and the springs
    #so we make a dictionary, keys are old IDs and values are new IDs
    
    #ball_dict
    balls_df.reset_index(drop=True, inplace=True)
    key=balls_df.ID
    value=balls_df.index
    ball_dict=dict(zip(key, value))
    #spring_dict
    springs_df.reset_index(drop=True, inplace=True)
    key=springs_df.ID
    value=springs_df.index
    spring_dict=dict(zip(key, value))

    #replacing old ids with new ids in balls_df
    balls_df.ID=balls_df.apply(lambda row : ball_dict[row['ID']], axis=1)
    balls_df.neighbours=balls_df.apply(lambda row : [ball_dict[item] for item in row['neighbours']], axis=1)
    balls_df.spring1=balls_df.apply(lambda row : [spring_dict[item] for item in row['spring1']], axis=1)
    balls_df.spring2=balls_df.apply(lambda row : [spring_dict[item] for item in row['spring2']], axis=1)
    
    
    #replacing old ids with new ids in springs_df
    springs_df.ID=springs_df.apply(lambda row : spring_dict[row['ID']], axis=1)
    springs_df.ball1=springs_df.apply(lambda row : ball_dict[row['ball1']], axis=1)
    springs_df.ball2=springs_df.apply(lambda row : ball_dict[row['ball2']], axis=1)
    
    return([balls_df, springs_df])

def init_trianglemesh_circularboundary_cropped(r_thres=None,nrow=3,ncol=None,spring_const=1,znoise=False,mu=0,sigma=0.01,thickness=2, base='soft', thin=False,viscoelastic_coeff=0.5):
    
    if ncol==None: # if not ncol not input, we try to make a square sheet first and then draw a circle inside it.
        ncol=int(nrow*1.73) #distance between rows is 1.73 (root of 3) units and distance between rows is 1 unit
        ncol= ncol if ncol%2!=0 else ncol+1 #just making ncol odd otherwise some errors arise (not yet debugged)

    [balls_df,springs_df]=init_trianglemesh(nrow=nrow,ncol=ncol,spring_const=spring_const,znoise=znoise,mu=mu,sigma=sigma,thickness=thickness,base=base,thin=thin,viscoelastic_coeff=viscoelastic_coeff)
    balls_df['r']=(balls_df.x**2+balls_df.y**2)**0.5
    if r_thres==None:
        r_thres=nrow*np.sqrt(3)/2 - 0.005
    
    
    #removing center
    #get center ball ID
    #get ID of neighbours
    #add those IDs to ball_indices
    centre=[int(nrow/2),int(ncol/2)] #check if there is a ball at that place
    flag=1 #indicates that centre does not correspond to a ball
    centre_ballID=-999
    while flag:
        centre_ball=balls_df.query('row=='+str(centre[0])+' and column==' +str(centre[1]) + ' and stack==0')
        if len(centre_ball)==0:
            centre[1]+=1 #shift to the next column
        else:
            [centre_ballID]=centre_ball.ID.values #now it will give us two values which we don't want so we should do something about it.
            flag=0
            
    ball_indices=balls_df.index[balls_df.ID.isin((balls_df.neighbours[centre_ballID]+[centre_ballID]))]
    
    #identifying balls outside the circle
    ball_indices=list(ball_indices)+list(balls_df.index[balls_df.r>r_thres])
    ball_indices=list(set(ball_indices))
    ball_ids=balls_df.ID[ball_indices]
    #now delete the balls which have ids as ball_ids
    balls_df=balls_df.drop(ball_indices)

    #now delete those rows in springs which have ids in ball_1 or ball_2
    #get the ids of these springs as well
    spring_indices=springs_df.index[(springs_df.ball1.isin(ball_ids)) | (springs_df.ball2.isin(ball_ids))]
    spring_ids=springs_df.ID[spring_indices]
    springs_df=springs_df.drop(spring_indices)

    #in the balls dataframe, remove these springs if they are present in spring1 or spring2 for any ball
    balls_df.spring1=balls_df.apply(lambda row : [item for item in row['spring1'] if item not in spring_ids], axis=1)
    balls_df.spring2=balls_df.apply(lambda row : [item for item in row['spring2'] if item not in spring_ids], axis=1)
    #in the balls dataframe, remove those neighbours whare present in ball_ids
    balls_df.neighbours=balls_df.apply(lambda row : [item for item in row['neighbours'] if not(item in ball_ids)], axis=1)
    
    #we will have to change the ids of the balls and the springs
    #so we make a dictionary, keys are old IDs and values are new IDs
    
    #ball_dict
    balls_df.reset_index(drop=True, inplace=True)
    key=balls_df.ID
    value=balls_df.index
    ball_dict=dict(zip(key, value))
    #spring_dict
    springs_df.reset_index(drop=True, inplace=True)
    key=springs_df.ID
    value=springs_df.index
    spring_dict=dict(zip(key, value))

    #replacing old ids with new ids in balls_df
    balls_df.ID=balls_df.apply(lambda row : ball_dict[row['ID']], axis=1)
    balls_df.neighbours=balls_df.apply(lambda row : [ball_dict[item] for item in row['neighbours']], axis=1)
    balls_df.spring1=balls_df.apply(lambda row : [spring_dict[item] for item in row['spring1']], axis=1)
    balls_df.spring2=balls_df.apply(lambda row : [spring_dict[item] for item in row['spring2']], axis=1)
    
    
    #replacing old ids with new ids in springs_df
    springs_df.ID=springs_df.apply(lambda row : spring_dict[row['ID']], axis=1)
    springs_df.ball1=springs_df.apply(lambda row : ball_dict[row['ball1']], axis=1)
    springs_df.ball2=springs_df.apply(lambda row : ball_dict[row['ball2']], axis=1)
    
    return([balls_df, springs_df])
    
def initialize_triangles_info(balls,springs):
    #### TO DO 
    # Add a triangles column to springs which tells which triangles a spring is part of 
    import networkx as nx
    #first we get graph from dataframes
    G=dftoGraph(balls,springs)
    #next we identify all the triangles
    triangles_df =get_polygons(G=G, solid=False)
    #next we calculate the edge lengths and areas of all the triangles
    triangles_df = update_triangles(polygons=triangles_df, G=G)
    #next we add a column to balls which records the triangles of which it is a part
    balls['triangles']=None
    for index,row in balls.iterrows():
        ball_id=balls.loc[index,'ID']
        triangles_df['bool']=triangles_df.apply(lambda row: ball_id in row['vertices'],axis=1)
        balls.at[index, 'triangles']=list(triangles_df.index[triangles_df['bool']])
    #next for each ball we calculate the 1/3 * sum of the areas of triangles of which it is a part
    balls['area']=balls.apply(lambda row: (np.sum(triangles_df.area[row['triangles']]))/3, axis=1)
    
    return([balls,springs])

def init_mesh(nrow=3,ncol=3,pattern='trianglemesh',triangle_areas=False,spring_const=1,znoise=False,mu=0,sigma=0.01,thickness=2, base='flat', thin=False,viscoelastic_coeff=0.5, r_thres=None):
    #this function generates a new mesh
    if pattern=='cubicmesh':
        [balls,springs]=init_cubicmesh(nrow=nrow,ncol=ncol,spring_const=spring_const,znoise=znoise,thickness=thickness,base=base,thin=thin,viscoelastic_coeff=viscoelastic_coeff)
    if pattern=='squaremesh':
        [balls,springs]=init_squaremesh(nrow=nrow,ncol=ncol,spring_const=spring_const,znoise=znoise) #add thickness to the square mesh
    if pattern=='trianglemesh':
        [balls,springs]=init_trianglemesh(nrow=nrow,ncol=ncol,spring_const=spring_const,znoise=znoise,thickness=thickness,base=base,thin=thin,viscoelastic_coeff=viscoelastic_coeff)
    if pattern=='circularboundary':
        [balls,springs]=init_trianglemesh_circularboundary(r_thres=r_thres,nrow=nrow,ncol=ncol,spring_const=spring_const,znoise=znoise,thickness=thickness,base=base,thin=thin,viscoelastic_coeff=viscoelastic_coeff)
    if pattern=='croppedcenter':
        [balls,springs]=init_trianglemesh_circularboundary_cropped(r_thres=r_thres,nrow=nrow,ncol=ncol,spring_const=spring_const,znoise=znoise,thickness=thickness,base=base,thin=thin,viscoelastic_coeff=viscoelastic_coeff)
    if triangle_areas:
        [balls,springs]=initialize_triangles_info(balls,springs)
    return([balls,springs])

def map_to_surface(balls, nrow=3, ncol=3):

    balls_x=pd.DataFrame(0.0, index=range(nrow), columns=range(ncol))
    balls_y=pd.DataFrame(0.0, index=range(nrow), columns=range(ncol))
    balls_z=pd.DataFrame(0.0, index=range(nrow), columns=range(ncol))
    for i in range(len(balls)):
        ID=balls.loc[i,'ID']
        row=balls.loc[ID,'row']
        column=balls.loc[ID,'column']
        balls_x.loc[row,column]=balls.loc[ID,'x']
        balls_y.loc[row,column]=balls.loc[ID,'y']
        balls_z.loc[row,column]=balls.loc[ID,'z']    
        
    return([balls_x,balls_y,balls_z])

def plot_surface(balls,nrow=3,ncol=3):
    
    fig=plt.figure()
    ax = plt.axes(projection='3d')
    #df=map_to_surface(balls)
    #df['balls_x']
    [balls_x,balls_y,balls_z]=map_to_surface(balls,nrow=nrow,ncol=ncol)
    ax.plot_surface(balls_x.values, balls_y.values, balls_z.values, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    print(balls_x)
    print(balls_y)
    print(balls_z)
    
def plot_base(balls,nrow=3,ncol=3):

    [balls_x,balls_y,balls_z]=map_to_surface(balls,nrow=nrow,ncol=ncol)
    balls_z=pd.DataFrame(0.0, index=range(nrow), columns=range(ncol))
    #print(balls_z_copy)

    fig=plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(balls_x.values, balls_y.values, balls_z.values, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def components_springforce(springs):
    total_f=-1*springs.k*springs.dl
    v=[springs.x2-springs.x1,springs.y2-springs.y1,springs.z2-springs.z1]
    norm=v[0]**2+v[1]**2+v[2]**2
    normed=norm.apply(lambda row: math.sqrt(row)) #normalised norm
    v=pd.DataFrame(v).T
    v=v.divide(normed,axis=0)
    f_components=v.mul(total_f,axis=0)
    
    return(f_components)

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
        
def compute_forces(balls,springs,tol=1e-4,dt=0.01,nrow=3,ncol=3,k=1,print_=False):
    #run the following script unless all the balls feel almost zero forces on them
    ncycle=0
    avg_disp=9999
    ball_positions=balls.loc[:,['x','y','z']]
    ball_positions.columns=range(3) # we can either change this or change every dataframe's colums to 'x','y','z'
    springs=update_springs(springs,balls.loc[:,['x','y','z']]) #because extension may not always involving changing the lengths of the springs

    while(avg_disp>tol):  #this loop cannot be reduced
        ncycle=ncycle+1
        if(print_):
            print('ncycle : '+str(ncycle))
        
        #get the components of each force 
        spring_f_components=components_springforce(springs) #O()
        
        #get the component of forces for each ball
        ball_f_components=balls.apply(lambda row: pd.DataFrame.sum(spring_f_components.loc[row['spring2'],:])-pd.DataFrame.sum(spring_f_components.loc[row['spring1'],:]),axis=1)

        #try this as well : don't make the output inside apply to dataframe and just transpose the numpy array
        ball_positions=ball_positions+dt*ball_f_components
        
        #probably we can make the code faster if we do not need to update the springs dataframe everytime
        springs=update_springs(springs, ball_positions) #if balls is updated before that then just send balls.loc[:,['x','y','z']]
        
        avg_disp=springs['dl'].abs().mean()*k*dt #this is of the same order of magnitude as the sum of mod of displacements of the balls
        if(print_):
            print('avg_disp : '+str(avg_disp))
        
        #break #for testing purposes
        
    ball_positions.columns=['x','y','z']
    balls.loc[:,['x','y','z']]=ball_positions

    return([balls,springs]) #returning a dictionary such that dictionary['balls']=balls and dictionary['springs']=springs 

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


def polygons_square(balls, springs, nrow, ncol):
    
    polygons_colnames=['ID', 'vertices', 'edges', 'Nbvertices'] #add Nb_neighbors, area, cell centre
    polygons=pd.DataFrame(0, index=range(0), columns=range(len(polygons_colnames)))
    polygons.columns=polygons_colnames
    polygons.index = range(polygons.shape[0])
    
    size=4 #size of each polygon
    
    polygon_id=-1
    for i in range(nrow-1):
        for j in range(ncol-1):
            polygon_id+=1
            ID=int(i*ncol+j) #so that floats dont come in the file
            vertices=[ID, ID+1, ID+ncol+1, ID+ncol]
            row=pd.DataFrame([polygon_id, vertices, [], size]).T
            #add to polygons
            row.columns=polygons.columns
            polygons=pd.concat([polygons,row])
            polygons.index = range(polygons.shape[0])
            
    return(polygons)

'''
def polygons_triangle(balls, springs, nrow, ncol):
    def idbp(row, column, stack): #IDbyPosition
        # return value for query comes in a weird format (ID_dirty), need to clean that
        ID_dirty=balls.query('row=='+str(row)+' and column==' +str(column) + ' and stack=='+str(stack)).ID.values 
        if len(ID_dirty)==0:
            return(None)
        [ID]=ID_dirty
        return(ID)
    
    polygons_colnames=['ID', 'vertices', 'edges', 'Nbvertices','stack'] #add Nb_neighbors, area, cell centre
    polygons=pd.DataFrame(0, index=range(0), columns=range(len(polygons_colnames)))
    polygons.columns=polygons_colnames
    polygons.index = range(polygons.shape[0])
    size=3 #size of each polygon
    
    polygon_id=-1
    for i in range(len(balls)):
        ID=int(balls.loc[i, 'ID']) #why do I need to add int in this line? the balls dataframe already has ids in int
        row=balls.loc[i, 'row']
        column=balls.loc[i, 'column']
        stack=balls.loc[i,'stack']
        neighbours=[idbp(row+1,column-1,stack), idbp(row+1,column+1,stack), idbp(row, column+2,stack)]
        neighbours=list(filter(None.__ne__, neighbours))
        if len(neighbours)<2: #a ball at the boundary of the mesh
            continue
        
        #adding the first polygon
        #make a query

        if neighbours[0] in balls.loc[balls.ID==neighbours[1], 'neighbours'].values[0]: #making sure polygon is made only if the three balls are neighbours of each other

            vertices=[ID,neighbours[1],neighbours[0]] #check if neighbours[1] and neighbours[0] are neighbours or not
            polygon_id+=1
            row=pd.DataFrame([polygon_id, vertices, [], size, stack]).T
            #add to polygons
            row.columns=polygons.columns
            polygons=pd.concat([polygons,row])
            polygons.index = range(polygons.shape[0])
        
        #adding the second polygon
        if len(neighbours)==3:
            if neighbours[1] in balls.loc[balls.ID==neighbours[2], 'neighbours'].values[0]: #making sure polygon is made only if the three balls are neighbours of each other

                vertices=[ID,neighbours[2],neighbours[1]] #check if neighbours[2] and neighbours[1] are neighbours or not
                polygon_id+=1
                row=pd.DataFrame([polygon_id, vertices, [], size,stack]).T
                #add to polygons
                row.columns=polygons.columns
                polygons=pd.concat([polygons,row])
                polygons.index = range(polygons.shape[0])
            
    return(polygons)

''' 

def polygons_triangle(balls, springs, nrow, ncol):
    #get a network from balls and springs 
    G=nx.Graph()
    G.add_nodes_from(list(balls.ID))
    G.add_edges_from(list(zip(springs.ball1, springs.ball2)))
    
    #get each triangle
    tri=nx.triangles(G)
    all_cliques= nx.enumerate_all_cliques(G)
    triad_cliques=[x for x in all_cliques if len(x)==3 ]
    
    #preparing the polygons dataframe
    polygons=pd.DataFrame({
            'vertices':triad_cliques,
            'Nbvertices':[3]*len(triad_cliques)
            })
    
    return(polygons)

'''
def get_polygons(balls, springs, pattern='squaremesh', nrow=3, ncol=3):
    
    #polygons.columns=['ID', 'vertices', 'edges', 'Nbvertices']
    
    if pattern=='squaremesh':
        polygons=polygons_square(balls, springs, nrow, ncol)
    elif pattern=='trianglemesh':
        polygons=polygons_triangle(balls, springs, nrow, ncol)
    
    return(polygons)
'''

#THIS
def integrate_springs(balls_df,springs_df,k=1,t_intervals=1000,tol=1e-4,verbose=False, base='soft'):
    
    #converting to numpy arrays to make calculations faster
    balls=balls_df[['x','y','z']].values
    springs=springs_df[['l0','l1','ball1','ball2','k']].values
    #getting ball-springs connections 
    balls_sp1=[]
    balls_sp2=[]
    for i in range(len(balls_df)):
        balls_sp1=balls_sp1+[balls_df.spring1[i]]
        balls_sp2=balls_sp2+[balls_df.spring2[i]]
    balls_sp1=np.array(balls_sp1) #an array of lists - ith row gives a list of springs in which ball i participates as the 1st ball
    balls_sp2=np.array(balls_sp2) #an array of lists - ith row gives a list of springs in which ball i participates as the 2st ball

    #using the solve_ivp function 
    #documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
    def comp_dballs(t,balls):#differential of the balls array
        balls=np.reshape(balls, (-1,3)) #-1 is the unspecified value which it calculates by itself
        v=balls[springs[:,2].astype(int)]-balls[springs[:,3].astype(int)] #springs has to be a global variable 
        #think of a way to avoid using springs,balls_sp1 and balls_sp2 as a global variable
        v_mod=np.sqrt(np.sum(v**2,axis=1).astype(np.float64))
        springs[:,1]=v_mod #this is the current length of the springs
        v_cap=v/v_mod[:,None] #dividing a 2d array by a 1d array, unit vector in direction of the spring
        f=(springs[:,4])[:,None]*(springs[:,1]-springs[:,0])[:,None]*v_cap #spring const * change in length * unit vector of force #springs[:,4] is spring const
        dballs=np.array([np.sum(f[balls_sp2[x],:],axis=0)-np.sum(f[balls_sp1[x],:],axis=0) for x in range(len(balls))]) #for each ball, change in displacement = the sum of forces from all the springs that the ball participates in
        
        ###########
        #Adding a hard surface below the sheet
        #How - for all coordinates where z would become negative, change them to 0 
        ###########
        if base=='hard':
            dballs[balls[:,2]<-dballs[:,2],2]=-balls[balls[:,2]<-dballs[:,2],2] 
        
        dballs=np.reshape(dballs,-1)
        return(dballs)
    
    #integrate the differential eqn
    balls_sol =solve_ivp(fun=comp_dballs, t_span=[0,1000], y0=np.reshape(balls,-1)) #need to reshape because this function only takes one dimensional inputs
    balls=np.reshape(balls_sol.y[:,-1],(-1,3)) #taking the last column of the time series
    dballs=comp_dballs(0,balls)
    avgdisp=np.absolute(dballs).mean()
    if verbose:
        print("Avg displacement : "+str(avgdisp))

    #update springs before returning
    v=balls[springs[:,2].astype(int)]-balls[springs[:,3].astype(int)] #springs has to be a global variable 
    v_mod=np.sqrt(np.sum(v**2,axis=1).astype(np.float64))
    springs[:,1]=v_mod #this is the current length of the springs

    balls_df[['x','y','z']]=balls
    springs_df['l1']=springs[:,1]
    
    #try to get the time series as well
    return([balls_df,springs_df,balls_sol])

def gaussian_surface(array, A=1, sigma_x=1, sigma_y=1):
    array=array.astype(np.float64)
    return(A*np.exp(-(array[:,0]**2/sigma_x**2+array[:,1]**2/sigma_y**2)))

def inverse_gaussian(array, A=0.5, sigma_x=1, sigma_y=1):
    array=array.astype(np.float64)
    return(-A*np.exp(-(array[:,0]**2/sigma_x**2+array[:,1]**2/sigma_y**2)))

def extend_springs(balls, springs, spring_types, f=gaussian_surface, A=1, sigma_x=1, sigma_y=1):
    #v = for each spring get coordinates of ball1 - coordinates of ball2
    v=balls[springs[:,2].astype(int)]-balls[springs[:,3].astype(int)]
    #v_mod contains lengths of springs
    v_mod=np.sqrt(np.sum(v**2,axis=1).astype(np.float64))
    #set l1 to length of springs
    springs[:,1]=v_mod
    #v_cap is a normalised vector
    v_cap=v/v_mod[:,None]

    for i in range(len(springs)):
        #to cause extension only in the xy planes and keeping the thickness of the sheet constant
        #we will avoid those iterations where the direction of the spring is perpendicular to x and y directions
        #inplane to make sure extension only in xy plane
        if spring_types[i]!='inplane':
            continue
        #get spring direction
        direction=np.tile(np.array([v_cap[i]]),(10,1))
        #get ten points between 0 and length of spring
        lengths=np.array(np.linspace(0,v_mod[i],10))
        #coordinates of the 2nd ball
        points=np.array([balls[int(springs[i,3])]])
        #10x3 array of coordinates
        points=np.tile(points,(10,1))
        points=points+direction*lengths[:,None]
        #for each spring, we now get 10 points between the position of the second ball 
        #and the desired direction of extension
        #extend the desired spring length (l0) by a proportion calculated from the gaussian curvature
        springs[i,0]=(1+np.mean(f(points[:,[0,1]], A=A, sigma_x=sigma_x, sigma_y=sigma_y)))*springs[i,0]
        
    return([balls,springs])

def bimaterial(balls_df, springs_df, gamma = 0.8):
    
    """only the top half layer gets extended, while the bottom one remains unchanged"""
    
    springs_df.loc[(springs_df['type']=='inplane') & (springs_df['z1'] > 0.5*np.max(springs_df.z1)), 'l0'] *= (gamma)
    springs_df=modify_diagonal_lengths(balls_df, springs_df)
    
    return(springs_df)

def radial_extension(x,y,gamma=0.6,nu=0.3569):
    return(gamma**(-nu))

def tangential_extension(x,y,gamma=0.6,nu=-1):
    return(gamma**(-nu))



def perp_direction(gamma=0.6,nu=0.3569):
    return(gamma**(-nu))

def nem_direction(gamma=0.6):
    return(gamma)

def nematic_extension(balls_df, springs_df, gamma=0.6,nu=0.3569, thin = False):
    #for a good cone, keep gamma=0.6 and nu=0.3569
    #for the best anticone I have till now, keep gamma=1.2 and nu=0.81705
    
    balls=balls_df[['x','y','z']].values
    springs=springs_df[['l0','l1','ball1','ball2','k']].values
    spring_types=springs_df['type'].values 
    
    v=balls[springs[:,2].astype(int)]-balls[springs[:,3].astype(int)]
    v_mod=np.sqrt(np.sum(v**2,axis=1).astype(np.float64))
    springs[:,1]=v_mod
    v_cap=v/v_mod[:,None]
    #spring midpoints
    mid_points=(balls[springs[:,2].astype(int)]+balls[springs[:,3].astype(int)])/2 
    mid_points[:,2]=0
    mid_points_mod=np.sqrt(np.sum(mid_points**2,axis=1).astype(np.float64))
    mid_points_mod[np.where(mid_points_mod==0)]=1 #hacky way to handle those points for which modulus of vector is zero on putting z=0
    radial_directions=mid_points/mid_points_mod[:,None]
    
    rad_ext_array=radial_extension(mid_points[:,0],mid_points[:,1],gamma=gamma,nu=nu) #for a cone, uncomment this line
    print(rad_ext_array)
    tang_ext_array=tangential_extension(mid_points[:,0],mid_points[:,1],gamma=gamma) #for a cone, uncomment this line
    print(tang_ext_array)
    
    #rad_ext_array=radial_extension(mid_points[:,0],mid_points[:,1],gamma=gamma) #for an anticone, uncomment this line
    #tang_ext_array=tangential_extension(mid_points[:,0],mid_points[:,1],gamma=gamma,nu=nu) #for an anticone, uncomment this line
    
    #vector component in the radial direcion - gets extended/contracted by whatever is the radial extension
    cos_theta=np.absolute((radial_directions*v_cap).sum(axis=1))
    #vector component in the tangential direction - gets extended/contracted by whatever is the tangential extension
    sin_theta=np.absolute(np.sqrt(np.abs((1-cos_theta**2)).astype(np.float64)))
    
    if(sum(np.isnan(sin_theta)) > 0):
        print(np.where(np.isnan(sin_theta)))
        print(cos_theta[np.isnan(sin_theta)])
        print(1-cos_theta[np.isnan(sin_theta)]**2)
    
    #desired extension is calculated as the length of vector with cos_theta and sin_theta components
    ext=np.sqrt((cos_theta*rad_ext_array)**2+(sin_theta*tang_ext_array)**2)
    #ext=ext+1
    ext=ext[:,None]
    #spring length multiplied by extension
    springs[:,0]=springs[:,0]*ext[:,0]
    
    balls_df[['x','y','z']]=balls
    
    #only the inplane extensions calculated
    springs_df['inplane_ext']=springs[:,0]
    
    #diagonal spring length is calculated by taking the inplane extension component and its height
    #easier than the update_diagonal_lengths function used previously
    springs_df['l0'] = np.sqrt(np.square(springs_df.inplane_ext)+np.square(springs_df.z1-springs_df.z2))

    
    return(springs_df)

#keep for reference, update_diagonal_springs used here
def nematic_extension_abhijeet(balls_df, springs_df, gamma=0.6,nu=0.3569, thin = False):
    #for a good cone, keep gamma=0.6 and nu=0.3569
    #for the best anticone I have till now, keep gamma=1.2 and nu=0.81705
    
    balls=balls_df[['x','y','z']].values
    springs=springs_df[['l0','l1','ball1','ball2','k']].values
    spring_types=springs_df['type'].values 
    
    v=balls[springs[:,2].astype(int)]-balls[springs[:,3].astype(int)]
    v_mod=np.sqrt(np.sum(v**2,axis=1).astype(np.float64))
    springs[:,1]=v_mod
    v_cap=v/v_mod[:,None]
    #spring midpoints
    mid_points=(balls[springs[:,2].astype(int)]+balls[springs[:,3].astype(int)])/2 
    mid_points[:,2]=0
    mid_points_mod=np.sqrt(np.sum(mid_points**2,axis=1).astype(np.float64))
    mid_points_mod[np.where(mid_points_mod==0)]=1 #hacky way to handle those points for which modulus of vector is zero on putting z=0
    radial_directions=mid_points/mid_points_mod[:,None]
    rad_ext_array=radial_extension(mid_points[:,0],mid_points[:,1],gamma=gamma,nu=nu) #for a cone, uncomment this line
    tang_ext_array=tangential_extension(mid_points[:,0],mid_points[:,1],gamma=gamma) #for a cone, uncomment this line
    #rad_ext_array=radial_extension(mid_points[:,0],mid_points[:,1],gamma=gamma) #for an anticone, uncomment this line
    #tang_ext_array=tangential_extension(mid_points[:,0],mid_points[:,1],gamma=gamma,nu=nu) #for an anticone, uncomment this line
    cos_theta=np.absolute((radial_directions*v_cap).sum(axis=1))
    sin_theta=np.absolute(np.sqrt((1-cos_theta**2).astype(np.float64)))
    ext=cos_theta*rad_ext_array+sin_theta*tang_ext_array
    #ext=ext+1
    ext=ext[:,None]
    indices=np.where(spring_types=='inplane')
    springs[indices,0]=springs[indices,0]*ext[indices,0]
    
    balls_df[['x','y','z']]=balls
    springs_df['l0']=springs[:,0]
    
    if thin==False:
        print('modifying diagonal lengths')
        springs_df=modify_diagonal_lengths(balls_df,springs_df)

    #return([balls_df,springs_df])
    return(springs_df)


def square_nematic_extension(balls_df, springs_df, gamma = 0.6, nu_perp = 0.3569, nu = -1, thin = False):
    
    """here the director pattern is concentric squares"""
    
    #identify positions of balls within the square
    #either in the top and bottom triangle, where director is aligned horizontally - balls marked with 'h'
    #or the left/right one where director is aligned vertically - balls marked with 'v'
    #balls which are sufficiently close to the x=y or x=-y lines are marked as edge 'e'
    balls_df['pos'] = np.where((balls_df.x > -balls_df.y) & (balls_df.x < balls_df.y), 'h', 'v')
    balls_df.loc[(balls_df.x < -balls_df.y) & (balls_df.x > balls_df.y), 'pos'] = 'h'
    
    s = min(springs_df[springs_df.type == 'inplane'].l1)
    balls_df.loc[(abs(balls_df.x + balls_df.y) < s/2) | (abs(balls_df.x - balls_df.y) < s/2), 'pos'] = 'e'
    
    #find springs connecting balls that are horizontal, vertical and at the edges
    #spring is 'vertical' if it connects two vertical or a vertical and an edge spring 
    #these get extended according to vertical director pattern
    #vice versa for horizontal springs
    #edge springs connect two 'edge' balls and do not get extended
    h_bool = (balls_df.iloc[springs_df.ball1].pos == 'h').values & (balls_df.iloc[springs_df.ball2].pos == 'h').values
    eh_bool = (balls_df.iloc[springs_df.ball1].pos == 'e').values & (balls_df.iloc[springs_df.ball2].pos == 'h').values
    he_bool = (balls_df.iloc[springs_df.ball1].pos == 'h').values & (balls_df.iloc[springs_df.ball2].pos == 'e').values
    h_bool = (eh_bool | he_bool | h_bool)


    v_bool = (balls_df.iloc[springs_df.ball1].pos == 'v').values & (balls_df.iloc[springs_df.ball2].pos == 'v').values
    ve_bool = (balls_df.iloc[springs_df.ball1].pos == 'v').values & (balls_df.iloc[springs_df.ball2].pos == 'e').values
    ev_bool = (balls_df.iloc[springs_df.ball1].pos == 'e').values & (balls_df.iloc[springs_df.ball2].pos == 'v').values
    v_bool = (v_bool | ve_bool | ev_bool)
    
    
    ####extension
    balls=balls_df[['x','y','z']].values
    springs=springs_df[['l0','l1','ball1','ball2','k']].values
    spring_types=springs_df['type'].values 

    #v holds vectors for each spring
    v=balls[springs[:,2].astype(int)]-balls[springs[:,3].astype(int)]
    #only interested in x and y coordinates, as in the end we only consider in-plane springs
    v = v[:,0:2]
    
    nem = nem_direction(gamma)
    perp = perp_direction(gamma, nu_perp)
    
    #calculate the reshaped vectors
    #for areas where director is vertical:
    #the vertical component of each vector (v*[0,1]) gets extended/contracted according to nematic extension/contraction
    #while the horizontal component (v*[1,0]) gets extended/contracted according to the perpendicular extension/contraction
    v_vertical_nematic = v*[1,0]*perp + v*[0,1]*nem
    #vice versa for horizontal alignemnt of director
    v_horizontal_nematic = v*[1,0]*nem + v*[0,1]*perp

    #calculate the lengths of resulting vectors
    vert_nem_lo = np.sqrt(v_vertical_nematic[:,0]**2 + v_vertical_nematic[:,1]**2)
    hor_nem_lo = np.sqrt(v_horizontal_nematic[:,0]**2 + v_horizontal_nematic[:,1]**2)
    
    h_indices=np.where(h_bool)
    v_indices=np.where(v_bool)

    #update the desired lengths of the springs of both the horizontal and vertical springs
    springs[h_indices,0]= hor_nem_lo[h_indices]
    springs[v_indices,0]= vert_nem_lo[v_indices]


    balls_df[['x','y','z']]=balls
    
    #the calculated change in length is in-plane only
    springs_df['inplane_ext']=springs[:,0]
    #for diagonal springs, calculate the desired length by taking the in-plane component 
    #and vertical component (which remains unchanged)
    springs_df['l0'] = np.sqrt(np.square(springs_df.inplane_ext)+np.square(springs_df.z1-springs_df.z2))
    
    return(springs_df)

def modify_diagonal_lengths(balls_df,springs_df):

	#for each ball in the top surface:
	#	
	#	Get edge that has ball1 or ball2 as the 'ball' and type is edge
	#	Save length of edge as lv and the other ball as ball_vertical
	# 	Get all edges that have ball1 or ball2 as the 'ball' and type is face
	#	For all edges:
	#		Get the ball1 if ball1 is not the ball1, else ball2. This is called ball_diagonal
	#		Get length of edge between ball_vertical and ball_diagonal as l_face
	#		Assign rt(l_face^2+lv^2) to length of edge
	#	return([balls,springs])

    for i in range(0,len(balls_df)):
        if balls_df['stack'][i]!=0:
            continue
        ball=balls_df['ID'][i] #same as i btw
        connected_springs=springs_df.loc[(springs_df['ball1']==ball) | (springs_df['ball2']==ball)]
        edge_spring=connected_springs.index[connected_springs['type']=='edge'].tolist()[0]
        ball_vertical=springs_df.ball1[edge_spring]+springs_df.ball2[edge_spring]-ball
        l_vertical=springs_df.l0[edge_spring] #vertical length
        face_springs=connected_springs.index[connected_springs['type']=='face'].tolist() #getting all the diagonal springs
        
        for j in range(0, len(face_springs)):
            ball_diagonal=springs_df.ball1[face_springs[j]]+springs_df.ball2[face_springs[j]]-ball
            #find the lo of spring connecting ball_vertical and ball_diagonal
            l_inplane=float(springs_df.l0[((springs_df['ball1']==ball_vertical) & (springs_df['ball2']==ball_diagonal)) | 
                           ((springs_df['ball2']==ball_vertical) & (springs_df['ball1']==ball_diagonal))])
            springs_df.loc[face_springs[j],'l0']=math.sqrt(l_vertical**2+l_inplane**2)

    return(springs_df)

def radial_like_extension(x,y,gamma=0.5):
    return(gamma)

def tangential_like_extension(x,y,gamma=-0.5):
    return(gamma)

def nematic_like_extension(balls_df, springs_df, gamma=-0.4,gamma_v=0.2, thin=False):
    #for a good cone, keep gamma=-0.4 and gamma_v=0.2
    #for a good anticone, keep gamma=-0.2 and gamma_v=-0.2

    balls=balls_df[['x','y','z']].values
    springs=springs_df[['l0','l1','ball1','ball2','k']].values
    spring_types=springs_df['type'].values 

    v=balls[springs[:,2].astype(int)]-balls[springs[:,3].astype(int)]
    v_mod=np.sqrt(np.sum(v**2,axis=1).astype(np.float64))
    springs[:,1]=v_mod
    v_cap=v/v_mod[:,None]
    mid_points=(balls[springs[:,2].astype(int)]+balls[springs[:,3].astype(int)])/2 
    mid_points[:,2]=0
    mid_points_mod=np.sqrt(np.sum(mid_points**2,axis=1).astype(np.float64))
    mid_points_mod[np.where(mid_points_mod==0)]=1 #hacky way to handle those points for which modulus of vector is zero on putting z=0
    radial_directions=mid_points/mid_points_mod[:,None]
    rad_ext_array=radial_like_extension(mid_points[:,0],mid_points[:,1],gamma=gamma_v)
    tang_ext_array=tangential_like_extension(mid_points[:,0],mid_points[:,1],gamma=gamma)
    cos_theta=np.absolute((radial_directions*v_cap).sum(axis=1))
    sin_theta=np.absolute(np.sqrt((1-cos_theta**2).astype(np.float64)))
    ext=cos_theta*rad_ext_array+sin_theta*tang_ext_array
    ext=ext+1
    ext=ext[:,None]
    indices=np.where(spring_types=='inplane')
    springs[indices,0]=springs[indices,0]*ext[indices,0]

    balls_df[['x','y','z']]=balls
    springs_df['l0']=springs[:,0]

    if thin==False:
        springs_df=modify_diagonal_lengths(balls_df,springs_df)
    
    return(springs_df)

def uniform_extension(balls_df, springs_df, gamma=0.2, thin=False):
    springs_df.loc[springs_df['type']=='inplane', 'l0']*=(1+gamma)
    if thin==False:
        springs_df=modify_diagonal_lengths(balls_df, springs_df)
    return(springs_df)

def r2_extension(balls_df, springs_df, gamma=-0.01, thin=False, debug = False):
    
    #In this function, we will shrink the material in tangential direction depending on r^2
    
    #first arrange balls_df by the ball_ID
    old_balls_index = balls_df.index
    old_springs_index = springs_df.index
    
    balls_df = balls_df.sort_values(by=['ID'])
    balls_df = balls_df.reset_index(drop = True)
    springs_df = springs_df.reset_index(drop = True)
    old_names = list(balls_df.ID)
    new_names = list(range(len(balls_df)))
    mapping = dict(zip(old_names, new_names))
    balls_df['ID'] = new_names
    new_spring_names = list(range(len(springs_df)))
    old_spring_names = list(springs_df['ID'])
    springs_df['ID'] = new_spring_names
    #print(new_names)
    springs_df['ball1'] = springs_df.apply(lambda row: mapping[row['ball1']], axis = 1)
    springs_df['ball2'] = springs_df.apply(lambda row: mapping[row['ball2']], axis = 1)
    
    #make the mapping 
    #change the ball1 and ball2 in springs
    #get the index for springs
    #reset index for balls and springs

    balls=balls_df[['x','y','z']].values
    springs=springs_df[['l0','l1','ball1','ball2','k']].values
    spring_types=springs_df['type'].values 

    v=balls[springs[:,2].astype(int)]-balls[springs[:,3].astype(int)]
    v_mod=np.sqrt(np.sum(v**2,axis=1).astype(np.float64))
    springs[:,1]=v_mod
    v_cap=v/v_mod[:,None]
    mid_points=(balls[springs[:,2].astype(int)]+balls[springs[:,3].astype(int)])/2 
    mid_points[:,2]=0
    #calculating distance from the center
    mid_points_mod=np.sqrt(np.sum(mid_points**2,axis=1).astype(np.float64)) 
    
    mid_points_mod[np.where(mid_points_mod==0)]=1 #hacky way to handle those points for which modulus of vector is zero on putting z=0
    radial_directions=mid_points/mid_points_mod[:,None]
    
    #this will be changed
    #rad_ext_array=radial_like_extension(mid_points[:,0],mid_points[:,1],gamma=gamma_v)
    rad_ext_array=mid_points_mod*0
    #this will be changed
    #tang_ext_array=tangential_like_extension(mid_points[:,0],mid_points[:,1],gamma=gamma)
    tang_ext_array=gamma*(mid_points_mod**2)
    
    cos_theta=np.absolute((radial_directions*v_cap).sum(axis=1))
    #rounding important because sometimes cos theta calculated as 1.00000000000002 which would give NAN for sin_theta
    cos_theta=np.round(list(cos_theta),10)
    sin_theta=np.absolute(np.sqrt((1-cos_theta**2).astype(np.float64)))
    sin_theta=np.round(list(sin_theta),10) #rounding sin_theta as well to make it consistent with cos_theta
    ext=cos_theta*rad_ext_array+sin_theta*tang_ext_array
    ext=ext+1
    ext=ext[:,None]
    indices=np.where(spring_types=='inplane')
    springs[indices,0]=springs[indices,0]*ext[indices,0]
    
    
    
    springs_df['l0']=springs[:,0]
    
    if thin==False:
        springs_df=modify_diagonal_lengths(balls_df,springs_df)
        
    springs_df.index = old_springs_index
    springs_df.ID = old_spring_names
    
    
    #print(balls_df)
    #print(springs_df)
    mapping = dict(zip(new_names, old_names))
    
    #change the ball1 and ball2 by previous mapping
    springs_df['ball1'] = springs_df.apply(lambda row: mapping[row['ball1']], axis = 1)
    springs_df['ball2'] = springs_df.apply(lambda row: mapping[row['ball2']], axis = 1)
    
    #reset index for springs
    return(springs_df)

def gradient_tang_radial(balls_df, springs_df, gamma_tang=-0.01, gamma_rad=0.5, thin=False, dependence_rad='uniform', dependence_tang='r2'):
    
    #In this function, we will shrink the material in tangential direction depending on r^2

    balls=balls_df[['x','y','z']].values
    springs=springs_df[['l0','l1','ball1','ball2','k']].values
    spring_types=springs_df['type'].values 

    v=balls[springs[:,2].astype(int)]-balls[springs[:,3].astype(int)]
    v_mod=np.sqrt(np.sum(v**2,axis=1).astype(np.float64))
    springs[:,1]=v_mod
    v_cap=v/v_mod[:,None]
    mid_points=(balls[springs[:,2].astype(int)]+balls[springs[:,3].astype(int)])/2 
    mid_points[:,2]=0
    #calculating distance from the center
    mid_points_mod=np.sqrt(np.sum(mid_points**2,axis=1).astype(np.float64)) 
    
    mid_points_mod[np.where(mid_points_mod==0)]=1 #hacky way to handle those points for which modulus of vector is zero on putting z=0
    radial_directions=mid_points/mid_points_mod[:,None]
    
    #this will be changed
    #rad_ext_array=radial_like_extension(mid_points[:,0],mid_points[:,1],gamma=gamma_v)
    if dependence_rad=='uniform':
        rad_ext_array=gamma_rad
    if dependence_rad=='r':
        rad_ext_array=gamma_rad*(mid_points_mod)
    if dependence_rad=='r2':
        rad_ext_array=gamma_rad*(mid_points_mod**2)
    #this will be changed
    #tang_ext_array=tangential_like_extension(mid_points[:,0],mid_points[:,1],gamma=gamma)
    if dependence_tang=='uniform':
        rad_ext_array=gamma_tang
    if dependence_tang=='r':
        rad_ext_array=gamma_tang*(mid_points_mod)
    if dependence_tang=='r2':
        rad_ext_array=gamma_tang*(mid_points_mod**2)
    
    cos_theta=np.absolute((radial_directions*v_cap).sum(axis=1))
    #rounding important because sometimes cos theta calculated as 1.00000000000002 which would give NAN for sin_theta
    cos_theta=np.round(list(cos_theta),10)
    sin_theta=np.absolute(np.sqrt((1-cos_theta**2).astype(np.float64)))
    sin_theta=np.round(list(sin_theta),10) #rounding sin_theta as well to make it consistent with cos_theta
    ext=cos_theta*rad_ext_array+sin_theta*tang_ext_array
    ext=ext+1
    ext=ext[:,None]
    indices=np.where(spring_types=='inplane')
    springs[indices,0]=springs[indices,0]*ext[indices,0]

    balls_df[['x','y','z']]=balls
    springs_df['l0']=springs[:,0]

    if thin==False:
        springs_df=modify_diagonal_lengths(balls_df,springs_df)
    
    return(springs_df)


def WriteImage(fileName, renWin, rgba=True):
    """
    Write the render window view to an image file.

    Image types supported are:
     BMP, JPEG, PNM, PNG, PostScript, TIFF.
    The default parameters are used for all writers, change as needed.

    :param fileName: The file name, if no extension then PNG is assumed.
    :param renWin: The render window.
    :param rgba: Used to set the buffer type.
    :return:
    """

    import os
    import vtk

    if fileName:
        # Select the writer to use.
        path, ext = os.path.splitext(fileName)
        ext = ext.lower()
        if not ext:
            ext = '.png'
            fileName = fileName + ext
        if ext == '.bmp':
            writer = vtk.vtkBMPWriter()
        elif ext == '.jpg':
            writer = vtk.vtkJPEGWriter()
        elif ext == '.pnm':
            writer = vtk.vtkPNMWriter()
        elif ext == '.ps':
            if rgba:
                rgba = False
            writer = vtk.vtkPostScriptWriter()

        elif ext == '.tiff':
            writer = vtk.vtkTIFFWriter()
        else:
            writer = vtk.vtkPNGWriter()

        windowto_image_filter = vtk.vtkWindowToImageFilter()
        windowto_image_filter.SetInput(renWin)
        windowto_image_filter.SetScale(1)  # image quality
        if rgba:
            windowto_image_filter.SetInputBufferTypeToRGBA()
        else:
            windowto_image_filter.SetInputBufferTypeToRGB()
            # Read from the front buffer.
            windowto_image_filter.ReadFrontBufferOff()
            windowto_image_filter.Update()

        writer.SetFileName(fileName)
        writer.SetInputConnection(windowto_image_filter.GetOutputPort())
        writer.Write()
    else:
        raise RuntimeError('Need a filename.')

def vtk_to_images(source_file_list=None, source_folder=None, ext=['', '.png', '.jpg', '.ps', '.tiff', '.bmp', '.pnm'], sortby=None,verbose=False,img_pattern=None,dest_folder='', skipby=0):
        
    import glob
    import os
    import vtk

    
    vtk_filenames=[]
    
    #collecting names of all vtk files 
    if source_folder != None:
        #add all the vtk files from this folder into files
        vtk_filenames = vtk_filenames + [f for f in glob.glob(source_folder+"*.vtk")]
        
    if source_file_list != None:
        vtk_filenames=vtk_filenames+source_file_list

    
    #sort files by some number 
    if sortby != None:
        sorter=np.array([float(x.replace(sortby[0],'.').split('.')[sortby[1]]) for x in vtk_filenames])
        #get indices of the sorted form of sorter array
        sorted_ind=np.argsort(sorter)
    else:
    	sorted_ind=range(len(vtk_filenames))
    
    if verbose:
        print(vtk_filenames)

    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    
    counter=0 #in case img_pattern is given
    
    #for v in vtk_filenames: #for each vtk file
    for i in range(len(vtk_filenames)): #for each vtk file
        
        # script adapted from:
        # for rendering paraview : https://vtk.org/Wiki/VTK/Examples/Python/vtkUnstructuredGridReader
        # for writing image : https://lorensen.github.io/VTKExamples/site/Python/IO/ImageWriter/

        #Skip files if skipby > 0 
        if skipby>0:
            if i%skipby != 0:
                continue

        v=vtk_filenames[sorted_ind[i]]

        # Read the source file.
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(v)
        reader.Update() # Needed because of GetScalarRange
        output = reader.GetOutput()
        scalar_range = output.GetScalarRange()
        
        # Create the mapper that corresponds the objects of the vtk file
        # into graphics elements
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(output)
        mapper.SetScalarRange(scalar_range)
        
        # Create the Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.RotateX(-70.0) #we can add these in the input to the function
        actor.RotateY(-10.0)
        actor.RotateZ(20.0)
        
        # Create the Renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        #renderer.SetBackground(1, 1, 1) # Set background to white #we can add these in the input to the function
        
        # Create the RendererWindow
        renderer_window = vtk.vtkRenderWindow()
        renderer_window.OffScreenRenderingOn()
        renderer_window.AddRenderer(renderer)
        renderer_window.Render()
        
        # Creating new image name
        v=dest_folder+v.split('/')[-1] #dest_folder/vtk_filename.vtk
        if img_pattern == None: #then the image takes the same name as the vtk file
            img_filenames = list(map(lambda x: v.split('.vtk')[0] + x, ext))
        else: #for storing images in a patterned way which helps in making the movie
            #if you are making the movie later, it is better to use the img_pattern in the get_movie function
            img_filenames = list(map(lambda x: dest_folder+'/'+img_pattern+str(counter) + x, ext))
        
        for f in img_filenames:
            #for each extension that you want out of ['', '.png', '.jpg', '.ps', '.tiff', '.bmp', '.pnm']
            WriteImage(f, renderer_window, rgba=False)
                
        counter+=1
        
def images_to_movie(source_folder=None, dest_folder='', framerate=1, moviename='movie.mp4',sortby=None, img_pattern='img', ext='.png', ):
    
    import os
    import numpy as np
    import glob
    import shutil
    
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
        
    tempfolder=dest_folder[:-1]+'temp/'
    if not os.path.exists(tempfolder):
    	os.mkdir(tempfolder)
    
    #You need to use sortby if the image names don't strictly follow the *1,*2,*3.. pattern

    if (sortby != None) and (source_folder != None):
        #sortby should be of the form [separator, position]
        #for example for thickness_0.6_15_31_seed_0.vtk, sortby=['_', 1]
        
        #get filelist from source_folder
        filelist = [f for f in glob.glob(source_folder+"*"+ext)]

        sorter=np.array([float(x.replace(sortby[0],'.').split('.')[sortby[1]]) for x in filelist])
        #get indices of the sorted form of sorter array
        sorted_ind=np.argsort(sorter)
        for i in range(len(sorted_ind)):
            file=filelist[sorted_ind[i]]
            new_name='img'+str(i)+ext #ext is also file.split('.')[-1]
            #rename the file with this new name 
            #os.rename(file, dest_folder+new_name)
            shutil.copyfile(file, tempfolder+new_name)
            
        img_pattern='img'
        source_folder=tempfolder

    os.system("ffmpeg -r " + str(framerate)+" -i " + source_folder + img_pattern + "%d" + ext + " -vcodec mpeg4 -y " + dest_folder + moviename)
    shutil.rmtree(tempfolder) #deleting temp directory
    
def vtk_to_movie(source_file_list=None, source_folder=None, framerate=3, moviename='movie.mp4',verbose=False,img_pattern='img',img_dest_folder='images/', mov_dest_folder=None, ext='.png', skipby=0,del_img_folder=True, sortby=None):
    
    import shutil
    import os

    if os.path.exists(img_dest_folder): # if this folder exists and contains images then it can mess up with making the video
    	shutil.rmtree(img_dest_folder)

    #convert vtk files to images
    vtk_to_images(source_folder=source_folder, dest_folder=img_dest_folder, ext=[ext], 
    skipby=skipby, sortby = None)
    #vtk_to_images(source_folder=source_folder, dest_folder=img_dest_folder, ext=[ext], 
    #skipby=skipby, verbose=verbose, img_pattern=img_pattern) #ext in array because the function has the option to make images for different extensions simultaneously

    #combine images to a movie
    images_to_movie(source_folder=img_dest_folder, dest_folder=mov_dest_folder, moviename=moviename, 
              framerate=framerate, ext=ext, img_pattern=img_pattern, sortby=None)
    #images_to_movie(source_folder=img_dest_folder, dest_folder=mov_dest_folder, moviename=moviename, 
    #          framerate=framerate, ext=ext, sortby=sortby, img_pattern=img_pattern)

    if del_img_folder: #this avoids having too much data in the project folder
        shutil.rmtree(img_dest_folder)

    if verbose:
        print('movie stored as '+moviename)
    
#vtk_to_movie(source_folder='test/vtk_files/',dest_folder='test/images/',ext='.jpg',moviename='movie.mp4', sortby=['_',1])


def get_vtk_timeseries_movie(balls_df,springs_df,time_series,vtk_dest_folder='vtk_timeseries/', mov_dest_folder='movie/', vtk_name_pattern=None, moviename='movie.mp4', nrow=3, ncol=5, pattern='trianglemesh', del_vtk_folder=True, ext='.png',framerate=800, skipby=100, verbose=False, sortby=None, makemovie=True, viscoelastic=True, frac_time=1):
    
    #for each timepoint:
    #    convert into df
    #    convert that df into vtk
    #    save that vtk file inside a folder with patterned names
    #send the folder to vtk_to_video()
    
    import os
    import shutil
    
    if os.path.exists(vtk_dest_folder):
        shutil.rmtree(vtk_dest_folder) #if a folder already exists, it can be confusing for further processing    
    os.mkdir(vtk_dest_folder)    

    if not os.path.exists(mov_dest_folder):
        os.mkdir(mov_dest_folder)
        
    springs=springs_df[['l0','l1','ball1','ball2','k']].values #converted to numpy just because some codes that I copied this from were in numpy
    
    counter=0
    
    for i in range(int(time_series.y.shape[1]*frac_time)):
        
        if skipby>0:
            if i%skipby!=0:
                continue

        #ith state of balls
        #treat viscoelastic and elastic cases differently because time_series is different for both
        if viscoelastic:
            balls_i=np.reshape(time_series.y[:,i],(-1,6))
            balls_i=balls_i[:,0:3]
        else:
            balls_i=np.reshape(time_series.y[:,i],(-1,3))
    
        #ith state of springs
        springs_i=springs
        v=balls_i[springs[:,2].astype(int)]-balls_i[springs[:,3].astype(int)] #v is spring vector
        v_mod=np.sqrt(np.sum(v**2,axis=1).astype(np.float64)) #v_mod is current length of spring
        springs_i[:,1]=v_mod
        
        #converting to df
        balls_df[['x','y','z']]=balls_i
        springs_df['l1']=springs_i[:,1]
        
        #convert df to vtk
        if vtk_name_pattern==None:
            vtk_name_pattern='timepoint'
        dfToVtk(balls_df, springs_df,filename=vtk_dest_folder+vtk_name_pattern+str(counter)+'.vtk', 
                nrow=nrow, ncol=ncol, pattern=pattern)
        counter+=1
        
    if makemovie == False:
        return

    vtk_to_movie(source_folder=vtk_dest_folder, mov_dest_folder=mov_dest_folder, ext=ext, 
            moviename=moviename, sortby=sortby, img_pattern=vtk_name_pattern, framerate=framerate,
                del_img_folder=False)
        
    if del_vtk_folder:
            shutil.rmtree(vtk_dest_folder)
            
    if verbose:
            print('movie saved as '+mov_dest_folder+moviename)

#get_vtk_timeseries_movie(folded_balls_df,folded_springs_df,time_series,nrow=nrow,ncol=ncol,del_vtk_folder=False,skipby=1000,vtk_dest_folder='vtk_timeseries/', mov_dest_folder='movie/', vtk_name_pattern='timepoint',moviename='movie_skipped_1000.mp4',framerate=1,verbose=False, sortby=['point',1])

def integrate_kv(balls_df,springs_df,k=1,t_intervals=1000,t_span=100, tol=1e-4,verbose=False, base='soft'):
    #kelvin-voigt elements 
    #converting to numpy arrays to make calculations faster

    balls=np.concatenate((balls_df[['x','y','z']].values,balls_df[['vx','vy','vz']].values), axis=1)
    springs=springs_df[['l0','l1','ball1','ball2','k','viscoelastic_coeff']].values
    #springs[:,5]=0 to make the springs completely elastic
    #getting ball-springs connections 
    balls_sp1=[]
    balls_sp2=[]
    for i in range(len(balls_df)):
        balls_sp1=balls_sp1+[balls_df.spring1[i]]
        balls_sp2=balls_sp2+[balls_df.spring2[i]]
    balls_sp1=np.array(balls_sp1) #an array of lists
    balls_sp2=np.array(balls_sp2) #an array of lists
    
    #using the solve_ivp function 
    #documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
    def comp_dballs(t,balls):#differential of the balls array
        #print('i',i)
        balls=np.reshape(balls, (-1,6)) #-1 is the unspecified value which it calculates by itself 
        ball_pos= balls[:,0:3]#xyz positions of balls
        #print('ball_pos', ball_pos[0,0], ball_pos[1,0])
        ball_vel= balls[:,3:]#velocities of balls
        #print('ball_vel', ball_vel[0,0], ball_vel[1,0])
        #calculating strain in springs
        sp_vector=ball_pos[springs[:,2].astype(int)]-ball_pos[springs[:,3].astype(int)] #spring vectors #springs array is a global variable here
        #print('sp_vector',sp_vector)
        sp_vector_mod=np.sqrt(np.sum(sp_vector**2,axis=1).astype(np.float64))
        #print('sp_vector_mod',sp_vector_mod)
        springs[:,1]=sp_vector_mod #this is the current length of the springs
        sp_vector_cap=sp_vector/sp_vector_mod[:,None]
        #print('sp_vector_cap',sp_vector_cap)
        sp_strain=springs[:,1]-springs[:,0] #spring strain is current length - natural length
        #print('sp_strain', sp_strain)
        #calculating strain rate in springs
        sp_strain_rate=np.sum((ball_vel[springs[:,2].astype(int)]-ball_vel[springs[:,3].astype(int)])*(sp_vector_cap), axis=1) 
        #print('sp_strain_rate with wrong signs',sp_strain_rate)
        #assigning correct sign to strain rate
        #sp_strain_rate=sp_strain_rate*np.sign(sp_strain) #if the length of the spring is increasing, then strain rate is negative only if current length of spring is smaller than the natural length
        #print('sp_strain_rate',sp_strain_rate)
        #calculating force for each spring
        #f=k*strain + eta*strain_rate
        f=((springs[:,4])[:,None]*(sp_strain)[:,None]+(springs[:,5])[:,None]*(sp_strain_rate)[:,None])*sp_vector_cap #spring const * change in length * unit vector of force #springs[:,4] is spring const
        #print('f', f[0])
        #calculating the change in the velocities of the balls because of this force
        d_vel=np.array([np.sum(f[balls_sp2[x],:],axis=0)-np.sum(f[balls_sp1[x],:],axis=0) for x in range(len(balls))])/balls_df.mass[:,None]
        #print(i, '\t' ,ball_pos[0,0], '\t' ,ball_pos[1,0], '\t' ,ball_vel[0,0], '\t' ,ball_vel[1,0], '\t' ,sp_strain, '\t' ,sp_strain_rate, '\t' ,f[0,0])
        #print('d_vel', d_vel)
        #finally, calculating the change in positions of the balls by moving them in the direction of their previous velocities
        d_pos=ball_vel
        #print('d_pos', d_pos)
        #Adding a hard surface below the sheet
        if base=='hard':
            d_pos[balls[:,2]<-d_pos[:,2],2]=-balls[balls[:,2]<-d_pos[:,2],2] 

        #combine ball_pos and ball_vel into balls
        dballs=np.reshape(np.concatenate((d_pos,d_vel), axis=1), -1)
        return(dballs)
    
    balls_sol =solve_ivp(fun=comp_dballs, t_span=[0,t_span], y0=np.reshape(balls,-1)) #need to reshape because this function only takes one dimensional inputs
    balls=np.reshape(balls_sol.y[:,-1],(-1,6)) #taking the last column of the time series
    ball_pos= balls[:,0:3]
    ball_vel= balls[:,3:]
    dballs=comp_dballs(0,balls)
    dballs=np.reshape(dballs, (-1,6))
    avgdisp=np.absolute(dballs[:,0:3]).mean()
    if verbose:
        print("Avg displacement : "+str(avgdisp))

    #update springs before returning
    sp_vector=ball_pos[springs[:,2].astype(int)]-ball_pos[springs[:,3].astype(int)]
    sp_vector_mod=np.sqrt(np.sum(sp_vector**2,axis=1).astype(np.float64))
    springs[:,1]=sp_vector_mod #this is the current length of the springs

    balls_df[['x','y','z','vx','vy','vz']]=balls
    springs_df['l1']=springs[:,1]
    
    #change the way of handling in vtk_timeseries functions
    
    return([balls_df,springs_df,balls_sol])

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

def get_length(pt1, pt2): 
    import numpy as np
    return np.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1])+(pt2[2]-pt1[2])*(pt2[2]-pt1[2]))

def triangle_area(a,b,c, mode = 'length'):
    
    import numpy as np
    
    if mode == 'length':
        l1=a
        l2=b
        l3=c
        
    if mode == 'coordinates':
        #get l1 l2 l3
        l1 = get_length(a,b)
        l2 = get_length(b,c)
        l3 = get_length(c,a)
        
    p=(l1+l2+l3)/2
    
    return(np.sqrt(p*(p-l1)*(p-l2)*(p-l3)))
    
def get_prism_volume(pt1, pt2, pt3):
    
    #assuming the base is the z=0 plane
    
    pt1_base = [pt1[0], pt1[1], 0]
    pt2_base = [pt2[0], pt2[1], 0]
    pt3_base = [pt3[0], pt3[1], 0]
    base_area = triangle_area(pt1_base, pt2_base, pt3_base, mode = 'coordinates')
    
    avg_ht = (pt1[2] + pt2[2] + pt3[2])/3
    
    vol = base_area*avg_ht
    
    return(vol)

def update_triangles(polygons=None, balls=None, springs=None, G=None):
    #we can also add gaussian and mean curvature
    #normal vector,vector1,vector2,vector3, natural_area or final_area
    #area_strain
    import numpy as np
    import networkx as nx
    import pandas as pd
    #in this function, we compute properties like 
    #vector1, vector2, vector3
    #length1, length2, length3
    #area, normal vector
    
    if G is None:
        #if G is None then I assume that balls and springs has been sent as argument
        G = dftoGraph(balls,springs) 
        
    polygons['pt1']=polygons.apply(lambda row: [G.nodes[row['vertices'][0]]['x'],G.nodes[row['vertices'][0]]['y'],G.nodes[row['vertices'][0]]['z']], axis = 1)
    polygons['pt2']=polygons.apply(lambda row: [G.nodes[row['vertices'][1]]['x'],G.nodes[row['vertices'][1]]['y'],G.nodes[row['vertices'][1]]['z']], axis = 1)
    polygons['pt3']=polygons.apply(lambda row: [G.nodes[row['vertices'][2]]['x'],G.nodes[row['vertices'][2]]['y'],G.nodes[row['vertices'][2]]['z']], axis = 1)
        
    polygons['projected_volume']=polygons.apply(lambda row: get_prism_volume(row['pt1'], row['pt2'], row['pt3']), axis=1)

    polygons['length1']=polygons.apply(lambda row: G[row['vertices'][0]][row['vertices'][1]]['l1'], axis=1)
    polygons['length2']=polygons.apply(lambda row: G[row['vertices'][1]][row['vertices'][2]]['l1'], axis=1)
    polygons['length3']=polygons.apply(lambda row: G[row['vertices'][2]][row['vertices'][0]]['l1'], axis=1)
    
    def get_vector(node1,node2):
        spring=G[node1][node2]
        return([spring['x2']-spring['x1'],spring['y2']-spring['y1'],spring['z2']-spring['z1']])
        
    polygons['vector1']=polygons.apply(lambda row: get_vector(row['vertices'][0],row['vertices'][1]), axis=1)
    polygons['vector2']=polygons.apply(lambda row: get_vector(row['vertices'][1],row['vertices'][2]), axis=1)
    polygons['vector3']=polygons.apply(lambda row: get_vector(row['vertices'][2],row['vertices'][0]), axis=1)
    
    #calculating area of triangle
    vect_triangle_area=np.vectorize(triangle_area)
    polygons['area']=vect_triangle_area(polygons['length1'],polygons['length2'],polygons['length3'])
    polygons['normal']=polygons.apply(lambda row: np.cross(row['vector1'], row['vector2']), axis = 1)
    
    return(polygons)
    
def get_box(min_x=-1,min_y=-1,max_x=1,max_y=1,min_z=0, max_z=2,filename='box.vtk'):
    
    balls=pd.DataFrame({'ID':list(range(8)),
                        'x':([min_x]*2+[max_x]*2)*2,
                        'y':([max_y]+[min_y]*2+[max_y])*2,
                        'z':[min_z]*4+[max_z]*4})
    springs=pd.DataFrame({'ID':list(range(12)),
                          'ball1':[0,0,0,1,1,2,2,3,4,4,5,6,],
                          'ball2':[1,3,4,2,5,3,6,7,7,5,6,7]})
    polygons=pd.DataFrame({'ID':list(range(6)),
                           'vertices':[[0,1,2,3],
                                       [4,5,6,7],
                                       [0,3,7,4],
                                       [0,1,5,4],
                                       [1,2,6,5],
                                       [2,6,7,3]],
                           'Nbvertices':[4]*6})
    dfToVtk(balls,springs,pattern='squaremesh',filename=filename,polygons=polygons)
    
    return([balls,springs])
    
def add_triangle_prop(text_vtk, prop_data, prop_name='prop', first=False):
    if first == True:
        nb_triangles=len(prop_data)
        text_vtk=text_vtk+"CELL_DATA "+str(nb_triangles)+'\n'
        
    text_vtk=text_vtk+"SCALARS "+prop_name+" float 1\nLOOKUP_TABLE default\n"
    text_vtk=text_vtk+'\n'.join([str(v) for v in prop_data])+'\n'
    
    
    return(text_vtk)


def get_vtk_timeseries_w_curvature(balls_df,springs_df,time_series,vtk_dest_folder='vtk_timeseries/', mov_dest_folder='movie/', vtk_name_pattern=None, moviename='movie.mp4', nrow=3, ncol=5, pattern='trianglemesh', del_vtk_folder=True, ext='.png',framerate=800, skipby=100, verbose=False, sortby=None, makemovie=True, viscoelastic=True):
    
    #for each timepoint:
    #    convert into df
    #    convert that df into vtk
    #    save that vtk file inside a folder with patterned names
    #send the folder to vtk_to_video()
    
    import os
    import shutil
    
    if os.path.exists(vtk_dest_folder):
        shutil.rmtree(vtk_dest_folder) #if a folder already exists, it can be confusing for further processing    
    os.mkdir(vtk_dest_folder)    

    if not os.path.exists(mov_dest_folder):
        os.mkdir(mov_dest_folder)
        
    springs=springs_df[['l0','l1','ball1','ball2','k']].values #converted to numpy just because some codes that I copied this from were in numpy
    
    counter=0
    
    for i in range(time_series.y.shape[1]):
        
        if skipby>0:
            if i%skipby!=0:
                continue

        #ith state of balls
        #treat viscoelastic and elastic cases differently because time_series is different for both
        if viscoelastic:
            balls_i=np.reshape(time_series.y[:,i],(-1,6))
            balls_i=balls_i[:,0:3]
        else:
            balls_i=np.reshape(time_series.y[:,i],(-1,3))
    
        #ith state of springs
        springs_i=springs
        v=balls_i[springs[:,2].astype(int)]-balls_i[springs[:,3].astype(int)] #v is spring vector
        v_mod=np.sqrt(np.sum(v**2,axis=1).astype(np.float64)) #v_mod is current length of spring
        springs_i[:,1]=v_mod
        
        #converting to df
        balls_df[['x','y','z']]=balls_i
        springs_df['l1']=springs_i[:,1]
        
        #convert df to vtk
        if vtk_name_pattern==None:
            vtk_name_pattern='timepoint'
        filename=vtk_dest_folder+vtk_name_pattern+str(counter)
        
        counter+=1
        
        def get_spring_stack(balls,springs_row):
            #print(balls['stack'][springs_row['ball1']] + balls['stack'][springs_row['ball2']])
            if (balls['stack'][springs_row['ball1']] + balls['stack'][springs_row['ball2']])==0:
                return(0)
            if (balls['stack'][springs_row['ball1']] + balls['stack'][springs_row['ball2']])==1:
                return(-9999)
            if (balls['stack'][springs_row['ball1']] + balls['stack'][springs_row['ball2']])==2:
                return(1)

        
        springs_df['stack']=springs_df.apply(lambda row: get_spring_stack(balls_df, row),axis=1)
        
        #when doing it for folded balls add the missing columns
        #some problem with stack == 1
        #balls_1=balls.loc[balls['stack']==1]
        #springs_1=springs.loc[springs['stack']==1]
        
        balls_0=balls_df.loc[balls_df['stack']==0]
        springs_0=springs_df.loc[springs_df['stack']==0]
        
        #adding z offset
        balls_0.z=balls_0.z+0.0001
        #balls_0.z=balls_0.z-0.0001
        
                
        
        #filename='elastic/init_thin'
        
        #dfToVtk(balls_1,springs_1,filename=filename+'.vtk')
        #vtk to ply function   
        #vtk_to_ply(filename+'.vtk')
        
        dfToVtk(balls_0,springs_0,filename=filename+'.vtk',lines=False)
        #vtk to ply function   
        vtk_to_ply(filename+'.vtk')
        
        
        ##########################################
        
        #-----------#
        # Importing #
        #-----------#
        #filename_without_extension="init_thin"
        #filename="init_thin.ply"
        #filename="torus.ply"
        #import scipy as sp
        #import numpy as np
        #import pandas as pd
        #import os 
        #import sys
        #import importlib
        #import matplotlib.pyplot as plt
        import MovieData_methods as MM
        import geometry_creation_methods as geometry_methods
        
        #importlib.reload(geometry_methods)
        #importlib.reload(MM)
        
        datapath = './'
        
        #Set filename of datafile.
        #filename = 'torus.ply'
        
        
        vertices, triangles = geometry_methods.load_ply_file(datapath + filename+'.ply')
        
        #reorient triangles so all normals face the same way
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
        
        #get the vtk file as a string
        with open(filename+'.vtk', 'r') as myfile:
            text_vtk = myfile.read()
            
        mod_text_vtk=text_vtk
        
        mod_text_vtk=add_triangle_prop(mod_text_vtk, list(triangles['gaussian_curvature']), prop_name="gaussian_curvature", first=True)
        
        mod_text_vtk=add_triangle_prop(mod_text_vtk, list(triangles['mean_curvature']), prop_name="mean_curvature")
        
        #save vtk modified file
        with open(filename+'.vtk', "w") as file:
                file.write(mod_text_vtk)
        
def get_graph_list(balls_df,springs_df,time_series, stack = None, skipby=100, viscoelastic=True):
    
    import copy

    all_timepoints=np.array(list(range(time_series.y.shape[1])))
    #condition= (np.mod(all_timepoints, 3)==0) & ((all_timepoints>init_timepoint)&(all_timepoints<=final_timepoint))
    condition= (np.mod(all_timepoints,skipby)==0)
    timepoints=all_timepoints[condition]
    #adding first and last timepoints
    timepoints=np.unique(np.insert(timepoints,[0,len(timepoints)],[0,time_series.y.shape[1]-1]))
    #timepoints= [26,30,40,50,60,70,80]
    Gs=[]
    #iteration
    
    springs=springs_df[['l0','l1','ball1','ball2','k']].values #converted to numpy just because some codes that I copied this from were in numpy
    
    for timepoint in timepoints:
        
        if viscoelastic:
            balls_i=np.reshape(time_series.y[:,timepoint],(-1,6))
            balls_i=balls_i[:,0:3]
        else:
            balls_i=np.reshape(time_series.y[:,timepoint],(-1,3))
    
        #ith state of springs
        springs_i=springs
        try:
            v=balls_i[springs[:,2].astype(int)]-balls_i[springs[:,3].astype(int)] #v is spring vector
        except TypeError:
            print('printing what gives TypeError')
            print('timepoint : ', str(timepoint))
            print(springs)
            break
        
        v_mod=np.sqrt(np.sum(v**2,axis=1).astype(np.float64)) #v_mod is current length of spring
        springs_i[:,1]=v_mod
        
        #converting to df
        balls_df[['x','y','z']]=balls_i
        springs_df['l1']=springs_i[:,1] #is this a repetitive step?
        springs_df=update_springs(springs_df,balls_df.loc[:,['x','y','z']])
        
        #converting df to graph
        G = dftoGraph(balls_df,springs_df)
        
        if stack is not None:
            #get subgraph of graph
            stack_nodes = set([n for n,d in G.nodes(data=True) if d['stack']==stack])
            G = G.subgraph(list(stack_nodes))

        #adding timepoint
        G.timepoint = timepoint
        
        Gs+=[copy.deepcopy(G)]
        
    return(Gs)
      
        
def plot_polygon_attributes(Gs, attribute = 'projected_volume', filename = 'plot.png', over = False, vol_box = 3036.055, ylim_top = 1, ylim_bottom = 0):
    
    xlabel = 'Time'
    ylabel = 'Vol under surface'
    y = []
    
    for G in Gs:
        
        #get polygons update is true
        triangles_df = get_polygons(G=G, solid=False, stack=0, compute_attributes = True)
        
        #if polygon_attribute is proj_volume 
        #   calculate sum
        if attribute == 'projected_volume':
            y+=[triangles_df['projected_volume'].sum()]
            
        #add to list of y values 
        
    x = [g.timepoint for g in Gs]
    x = np.divide(x, x[-1])
    y = np.array(y)
    vol_under = y
    
    if over:
        vol_over = ( vol_box - vol_under)/vol_box
        y = vol_over
        
    #plot
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    plt.grid()
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_ylim(top=ylim_top,bottom=ylim_bottom)
    
    ax.plot(x, y,color="blue",linewidth=2,linestyle='-');
    plt.savefig(filename)
        
    return([x,y])



'''Visualisation stuff'''

def mask_outside_polygon(poly_verts, ax=None):
    
    """reference:
    https://stackoverflow.com/questions/3320311/fill-outside-of-polygon-mask-array-where-indicies-are-beyond-a-circular-bounda
    """
    
    """
    Plots a mask on the specified axis ("ax", defaults to plt.gca()) such that
    all areas outside of the polygon specified by "poly_verts" are masked.  

    "poly_verts" must be a list of tuples of the verticies in the polygon in
    counter-clockwise order.

    Returns the matplotlib.patches.PathPatch instance plotted on the figure.
    """

    if ax is None:
        ax = plt.gca()

    # Get current plot limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xlim = (xlim[0] + 0.5, xlim[1] - 0.5)
    ylim = (ylim[0] + 0.5, ylim[1] - 0.5)

    # Verticies of the plot boundaries in clockwise order
    bound_verts = [(xlim[0], ylim[0]), (xlim[0], ylim[1]), 
                   (xlim[1], ylim[1]), (xlim[1], ylim[0]), 
                   (xlim[0], ylim[0])]

    # A series of codes (1 and 2) to tell matplotlib whether to draw a line or 
    # move the "pen" (So that there's no connecting line)
    bound_codes = [mpath.Path.MOVETO] + (len(bound_verts) - 1) * [mpath.Path.LINETO]
    poly_codes = [mpath.Path.MOVETO] + (len(poly_verts) - 1) * [mpath.Path.LINETO]

    # Plot the masking patch
    path = mpath.Path(bound_verts + poly_verts, bound_codes + poly_codes)
    patch = mpatches.PathPatch(path, facecolor='white', edgecolor='none', alpha = 1, zorder = 5)
    patch = ax.add_patch(patch)
    
    print(patch)

    # Reset the plot limits to their original extents
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return patch
