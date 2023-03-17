#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday April 8 2019

@author: Joris Paijmans - paijmans@pks.mpg.de

Several functions to create different geometries in different file formats.
1. polygonal mesh       -> Double Connected Edge List (DCEL)
2. DCEL                 -> Vertex model NetCDF files.
3. Voronoi tesselation  -> DCEL
4. Vertex Model NetCDF  -> Vertices + DCEL
5. Vertex Model NetCDF  -> MovieData tissue miner curvced surface shear decompostion.
"""
import numpy as np
import scipy as sp
import pandas as pd
#import netCDF4 as netcdf
import sys
import csv
import os
import pickle
import linecache

from scipy.spatial import SphericalVoronoi

import MovieData_methods as MM


dcel_table_columns = ['vertex_id', 'face_id', 'left_dbond_id', 'conj_dbond_id']

def load_obj_file(filename):
    """ Load a triangular mesh with the ASCII OBJ file format.
    
    Parameters
    ----------
    filename : string
        Path and filename to OBJ file.
        OBJ File format:
        Each vertex position is given by a row, preceded with a 'v' and 3 floats x,y and z.
        Faces (triangles) are preceded with a 'f' and 3 int with vertex ids.
    
    Returns
    -------
    vertices : pandas.DataFrame (float) Mx3
        M vertex positions ['x_pos', 'y_pos', 'z_pos']
    triangles : pandas.DataFrame (int) Nx3
        N triangles each defined by its three vertex ids.    
    
    """
    #Load obj file into Pandas dataframe.
    obj_data = pd.read_csv(filename, sep=' ', skiprows=[0], header=-1)
    #Extract vertices and faces.
    vertices  = obj_data[obj_data[0] == 'v'].copy()
    triangles = obj_data[obj_data[0] == 'f'].copy()
    #Drop column indicating data type.
    vertices.drop(0, axis = 1, inplace = True)
    triangles.drop(0, axis = 1, inplace = True)
    #Reset index and reset vertex ids such that they start at 0, not 1.
    triangles.reset_index(drop=True, inplace=True)
    triangles = triangles.astype(int)
    triangles -= 1
    #Reset column nanems.
    vertices.rename({1:'x_pos', 2:'y_pos', 3:'z_pos'}, axis=True, inplace=True)
    triangles.rename({1:'vertex_id_1', 2:'vertex_id_2', 3:'vertex_id_3'}, axis=True, inplace=True)
    
    return vertices, triangles
    
    
def load_ply_file(filename):
    """ Load a triangular mesh with the ASCII PLY file format.
    
    Parameters
    ----------
    filename : string
        Path and filename to PLY file.
        PLY File format:
        HEADER: contains at least the strings 'element vertex', 'element face', 'end_header' followed by and int
        giving the number of vertices, faces and the header ending, respectively.
        Each vertex position is given by a rows of floats x,y and z
        Each faces (triangle) is given by number of vertices (3) and the vertex ids.
    
    Returns
    -------
    vertices : pandas.DataFrame (float) Mx3
        M vertex positions ['x_pos', 'y_pos', 'z_pos']
    triangles : pandas.DataFrame (int) Nx3
        N triangles each defined by its three vertex ids.    
    
    """
    #Read file header and retrieve number of vertices, faces and linenumber where header ends.
    vertex_number_ident = 'element vertex'
    face_number_ident   = 'element face'
    header_ending_ident = 'end_header'
    for line_nbr, line in enumerate(open(filename)):

        if vertex_number_ident in line:
            vertex_number = int( line[line.find(vertex_number_ident)+len(vertex_number_ident):] )
            
        if face_number_ident in line:
            face_number = int( line[line.find(face_number_ident)+len(face_number_ident):] )
            
        if header_ending_ident in line:
            header_ending = line_nbr
            break       

    #Specify row numbers where vertex and face definitions start.
    vertex_start_lnbr = header_ending + 2
    face_start_lnbr   = vertex_start_lnbr + vertex_number
    #Load vertex positions from file.
    vertices = [list(map(float, linecache.getline(filename, lnbr).split()))
         for lnbr in range(vertex_start_lnbr, vertex_start_lnbr + vertex_number)]
    vertices = np.array(vertices)
    #Load face defintions from file.
    faces = [list(map(int, linecache.getline(filename, lnbr).split()))
         for lnbr in range(face_start_lnbr, face_start_lnbr + face_number)]
    faces = np.array(faces)
    #Remove first colums from faces, assert all faces are triangles.
    assert len(faces.shape) == 2, print('Not all loaded faces have the same vertex number')
    faces = faces.T[1:].T
    
    #Convert numpy arrays to Pandas DataFrames.
    vertices = pd.DataFrame(vertices).rename({0:'x_pos', 1:'y_pos', 2:'z_pos'}, axis=True)
    faces = pd.DataFrame(faces).rename({0:'vertex_id_1', 1:'vertex_id_2', 2:'vertex_id_3'}, axis=True)
    
    return vertices, faces
    
    
def order_face_vertices_by_left_hand_rule(vertices, triangles, center = np.array([0., 0., 0.])):
    """ Order the vertices around each face by the left hand rule w.r.t a vector from center to

    Given vertices and a list of triangles, check whether vertices of each polygon are ordered by the left hand rule.

    Parameters
    ----------

    vertices  : pandas.DataFrame (float) Nx3, columns: ['x_pos', 'y_pos', 'z_pos']
        N vertex positions
    triangles : pandas.DataFrame (int) Mx3, columns: ['vertex_id_1', 'vertex_id_2', 'vertex_id_3']
        M triangles specified by their 3 vertex ids.
    center: np.array (float), 3x1
        Center position from which the outward normal to each triangle is calculated.

    Return
    ------
    sorted_triangles : pandas.DataFrame (int) Mx3
    """
    
    vertices_array = vertices.values

    triangles_pos_1 = vertices_array[triangles['vertex_id_1'].values]
    triangles_pos_2 = vertices_array[triangles['vertex_id_2'].values]
    triangles_pos_3 = vertices_array[triangles['vertex_id_3'].values]

    triangles_E12 = triangles_pos_2 - triangles_pos_1
    triangles_E13 = triangles_pos_3 - triangles_pos_1

    triangles_normal_vector = np.cross(triangles_E12, triangles_E13)

    #Define orientation vector to determine sign of normal vector.
    triangles_orientation_vector = triangles_pos_1 - center

    triangles_not_outward_normal = np.array([normal.dot(orientation) < 0 for normal, orientation
                                             in zip(triangles_normal_vector, triangles_orientation_vector)])

    sorted_triangles = triangles
    if sum(triangles_not_outward_normal) > 0:
        print(sum(triangles_not_outward_normal), 'triangles will have their vertex indices re-ordered to produce outward normals.')

        sorted_triangles = triangles.copy()

        swap_vertex_id_2 = sorted_triangles.loc[triangles_not_outward_normal, 'vertex_id_2'].values.copy()
        sorted_triangles.loc[triangles_not_outward_normal, 'vertex_id_2'] = sorted_triangles.loc[triangles_not_outward_normal, 'vertex_id_3'].values
        sorted_triangles.loc[triangles_not_outward_normal, 'vertex_id_3'] = swap_vertex_id_2

    return sorted_triangles


def create_DCEL_from_polygonal_mesh(faces, triangle_mesh = True):
    """ Create a doubly connected edge list from a polygonal mesh described by the vertex ids of each polygon.

    Parameters
    ----------
    faces : numpy.ndarray (int) Mx3
        List of vertex ids of M faces
    triangle_mesh : {'True', 'False'}
        Whether the faces are a triangular mesh

    Return
    ------
    dcel_table : pandas.DataFrame (int) Nx5
        Table containing for each directed bond:
        - right vertex id
        - left dbond id
        - conjugate dbond id
        - face id
        - edge id
    """
    #Set number of faces.
    Nfaces = len(faces)
    #Set number of internal dbonds.
    Ndbonds_int = 3 * Nfaces

    assert faces.shape == (Nfaces, 3) and triangle_mesh == True, 'Non triangle faces not implemented.'

    faces = np.array(faces)

    # TODO: Check correct orientation of vertices (left hand rule)
    # TODO: Check consistency of input data. (Every edge exists only once in each face!)

    ### Every pair op vertex ids in faces creates an internal dbond. Create list: Ndbonds_int x 2:
    dbonds_vertex_id_pair_int = [[[triple[0], triple[1]], [triple[1], triple[2]], [triple[2], triple[0]]]
                                 for triple in faces]
    dbonds_vertex_id_pair_int = np.array(dbonds_vertex_id_pair_int)
    dbonds_vertex_id_pair_int = np.reshape(dbonds_vertex_id_pair_int, (Ndbonds_int, 2))

    ### Create unique id for each internal dbond and its conj dbond based on their vertex id pairs.
    dbonds_id_int = np.array([pairing_function([pair_id[0], pair_id[1]]) for pair_id in dbonds_vertex_id_pair_int])
    dbonds_conjid = np.array([pairing_function([pair_id[1], pair_id[0]]) for pair_id in dbonds_vertex_id_pair_int])

    #Check whether every dbond_id is unique. I fnot, either more than 2 triangles are connected to an edge or
    #the vertices are not ordered consistently.
    dbonds_id_int_unique = np.unique(dbonds_id_int)
    if len(dbonds_id_int) != len(dbonds_id_int_unique):
        print('Warning: Some internal edges in the mesh are not unique! Non unique edges:', len(dbonds_id_int) - len(dbonds_id_int_unique) )
        
        sorted_dbonds_id_int = dbonds_id_int.copy()
        sorted_dbonds_id_int.sort()
        diff_sorted_dbonds_id_int = np.diff(sorted_dbonds_id_int)
        error_sorted_dbonds_idx = np.where(diff_sorted_dbonds_id_int == 0)[0]
        error_dbonds_id  = sorted_dbonds_id_int[error_sorted_dbonds_idx]      
        error_dbonds_idx = [np.where(dbonds_id_int == error_dbond_id)[0] for error_dbond_id in error_dbonds_id]
        error_dbonds_idx = np.concatenate(error_dbonds_idx)
        print(error_dbonds_idx)
        
        error_faces_idx = (error_dbonds_idx/3).astype(int)
        error_faces_unique_idx = np.unique(error_faces_idx)
        
        error_faces_idx = list(error_faces_idx)
        print(error_faces_idx)
        print(faces[error_faces_idx])
        
        error_faces_multiples_id = [face_idx for face_idx in error_faces_unique_idx if error_faces_idx.count(face_idx) > 1]
        
        print(error_faces_multiples_id)
        print(faces[error_faces_multiples_id])

    ### Determine which edges are internal (both dbonds in dbonds_id_int) or external (only one dbond in dbonds_id_int)
    # Brute force method of finding the conj dbond for each dbond. Scaling: N**2/2
    dbonds_conj_index = [np.where(dbonds_conjid == dbond_id)[0] for dbond_id in dbonds_id_int]
    dbonds_conj_index = np.array(dbonds_conj_index)

    # Count number of conj dbonds for each dbond. Should be either 1 or 0.
    dbonds_conj_count = np.array([len(conj_idx) for conj_idx in dbonds_conj_index])
    # For 1 this dbond is internal, for 0 it is at the mesh boundary.
    dbonds_boundary_idx = np.where(dbonds_conj_count == 0)[0]
    dbonds_internal_idx = np.where(dbonds_conj_count == 1)[0]
    dbonds_strange_idx = np.where(dbonds_conj_count > 1)[0]
    print('Detected are:')
    print('Internal dbonds:', len(dbonds_internal_idx))
    print('Boundary dbonds:', len(dbonds_boundary_idx))
    print('Strange dbonds (dbonds that have more than 2 neighbours):', len(dbonds_strange_idx))

    # Set number of dbonds on the tissue boundary.
    Ndbonds_ext = len(dbonds_boundary_idx)
    Ndbonds_tot = Ndbonds_int + Ndbonds_ext

    #If there are external dbonds, the surface is not closed.
    closed_surface = True
    if Ndbonds_ext > 0:
        closed_surface = False

    # Create indices for internal and external dbonds
    dbonds_idx_int = np.array(range(Ndbonds_int))
    dbonds_idx_ext = np.array(range(Ndbonds_int, Ndbonds_tot))

    # Reserve output arrays memory:
    dbonds_right_vertex = np.array([-1] * Ndbonds_tot)
    dbonds_left_vertex  = np.array([-1] * Ndbonds_tot)
    dbonds_face_id      = np.array([-1] * Ndbonds_tot)
    dbonds_conjdb       = np.array([-1] * Ndbonds_tot)
    dbonds_leftdb       = np.array([-1] * Ndbonds_tot)
    dbonds_rightdb      = np.array([-1] * Ndbonds_tot)


    ### Set dbonds left and right vertices to interior and exterior dbonds.
    dbonds_right_vertex_int, dbonds_left_vertex_int = dbonds_vertex_id_pair_int.T
    dbonds_right_vertex[:Ndbonds_int] = dbonds_right_vertex_int
    dbonds_left_vertex[:Ndbonds_int]  = dbonds_left_vertex_int

    if not closed_surface:
        dbonds_left_vertex_ext, dbonds_right_vertex_ext = dbonds_vertex_id_pair_int[dbonds_boundary_idx].T
        dbonds_right_vertex[Ndbonds_int:Ndbonds_tot] = dbonds_right_vertex_ext
        dbonds_left_vertex[Ndbonds_int:Ndbonds_tot] = dbonds_left_vertex_ext


    ### Set internal dbonds left and right dbond
    faces_dbonds_idx_int = np.reshape(dbonds_idx_int, (Nfaces, 3))

    dbonds_leftdb_int  = np.array([[db_idx[1], db_idx[2], db_idx[0]] for db_idx in faces_dbonds_idx_int])
    dbonds_leftdb[:Ndbonds_int]  = np.reshape(dbonds_leftdb_int,  (Ndbonds_int))

    dbonds_rightdb_int = np.array([[db_idx[2], db_idx[0], db_idx[1]] for db_idx in faces_dbonds_idx_int])
    dbonds_rightdb[:Ndbonds_int] = np.reshape(dbonds_rightdb_int, (Ndbonds_int))

    # Set left and righ dbond of external dbonds.
    # For each external dbond, find the external dbond that has the right_vertex_id equal this dbond left_vertex_id
    if not closed_surface:
        dbonds_leftdb_ext_idx = [np.where(dbonds_right_vertex_ext == lvid)[0][0] for lvid in dbonds_left_vertex_ext]
        dbonds_leftdb_ext     = np.array([dbonds_idx_ext[idx] for idx in dbonds_leftdb_ext_idx])

        dbonds_rightdb_ext_idx = [np.where(dbonds_left_vertex_ext == rvid)[0][0] for rvid in dbonds_right_vertex_ext]
        dbonds_rightdb_ext     = np.array([dbonds_idx_ext[idx] for idx in dbonds_rightdb_ext_idx])

        dbonds_leftdb[Ndbonds_int:Ndbonds_tot]  = dbonds_leftdb_ext
        dbonds_rightdb[Ndbonds_int:Ndbonds_tot] = dbonds_rightdb_ext


    ### Set dbonds face id. (exterior dbonds have face_id -1).
    dbonds_face_id_int = np.array([[face_id] * 3 for face_id in range(Nfaces)])
    dbonds_face_id[:Ndbonds_int] = np.reshape(dbonds_face_id_int, (Ndbonds_int))


    ### Set dbonds conj dbond.
    # Of the interior dbonds.
    dbonds_conj_int_index = dbonds_conj_index[dbonds_internal_idx]
    dbonds_conjdb[dbonds_internal_idx] = np.array([conj_index[0] for conj_index in dbonds_conj_int_index])

    if not closed_surface:
        # Of the interior dbonds at the tissue boundary.
        dbonds_conjdb[dbonds_boundary_idx] = dbonds_idx_ext
        # Of the dbonds at the tissue exterior.
        dbonds_conjdb[dbonds_idx_ext] = dbonds_boundary_idx

    ### Store all data in pandas.DataFrame
    dcel_table = pd.DataFrame(-1, index=range(Ndbonds_tot), columns=dcel_table_columns)
    dcel_table.index.name = 'dbond_id'

    dcel_table['vertex_id']     = dbonds_right_vertex
    dcel_table['face_id']       = dbonds_face_id
    dcel_table['conj_dbond_id'] = dbonds_conjdb
    dcel_table['left_dbond_id'] = dbonds_leftdb

    if triangle_mesh:
        dcel_table.rename({'face_id': 'triangle_id'}, inplace=True, axis=1)

    # Include bond id for lookup of the geomteric properties of this edge.
    dcel_table['bond_id'] = dcel_table.index
    conjugate_edges = dcel_table[dcel_table.index > dcel_table['conj_dbond_id']]['conj_dbond_id'].copy()
    dcel_table.loc[conjugate_edges.index, 'bond_id'] = conjugate_edges.values

    return dcel_table


def sortVertexCellIds_cc(vid, dcel_table):
    """ Sort cell_Ids connected to vertex vid in counterclockwise direction, starting with lowest cellId.

    Parameters
    ---------
    vid : int
        Index of vertex
    dcel_table : pandas.DataFrame (int)
        Doubly connected edge list where index is the id of each dbond, and contains columns:
        - cell_id : cell id of dbond.
        - vertex_id : right vertex of dbond
        - conj_dbond_id : conjugate dbond of each dbond.

    Returns
    -------
    sorted_cell_ids : List
        For each vertex, return a list of cell ids ordered in counter clockwise direction.

    """

    bids = np.where(dcel_table['vertex_id'] == vid)[0]
    cids = dcel_table['cell_id'].iloc[bids].values
    cnjb_bids = dcel_table['conj_dbond_id'].iloc[bids].values
    cnjb_cids = dcel_table['cell_id'].iloc[cnjb_bids].values

    sorted_cids = [cids.min()]
    while (len(sorted_cids) < len(cids)):
        sorted_cids.append(cids[np.where(cnjb_cids == sorted_cids[-1])[0]][0])

    return sorted_cids



def create_doubly_connected_edge_list_from_positions_on_sphere(vertex_positions_xyz):
    """
    Given set of points xyz that lie on a sphere, perform voronoi tesselation and order vertices in counterclockwize direction.
    :param vertex_positions_xyz: List of vertex positions [x, y, z].

    Output is a doubly connected edge list with:
    bond_rv   = right vertex of each dbond.
    bond_cell = cell id associated to each dbond.
    bond_cb   = conjugate dbond id of each dbond.
    bond_rb   = righ (previous) dbond of each dbond.
    bond_lb   = left (next) dbond of each dbond
    """

    # Create voronoi tesselation of the sphere given vertices (cell centers) that lie on a sphere.
    sv = SphericalVoronoi(vertex_positions_xyz)
    sv.sort_vertices_of_regions()

    # Since vertices that define a polygon can be ordered clockwize or cc, make sure they are all counter-clockwize.
    for cellId, xyz_cellCenter, vertexId_cell in zip(range(N), sv.points, sv.regions):

        xyz_cellVertices = sv.vertices[vertexId_cell]

        ErrorCntr = 0
        for xyz_thisvx, xyz_nextvx in zip(xyz_cellVertices, np.roll(xyz_cellVertices, -1, axis=0)):

            n = np.cross(xyz_nextvx - xyz_thisvx, xyz_cellCenter - xyz_thisvx)
            n_norm = np.sqrt(sum(n * n))
            r_norm = np.sqrt(sum(xyz_thisvx * xyz_thisvx))

            triangle_sign = np.dot(n / n_norm, xyz_thisvx / r_norm)

            # If the normal vector is directed into the sphere, swap the order of the cells around the vertex.
            if triangle_sign < 0:
                ErrorCntr += 1
                # print(cellId, len(vertexId_cell), triangle_sign)

        if ErrorCntr == 0:
            pass
            # orderedCellVertexIds[cellId] = vertexId_cell
        elif ErrorCntr == len(vertexId_cell):
            # orderedCellVertexIds[cellId] = vertexId_cell.reverse();
            vertexId_cell.reverse()
        else:
            print("Error with vertex order of cell Id", cellId, ":", len(vertexId_cell), ErrorCntr)

    # Given vertex topology in sv.regions, set the bond topology needed for the vertex model.
    # Therefore, for each bond (id given by the position in the list), give the
    # right_vertex, cell_id, conjugate_bond_id and right_bond_id.
    Nbonds = 3 * len(sv.vertices)
    bond_rv = np.zeros(Nbonds, np.int64)
    bond_cell = np.zeros(Nbonds, np.int64)
    bond_cb = np.zeros(Nbonds, np.int64)
    bond_rb = np.zeros(Nbonds, np.int64)
    bond_lb = np.zeros(Nbonds, np.int64)

    Ncntr = 0
    Cell_id = 0
    for cellVertexIds in sv.regions:
        Ncell = len(cellVertexIds)
        bond_rv[Ncntr:Ncntr + Ncell] = np.array(cellVertexIds).astype(np.int64)
        bond_cell[Ncntr:Ncntr + Ncell] = Cell_id
        bond_rb[Ncntr:Ncntr + Ncell] = np.roll(list(range(Ncntr, Ncntr + Ncell)), 1)
        bond_lb[Ncntr:Ncntr + Ncell] = np.roll(list(range(Ncntr, Ncntr + Ncell)), -1)

        Ncntr += Ncell
        Cell_id += 1

        vertexToCell = [[] for _ in range(len(sv.vertices))]

    ### Set conjugate bond for each dbond.

    # For each vertex, find the three cell ids surrounding the vertex.
    for cellVertexIds, cellId in zip(sv.regions, range(len(sv.regions))):
        for vertexId in cellVertexIds:
            vertexToCell[vertexId].append(cellId)

    # Sort cell ids around the vertex in counter-clockwize direction by hack if normal vector
    # to the triangle connecting the three cell centers, points outward from the sphere.
    correctedVertexToCell = [[] for _ in range(len(sv.vertices))]
    for vertexId, cellIds in zip(range(len(sv.vertices)), vertexToCell):
        sortedCellIds = np.roll(cellIds, -np.where(np.array(cellIds) == min(cellIds))[0])

        cells_xyz = sv.points[sortedCellIds]

        n = np.cross(cells_xyz[1] - cells_xyz[0], cells_xyz[2] - cells_xyz[0])

        n_norm = np.sqrt(sum(n * n))
        r_norm = np.sqrt(sum(cells_xyz[0] * cells_xyz[0]))

        triangle_sign = np.dot(n / n_norm, cells_xyz[0] / r_norm)

        # If the normal vector is directed into the sphere, swap the order of the cells around the vertex.
        if triangle_sign < 0:
            temp = sortedCellIds[1]
            sortedCellIds[1] = sortedCellIds[2]
            sortedCellIds[2] = temp

        correctedVertexToCell[vertexId] = sortedCellIds

    vertexToCell = correctedVertexToCell

    # Associate conjugate bond to each bond.
    for vertexId, orderedCellIds in zip(range(len(sv.vertices)), vertexToCell):
        # Get bond ids and their cell ids of bonds connected to this vertexId.
        bond_ids = np.where(bond_rv == vertexId)[0]
        randomCellIds = bond_cell[bond_ids]

        # Compare the cell Id of each bond with the ordered list (cc) of cell ids to establish the correct order of the bonds.
        bondId_firstCell = bond_ids[np.where(np.array(randomCellIds) == orderedCellIds[0])[0]]
        bondId_secondCell = bond_ids[np.where(np.array(randomCellIds) == orderedCellIds[1])[0]]
        bondId_thirdCell = bond_ids[np.where(np.array(randomCellIds) == orderedCellIds[2])[0]]

        # Find the right_bond (previous bond in the cell going cc) of each bond,
        # they are the conjugate bonds of the bond in the opposite cell.
        rbondId_firstCell = bond_rb[bondId_firstCell]
        rbondId_secondCell = bond_rb[bondId_secondCell]
        rbondId_thirdCell = bond_rb[bondId_thirdCell]

        bond_cb[bondId_firstCell] = rbondId_thirdCell
        bond_cb[bondId_secondCell] = rbondId_firstCell
        bond_cb[bondId_thirdCell] = rbondId_secondCell

    ### Check double connected edge list consistency.

    # Check topological consistency.
    if len(set(bond_cb)) != len(bond_cb):
        print("Error: Set of conjugate bonds is not unique!")

    for bid in range(Nbonds):

        vertexId_bond = bond_rv[bid]
        rbondId_bond = bond_rb[bid]

        cbond_rbond = bond_cb[rbondId_bond]

        # print(vertexId_bond, rbondId_bond, cbond_rbond, bond_rv[cbond_rbond])

        if (vertexId_bond != bond_rv[cbond_rbond]):
            print("Error in bond topology: rightvertex of bond ", bid,
                  " does not match rightvertex of conjugate bond of the right_bond of bond.")
            print(vertexId_bond, rbondId_bond, cbond_rbond)

    return sv.vertices, bond_rv, bond_cell, bond_cb, bond_rb, bond_lb


# Create Vertex Model NetCDF file that can be read by the vertex model code by Matthias.
# Input is the DCEL.
def create_vertex_model_netCDF_from_doubly_connected_edge_list(file_name,
                                                               vertices,
                                                               dcel_table,
                                                               net_props={}):

    print('Creating netCDF file named ', file_name, '.')

    vertices = vertices.values

    dbond_cell         = dcel_table['cell_id'].values
    dbond_conj_db      = dcel_table['conj_dbond_id'].values
    dbond_right_vertex = dcel_table['vertex_id'].values

    dbond_left_db = []
    if 'left_dbond_id' in dcel_table:
        dbond_left_db = dcel_table['left_dbond_id'].values

    dbond_right_db = []
    if 'right_dbond_id' in dcel_table:
        dbond_right_db = dcel_table['right_dbond_id'].values

    xper = []
    yper = []
    if 'xPer' in dcel_table.columns and 'yPer' in dcel_table.columns:
        xper = dcel_table['xPer'].values
        yper = dcel_table['yPer'].values

    # Network dimension: 2 or 3.
    network_dimension = vertices.shape[-1]

    # Open NetCDF file.
    rootgrp = netcdf.Dataset(file_name, "w", format="NETCDF4")

    # Set file attributes
    NetCDFversion = 6
    rootgrp.Type = "Network"
    rootgrp.Version = NetCDFversion

    network_cells_id = np.unique(dbond_cell)

    # Set dimensions of possible arrays (variables)
    vertexDim = rootgrp.createDimension("vertex index", len(vertices))
    bondDim   = rootgrp.createDimension("bond index", len(dbond_cell))
    cellDim   = rootgrp.createDimension("cell index", len(network_cells_id))

    # General variables
    frame_time_var = rootgrp.createVariable("Network::time", "f8")
    frame_time_var[:] = net_props['time'] if 'time' in net_props.keys() else 0
    cellNumberVar = rootgrp.createVariable("Cell::number", "i4")
    cellNumberVar[:] = len(network_cells_id)

    ### Vertex positions x,y and z.
    vertexXVar = rootgrp.createVariable("Vertex::x", "f8", ("vertex index",))
    vertexXVar[:] = vertices.T[0]
    vertexYVar = rootgrp.createVariable("Vertex::y", "f8", ("vertex index",))
    vertexYVar[:] = vertices.T[1]
    if network_dimension == 3:
        vertexZVar = rootgrp.createVariable("Vertex::z", "f8", ("vertex index",))
        vertexZVar[:] = vertices.T[2]

    ### DBOND variables: Set dbond right-vertex_id, cell_id and conj_dbond_id.
    bondRightVertexVar = rootgrp.createVariable("Bond::rightVertex", "i4", ("bond index",))
    bondRightVertexVar[:] = dbond_right_vertex
    bondCellVar = rootgrp.createVariable("Bond::cell", "i4", ("bond index",))
    bondCellVar[:] = dbond_cell
    bondConjBondVar = rootgrp.createVariable("Bond::conjBond", "i4", ("bond index",))
    bondConjBondVar[:] = dbond_conj_db
    # Set possible left or right dbond.
    assert len(dbond_right_db) != 0 or len(dbond_left_db) != 0
    if len(dbond_right_db) != 0:
        bondRightBondVar = rootgrp.createVariable("Bond::rightBond", "i4", ("bond index",))
        bondRightBondVar[:] = dbond_right_db
    if len(dbond_left_db) != 0:
        bondLeftBondVar = rootgrp.createVariable("Bond::leftBond", "i4", ("bond index",))
        bondLeftBondVar[:] = dbond_left_db
        # Set possible periodic boundary dbonds.
    if xper != [] and yper != []:
        bondXPerVar = rootgrp.createVariable("Bond::xPer", "i4", ("bond index",))
        bondYPerVar = rootgrp.createVariable("Bond::yPer", "i4", ("bond index",))
        bondXPerVar[:] = xper
        bondYPerVar[:] = yper

    ### FRAME properties:
    xWidthVar = rootgrp.createVariable("Frame::xWidth", "f8")
    xWidthVar[:] = net_props['xWidth'] if 'xWidth' in net_props.keys() else 1.0
    yWidthVar = rootgrp.createVariable("Frame::yWidth", "f8")
    yWidthVar[:] = net_props['yWidth'] if 'yWidth' in net_props.keys() else 1.0
    skewvar = rootgrp.createVariable("Frame::skewVariable", "f8")
    skewvar[:] = net_props['skew'] if 'skew' in net_props.keys() else 0.0

    ### CELL properties:
    cellIdsVar = rootgrp.createVariable("Cell::cellId", "i4", ("cell index",))
    cellIdsVar[:] = network_cells_id
    cellTypeVar = rootgrp.createVariable("Cell::cellType", "i4", ("cell index",))
    cellTypeVar[:] = 0

    rootgrp.close()


def pairing_function(id_list):
    """ Cantor pairing function generates unique numbers from sets of numbers.
    Parameters
    ----------
    id_list : List (int)
        List containing integers.

    Return
    ------
    cantor_number: int
    """
    y = id_list.pop()
    x = id_list[-1]

    if (len(id_list) > 1):
        x = pairingFunction(id_list)

    return int(0.5 * (x + y) * (x + y + 1) + y)


# Given the vertex model NetCDF output files at the specified filepath, calculate the moviedata output table files vertices, dbonds and cells.
def create_moviedata_vertices_dbonds_cells_tables_from_vertex_model_netCDF_files(datapath_vertex_model_output,
                                                                                 datapath_root,
                                                                                 datapath_frames,
                                                                                 remove_periodicity = True):

    # Find all netcdf files in given directory.
    filename_identifier_end = '.vm.nc'
    filenames = [f for f in os.listdir(datapath_vertex_model_output) if
                 os.path.isfile(datapath_vertex_model_output + f)]
    network_filenames = [f for f in filenames if f.endswith(filename_identifier_end)]
    network_filenames.sort()

    frames = pd.DataFrame({'frame': range(len(network_filenames)),
                           'time_sec': 3600 * np.arange(0.0, len(network_filenames), 1.0)})
    for frameIdx, net_filename in enumerate(network_filenames):

        vertices, dcel_table, network_props =\
            load_vertex_model_netCDF(datapath_vertex_model_output + net_filename, remove_periodicity = remove_periodicity)

        # Generate tables for MovieData input.
        frames.loc[frameIdx, 'time_sec'] = 1 * network_props['time']

        cell_ids = np.unique(dcel_table['cell_id'].values)
        cell_ids = np.delete(cell_ids, np.where(cell_ids == MM.BOUNDARY_CELL_ID)[0])
        cells = pd.DataFrame( [], index=cell_ids )
        cells.index.name = 'cell_id'

        # Save network files to disk for each seperate frame.
        MM.table_io_per_frame(datapath_frames, 'vertices', frameIdx, network_type='cells', action='save', table=vertices)
        MM.table_io_per_frame(datapath_frames, 'dbonds'  , frameIdx, network_type='cells', action='save', table=dcel_table)
        MM.table_io_per_frame(datapath_frames, 'cells'   , frameIdx, network_type='cells', action='save', table=cells)

    frames.to_pickle(datapath_root + 'frames.pkl')


def load_vertex_model_netCDF(netcdf_filename, remove_periodicity = False):
    """ Convert vertex model output into vertices and DCEL table.

    Parameters
    ----------
    netcdf_filename : string
        path and filename of vertex model output netCDF file.
    remove_periodicity : {'False', 'True'}
        Whether to remove cells on the periodic boundary. They are converted the tissue boundary.

    Return
    ------
    vertices : pandas.Dataframe (float) Mx3
        Positions of the M vertices in the cell network.
    dcel_table : pandas.DataFrame (int) Nx4
        Double connected edge list describing the cell network topology.
    network_properties : dictionary
        Dictionary containing the network properties:
        - time   - Time of the frame
        - xWidth - box horizontal dimension
        - yWidth - box vertical dimension
        - skew   - Skew in the periodic boundary condition in case of simple shear.
    """

    # Given the table of the right bond for each bond, recursively find all the bonds related to a cell.
    def sortCellBondIds_c(sortedbondList, dbond_right_db):

        next_bondId = dbond_right_db[sortedbondList[-1]]
        if next_bondId != sortedbondList[0]:
            sortedbondList.append(next_bondId)
            sortCellBondIds_c(sortedbondList, dbond_right_db)

        return sortedbondList

    # Cells with periodic boundary conditions are difficult: remove them!
    def remove_cells_on_boundary(dbond_cell_id, cell_ids, xPer, yPer):

        # Loop over all cells and find cells with dbonds that cross the periodic boundary.
        for cell_idx, cell_id in enumerate(cell_ids):

            cell_dbonds_index = np.where(dbond_cell_id == cell_id)[0]

            cell_dbonds_xper = xPer[cell_dbonds_index]
            cell_dbonds_yper = yPer[cell_dbonds_index]

            # Continue to next cell if this cell does not contain periodic dbonds.
            if (cell_dbonds_xper == 0).all() and (cell_dbonds_yper == 0).all():
                continue

            dbond_cell_id[cell_dbonds_index] = MM.BOUNDARY_CELL_ID
            cell_ids[cell_idx]               = MM.BOUNDARY_CELL_ID

        cell_ids = np.delete(cell_ids, np.where(cell_ids == MM.BOUNDARY_CELL_ID)[0])

        return dbond_cell_id, cell_ids

    # Open netCDF file.
    f = netcdf.Dataset(netcdf_filename, 'r')


    ### Load network properties.
    network_properties = {'time':1, 'xWidth':1, 'yWidth':1, 'skew':0}

    network_properties['time']   = f.variables['Network::time'][:].data
    network_properties['xWidth'] = f.variables['Frame::xWidth'][:].data
    network_properties['yWidth'] = f.variables['Frame::yWidth'][:].data
    network_properties['skew']   = f.variables['Frame::skewVariable'][:].data


    ### Make vertices table.

    # Load vertice postitions. For original vertex model, set all z-positions to 0.
    v_x = f.variables['Vertex::x'][:].copy()
    v_y = f.variables['Vertex::y'][:].copy()
    try:
        v_z = f.variables['Vertex::z'][:].copy()
    except:
        v_z = np.array([0.] * len(v_x))

    vertices = pd.DataFrame({'x_pos': v_x, 'y_pos': v_y, 'z_pos': v_z})


    ### Make DCEL table

    # Load dbond topology.
    dbond_right_vertex = f.variables['Bond::rightVertex'][:].copy()
    dbond_cell_id      = f.variables['Bond::cell'][:].copy()
    dbond_conj_db      = f.variables['Bond::conjBond'][:].copy()
    dbond_right_db     = f.variables['Bond::rightBond'][:].copy()

    # Load cell properties. When cell ids are not part of database, generate them.
    try:
        cell_ids = f.variables['Cell::cellId'][:].copy()
    except:
        cell_ids = np.unique(dbond_cell_id)

    # Check if cdf files contain data on periodic dbonds.
    periodic_boundaries = False
    try:
        xPer = f.variables['Bond::xPer'][:].copy()
        yPer = f.variables['Bond::yPer'][:].copy()
        if (xPer != 0).any() or (yPer != 0).any():
            periodic_boundaries = True
    except:
        print('No indicators for dbond periodicity were found.')

    # If periodic boundary conditions and remove_periodicity enabled, remove cells that lie on the boundary.
    if periodic_boundaries and remove_periodicity:
        dbond_cell_id, cell_ids = remove_cells_on_boundary(dbond_cell_id, cell_ids, xPer, yPer)

    # Create dictionary of cellId - counter clockwise ordered bond ids.
    sortedCellBondIds_cc = {}
    for cellId in cell_ids:
        # Pick first bond related to cellId
        firstBondId = np.where(dbond_cell_id == cellId)[0][0]
        # Find all bonds related to cellId, change order from clockwise to counter clockwise.
        sortedCellBondIds_cc[cellId] = np.flip(sortCellBondIds_c([firstBondId], dbond_right_db), axis=0)

    # Convert right_dbond to left_dbond id.
    dbond_left_db = np.array([-1] * len(dbond_right_db))
    for cid in cell_ids:
        bids_cc_i = sortedCellBondIds_cc[cid]
        bids_cc_ip1 = np.roll(bids_cc_i, -1)
        dbond_left_db[bids_cc_i] = bids_cc_ip1

    dcel_table = pd.DataFrame({'cell_id': dbond_cell_id,
                               'conj_dbond_id': dbond_conj_db,
                               'vertex_id': dbond_right_vertex,
                               'left_dbond_id': dbond_left_db,
                               'right_dbond_id': dbond_right_db})

    if periodic_boundaries:
        dcel_table['xPer'] = xPer
        dcel_table['yPer'] = yPer

    return vertices, dcel_table, network_properties
