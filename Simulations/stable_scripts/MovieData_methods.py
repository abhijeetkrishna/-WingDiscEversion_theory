#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday Oct 18 2018

@author: Joris Paijmans - paijmans@pks.mpg.de
"""

import numpy as np
import pandas as pd
import sys
import csv
import sys
import os
import pwd
import pickle
import scipy.linalg as lin

import MovieData_methods as MM
import MovieData_differential_geometry_methods as MMdg

### MOVIE DATA GLOBALLY DEFINED CONSTANTS

#Cell Id of area outside the tissue boundary.
BOUNDARY_CELL_ID           = 10000
BOUNDARY_TRIANGLE_ID       = -1
#Cells with more than this number of vertices will be removed from network.
#Assumming this is a hole or fld in the tissue.
CELL_MAXIMUM_VERTEX_NUMBER = 20

#Directory name of directory to place all the frames.
DATAPATH_ROOT_FRAMES = 'frames'

#Small number to prevent division by zero etc.
EPSILON = 1e-12

#Valid strings for saving and loading different network types.
valid_networks             = ['cells', 'dual_triangles', 'subcellular_triangles', 'cell_tensor', 'triangle_tensor', 'tissue_average']
valid_file_actions         = ['load', 'save']

valid_cell_table_names     = ['cells', 'vertices', 'dbonds', 'sorted_cell_dbonds_per_frame', 'shear_tensor',
                              'elongation_tensor', 'deformation_rate', 'curvature_tensor']
valid_triangle_table_names = ['triangles', 'vertices', 'dbonds', 'deformation_rate', 'shear_tensor', 'bond_mean_curvature',
                              'bond_int_curvature_tensor', 'elongation_tensor', 'curvature_tensor']

### Define table columns
#For tissue_deformation_rate_tensor
tissue_deformation_rate_tensor_columns = ['time', 'dt', 'delta_area', 'shear_xx', 'shear_xy', 'delta_psi',
                                          'shear_corotational_xx', 'shear_corotational_xy', \
                                          'covariance_elongation_xx', 'covariance_elongation_xy',
                                          'covariance_corotation_xx', 'covariance_corotation_xy', \
                                          'shear_rot_corr_xx', 'shear_rot_corr_xy', 'shear_area_corr_xx',
                                          'shear_area_corr_xy']
#For triangles_deformation_rate_tensor
triangles_deformation_rate_tensor_columns = ['triangle_id', 'delta_area', 'shear_xx', 'shear_xy', 'delta_psi',
                                             'shear_corotational_xx', 'shear_corotational_xy',
                                             'delta_elongation_xx', 'delta_elongation_xy',
                                             'delta_elongation_twophi']

#For text coloring.
class txtcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    

#Calculates the difference angles, angle2 - angle1
def angle_difference(angle1, angle2):
    return np.angle( np.exp(1j*(angle2 - angle1)) )


def rotation(tensor_array, angle_array, unit_axis_array):
    """ Return the vector 'vec' rotated by an angle 'angle' around the axis 'axis' in 3d
    and return also the associated rotation matrix 'rot'.
    The computation goes as follow:
    let 'u' be the unit vector along 'axis', i.e. u = axis/norm(axis)
    and A = I × u be the skew-symmetric matrix associated to 'u', i.e.
    the cross product of the identity matrix with 'u'
    then rot = exp(θ A) is the 3d rotation matrix.
    """

    # Determine dimension and rank of input tensor.
    dim = tensor_array.shape[-1]
    rank = len(tensor_array.shape) - 1

    assert rank == 1 or rank == 2

    rot_array = \
        np.array([lin.expm(np.cross(np.eye(3), axis * angle)) for axis, angle in zip(unit_axis_array, angle_array)])

    rotated_tensor_array = np.array([])
    if rank == 1:
        rotated_tensor_array = np.array([np.dot(rot, tensor) for rot, tensor in zip(rot_array, tensor_array)])
    elif rank == 2:
        rotated_tensor_array = np.array([np.dot(rot, tensor).dot(rot.T) for rot, tensor in zip(rot_array, tensor_array)])

    return rotated_tensor_array


def unit_vector(vectors):
    """ Normalize array of vectors
    
    Parameters
    ----------
    vectors : numpy.array / list N x d
        Array or list with M d-dimensional vectors.
        
    Results
    -------
    unit_vectors : numpy.array
        M normalized vectors
    
    """
    
    vector_length = np.sqrt(np.sum(vectors**2, axis=1))
    
    unit_vectors = [vec/l for vec, l in zip(vectors, vector_length)]
    
    return np.array(unit_vectors)



#Check if dbonds table topology is consistent.
def check_dbonds_consistency(dbonds, strict = True):

    next_dbond_vertex_id = dbonds.loc[dbonds['left_dbond_id']]['vertex_id'].values
    conj_dbond_vertex_id = dbonds.loc[dbonds['conj_dbond_id']]['vertex_id'].values
    
    dbonds_table_consistency = sum(next_dbond_vertex_id == conj_dbond_vertex_id) == len(dbonds)
    
    if dbonds_table_consistency:
        print( 'dbond table consistency: ' + txtcolors.OKGREEN + 'PASSED' + txtcolors.ENDC + ', Consistent dbonds: ' + txtcolors.BOLD\
              + str(sum(next_dbond_vertex_id == conj_dbond_vertex_id)) + txtcolors.ENDC + ', total dbonds: ' + txtcolors.BOLD + str(len(dbonds)) + txtcolors.ENDC )
    else:
        print('dbond table consistency: ' + txtcolors.FAIL + 'FAILED' + txtcolors.ENDC + ', Consistent dbonds: ' + txtcolors.BOLD\
              + str(sum(next_dbond_vertex_id == conj_dbond_vertex_id)) + txtcolors.ENDC + ', total dbonds: ' + txtcolors.BOLD + str(len(dbonds)) + txtcolors.ENDC)
        
    if strict:
        assert dbonds_table_consistency


# Sort cell_Ids connected to vertex vid in counterclockwise direction, starting with lowest cellId.
def sortVertexCellIds_cc(vertex_id, dbonds):

    selected_dbonds = dbonds['vertex_id'] == vertex_id
    connected_cell_ids, connected_conj_dbond_ids =\
    dbonds[selected_dbonds][['cell_id', 'conj_dbond_id']].values.T

    conj_dbonds_cell_ids = dbonds.iloc[connected_conj_dbond_ids]['cell_id'].values

    if sum(conj_dbonds_cell_ids == BOUNDARY_CELL_ID) > 1:
        return conj_dbonds_cell_ids

    sorted_cell_ids = [connected_cell_ids.min()]
    while(len(sorted_cell_ids) < len(connected_cell_ids)):
        sorted_cell_ids.append(connected_cell_ids[np.where(conj_dbonds_cell_ids==sorted_cell_ids[-1])[0]][0])

    return sorted_cell_ids


#Calculate the center of mass given vertex ids in dual lattice, and return id for new dual vertex.
def calculateCenterOfMassAndID_dual(dual_vertex_ids, dual_vertices):

    center_of_mass_xyz = sum( dual_vertices.loc[dual_vertex_ids][['x_pos','y_pos','z_pos']].values ) / len(dual_vertex_ids)

    #center_of_mass_id  = MM.generate_center_of_mass_id(dual_vertex_ids)
    center_of_mass_id = dual_vertices.index.max() + 1

    return center_of_mass_xyz, center_of_mass_id


#Calculate the center of mass of a cell.
def calculateCenterOfMass_cell(cell_vertex_indices, vertices):
    
    cell_center_xyz = sum( vertices.iloc[cell_vertex_indices][['x_pos','y_pos','z_pos']].values ) / len(cell_vertex_indices)

    return cell_center_xyz


#Caculate the positions of vertices in the dual vertex network from cells.
def calculate_dual_vertex_positions_from_cells(dbonds, vertices, sorted_dbonds_per_cell_id):

    cell_ids = [cell_id for cell_id, temp in sorted_dbonds_per_cell_id.items()]

    dual_vertices = dict(zip(cell_ids, [[0., 0., 0.]] * len(cell_ids)))
    for cell_id, sorted_dbonds_ids in sorted_dbonds_per_cell_id.items():
        cell_vertex_indices = dbonds.iloc[ sorted_dbonds_ids ]['vertex_id'].values
        dual_vertices[cell_id] = MM.calculateCenterOfMass_cell(cell_vertex_indices, vertices)

    dual_vertices = pd.DataFrame.from_dict(data=dual_vertices, orient='index')
    dual_vertices.columns = ['x_pos','y_pos','z_pos']

    return dual_vertices


#Set-up a dictionary which given a vertex id, returns the idices of vertices it is connected to.
def make_dual_vertex_network_from_cells(dbonds, sorted_dbonds_per_cell_id, cell_id_to_vertex_index):
    dual_vertex_network = {}
    for cell_id, vertex_id in cell_id_to_vertex_index.items():

        if cell_id == BOUNDARY_CELL_ID:
            continue

        cell_sorted_conj_dbonds_id = dbonds.iloc[ sorted_dbonds_per_cell_id[cell_id] ]['conj_dbond_id'].values
        cell_sorted_cell_neighbour_ids = dbonds.iloc[cell_sorted_conj_dbonds_id]['cell_id'].values

        dual_vertex_network[vertex_id] = np.array([cell_id_to_vertex_index[cid] for cid in cell_sorted_cell_neighbour_ids])

    return dual_vertex_network


#Given the vertices, find all the neighbouring cell ids in counter-clockwise direction as well as the order of each vertex.
def get_vertex_connected_cell_ids_cc(dbonds):

    sorted_cell_ids_per_2_vertex_id = {}
    sorted_cell_ids_per_3_vertex_id = {}
    sorted_cell_ids_per_higher_order_vertex_id = {}

    print('Sort cell_ids in counter-clockwise direction around each vertex.')
    
    vertex_ids = np.unique(dbonds['vertex_id'].values)
    for vertex_index in vertex_ids:

        if(vertex_index%1000 == 0):
            print(vertex_index, ":", len(vertex_ids))

        sorted_cell_ids = MM.sortVertexCellIds_cc(vertex_index, dbonds)

        if len(sorted_cell_ids) == 2:
            sorted_cell_ids_per_2_vertex_id[vertex_index] = sorted_cell_ids
        elif len(sorted_cell_ids) == 3:
            sorted_cell_ids_per_3_vertex_id[vertex_index] = sorted_cell_ids
        elif len(sorted_cell_ids) > 3:
            sorted_cell_ids_per_higher_order_vertex_id[vertex_index] = sorted_cell_ids

    return sorted_cell_ids_per_2_vertex_id, sorted_cell_ids_per_3_vertex_id, sorted_cell_ids_per_higher_order_vertex_id


# Load and save the files containing network data of each movie frame.
def table_io_per_frame(datapath_frames, table_name, frameIdx, network_type = 'cells', action = 'load', table = []):
           
    assert network_type in valid_networks, "Wrong network type given: " + str(network_type)+". Allowed types: "+str(valid_networks)
    assert action in valid_file_actions, "Wrong file action given:" + str(action)+". Allowed actions: "+str(valid_file_actions)
    
    input_filename = ''
    if network_type == 'cells':
        assert table_name in valid_cell_table_names, "Wrong table name for cells network given. Allowed: "+str(valid_cell_table_names)
        network_filename = 'cell_network_' + table_name + '_' + "%04d" % (frameIdx,) + '.pkl'
    elif network_type == 'dual_triangles':
        assert table_name in valid_triangle_table_names, "Wrong table name for dual-triangles network given. Allowed: "+str(valid_triangle_table_names)
        network_filename = 'dual_triangle_network_' + table_name + '_' + "%04d" % (frameIdx,) + '.pkl'
    elif network_type == 'subcellular_triangles':
        assert table_name in valid_triangle_table_names, "Wrong table name for subcellular-triangles network given. Allowed: "+str(valid_triangle_table_names)
        network_filename = 'subcellular_triangle_network_' + table_name + '_' + "%04d" % (frameIdx,) + '.pkl'
    else:
        assert table_name in valid_triangle_table_names, "Wrong table name for network_average given. Allowed: " + str(valid_triangle_table_names)
        network_filename = 'tissue_average_' + table_name + '_' + "%04d" % (frameIdx,) + '.pkl'
    
    datapath_frames_network = datapath_frames + network_type + '/'
    datapath_filename = datapath_frames_network + network_filename
    
    if action == 'load':
    
        #Check if file to load exists.
        if not os.path.isfile(datapath_filename):
            print('Failed to load file ', datapath_filename)
            return False

        # Read data from picke file and put in dictionary data.
        print('Loading file', network_filename)
        with open(datapath_filename, "rb") as f:
            table = pickle.load(f)

        # Remove dummy cell from cells table.
        if table_name == 'cells':
            if (table.index == BOUNDARY_CELL_ID).any():
                table.drop(index=BOUNDARY_CELL_ID, inplace=True)

    if action == 'save':
        
        #Make directories if not exsisting yet.
        if not os.path.isdir(datapath_frames):
            os.mkdir(datapath_frames)

        if not os.path.isdir(datapath_frames_network):
            os.mkdir(datapath_frames_network)           
            
        # Save data to pickle file.
        print('Saving file', network_filename)
        with open(datapath_filename, "wb") as f:
            pickle.dump(table, f)
                
    return table


# For each cell in cells, calculate its area and mean unit normal vector given the centroid.
def calculate_cell_area_and_unit_normal_vector(vertices, cells, dbonds, sorted_dbonds_per_cell_id):
    # Make cell_id to vertex position [x,y,z] dictionary
    # Make cell_id to vertex index dictionary.
    vertices_indices_per_cell_list = [dbonds.iloc[sorted_dbonds_per_cell_id[cell_id]]['vertex_id'].values for cell_id in
                                      cells.index]
    sorted_vertices_per_cell_id = dict(zip(cells.index, vertices_indices_per_cell_list))

    cells_vertices_xyz = [vertices.iloc[sorted_vertices_per_cell_id[cell_id]][['x_pos', 'y_pos', 'z_pos']].values for
                          cell_id in cells.index]
    cells_vertices_xyz_per_cell = dict(zip(cells.index, cells_vertices_xyz))

    cells_normal_vector = []
    cells_area = []
    for cell_id, vertices_xyz in cells_vertices_xyz_per_cell.items():

        # cells with only two dbonds have no defined area or normal vector; here we put them to zero.
        if len(vertices_xyz) < 3:
            cells_area.append(0.0)
            cells_normal_vector.append(np.array([0.0, 0.0, 0.0]))
            continue

        cell_center_xyz = cells.loc[cell_id][['center_x', 'center_y', 'center_z']].values

        next_vertices_xyz = np.roll(vertices_xyz, -1, axis=0)

        # Find vectors between adjacent vertices (in cc direction)
        inter_vertex_vectors_cc = next_vertices_xyz - vertices_xyz
        vertex_to_cell_center_vector = cell_center_xyz - vertices_xyz

        cell_triangles_normal_vectors = np.cross(inter_vertex_vectors_cc, vertex_to_cell_center_vector)

        # Calculate cell unit normal vector from the area weighted average of normal vectors on triangles.
        cell_normal_vector = sum(cell_triangles_normal_vectors)
        cell_normal_vector /= np.sqrt(cell_normal_vector.dot(cell_normal_vector) + EPSILON)

        # Calculate cell area from the area of the triangles (half norm of triangle normal vectrors).
        cell_area = 0.5 * sum([np.sqrt(normal.dot(normal)) for normal in cell_triangles_normal_vectors])

        cells_area.append(cell_area)
        cells_normal_vector.append(cell_normal_vector)

    return np.array(cells_area), np.array(cells_normal_vector)


# Update triangels table with area and unit normal vector given vertices and table with triangle vertex ids.
def calculate_triangle_area_and_unit_normal_vector(vertices, triangles):
    print('Calculate area and unit-normal for all triangles...')

    triangles_pos_1 = vertices[['x_pos', 'y_pos', 'z_pos']].loc[triangles['vertex_id_1'].values].values
    triangles_pos_2 = vertices[['x_pos', 'y_pos', 'z_pos']].loc[triangles['vertex_id_2'].values].values
    triangles_pos_3 = vertices[['x_pos', 'y_pos', 'z_pos']].loc[triangles['vertex_id_3'].values].values

    triangles_E12 = triangles_pos_2 - triangles_pos_1
    triangles_E13 = triangles_pos_3 - triangles_pos_1

    triangles_normal_vector = np.cross(triangles_E12, triangles_E13)
    triangles_normal_magnitude = np.sqrt(np.einsum('ij, ij->i', triangles_normal_vector, triangles_normal_vector))
    triangles_unit_normal_vector = np.divide(triangles_normal_vector.transpose(),
                                             triangles_normal_magnitude + EPSILON).transpose()

    # Express the orientation of the normal vector in two angles with the X and Y axis.
    # These angles is how much to ratate along the respective axis to rotate the z unit vector to the triangle normal.
    nx, ny, nz = zip(*triangles_unit_normal_vector)
    triangles_normal_ThetaX = -np.arctan2(ny, nz)
    triangles_normal_ThetaY = np.arctan2(nx, np.sqrt(1 - np.multiply(nx, nx)))

    triangles['dr_12_x'] = triangles_E12.T[0]
    triangles['dr_12_y'] = triangles_E12.T[1]
    triangles['dr_12_z'] = triangles_E12.T[2]
    
    triangles['dr_13_x'] = triangles_E13.T[0]
    triangles['dr_13_y'] = triangles_E13.T[1]
    triangles['dr_13_z'] = triangles_E13.T[2]

    triangles['area'] = 0.5 * triangles_normal_magnitude
    
    triangles['normal_x'] = triangles_unit_normal_vector.T[0]
    triangles['normal_y'] = triangles_unit_normal_vector.T[1]
    triangles['normal_z'] = triangles_unit_normal_vector.T[2]
  
    triangles['rotation_angle_x'] = triangles_normal_ThetaX
    triangles['rotation_angle_y'] = triangles_normal_ThetaY

    return triangles


# Function returns principle curvatures kappa1 and kappa2 given the mean and gaussian curvature defined on a set of
# patches (triangles, cells etc.). If H^2 < K, and the principle curvatures are negative, we set the imaginary part to 0.
# Input and output as double column as pandas dataframe.
# Output as dataframe. When ordered = 'value', kappa1 >= kappa2,
#                                   = 'magnitude' abs(kappa1) > abs(kappa2)
def calculate_patches_principle_curvatures(patches, ordered = 'value'):

    assert 'mean_curvature' in patches.columns and 'gaussian_curvature' in patches.columns

    # First calculate principe curvatures from gaussian and mean curvatures (make sure principle curvatures are not imaginary).
    # kappa1 = H + sqrt(H**2 - K), kappa2 = H - sqrt(H**2 - K)
    patches_imaginary_kappa = patches['mean_curvature'] ** 2 < patches['gaussian_curvature']
    patches_sqrt_term = patches['mean_curvature'] ** 2 - patches['gaussian_curvature']
    patches_sqrt_term[patches_imaginary_kappa] = 0.0
    print('Fraction of Cij with imaginary Eigenvalues:', sum(patches_imaginary_kappa) / len(patches))

    patches_kappa1 = (patches['mean_curvature'] + np.sqrt(patches_sqrt_term)).values
    patches_kappa2 = (patches['mean_curvature'] - np.sqrt(patches_sqrt_term)).values

    if ordered == 'magnitude':
        #Make sure kappa_1 corresponds to eigenvalue with heighest magnitude.
        patches_mag_k1_smaller_than_mag_k2 = abs(patches_kappa1) < abs(patches_kappa2)
        kappa1_values_to_swap = patches_kappa1[patches_mag_k1_smaller_than_mag_k2]
        patches_kappa1[patches_mag_k1_smaller_than_mag_k2] = patches_kappa2[patches_mag_k1_smaller_than_mag_k2]
        patches_kappa2[patches_mag_k1_smaller_than_mag_k2] = kappa1_values_to_swap

    patches_principle_curvatures           = pd.DataFrame(index=patches.index)
    patches_principle_curvatures['kappa1'] = patches_kappa1
    patches_principle_curvatures['kappa2'] = patches_kappa2

    return patches_principle_curvatures


#Calculate the integrated curvature tensor on each of the dbonds in the network.
#Each dbond spans a cylinder, which is inegrated over its length and angle between adjacent normal vectors.
#Output in Euclidean frame. Integrated curvature tensor of dbond is assumed equal to its conjugate.
def calculate_integrated_curvature_tensor_on_dbonds(triangles, triangle_dbonds, triangle_vertices, dbonds_mean_curvature):

    print('Calculate integrated curvature tensor on each dbond.')

    #First select all dbonds that are not on the boundary (the integrated curvature tensor on the boundary is assumed zero).
    dbonds_conj_dbond_id     = triangle_dbonds['conj_dbond_id'].values
    dbonds_not_at_boundary   = (triangle_dbonds['triangle_id'] > -1).values & (triangle_dbonds.iloc[dbonds_conj_dbond_id]['triangle_id'] > -1).values
    triangle_internal_dbonds = triangle_dbonds[dbonds_not_at_boundary]

    #Get index of internal dbonds with has a lower index than its conjugate dbond: these are called bonds.
    bonds_index    = triangle_internal_dbonds[triangle_internal_dbonds.index < triangle_internal_dbonds['conj_dbond_id']].index
    triangle_bonds = triangle_internal_dbonds.loc[bonds_index]

    # Calculate integrated curvature tensor on each bond (aligned along z-axis)
    bonds_int_curvature_tensor_z_axis = np.array([ 0.5*length*np.array([[ -0.5*(np.sin(2*sign_theta) - 2*sign_theta), -np.sin(sign_theta)**2, 0.0],
                                                                        [ -np.sin(sign_theta)**2,  0.5*(np.sin(2*sign_theta) + 2*sign_theta), 0.0],
                                                                        [                    0.0,                   0.0,                      0.0]])
                                                   for sign_theta, length in dbonds_mean_curvature[['signed_angle', 'length']].loc[bonds_index].values])

    # Find normal vector on bond and z-axis
    bonds_right_vid = triangle_bonds['vertex_id'].values
    bonds_left_vid  = triangle_dbonds.iloc[ triangle_bonds['left_dbond_id'].values ]['vertex_id'].values
    triangle_vertices_positions = triangle_vertices[['x_pos', 'y_pos', 'z_pos']].values

    bonds_dist_vector      = triangle_vertices_positions[bonds_left_vid] - triangle_vertices_positions[bonds_right_vid]
    bonds_length           = np.sqrt(np.sum(bonds_dist_vector ** 2, axis=1))
    bonds_unit_dist_vector = bonds_dist_vector / (bonds_length.reshape(-1,1) + MM.EPSILON)

    zaxis_bonds_normal      = np.cross(np.array([0., 0., 1.]), bonds_unit_dist_vector)
    zaxis_bonds_normal_norm = np.sqrt(np.sum(zaxis_bonds_normal ** 2, axis=1))
    zaxis_bonds_unit_normal = zaxis_bonds_normal / (zaxis_bonds_normal_norm + MM.EPSILON).reshape(-1, 1)
    #zero_norm_normal_idx    = np.where(zaxis_bonds_normal_norm < MM.EPSILON)[0]
    #zaxis_bonds_unit_normal[zero_norm_normal_idx] = np.array([[1., 0. , 0.]] * len(zero_norm_normal_idx))

    # Find angle between bond and z-axis.
    bonds_zaxis_cos_angle = bonds_unit_dist_vector.T[2]
    bonds_zaxis_sin_angle = np.sqrt(np.sum(zaxis_bonds_normal ** 2, axis=1))
    bonds_zaxis_angle     = np.arccos(bonds_zaxis_cos_angle)

    # Align the long axis of the int curvature (z-axis) with the dbond axis.
    bonds_int_curvature_tensor_long_axis_rotated = MM.rotation(bonds_int_curvature_tensor_z_axis, bonds_zaxis_angle, zaxis_bonds_unit_normal)


    # For each dbond, find the mean of the adjacent triangle normals.
    bonds_triangle_id = triangle_bonds['triangle_id']
    conj_bonds_triangle_id = triangle_dbonds['triangle_id'].loc[triangle_bonds['conj_dbond_id'].values]

    bonds_left_triangle_normal  = triangles.loc[bonds_triangle_id][['normal_x', 'normal_y', 'normal_z']].values
    bonds_right_triangle_normal = triangles.loc[conj_bonds_triangle_id][['normal_x', 'normal_y', 'normal_z']].values

    bonds_mean_normal = bonds_left_triangle_normal + bonds_right_triangle_normal
    bonds_mean_normal_norm = np.sqrt(np.sum(bonds_mean_normal ** 2, axis=1))
    bonds_mean_unit_normal = bonds_mean_normal / (bonds_mean_normal_norm.reshape(-1, 1) + MM.EPSILON)

    # For the integrated curvature tensor, find eigenvector perpendicular to cylinder.
    # This is the eigen vector with the eigenvalue lambda_2 such that: lambda_1 < lambda_2 < lambda_3.
    bonds_int_curvature_tensor_eigenvector_r = np.array([[0., 0., 0.]] * len(bonds_index))
    for bond_index, int_curvature_tensor in enumerate(bonds_int_curvature_tensor_long_axis_rotated):

        int_curvature_tensor_eig = np.linalg.eig(int_curvature_tensor)

        eigenvalues_abs = abs(int_curvature_tensor_eig[0])
        # Find eigenvalue that is in between the three eigenvalues.
        mid_eigenval_idx = np.argmax(abs((eigenvalues_abs - min(eigenvalues_abs)) * (eigenvalues_abs - max(eigenvalues_abs))))

        bonds_int_curvature_tensor_eigenvector_r[bond_index] = int_curvature_tensor_eig[1][:, mid_eigenval_idx]

    # Find angle between eigenvec_r of the rotated int curvature tensor and the mean normal between adjacent triangles of each dbond.
    # Find cos(theta) of angle between bonds.
    bond_mean_n_int_curv_eigenvec_r_cos_theta = np.array([eigenvec_r.dot(mean_n)
        for eigenvec_r, mean_n in zip(bonds_int_curvature_tensor_eigenvector_r, bonds_mean_unit_normal)])
    #Find singned sin(theta).
    bond_mean_n_int_curv_eigenvec_r_normal = np.cross(bonds_int_curvature_tensor_eigenvector_r, bonds_mean_unit_normal)
    bond_mean_n_int_curv_eigenvec_r_sin_theta = np.sqrt(np.sum(bond_mean_n_int_curv_eigenvec_r_normal ** 2, axis=1))
    bond_mean_n_int_curv_eigenvec_r_sin_theta_sign = np.sign(np.array([unit_l.dot(eigen_r_mean_n_normal)
        for unit_l, eigen_r_mean_n_normal in zip(bonds_unit_dist_vector, bond_mean_n_int_curv_eigenvec_r_normal)]))
    bond_mean_n_int_curv_eigenvec_r_sin_theta *= bond_mean_n_int_curv_eigenvec_r_sin_theta_sign
    bond_mean_n_int_curv_eigenvec_r_theta = np.arctan2(bond_mean_n_int_curv_eigenvec_r_sin_theta,
                                                       bond_mean_n_int_curv_eigenvec_r_cos_theta)

    # Rotate the int curvature tensor around the axis, such that eigenvector_r matched the mean normal vectors.
    bonds_int_curvature_tensor = MM.rotation(bonds_int_curvature_tensor_long_axis_rotated,
                                             bond_mean_n_int_curv_eigenvec_r_theta,
                                             bonds_unit_dist_vector)
    dbonds_int_curvature_tensor_dict = dict(zip(bonds_index, bonds_int_curvature_tensor))

    #Assign zero tensor to boundary dbonds.
    boundary_dbonds_id = np.setdiff1d(triangle_dbonds.index, triangle_dbonds[dbonds_not_at_boundary].index, assume_unique=True)
    for dbond_id in boundary_dbonds_id:
        dbonds_int_curvature_tensor_dict[dbond_id] = np.zeros((3,3))

    #Assign tensors to the conjugate dbonds as well; they are equal to the tensors on the bonds.
    conj_bonds_index = triangle_dbonds[triangle_dbonds.index > triangle_dbonds['conj_dbond_id']].index
    for conj_bond_id in conj_bonds_index:
        bond_id = triangle_dbonds['conj_dbond_id'].loc[conj_bond_id]
        dbonds_int_curvature_tensor_dict[conj_bond_id] = dbonds_int_curvature_tensor_dict[bond_id]

    return dbonds_int_curvature_tensor_dict


def calculate_average_curvature_tensor_on_patches(patches_triangle_ids_dict, dbonds_int_curvature_tensor_dict,
                                                  triangles, triangle_dbonds, patches_mean_gaussian_curvatures = []):
    # Set number of patches.
    Npatches = len(patches_triangle_ids_dict)

    print('Calculate integrated curvature tensor on', Npatches, 'patches.')

    # Calculate the area weighted mean and gaussian curvatures on each patch.
    if len(patches_mean_gaussian_curvatures) == 0:
        patches_mean_gaussian_curvatures =\
            MM.calculate_mean_and_gaussian_curvature_on_patches_from_triangles(patches_triangle_ids_dict, triangles)

    # Find all bond ids of the different patches of triangles. Make sure each bond id appears only once.
    patches_bonds_id = [np.unique(np.concatenate([triangle_dbonds[triangle_dbonds['triangle_id'] == triangle_id]['bond_id'].values
                                         for triangle_id in patch_triangle_ids]))
                         for patch_id, patch_triangle_ids in patches_triangle_ids_dict.items()]
    
    # Add up all intrinsic curvature tensors for the different patches of bond ids.
    patches_total_int_curvature_tensor = np.array([sum([dbonds_int_curvature_tensor_dict[bid] for bid in patch_bonds_id])
                                                   for patch_bonds_id in patches_bonds_id])
    patches_total_int_curvature_tensor_dict =\
        dict(zip(patches_triangle_ids_dict.keys(), patches_total_int_curvature_tensor))

    # Get sorted eigenvectors from largest to smallest absolute values from the total integrated curvature tensor.
    patches_int_curvature_basis_eigenval_dict, patches_int_curvature_principle_basis_dict = \
        MM.calculate_tensor_eigenvalues_and_eigenvector(patches_total_int_curvature_tensor_dict, 'sorted_basis')

    # Get int curvature tensor principle basis vectors from dictionary.
    patches_int_curvature_tensor_principle_basis_vectors =\
        np.array([basis for patch_id, basis in patches_int_curvature_principle_basis_dict.items()])

    # Define curvature tensor in principle basis with eigenvalues.
    patches_principle_curvatures = MM.calculate_patches_principle_curvatures(patches_mean_gaussian_curvatures, ordered='magnitude')
    patches_int_curvature_tensor_principle_basis = np.array([np.diag([kmax, kmin, 0.0])
                                        for kmax, kmin in patches_principle_curvatures[['kappa1', 'kappa2']].values])

    # Express curvature tensor in euclidean basis.
    patches_curvature_tensor_euclidean_basis = \
        MMdg.tensor_basis_transformation(patches_int_curvature_tensor_principle_basis, patches_int_curvature_tensor_principle_basis_vectors)

    return dict(zip(patches_triangle_ids_dict.keys(), patches_curvature_tensor_euclidean_basis))


#Update the triangles table with the mean and gaussian curvature.
def calculate_triangle_mean_and_gaussian_curvature(dbonds, vertices, triangles, dbonds_mean_curvature):
    
    print('Calculate mean and gaussian curvature on all triangles...')
    
    dbonds_by_vertex_id = dbonds.groupby('vertex_id')

    triangles_mean_curvature     = np.array([0.] * len(triangles))
    triangles_gaussian_curvature = np.array([0.] * len(triangles))
    
    for triangle_id in triangles.index:

        triangle_area = triangles.at[triangle_id, 'area']

        if triangle_area == 0.0:
            continue

        triangle_dbonds_ids = dbonds[ dbonds['triangle_id'] == triangle_id ].index.values.astype(int)
        triangle_vertex_ids = triangles.iloc[triangle_id][['vertex_id_1', 'vertex_id_2', 'vertex_id_3']].values.astype(int)
        #Find total area of the triangles connected to each vertex.
        triangle_vertices_connected_triangles_id = [dbonds[dbonds['vertex_id'] == vid]['triangle_id'].values
                                                    for vid in triangle_vertex_ids]
        triangle_vertices_connected_triangles_id =  [[tid for tid in tids if tid > -1] for tids in triangle_vertices_connected_triangles_id]

        triangle_vertices_connected_triangles_total_area = [triangles.iloc[tids]['area'].sum()
                                                            for tids in triangle_vertices_connected_triangles_id]

        triangle_angular_defects = [vertices['angular_defect'].at[vid] for vid in triangle_vertex_ids]
        
        triangle_dbonds_mean_curvature = [dbonds_mean_curvature['int_mean_curvature'].iat[dbond_id] for dbond_id in triangle_dbonds_ids]
            
        triangle_vertex_order = [len( dbonds_by_vertex_id.get_group(vid) ) for vid in triangle_vertex_ids]

        ### There are several definitions for the Gaussian (and mean) curvature on the triangle.
        # We can weight the angular defect \beta_i between the N_i connected triangles around vertex i as:
        # 1. Uniformly     : \beta_i / N_i
        # 2. Area weighted : \beta_i / A_i^tot (A^i_tot is total area of connected triangles at vertex i)
        # 3. Angle weighted: \beta_i * \alpha_ij / \alpha_i^tot 
        #    (\alpha_i^tot is the sum of the angles, sum_j \alpha_ij, that the connected triangles make with the vertex).
        # Def 1:
        #triangles_gaussian_curvature[triangle_id] = sum([triangle_angular_defects[idx] / triangle_vertex_order[idx] for idx in range(3)]) / triangle_area
        # Def 2:
        triangles_gaussian_curvature[triangle_id] = sum([triangle_angular_defects[idx] / triangle_vertices_connected_triangles_total_area[idx] for idx in range(3) ])

        ## Mean curvature definition.
        triangles_mean_curvature[triangle_id] = 0.5 * sum(triangle_dbonds_mean_curvature) / triangle_area

    triangles['gaussian_curvature'] = triangles_gaussian_curvature
    triangles['mean_curvature']     = triangles_mean_curvature

    return triangles


#For each dbond, calculate the mean curvature as 0.5 * bond_length * CosInv(n1.n2), where n1,n2 are the unit normal vectors on the adjacent triangles,
#given tables of directed bonds, vertices and triangles (for their normal vectors).
def calculate_mean_curvature_on_dbonds(dbonds, vertices, triangles):

    print('Calculate mean curvature on all dbonds...')
    
    int_mean_curvature_on_dbond = dict(zip( dbonds.index, np.zeros(len(dbonds)) ))
    int_signed_angle_on_dbond   = dict(zip( dbonds.index, np.zeros(len(dbonds)) ))
    dbond_length_dict           = dict(zip( dbonds.index, np.zeros(len(dbonds)) ))
    for dbond_id, dbond in dbonds.iterrows():

        #Only make calculation for one of the dbond in the pair.
        if dbond_id > dbond['conj_dbond_id']:
            continue

        conj_triangle_id = dbonds.iloc[ dbond['conj_dbond_id'] ]['triangle_id']
        this_triangle_id = dbond['triangle_id']

        #Calculate the length between the vertices the dbond connects.
        right_vertex_id = dbond['vertex_id']
        left_vertex_id  = dbonds.iloc[ dbond['left_dbond_id'] ]['vertex_id']
        dR = vertices.loc[left_vertex_id, ['x_pos','y_pos','z_pos']].values - vertices.loc[right_vertex_id, ['x_pos','y_pos','z_pos']].values
        dbond_length = np.sqrt( dR.dot(dR) )

        #dbonds at the boundary, are assumed to have no mean curvature.
        if conj_triangle_id < 0 or this_triangle_id < 0:
            dbond_length_dict[dbond_id]               = dbond_length
            dbond_length_dict[dbond['conj_dbond_id']] = dbond_length
            continue

        #Obtain normal vectors on both triangles on the sides.
        n1 = triangles.loc[this_triangle_id, ['normal_x', 'normal_y', 'normal_z']].values
        n2 = triangles.loc[conj_triangle_id, ['normal_x', 'normal_y', 'normal_z']].values

        #Determine the sign of the angle between the normal vectors.
        #If the cross product of the normal vectors is in the same direction as the dbond, its +1, else -1.
        cross_n1n2 = np.cross(n1, n2)
        sign = 1. if cross_n1n2.dot(dR) >= 0 else -1
        
        #Store mean curvature on the dbond and its conjugate dbond.
        innerprod_n1n2 = n1.dot(n2)
        if abs(innerprod_n1n2) > 1.: innerprod_n1n2 /= innerprod_n1n2
        signed_angle_n1n2 = sign * np.arccos(innerprod_n1n2)
        bond_mean_curvature = 0.5 * signed_angle_n1n2 * dbond_length



        int_mean_curvature_on_dbond[dbond_id]                 = bond_mean_curvature
        int_mean_curvature_on_dbond[ dbond['conj_dbond_id'] ] = bond_mean_curvature
        int_signed_angle_on_dbond[dbond_id]                   = signed_angle_n1n2
        int_signed_angle_on_dbond[ dbond['conj_dbond_id'] ]   = signed_angle_n1n2
        dbond_length_dict[dbond_id]                           = dbond_length
        dbond_length_dict[ dbond['conj_dbond_id'] ]           = dbond_length

    int_mean_curvature_on_dbond_df = pd.DataFrame([], index=dbonds.index)
    int_mean_curvature_on_dbond_df['int_mean_curvature'] = pd.DataFrame.from_dict(int_mean_curvature_on_dbond, orient='index')
    int_mean_curvature_on_dbond_df['signed_angle']       = pd.DataFrame.from_dict(int_signed_angle_on_dbond, orient='index')
    int_mean_curvature_on_dbond_df['length']             = pd.DataFrame.from_dict(dbond_length_dict, orient='index')

    return int_mean_curvature_on_dbond_df

    
#Calculate the angular defect for every vertex and update the vertices table. 
def calculate_angular_defect_per_vertex(dbonds, vertices, triangles):

    print('Calculate angular defect for all vertices...')

    # Function returns a signed magnitude of the cross product between vectors_a and vectors_b
    def determine_sign_vector_pair(vectors_a, vectors_b, orientation_vector):
        # For each vectors pair, calculate its normal vector.
        vectors_pair_normal_vector = np.cross(vectors_a, vectors_b)
        vectors_pair_normal_length = np.sqrt(
            np.einsum('ij, ij->i', vectors_pair_normal_vector, vectors_pair_normal_vector))

        # Determine sign of normal vector (outward or inward) by comparing to an orientation vector.
        vectors_pair_sign = vectors_pair_normal_vector.dot(orientation_vector)
        vectors_pair_sign = np.sign(vectors_pair_sign)

        return vectors_pair_sign * vectors_pair_normal_length

    ### Before we can calculate angular defect per vertex, we need to sort all the vertices connected to each vertex
    # in the network in counter-clockwise direction. This requires determining the angles between the vectors
    # from each vertex to its neigbours.

    #Group dbonds by vertex_id and get all vertex positions.
    dbonds_by_vertex_id = dbonds.groupby('vertex_id')
    vertices_xyz = vertices[['x_pos', 'y_pos', 'z_pos']].values

    #Orientation vector defines the direction of the surface at every vertex position.
    #We define the orientation vector als the normal vector on a triangle the vertex is connected to.
    #First make dict: vertex_id -> triangle_id.
    vertex_id_to_triangle_id = dict(zip(vertices.index, [0] * len(vertices)))

    for triangle_id, row in triangles[['vertex_id_1', 'vertex_id_2', 'vertex_id_3']].iterrows():
        if triangle_id < 0:
            continue

        vid1, vid2, vid3 = row
        vertex_id_to_triangle_id[vid1] = triangle_id
        vertex_id_to_triangle_id[vid2] = triangle_id
        vertex_id_to_triangle_id[vid3] = triangle_id

    triangle_id_array = np.array([triangle_id for vertex_id, triangle_id in vertex_id_to_triangle_id.items()])

    vertices_orientation_vector = triangles[['normal_x', 'normal_y', 'normal_z']].loc[triangle_id_array].values

    # Make vertex_id -> connected vertex ids dictionary.
    connected_dbonds_left_dbond_id = [dbonds_by_vertex_id.get_group(vid)['left_dbond_id'].values for vid in vertices.index]
    connected_vertex_ids = [dbonds.iloc[connected_left_dbond_ids]['vertex_id'].values
                            for connected_left_dbond_ids in connected_dbonds_left_dbond_id]
    connected_vertex_ids_per_vertex_id = dict(zip(vertices.index, connected_vertex_ids))

    # Make connected_vertices_xyz - vertex_xyz to vertex_id dictionary.
    dR_xyz_connected_vertices = [vertices_xyz[connected_vids] - vertices_xyz[vid] for vid, connected_vids in
                                 connected_vertex_ids_per_vertex_id.items()]

    # Find sin_theta and cos_theta between dR0 and dR vectors for all vertices and use it to calculate angle: 0->2*pi.
    dR_connected_sin_theta_per_vertex = [determine_sign_vector_pair([dR_xyz[0]] * len(dR_xyz), dR_xyz, orientation_vector)
                                         for dR_xyz, orientation_vector in zip(dR_xyz_connected_vertices, vertices_orientation_vector)]
    dR_connected_cos_theta_per_vertex = [vectors.dot(vectors[0]) for vectors in dR_xyz_connected_vertices]
    angles_connected_vertices_per_vertex = [np.arctan2(sin_theta, cos_theta) for sin_theta, cos_theta in
                                            zip(dR_connected_sin_theta_per_vertex, dR_connected_cos_theta_per_vertex)]
    #Add 2*pi to every angle that is negative.
    for angles_connected in angles_connected_vertices_per_vertex:
        angles_connected_negative_index = np.where(angles_connected < 0)[0]
        angles_connected[angles_connected_negative_index] = angles_connected[angles_connected_negative_index] + 2*np.pi

    # Find counter-clockwise order of the vertices connected to each vertex.
    vertex_order_per_vertex = [np.argsort(angles_connected_vertices) for angles_connected_vertices in
                               angles_connected_vertices_per_vertex]

    # Sort vertex_ids around each vertex in cc direction.
    # sorted_connected_vertex_ids_per_vertex_id = [connected_vertex_ids[vertex_order]
    #                                             for connected_vertex_ids, vertex_order in zip(connected_vertex_ids, vertex_order_per_vertex)]

    # Sort dR from each vertex in cc direction.
    sorted_dR_connected_vertices = [dR_connected_vertex[vertex_order]
                                    for dR_connected_vertex, vertex_order in zip(dR_xyz_connected_vertices, vertex_order_per_vertex)]

    # Calculate the angles between consecutive edges connected to each vertex.
    dR_consecutive_sin_theta_per_vertex = [determine_sign_vector_pair(sorted_vectors, np.roll(sorted_vectors, -1, axis=0), orientation_vector)
                                           for sorted_vectors, orientation_vector in zip(sorted_dR_connected_vertices, vertices_orientation_vector)]
    dR_consecutive_cos_theta_per_vertex = [np.sum(np.roll(sorted_vectors, -1, axis=0) * sorted_vectors, axis=1)
                                           for sorted_vectors in sorted_dR_connected_vertices]
    angles_consecutive_vertices_per_vertex = [np.arctan2(sin_theta, cos_theta)
                                              for sin_theta, cos_theta in zip(dR_consecutive_sin_theta_per_vertex,
                                                                              dR_consecutive_cos_theta_per_vertex)]
    # Add 2*pi to every angle that is negative.
    for consecutive_angles in angles_consecutive_vertices_per_vertex:
        consecutive_angles_negative_index = np.where(consecutive_angles < 0)[0]
        consecutive_angles[consecutive_angles_negative_index] = consecutive_angles[consecutive_angles_negative_index] + 2*np.pi

    total_angle_per_vertex = [np.sum(angles) for angles in angles_consecutive_vertices_per_vertex]
    total_angle_defect_per_vertex_id = np.array([MM.angle_difference(vertex_angle, 2 * np.pi) for vertex_angle in total_angle_per_vertex])

    #For angular defects close to zero (or pi), set these defects to zero.
    angle_defect_close_to_zero_index = \
        np.where([(np.abs(ang_def) < EPSILON) or (np.abs(MM.angle_difference(ang_def, np.pi)) < EPSILON) for ang_def in total_angle_defect_per_vertex_id])[0]
    total_angle_defect_per_vertex_id[angle_defect_close_to_zero_index] = 0.0

    #Create dictionary and make pandas DataFrame and add to vertices table.
    vertices['angular_defect'] = total_angle_defect_per_vertex_id
    
    #total_angle_defect_per_vertex_id = dict( zip( vertices.index, total_angle_defect_per_vertex_id) )
    #angular_defect_per_vertex_table = pd.DataFrame.from_dict(total_angle_defect_per_vertex_id, orient='index')
    #vertices = pd.concat([vertices, angular_defect_per_vertex_table], axis=1)
    #vertices.rename(columns={0:'angular_defect'}, inplace=True)
    
    return vertices


#This function calculates the elongation tensor in 3D:
# Rot_x . Rot_y . Rot_z |q| Diag(-1, 1, 0) Inverse( Rot_x . Rot_y . Rot_z ) or
# Rot_x . Rot_y Q_2d . Inverse( Rot_x . Rot_y )
#It adds the elongation norm and orientation to the triangle table.
def calculate_triangle_elongation_tensor(triangles):

    print('Calculate elongation tensor for all triangels.')

    triangles_dR12_xy, triangles_dR13_xy = MM.rotate_triangles_into_xy_plane(triangles)

    splitTrianglesS = MM.calculate_triangle_state_tensor_symmetric_antisymmetric(triangles_dR12_xy, triangles_dR13_xy)

    # Calculate the norm of nematic tensor and area of all triangles.
    triangles_elongation_norm = np.array(list(map(lambda S: MM.Qnorm(*S), splitTrianglesS)))
    triangles_elongation_norm[ np.where(np.isnan(triangles_elongation_norm)) ] = 0.0

    #In-plane angle of nematic is twice its orientation angle.
    elongation_twophi = splitTrianglesS.T[2]
    triangles['rotation_angle_z']         = splitTrianglesS.T[0]
    triangles['elongation_tensor_norm'] = triangles_elongation_norm
    triangles['elongation_tensor_twophi'] = elongation_twophi

    #For all triangles, calculate the components of the elongation tensor qxx and qxy.
    cos_two_phi, sin_two_phi = np.cos( elongation_twophi ), np.sin( elongation_twophi )
    triangles_q_xx, triangles_q_xy = triangles_elongation_norm * cos_two_phi, triangles_elongation_norm * sin_two_phi

    #Generate the 2d elongation tensors in the xy-plane, written in 3d matrix form.
    triangles_q_2d = [ np.array([[qxx,  qxy, 0.],
                                 [qxy, -qxx, 0.],
                                 [ 0.,   0., 0.]]) for qxx, qxy in zip(triangles_q_xx, triangles_q_xy) ]

    triangles_q_3d = MM.transform_matrices_from_xy_plane_to_triangle_plane(triangles_q_2d,
                                                                           triangles['rotation_angle_x'].values,
                                                                           triangles['rotation_angle_y'].values)

    triangles_elongation_tensor = dict( zip(triangles.index, triangles_q_3d) )

    return triangles_elongation_tensor, triangles


#This function calculates the deformation rate tensor and the 3d shear tensor for the triangles present in
#two consecutive frames, and calculates their area weighted average for the whole tissue.
def calculate_triangle_deformation_rate_tensor(selected_triangles_this_frame,
                                               selected_triangles_next_frame,
                                               delta_t, N_interpolation_steps):

    # Rotate selected triangles into the xy-plane for both consecutive frames.
    E12_xy_this_frame, E13_xy_this_frame = MM.rotate_triangles_into_xy_plane(selected_triangles_this_frame)
    E12_xy_next_frame, E13_xy_next_frame = MM.rotate_triangles_into_xy_plane(selected_triangles_next_frame)

    # Create triangle_deformation_rate_tensor DataFrame and fill it with triangle shear calculation.
    triangles_deformation_rate_tensor = pd.DataFrame(0.0, index=selected_triangles_this_frame.index,
                                                     columns=MM.triangles_deformation_rate_tensor_columns)
    triangles_deformation_rate_tensor['triangle_id'] = selected_triangles_this_frame['triangle_id']

    triangles_deformation_rate_tensor, tissue_deformation_rate_tensor = \
        MM.calculate_triangle_deformation_with_interpolation(E12_xy_this_frame, E13_xy_this_frame,
                                                             E12_xy_next_frame, E13_xy_next_frame,
                                                             triangles_deformation_rate_tensor,
                                                             N_interpolation_steps)

    # Convert deformation gradient properties to rates per hours.
    triangles_deformation_rate_tensor[MM.triangles_deformation_rate_tensor_columns[1:]] /= delta_t
    tissue_deformation_rate_tensor[MM.tissue_deformation_rate_tensor_columns[2:]]       /= delta_t

    # Generate 3d shear tensor matrices.
    triangles_v_xx = triangles_deformation_rate_tensor['shear_xx']
    triangles_v_xy = triangles_deformation_rate_tensor['shear_xy']
    triangles_shear_2d = [np.array([[vxx,  vxy, 0.],
                                    [vxy, -vxx, 0.],
                                    [ 0.,   0., 0.]]) for vxx, vxy in zip(triangles_v_xx, triangles_v_xy)]

    triangles_shear_3d =\
        MM.transform_matrices_from_xy_plane_to_triangle_plane(triangles_shear_2d,
                                                              selected_triangles_this_frame['rotation_angle_x'].values,
                                                              selected_triangles_this_frame['rotation_angle_y'].values)

    triangles_shear_tensor = dict(zip(selected_triangles_this_frame.index, triangles_shear_3d))

    return triangles_deformation_rate_tensor, tissue_deformation_rate_tensor, triangles_shear_tensor


#Given triangle 3x3 deformation rate tensor u_ij, calculate the triangle vorticity vector \omega.
# omega^1 = u_23 - u_32
# omega^2 = u_31 - u_13
# omega^3 = u_12 - u_21
# TODO: Check defintion of levi-civita symol: e_ij = sqrt(g) [[]0,1],[-1,0]]
def calculate_triangle_vorticity_vector(triangles_deformation_rate_tensor_Uij):

    triangles_vorticity = np.array([[0., 0., 0.] * len(triangles_deformation_rate_tensor_Uij)])

    triangles_vorticity_1 = triangles_deformation_rate_tensor_Uij.T[1][2] - triangles_deformation_rate_tensor_Uij.T[2][1]
    triangles_vorticity_2 = triangles_deformation_rate_tensor_Uij.T[2][0] - triangles_deformation_rate_tensor_Uij.T[0][2]
    triangles_vorticity_3 = triangles_deformation_rate_tensor_Uij.T[0][1] - triangles_deformation_rate_tensor_Uij.T[1][0]

    triangles_vorticity_vector = np.array([triangles_vorticity_1, triangles_vorticity_2, triangles_vorticity_3]).T

    return triangles_vorticity_vector


#Calculate the neigbour numbers of the cells given by their cell ids.
def calculate_cell_neighbour_number(cell_ids, sorted_dbonds_per_cell):
     
    neignbour_number_per_cell = {}
    for cell_id in cell_ids:
        sorted_dbonds = sorted_dbonds_per_cell[cell_id]
        neignbour_number_per_cell[cell_id] = len(sorted_dbonds)

    neignbour_number_per_cell = pd.DataFrame.from_dict(neignbour_number_per_cell, orient='index')
    
    return neignbour_number_per_cell


# Calculate the shear on each triangle, decompose it into elongation, corotation and correlation
# and calculate tissue wide area weighted averages.
def calculate_triangle_deformation_with_interpolation(E12_xy_this_frame, E13_xy_this_frame, E12_xy_next_frame,
                                                      E13_xy_next_frame, triangles_deformation_rate_tensor, \
                                                      N_interpolation_steps=100):
    # Decompose triangle vectors dR12 and dR13 into state properties without checks.
    def calculate_triangle_state_tensor_orientation_area_elongation(E12_xy, E13_xy):
        splitTrianglesS = MM.calculate_triangle_state_tensor_symmetric_antisymmetric(E12_xy, E13_xy)
        triangles_q_norm = np.array(list(map(lambda S: MM.Qnorm(*S), splitTrianglesS)))
        triangles_q_xx = triangles_q_norm * np.cos(splitTrianglesS.T[2])
        triangles_q_xy = triangles_q_norm * np.sin(splitTrianglesS.T[2])

        # theta, area, two_phi, q_norm, q_xx, q_xy
        return splitTrianglesS.T[0], splitTrianglesS.T[1] - splitTrianglesS.T[3], splitTrianglesS.T[2],\
               triangles_q_norm, triangles_q_xx, triangles_q_xy

    # Given triangle tensors dR = [dR12, dR13] in this and next frame,
    # calculate the shear tensor between frames and decompose in dArea, dPsi and the shear nematic dnemUxx, dnemUxy.
    def calculate_triangle_shear_tensor(E12_xy_this_frame, E13_xy_this_frame, E12_xy_next_frame, E13_xy_next_frame):
        # Create triangle state matrix [E12, E13]
        E12E13_next_frame = np.array(list(zip(zip(E12_xy_next_frame.T[0], E13_xy_next_frame.T[0]),
                                              zip(E12_xy_next_frame.T[1], E13_xy_next_frame.T[1]))))

        # Invert triangle tensors in this frame.
        # First check if triangles have non-zero determinant, in case of one or two vertex cells.
        #E12E13_this_frame_determinant = E12_xy_this_frame.T[0] * E13_xy_this_frame.T[1] - E13_xy_this_frame.T[0] * E12_xy_this_frame.T[1]
        #zero_determinant_index = np.where(E12E13_this_frame_determinant == 0.0)[0]
        #E12_xy_this_frame.T[0][zero_determinant_index] += EPSILON
        #E13_xy_this_frame.T[1][zero_determinant_index] += EPSILON

        E12E13_this_frame = np.array(list(zip(zip(E12_xy_this_frame.T[0], E13_xy_this_frame.T[0]),
                                              zip(E12_xy_this_frame.T[1], E13_xy_this_frame.T[1]))))
        E12E13_this_frame_inverse = list(map(np.linalg.inv, E12E13_this_frame))

        # E12E13_this_frame_inverse_den = E12_xy_this_frame.T[1] * E13_xy_this_frame.T[0] + E12_xy_this_frame.T[0] * E13_xy_this_frame.T[1]
        # E12E13_this_frame_inverse_xx = E13_xy_this_frame.T[1] / E12E13_this_frame_inverse_den
        # E12E13_this_frame_inverse_xy = E13_xy_this_frame.T[0] / E12E13_this_frame_inverse_den
        # E12E13_this_frame_inverse_yx = E12_xy_this_frame.T[1] / E12E13_this_frame_inverse_den
        # E12E13_this_frame_inverse_yy = E12_xy_this_frame.T[0] / E12E13_this_frame_inverse_den
        # E12E13_this_frame_inverse = zip( zip(E12E13_this_frame_inverse_xx, E12E13_this_frame_inverse_xy),\
        #                                 zip(E12E13_this_frame_inverse_yx, E12E13_this_frame_inverse_yy) )

        # Calculate deformation tensors for the triangles: M = R'.(R)^-1
        triangles_transformation_tensor = list(map(np.dot, E12E13_next_frame, E12E13_this_frame_inverse))
        triangles_transformation_tensor_transposed = list(map(np.transpose, triangles_transformation_tensor))

        # Calculate shear tensor the triangles.
        triangles_shear_tensor = list(
            map(lambda M_transposed: np.subtract(M_transposed, [[1, 0], [0, 1]]), triangles_transformation_tensor_transposed))

        # Decompose shear tensors in a trace, rotation and pure shear (nematic): U = A + Bnem
        # dU = 1/2dUkk * del_ij + dnemU_ij - dpsi * eps_ij
        triangles_shear_tensor_decomposed = np.array(list(map(MM.splitTensor_components, triangles_shear_tensor)))

        delUkk = np.multiply(triangles_shear_tensor_decomposed.T[0], 2)
        delnemUxx = triangles_shear_tensor_decomposed.T[2]
        delnemUxy = triangles_shear_tensor_decomposed.T[3]
        delPsi = np.multiply(triangles_shear_tensor_decomposed.T[1], -1)

        return delUkk, delnemUxx, delnemUxy, delPsi


    print('Performing triangle deformation interpolation for', len(E12_xy_this_frame), 'triangles with', N_interpolation_steps, 'interpolation steps.' )

    #Initialize tissue_deformation_rate_tensor Series.
    tissue_deformation_rate_tensor = pd.Series(0.0, index=MM.tissue_deformation_rate_tensor_columns)

    # Calculate the difference between triangle vectors between this and the next movie frame.
    delta_E12_xy = E12_xy_next_frame - E12_xy_this_frame
    delta_E13_xy = E13_xy_next_frame - E13_xy_this_frame
    delta_E12_xy_per_step = delta_E12_xy / N_interpolation_steps
    delta_E13_xy_per_step = delta_E13_xy / N_interpolation_steps

    # Now we linearly interpolate the triangle trajectories into N_interpolation_steps
    prev_E12_xy = E12_xy_this_frame
    prev_E13_xy = E13_xy_this_frame
    prev_theta, prev_area, prev_q_twophi, prev_q_norm, prev_q_xx, prev_q_xy = \
        calculate_triangle_state_tensor_orientation_area_elongation(prev_E12_xy, prev_E13_xy)
    start_total_area = sum(prev_area)
    prev_total_area = start_total_area
    for step in range(1, N_interpolation_steps + 1):
        next_E12_xy = prev_E12_xy + delta_E12_xy_per_step
        next_E13_xy = prev_E13_xy + delta_E13_xy_per_step

        # Find all triangle vectors which should be swapped.
        triangles_normal = np.cross(next_E12_xy, next_E13_xy)
        triangles_wrong_order_index = np.where(triangles_normal < 0)[0]
        # Swap the wrongly ordered triangles.
        temp = next_E12_xy[triangles_wrong_order_index]
        next_E12_xy[triangles_wrong_order_index] = next_E13_xy[triangles_wrong_order_index]
        next_E13_xy[triangles_wrong_order_index] = temp

        ### For all triangles, calculate deformation gradient tensor for this interpolation step.
        # Decompose triangle vectors in the next step into triangle state tensor.
        next_theta, next_area, next_q_twophi, next_q_norm, next_q_xx, next_q_xy = \
            calculate_triangle_state_tensor_orientation_area_elongation(next_E12_xy, next_E13_xy)

        # Decompose triangles transformation tensor u^m_ij into trance, traceless-symmetric and anti-symmetric parts.
        u_kk, u_nem_xx, u_nem_xy, del_psi = calculate_triangle_shear_tensor(prev_E12_xy, prev_E13_xy,
                                                                            next_E12_xy, next_E13_xy)
        # Calculate the mean corotation j_nematic
        del_phi = np.array(list(map(MM.angle_difference, prev_q_twophi, next_q_twophi))) / 2.0
        # g = np.sinh(2 * q_mean_norm) / (2 * q_mean_norm)
        # del_nem_j_prefac = -2*(g * del_theta + (1 - g) * del_phi)
        c = np.tanh(2 * prev_q_norm) / (2 * prev_q_norm + EPSILON)
        del_nem_j_prefac = -2 * (c * del_psi + (1 - c) * del_phi)
        j_nematic_xx = del_nem_j_prefac * -1 * prev_q_xy
        j_nematic_xy = del_nem_j_prefac * prev_q_xx

        triangles_deformation_rate_tensor['delta_area'] += u_kk
        triangles_deformation_rate_tensor['delta_elongation_xx'] += next_q_xx - prev_q_xx
        triangles_deformation_rate_tensor['delta_elongation_xy'] += next_q_xy - prev_q_xy
        triangles_deformation_rate_tensor['delta_elongation_twophi'] += del_phi
        triangles_deformation_rate_tensor['shear_xx']  += u_nem_xx
        triangles_deformation_rate_tensor['shear_xy']  += u_nem_xy
        triangles_deformation_rate_tensor['delta_psi'] += del_psi
        triangles_deformation_rate_tensor['shear_corotational_xx'] += j_nematic_xx
        triangles_deformation_rate_tensor['shear_corotational_xy'] += j_nematic_xy


        ### Calculate tissue wide area weighted averages of deformation rate tensor.

        # Calculate tissue shear and rotation.
        tissue_deformation_rate_tensor['shear_xx']  += np.sum(u_nem_xx * prev_area) / prev_total_area
        tissue_deformation_rate_tensor['shear_xy']  += np.sum(u_nem_xy * prev_area) / prev_total_area
        tissue_deformation_rate_tensor['delta_psi'] += np.sum(del_psi * prev_area) / prev_total_area
        # tissue_deformation_rate_tensor['shear_corotational_xx'] += ...
        # tissue_deformation_rate_tensor['shear_corotational_xy'] += ...

        # Calculate covariance between elongation and area change.
        tissue_deformation_rate_tensor['covariance_elongation_xx'] += \
            np.sum(u_kk * prev_q_xx * prev_area) / prev_total_area
        tissue_deformation_rate_tensor['covariance_elongation_xy'] += \
            np.sum(u_kk * prev_q_xy * prev_area) / prev_total_area
        # Calculate covariance between corotation and area change.
        tissue_deformation_rate_tensor['covariance_corotation_xx'] += np.sum(j_nematic_xx * prev_area) / prev_total_area
        tissue_deformation_rate_tensor['covariance_corotation_xy'] += np.sum(j_nematic_xy * prev_area) / prev_total_area


        ### Safe next triangle as this triangle for next step.
        prev_theta, prev_area, prev_q_twophi, prev_q_norm, prev_q_xx, prev_q_xy = next_theta, next_area, next_q_twophi, next_q_norm, next_q_xx, next_q_xy
        prev_E12_xy = next_E12_xy
        prev_E13_xy = next_E13_xy
        prev_total_area = sum(next_area)


    tissue_deformation_rate_tensor['delta_area'] = (prev_total_area - start_total_area) / start_total_area

    return triangles_deformation_rate_tensor, tissue_deformation_rate_tensor


# Function takes triangles table, and rotates the triangle vectors E12 and E13 into the xy plane using the
# angles rotation_angle_x and rotation_angle_y.
def rotate_triangles_into_xy_plane(triangles, to_xy_plane = True):
    # For all triangles, get triangle vectors E12 and E13 and normal vector orientation.
    triangles_E12_xyz = triangles[['dr_12_x', 'dr_12_y', 'dr_12_z']].values
    triangles_E13_xyz = triangles[['dr_13_x', 'dr_13_y', 'dr_13_z']].values
    triangles_normal_ThetaX = triangles['rotation_angle_x'].values
    triangles_normal_ThetaY = triangles['rotation_angle_y'].values

    # Rotate the triangle vectors into the XY plain.
    # Minus signs for angles because you rotate normal vector towards z-unit vector.
    cosX, sinX = np.cos(-triangles_normal_ThetaX), np.sin(-triangles_normal_ThetaX)
    cosY, sinY = np.cos(-triangles_normal_ThetaY), np.sin(-triangles_normal_ThetaY)

    # Define Rot_x . Rot_y rotation matrix for every triangle.
    Inv_rotation_x = [np.array(((1, 0, 0), (0, cX, -sX), (0, sX, cX))) for cX, sX in zip(cosX, sinX)]
    Inv_rotation_y = [np.array(((cY, 0, sY), (0, 1, 0), (-sY, 0, cY))) for cY, sY in zip(cosY, sinY)]
    triangles_Inverse_RotX_RotY = [np.dot(Inv_RotY, Inv_RotX) for Inv_RotX, Inv_RotY in
                                   zip(Inv_rotation_x, Inv_rotation_y)]

    # Find triangle side vectors in the xy plane.
    triangles_E12_xy = np.array(list(map(np.dot, triangles_Inverse_RotX_RotY, triangles_E12_xyz)))
    triangles_E13_xy = np.array(list(map(np.dot, triangles_Inverse_RotX_RotY, triangles_E13_xyz)))
    # Make 2d vectors out of in-plane vectors.
    triangles_E12_xy = np.stack((triangles_E12_xy.T[0], triangles_E12_xy.T[1]), axis=-1)
    triangles_E13_xy = np.stack((triangles_E13_xy.T[0], triangles_E13_xy.T[1]), axis=-1)

    return triangles_E12_xy, triangles_E13_xy


#Given an array of 3d matrices describing a quantity in the xy-plane,
#function returns matrix transformed to the plane of the triangle.
def transform_matrices_from_xy_plane_to_triangle_plane(matrices_2d, triangles_normal_ThetaX, triangles_normal_ThetaY):

    #Check if input arrays have correct dimensions.
    assert len(matrices_2d) == len(triangles_normal_ThetaX) and len(matrices_2d) == len(triangles_normal_ThetaY)

    # Rotate the triangle vectors into the XY plain.
    cosX, sinX = np.cos(triangles_normal_ThetaX), np.sin(triangles_normal_ThetaX)
    cosY, sinY = np.cos(triangles_normal_ThetaY), np.sin(triangles_normal_ThetaY)

    # Define Rot_x.Rot_y and its inverse rotation matrix for every triangle.
    rotation_x = [np.array(((1, 0, 0), (0, cX, -sX), (0, sX, cX))) for cX, sX in zip(cosX, sinX)]
    rotation_y = [np.array(((cY, 0, sY), (0, 1, 0), (-sY, 0, cY))) for cY, sY in zip(cosY, sinY)]

    triangles_RotX_RotY = [np.dot(RotX, RotY) for RotX, RotY in zip(rotation_x, rotation_y)]
    triangles_Inverse_RotX_RotY = [ np.linalg.inv(RxRy) for RxRy in triangles_RotX_RotY ]

    matrices_3d = np.array([triangles_RotX_RotY[idx].dot( matrices_2d[idx].dot(triangles_Inverse_RotX_RotY[idx]))
                               for idx in range(len(matrices_2d))])

    return matrices_3d


#Calculate the area weighted average of the triangle tensors expressed in the eucledean basis.
#Average over the triangles specified in grouped_triangles_index.
def calculate_area_weighted_mean_tensor_from_triangle_tensors(triangles, triangles_tensor_3d_dict, grouped_triangles_index_dict):

    #Get indices of all triangles that have a tensor defined.
    triangles_with_tensor_index    = np.array([triangle_index for triangle_index, tensor in triangles_tensor_3d_dict.items()])
    triangles_without_tensor_index = np.setdiff1d(triangles.index, triangles_with_tensor_index)

    #Set non specified tensors to zero.
    for tid in triangles_without_tensor_index:
        triangles_tensor_3d_dict[tid] = np.zeros((3,3))

    #For all groups of triangles, take the area weighted average of their triangle 3D tensors.
    averaged_tensors_dict = dict(zip(grouped_triangles_index_dict.keys(), np.zeros((len(grouped_triangles_index_dict), 3, 3))))
    for group_id, triangles_index in grouped_triangles_index_dict.items():

        triangles_tensor = np.array([triangles_tensor_3d_dict[tid] for tid in triangles_index])
        triangles_area   = triangles['area'].loc[triangles_index].values

        averaged_tensors_dict[group_id] = np.array(sum([area * tensor for area, tensor in
                                                        zip(triangles_area, triangles_tensor)])) / sum(triangles_area + EPSILON)

    return averaged_tensors_dict


def calculate_tensor_eigenvalues_and_eigenvector(tensors_dict, return_eigen_property = 'max'):
    """ Get tensor eigenvalues and eigenvectors with specified property.

        Parameters
        ----------
        tensor_dict : Dictionary
            Dictionary of tensor_id -> tensor (given by dxd matrix)
        return_eigen_property : {'max', 'min', 'abs_max', 'abs_min', 'sorted_basis'}
            Specify which eigenvalues/ eigenvectors to return. Options are: 
                - max (def)    : return eigenvalue/vec with highest value
                - min          : return eigenvalue/vec with lowest value
                - abs_max      : return eigenvalue/vec with highest norm
                - abs_min      : return eigenvalue/vec with smallest norm
                - sorted_basis : return base vectors and values sorted by the magnitude of the eigenvector.

        Returns
        -------
        tensors_spec_eigenvalue : Dictionary
            tensor_id -> requested eigenvalue

        tensors_spec_eigenvector : Dictionary
            tensor_id -> requested eigenvector
    """
    
    
    if return_eigen_property == 'sorted_basis':
        tensors_spec_eigenvalue  = dict(zip(tensors_dict.keys(), np.array([np.zeros(3)] * len(tensors_dict))))
        tensors_spec_eigenvector = dict(zip(tensors_dict.keys(), np.array([np.zeros((3,3))] * len(tensors_dict))))
    else:
        tensors_spec_eigenvalue  = dict(zip(tensors_dict.keys(), np.array( [0.0] * len(tensors_dict) ) ))
        tensors_spec_eigenvector = dict(zip(tensors_dict.keys(), np.array( [[0.0, 0.0, 0.0]] * len(tensors_dict) )))

    #For each tensor find eigenvector and value with requested property.
    for tensor_id, tensor in tensors_dict.items():

        tensor_eigen      = np.linalg.eig(tensor)
        real_eigenvalues  = np.real(tensor_eigen[0])
        real_eigenvectors = np.real(tensor_eigen[1])

        specified_eigenvalue_index = 0
        if return_eigen_property == 'max':
            specified_eigenvalue_index = np.argmax(real_eigenvalues)
        elif return_eigen_property == 'abs_max':
            specified_eigenvalue_index = np.argmax(abs(real_eigenvalues))
        elif return_eigen_property == 'abs_min':
            specified_eigenvalue_index = np.argmin(abs(real_eigenvalues))
        elif return_eigen_property == 'min':
            specified_eigenvalue_index = np.argmin(real_eigenvalues)
        elif return_eigen_property == 'sorted_basis':
            specified_eigenvalue_index = np.argsort(-abs(real_eigenvalues))

        tensors_spec_eigenvalue[tensor_id]  = real_eigenvalues[specified_eigenvalue_index]
        if isinstance(specified_eigenvalue_index, np.ndarray):
            tensors_spec_eigenvector[tensor_id] = np.array([real_eigenvectors[:, idx] for idx in specified_eigenvalue_index])
        else:
            tensors_spec_eigenvector[tensor_id] = real_eigenvectors[:, specified_eigenvalue_index]

    return tensors_spec_eigenvalue, tensors_spec_eigenvector


#Given triangles of this frame, and there deformation rate to the next frame as a triangle_index -> quantity dictionary,
#calculate the area weighted average for each cell from the triangle quantities.
#Missing triangles deformation rate scalars are set to zero.
def calculate_cell_scalar_from_triangle_scalars(cell_ids, triangles, triangles_deformation_rate):
    
    #Get indices of all triangles that have a deformation scalar defined.
    triangles_with_scalar_index    = np.array([triangle_index for triangle_index, scalar in triangles_deformation_rate.items()])
    #Triangles without a scalar defined, set to zero.
    triangles_without_scalar_index = np.setdiff1d(triangles.index, triangles_with_scalar_index)
    for tid in triangles_without_scalar_index:
        triangles_deformation_rate[tid] = 0.0

    #For all cells, take the area weighter average of their triangle scalars.
    cell_mean_triangle_scalar = np.array( [0.0] * len(cell_ids) )
    for cell_idx, cell_id in enumerate(cell_ids):

        #Select overlapping triangles with this cell.
        cell_triangles_index = triangles[ triangles['cell_id'] == cell_id ].index

        #Zero-area cells have less than three triangles, their elongation is put to zero.
        if len(cell_triangles_index) < 3:
            continue

        cell_triangles_scalar = np.array([triangles_deformation_rate[tidx] for tidx in cell_triangles_index])
        cell_triangles_area   = triangles['area'].loc[cell_triangles_index].values

        cell_mean_triangle_scalar[cell_idx] = np.array(sum([area * scalar for area, scalar in
            zip(cell_triangles_area, cell_triangles_scalar)])) / sum(cell_triangles_area + EPSILON)

    return cell_mean_triangle_scalar


#Given cell, triangle and triangle scalar tables, calculate the area weighted mean a scalar defined on the triangles.
def update_cell_table_with_scalar_from_triangle_scalars(cells, triangles, triangles_scalar, triangles_scalar_name):

    print('Update cells table with the area weighted', triangles_scalar_name, 'from the triangles.')

    #Calculate cell elongation from the area weighted mean of the 3d triangle elongation tensors.
    triangles_scalar_dict = dict(zip( triangles_scalar.index, triangles_scalar[triangles_scalar_name].values ))
    cell_mean_scalar = MM.calculate_cell_scalar_from_triangle_scalars(cells.index, triangles, triangles_scalar_dict)

    cells[triangles_scalar_name] = cell_mean_scalar

    return cells


# Given cells and triangle tables and a dictionary with triangle tensors,
# calculate area weighted mean of these tensors for every cell and return as dictionary.
def calculate_cell_tensor_from_triangle_tensors(cells, triangles, triangles_tensor, triangles_tensor_name, eigenvalue_property='max'):

    print('Update cells table with the area weighted', triangles_tensor_name, 'from the triangles.')

    #Group triangle index by cell_id:
    cells_triangles_idx = np.array([triangles[triangles['cell_id'] == cell_id].index.values for cell_id in cells.index])
    cells_triangles_index_dict = dict(zip(cells.index, cells_triangles_idx))

    #Calculate area weighted mean tensor on each cell.
    cells_mean_tensor_3d_dict = MM.calculate_area_weighted_mean_tensor_from_triangle_tensors(triangles, triangles_tensor, cells_triangles_index_dict)

    #Find eigenvalue and eigenvecor with property eigenvalue_property from averaged tensors.
    cells_tensor_eigenvalue, cells_tensor_eigenvector = MM.calculate_tensor_eigenvalues_and_eigenvector(cells_mean_tensor_3d_dict, eigenvalue_property)
    cells[triangles_tensor_name + '_norm_' + eigenvalue_property] = np.array([value for id, value in cells_tensor_eigenvalue.items()])

    return cells, cells_mean_tensor_3d_dict


#Given cells and triangle tables, calculate area weighted mean and gaussian curvature of each cell and update cells table.
def update_cell_table_with_curvature_from_triangle_curvatures(cells, triangles):

    print('Update cells table with the area weighted mean and gaussian curvature from the triangles.')

    # Calculate cell mean and gaussian curvature from the area weighted mean of the triangle curvatures.
    trs = triangles.groupby('cell_id')

    cells_mean_curvature =\
        [(trs.get_group(cell_id)['area'] * trs.get_group(cell_id)['mean_curvature']).sum() / (trs.get_group(cell_id)['area'].sum() + EPSILON)
         for cell_id in cells.index]
    cells_gaussian_curvature =\
        [(trs.get_group(cell_id)['area'] * trs.get_group(cell_id)['gaussian_curvature']).sum() / (trs.get_group(cell_id)['area'].sum() + EPSILON)
         for cell_id in cells.index]

    cells['mean_curvature'] = np.array(cells_mean_curvature)
    cells['gaussian_curvature'] = np.array(cells_gaussian_curvature)

    return cells


# Given a dictionary of patches, as triangle indices indexed with patch ids, calculate the area weighted means
# of the gaussian and mean curvature on these patches from the values on the triangles. Return pandas dataframe with values.
def calculate_mean_and_gaussian_curvature_on_patches_from_triangles(patches_triangle_ids_dict, triangles):

    print('Calculate mean and gaussian curvature on', len(patches_triangle_ids_dict), 'patches.')

    #Find the areas of the triangles in each patch.
    patches_triangle_areas = np.array([triangles.iloc[triangle_ids]['area'].values for patch_id, triangle_ids in patches_triangle_ids_dict.items()])

    patches_mean_curvature =\
        [(patches_triangle_areas[patch_idx] * triangles.loc[triangle_ids]['mean_curvature']).sum() / (patches_triangle_areas[patch_idx].sum() + EPSILON)
         for patch_idx, (patch_id, triangle_ids) in enumerate(patches_triangle_ids_dict.items())]
    patches_gaussian_curvature =\
        [(patches_triangle_areas[patch_idx] * triangles.loc[triangle_ids]['gaussian_curvature']).sum() / (patches_triangle_areas[patch_idx].sum() + EPSILON)
         for patch_idx, (patch_id, triangle_ids) in enumerate(patches_triangle_ids_dict.items())]

    patches                       = pd.DataFrame(index=patches_triangle_ids_dict.keys())
    patches['mean_curvature']     = patches_mean_curvature
    patches['gaussian_curvature'] = patches_gaussian_curvature

    return patches


# Given decomposed triange State Tensor, return the norm of nematic tensor.
def Qnorm(theta, AabsSq, twophi, BabsSq):
    return np.arcsinh(np.sqrt(BabsSq) / np.sqrt(AabsSq - BabsSq + EPSILON))


# Split the tensor T into trace, anti-symmetric part and a nematic part: T = A + B_nematic
# Return angles and norms of both tensors.
def splitTensor_angleNorm(T):
    # T = A + B
    # A = [[a,-b],
    #     [b, a]]
    # B = [[c, d],
    #     [d,-c]]
    a = 0.5 * (T[0, 0] + T[1, 1])
    b = 0.5 * (T[1, 0] - T[0, 1])
    c = 0.5 * (T[0, 0] - T[1, 1])
    d = 0.5 * (T[0, 1] + T[1, 0])

    theta = np.arctan2(b, a)
    AabsSq = a * a + b * b

    twophi = theta + np.arctan2(d, c)
    BabsSq = c * c + d * d

    return theta, AabsSq, twophi, BabsSq


# Split the tensor T into trace, anti-symmetric part and a nematic part: T = A + B_nematic
# Return components of both tensors.
def splitTensor_components(T):
    # T = A + B
    # A = [[a,-b],
    #     [b, a]]
    # B = [[c, d],
    #     [d,-c]]
    a = 0.5 * (T[0, 0] + T[1, 1])
    b = 0.5 * (T[1, 0] - T[0, 1])
    c = 0.5 * (T[0, 0] - T[1, 1])
    d = 0.5 * (T[0, 1] + T[1, 0])

    return a, b, c, d


# Given the two in-plane side-vectors of a triangle E12 and E13, calculate the triangle state tensor S = R.Inverse[C].
# Return S decomposed into the norms and angles of its symmetric and anti-symmetric parts.
def calculate_triangle_state_tensor_symmetric_antisymmetric(E12_xy, E13_xy, A0=1):
    l = np.sqrt(4 * A0 / np.sqrt(3))
    EquiTriangle = [[l, 0.5 * l],
                    [0, 0.5 * np.sqrt(3) * l]]

    EquiTriangleInv = np.linalg.inv(EquiTriangle)

    # Convert triangle vectors into triangle state tensor R.
    triangle_state_tensors_R = np.stack((E12_xy, E13_xy), axis=-1)
    # Convert triangle tensor R in triangle tensor S = R.C^{-1}
    triangle_state_tensors_S = np.array(
        list(map(lambda triangleR: np.dot(triangleR, EquiTriangleInv), triangle_state_tensors_R)))

    # Split triangle tensor S in traceless symmetric part and trace anti-symmetric part: theta, AabsSq, twophi, BabsSq.
    if len(np.shape(triangle_state_tensors_S)) > 2:
        triangles_state_tensor_decomposed = np.array(list(map(MM.splitTensor_angleNorm, triangle_state_tensors_S)))
    else:
        triangles_state_tensor_decomposed = np.array(MM.splitTensor_angleNorm(triangle_state_tensors_S))

    return triangles_state_tensor_decomposed


#Generate unique triangle ids using the cell_ids, given dbonds and triangles. Triangle table is updated.
#Here, we use a pairingfunction on the cell_id of the triangle and the cell_id of its neighbour.
#These two numbers are unique if all the cells are convex.
#TODO: triangle ids are not unique due to zero-area cells and boundary cells.
def generate_triangle_ids(triangle_dbonds, triangles):

    #Get all dbonds that live on the cell boundaries and are not part of the tissue boundary.
    #The triangle_id of the conjugate dbond of these dbond is the neighbour triangle id of each triangle.
    triangle_inner_dbonds   = triangle_dbonds[triangle_dbonds['triangle_id'] != -1]
    triangles_neighbour_tid = triangle_dbonds.iloc[ triangle_inner_dbonds['conj_dbond_id'][0::3].values ]['triangle_id'].values

    #Add tissue exterior triangle (id = -1), and find the cell_id of all neighbouring triangles.
    triangles.loc[-1] = np.array([-1] * len(triangles.columns)).astype(dtype=int)
    triangles.loc[-1, 'cell_id'] = MM.BOUNDARY_CELL_ID

    triangles_neighbour_cell_id = triangles.loc[ triangles_neighbour_tid ]['cell_id'].values

    #Remove tissue exterior triangle.
    triangles.drop(index=-1, inplace=True)

    #Find cell_id of each triangle, and create unique triangle id from its cell_id and neighbour_cell_id.
    triangles_cell_id = triangles['cell_id'].values
    triangles_id      = np.array([ MM.pairingFunction([cell_id, neighbour_cell_id]) for cell_id, neighbour_cell_id in zip(triangles_cell_id, triangles_neighbour_cell_id)])

    print( 'Total triangles:', len(triangles_id), ', triangles with unique id:', len(set(triangles_id)) )
    
    #Add the triangle ids to the triangles table.
    triangles['triangle_id'] = triangles_id

    return triangles


#Generate unique triangle id string for each triangle,
#given a set of unique ids for each triangle.
def generate_triangle_id_strings(triangles_unique_ids, sort_ids = True):

    N_triangles = len(triangles_unique_ids)

    #For every triangle, shift the given ids such that the smallest is in front.
    if sort_ids:
        triangles_shifted_ids = [np.roll(unique_ids, -np.argmin(unique_ids)) for unique_ids in triangles_unique_ids]
    else:
        triangles_shifted_ids = triangles_unique_ids

    #Convert every integer to a string.
    triangles_shifted_strings = [list(map(str, ids)) for ids in triangles_shifted_ids]
    #Merge the strings for every triangle to one: this is the unique id string for each triangle.
    triangles_id = ['_']*N_triangles
    for triangle_idx, strings in enumerate(triangles_shifted_strings):
        triangles_id[triangle_idx] = triangles_id[triangle_idx].join(strings)

    return triangles_id
    
    
#Cantor pairing function generates unique numbers from sets of numbers.
def pairingFunction(id_list):

    y = id_list.pop()
    x = id_list[-1]

    if(len(id_list) > 1):
        x = pairingFunction(id_list)

    return int(0.5 * ( x + y ) * ( x + y + 1 ) + y)


# Load frames file to find frame indices from movie.
def load_network_frames(path='./'):
    frames_filename = 'frames.pkl'

    if not os.path.isfile(path + frames_filename):
        print('File', frames_filename, 'not found in current directory. Exit.')
        sys.exit()

    with open(path + frames_filename, "rb") as f:
        frames = pickle.load(f)

    return frames


# Return vertex scaling pixel:micrometer DataFrame from current directory.
def load_vertex_scaling(path):
    vertex_scaling_filename = 'vertex_scaling.txt'

    if not os.path.isfile(path + vertex_scaling_filename):
        print('File', vertex_scaling_filename, 'not found in path', path,
              '. Continue without scaling vertex positions.')
        vertex_scaling = pd.DataFrame({'x_scaling': [1.0], 'y_scaling': [1.0], 'z_scaling': [1.0]})
    else:
        print('Found vertex scaling file.')
        vertex_scaling = pd.read_csv(path + vertex_scaling_filename, sep=' ', header=0)

    return vertex_scaling


#Return correctly formatted path (beginning and endin with '/')
def check_path_existence( input_path ):
    
    #Remove leading and trailing white spaces.
    input_path = input_path.strip()
    
    #Check if path exists.
    input_path_existence = os.path.exists(input_path)
    
    if not input_path_existence:
        print('The given path of the movie data is invalid. Please check again. Path given:', input_path)
        sys.exit()
        
    #Make sure path ends with a '/'.
    if not input_path[-1] == '/':
        input_path = input_path + '/'
        
    return input_path