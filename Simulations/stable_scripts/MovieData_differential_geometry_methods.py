#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Dec 18 2018

@author: Joris Paijmans - paijmans@pks.mpg.de

Define all functions using differential geometry for the MovieData program.
"""

import numpy as np
import pandas as pd
import sys
import csv
import sys
import os
import pwd
import pickle

import MovieData_methods as MM
import MovieData_differential_geometry_methods as Mdgm

#Function returns the full covariant metric for each triangle with base vectors E12, E13 and n as:
# g_ij = [[E12.E12, E12.E13],  g_ij = [[E12.E12, E12.E13, 0.],
#         [E13.E12, E13.E13]]          [E13.E12, E13.E13, 0.],
#                                      [     0.,      0., 1.]]
# (dimension = 2)              (dimension = 3)
#If contravariant is set to True, full contravariant version (inverse) is retured.
def get_triangles_metric_g_ij(triangles, contravariant = False, dimension = 2):

    triangles_E12 = triangles[['dr_12_x', 'dr_12_y' ,'dr_12_z']].values
    triangles_E13 = triangles[['dr_13_x', 'dr_13_y' ,'dr_13_z']].values

    triangles_g_ij = np.array([] * len(triangles))
    
    if dimension == 2:
        #In-plane only 2x2 respresentation.
        triangles_g_ij = [[ np.sum(triangles_E12 * triangles_E12, axis=1), np.sum(triangles_E12 * triangles_E13, axis=1)],
                             [ np.sum(triangles_E13 * triangles_E12, axis=1), np.sum(triangles_E13 * triangles_E13, axis=1)]]
    else:
        #Full 3x3 representation.
        N_tri = len(triangles)
        triangles_g_ij = [[ np.sum(triangles_E12 * triangles_E12, axis=1), np.sum(triangles_E12 * triangles_E13, axis=1), np.array([0.]*N_tri)],
                             [ np.sum(triangles_E13 * triangles_E12, axis=1), np.sum(triangles_E13 * triangles_E13, axis=1), np.array([0.]*N_tri)],
                             [np.array([0.]*N_tri), np.array([0.]*N_tri), np.array([1.]*N_tri)]]

    triangles_g_ij = np.array(triangles_g_ij).transpose()

    #Zero area triangles produces metrics that have no inverse, we therefore add an epsilon difference to these triangle metrics.
    triangles_small_det_gij_idx = np.where(abs(np.linalg.det(triangles_g_ij)) < 1e-12)[0]
    triangles_g_ij[triangles_small_det_gij_idx] += MM.EPSILON * np.eye(dimension)

    if contravariant:
        triangles_g_ij = np.linalg.inv(triangles_g_ij)
    
    return triangles_g_ij


# Return the covariant base vectors for all triangles.
# If contravariant is set to true; give base vectors in contravariant form.
# Id dimensin is set to 3, also return unit normal vector, if set to two don't include it.
def get_triangles_local_basis_3d(triangles, triangles_inverse_metric = [], contravariant = False, dimension = 2, transpose = False):
    """ Get the local basis for each triangle
    
    Parameters
    ----------
    triangles : pandas.DataFrame, columns = {['dr_12_x', 'dr_12_y', 'dr_12_z'], ['dr_13_x', 'dr_13_y' ,'dr_13_z']}
    triangles_inverse_metric : numpy.array (float)
        Optional: The contravariant metric in the local triangle basis.
    contravariant : {False, True}
        Return triangle basis in contravariant form.
    dimension : {2, 3}
        Return ony the tangential basis vectors (2) or the tangential and norml basis vectors (3).
    transpose : {False, True}
        Return transposed version of the basis vectors Rij -> Rji
        
    Returns
    -------
        triangles_basis : numpy.array (float)
            The local covariant basis of each triangle
    """
    

    if dimension == 2:
        triangles_basis = np.transpose(np.array([triangles[['dr_12_x', 'dr_12_y', 'dr_12_z']].values,
                                                triangles[['dr_13_x', 'dr_13_y' ,'dr_13_z']].values]), axes = (1,0,2))
    else:
        triangles_basis = np.transpose(np.array([triangles[['dr_12_x', 'dr_12_y', 'dr_12_z']].values,
                                                triangles[['dr_13_x', 'dr_13_y' ,'dr_13_z']].values,
                                                triangles[['normal_x', 'normal_y', 'normal_z']].values]), axes = (1,0,2))
    if contravariant:
        if triangles_inverse_metric == [] or len(triangles_inverse_metric) != len(triangles):
            triangles_inverse_metric = Mdgm.get_triangles_metric_g_ij(triangles, contravariant=True, dimension = dimension)

        triangles_basis = np.array([inverse_g_ij.dot(E12_E12_normal)
            for inverse_g_ij, E12_E12_normal in zip(triangles_inverse_metric, triangles_basis)])

    if transpose:
        triangles_basis = np.transpose(triangles_basis, axes=(0, 2, 1))

    return triangles_basis


# Function returns the deformation rate tensor of each triangle given the
# base vectors in this frame, \vec{E}_alpha, and the change in base vectors to the next frame \delta \vec{E}_alpha.
# U_\alpha\beta = \delta \vec{E}_alpha \dot \vec{E}_beta
# IT can give the full, symmetric or skew-symmetric deformation rate tensor.
def get_triangles_deformation_rate_tensor_Uij(triangles_this_frame, triangles_next_frame, delta_t = 1.0, type='full'):

    assert type in ['full', 'symmetric', 'skew_symmetric'], 'Wrong deformation rate tensor type given: '+str(type)

    triangles_this_frame_E12 = triangles_this_frame[['dr_12_x', 'dr_12_y', 'dr_12_z']].values
    triangles_this_frame_E13 = triangles_this_frame[['dr_13_x', 'dr_13_y', 'dr_13_z']].values
    triangles_this_frame_N   = triangles_this_frame[['normal_x', 'normal_y', 'normal_z']].values

    triangles_next_frame_E12 = triangles_next_frame[['dr_12_x', 'dr_12_y', 'dr_12_z']].values
    triangles_next_frame_E13 = triangles_next_frame[['dr_13_x', 'dr_13_y', 'dr_13_z']].values
    triangles_next_frame_N   = triangles_next_frame[['normal_x', 'normal_y', 'normal_z']].values

    triangles_delta_E12 = triangles_next_frame_E12 - triangles_this_frame_E12
    triangles_delta_E13 = triangles_next_frame_E13 - triangles_this_frame_E13
    triangles_delta_N   = triangles_next_frame_N - triangles_this_frame_N

    triangles_U_ij = [[np.sum(triangles_delta_E12 * triangles_this_frame_E12, axis=1),
                       np.sum(triangles_delta_E13 * triangles_this_frame_E12, axis=1),
                       np.sum(triangles_delta_N   * triangles_this_frame_E12, axis=1)],
                      [np.sum(triangles_delta_E12 * triangles_this_frame_E13, axis=1),
                       np.sum(triangles_delta_E13 * triangles_this_frame_E13, axis=1),
                       np.sum(triangles_delta_N   * triangles_this_frame_E13, axis=1)],
                      [np.sum(triangles_delta_E12 * triangles_this_frame_N, axis=1),
                       np.sum(triangles_delta_E13 * triangles_this_frame_N, axis=1),
                       np.sum(triangles_delta_N   * triangles_this_frame_N, axis=1)]]

    triangles_U_ij = np.array(triangles_U_ij).transpose() / delta_t

    if type == 'symmetric':
        triangles_U_ij = np.array([0.5 * (U_ij + U_ij.T) for U_ij in triangles_U_ij])
    elif type == 'skew_symmetric':
        triangles_U_ij = np.array([0.5 * (U_ij - U_ij.T) for U_ij in triangles_U_ij])

    return triangles_U_ij


#Perform basis transformation using basis matrices triangles_basis1 and triangles_basis2.
def tensor_basis_transformation(triangles_tensor, triangles_basis1, triangles_basis2 = [], return_transposed = False):
    """ Perform a basis transformation on each tensor defined in the basis of each triangle.
    
    Parameters
    ----------
    triangles_tensor : numpy.array (float) Mxrxd
        M tensors with dimension d and rank r={1,2}.
    triangles_basis1 : numpy.array (float) Mxfxd
        M local basis basis, with f bases vectors with dimension d.
    triangles_basis2 : numpy.array (float) Mxfxd
        M local basis basis, with f bases vectors with dimension d.
        
    Returns
    -------
    triangles_tensor_transformed : numpy.array (float) Mxfxr
        transformed tensors with Tij = Tab Ria Rjb    
    
    """
    triangles_tensor = np.array(triangles_tensor)
    triangles_basis1 = np.array(triangles_basis1)
    triangles_basis2 = np.array(triangles_basis2)

    #Determine dimension and rank of input tensor.
    dim  = triangles_tensor.shape[-1]
    rank = len(triangles_tensor.shape) - 1

    assert rank == 1 or rank == 2

    #If only one basis is given, assume both input basis are equal.
    if len(triangles_basis2) == 0:
        triangles_basis2 = triangles_basis1

    triangles_tensor_trans = []
    if rank == 2:
        triangles_tensor_trans = \
            [np.sum(M[i, j] * np.outer(basis1[i], basis2[j]) for i in range(dim) for j in range(dim))
             for M, basis1, basis2 in zip(triangles_tensor, triangles_basis1, triangles_basis2)]
    else:
        triangles_tensor_trans = \
            [np.sum(V[i] * basis[i] for i in range(dim))
             for V, basis in zip(triangles_tensor, triangles_basis1)]

    return np.array(triangles_tensor_trans)


#Given a tensor and the covariant form of the basis vectors, calculate tensor trace.
def tensor_trace(triangles_cov_tensor, triangles_contr_metric):

    triangles_tensor = np.array(triangles_cov_tensor)
    triangles_basis  = np.array(triangles_contr_metric)

    tensor_trace = np.array([contr_metric.dot(cov_tensor) for cov_tensor, contr_metric in zip(triangles_cov_tensor, triangles_contr_metric)])

    return np.trace(tensor_trace.T)


#Given tensor and local contravariant form of the local basis, return traceless tensor in that basis.
def tensor_remove_trace(triangles_cov_tensor, triangles_cov_metric, triangles_contr_metric):

    triangles_tensor_trace = Mdgm.tensor_trace(triangles_cov_tensor, triangles_contr_metric)

    dim = triangles_cov_tensor.shape[-1]

    triangles_cov_tensor_traceless = [tensor - 1./dim * trace * g_ij
                                      for tensor, trace, g_ij in zip(triangles_cov_tensor, triangles_tensor_trace, triangles_cov_metric)]

    return np.array(triangles_cov_tensor_traceless)


# Calculate nematic tensor from a covariant unit vector, the nematic eigenvalues and the covariant metric tensor.
def calculate_cov_nematic_tensor(cov_unit_vector, cov_metric_tensor, nematic_eigen_values=[]):
    if len(nematic_eigen_values) == 0:
        nematic_eigen_values = np.ones(len(cov_unit_vector))

    nematic_unit_tensor = [np.array([[pi[0] * pi[0], pi[0] * pi[1]],
                                     [pi[1] * pi[0], pi[1] * pi[1]]]) - 0.5 * g_ij
                           for g_ij, pi in zip(cov_metric_tensor, cov_unit_vector)]

    nematic_unit_tensor = np.array(nematic_unit_tensor)

    return np.array([q * nematic for q, nematic in zip(nematic_eigen_values, nematic_unit_tensor)])