
from methods import *

#Taking arguments from command line
map_index_dest=sys.argv[1]
task_id=int(sys.argv[2])
print("map_index_dest : " + str(map_index_dest))
print("task id : " + str(task_id))

#getting the dictionary of variables and their default values
var_dict = {
    'thickness':0.1,
    #'R_initial':1,
    'theta_max': 0.8662, #32*np.pi/180,
    #'theta_ref': 0.4029,
    'angle_of_rotation':0.1,
    'mesh_refine_factor':80,
    'theta_DV' : 0.1931,
    'R':1,
}

#reading the map_index csv
print('reading the map_index csv')
map_index=pd.read_csv(map_index_dest)
input_var=list(map_index.columns) #variables for which values are being imported
for i in range(len(input_var)):
    var_dict[input_var[i]]=map_index.loc[task_id,str(input_var[i])]

#initializing variables
print('initializing variables')
for key,val in var_dict.items():
    exec(key + '=val')

###################################
# Get spherical mesh from meshzoo #
###################################

output_pickle = f"pickle/IcoSph_mesh_refine_factor_{mesh_refine_factor}.pkl"
output_file = f"vtk/IcoSph_mesh_refine_factor_{mesh_refine_factor}.ply"
try:
    [sphere_balls_df,sphere_edges_df,sphere_cells_df] = pickle.load(open(output_pickle,"rb"))
except:
    [sphere_balls_df,sphere_edges_df,sphere_cells_df] = get_meshzoo_icosa_sphere(refine_factor = mesh_refine_factor,
                                                                                 output_pickle = output_pickle, 
                                                                                 output_file = output_file
                                                                                )
#########################
# Add thickness to mesh #
#########################

output_pickle = f"pickle/IcoSph_mesh_wo_basis_vectors_refine_factor_{mesh_refine_factor}_thickness_{thickness}_angle_of_rotation_{angle_of_rotation}_theta_max_{theta_max}.pkl"
output_file = f"vtk/IcoSph_mesh_wo_basis_vectors_refine_factor_{mesh_refine_factor}_thickness_{thickness}_angle_of_rotation_{angle_of_rotation}_theta_max_{theta_max}.vtk"
try:
    [balls_df, springs_df] = pickle.load(open(output_pickle,"rb"))
except:
    thickness_polynomial_coeffs = np.array([1])*thickness
    thickness_polynomial_obj = np.poly1d(thickness_polynomial_coeffs)

    [balls_df, springs_df] = add_thickness_to_mesh(v_df = sphere_balls_df[['x', 'y', 'z']], edges_df = sphere_edges_df, #cells_df = cells_df,
                                                               thickness_polynomial_obj = thickness_polynomial_obj,
                                                               rotate_mesh_bool=True, angle_of_rotation = angle_of_rotation,
                                                               crop_mesh_bool=True, theta_max=theta_max,
                                                               output_pickle = output_pickle, 
                                                               output_vtk = output_file,
                                                              )

#############################
# Add basis vectors to mesh #
#############################

#add basis vectors to mesh
output_pickle = f"pickle/IcoSph_mesh_w_basis_vectors_refine_factor_{mesh_refine_factor}_thickness_{thickness}_angle_of_rotation_{angle_of_rotation}_theta_max_{theta_max}_theta_DV_{theta_DV}.pkl"
try:
    [balls_df, springs_df] = pickle.load(open(output_pickle,"rb"))
except:
    #add basis vectors
    [balls_df, springs_df] = add_basis_vectors_to_Sph(balls_df, springs_df, theta_DV=theta_DV)
    #save
    pickle.dump([balls_df,springs_df], open(output_pickle, 'wb'))


# to see how to plot, look at the notebook in this folder