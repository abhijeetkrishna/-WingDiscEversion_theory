from functions import *
from wd_functions import *
import glob
from pathlib import Path

#Taking arguments from command line
map_index_dest=sys.argv[1]
task_id=int(sys.argv[2])

#getting the dictionary of variables and their default values
var_dict = {
    'seed':0,
    'nb_iterations':20,
    'thickness':0.1,
    'R_initial':1,
    'theta_DV':0.0897495,
    'DV_present':True,
    #'outDV_gradient':True,
    'volume_conservation':True,
    'k_type':'k_c',
    'theta_max': 0.4029, #32*np.pi/180,
    'overwrite_old_simulation':True,
}

#reading the map_index csv
print('reading the map_index csv')
map_index=pd.read_csv(map_index_dest)
input_var=list(map_index.columns) #variables for which values are being imported
for i in range(len(input_var)):
    if input_var[i] == 'folder_name':
        dirname = map_index.loc[task_id,str(input_var[i])]
        continue
    var_dict[input_var[i]]=map_index.loc[task_id,str(input_var[i])]

#initializing variables
print('initializing variables')
for key,val in var_dict.items():
    exec(key + '=val')
#set the seed
np.random.seed(seed)

#######################################
#changing reference thickness of mesh #
#######################################

thick_mesh_file = 'gmsh_r_30_rotated_spherical_cap_theta_' + str(np.round(theta_max*180/np.pi,2)) + '_thickness_'+str(thickness)
mesh_file = "gmsh_sphere_lattice_r_30.pickle"

#Uncomment the following two lines to get differential thickness
#thickness_polynomial_coeffs = np.array([-0.22, 1.22])*thickness
#thickness_polynomial_coeffs = np.array([-0.1397, 0, 1.1397])*thickness
thickness_polynomial_coeffs = np.array([1])*thickness
thickness_polynomial_obj = np.poly1d(thickness_polynomial_coeffs)

if not(os.path.isfile(thick_mesh_file + '.pickle')):
    print('generating mesh for thickness : ' + str(thickness))
    #print('getting thick mesh out of thin mesh')
    init_gmsh_spherical_cap_mesh(mesh_file = mesh_file,theta_max = theta_max,R = 1, nstack = 2,
             thickness = thickness, thickness_polynomial_obj = thickness_polynomial_obj, 
             output_pickle = thick_mesh_file+'.pickle', output_vtk = thick_mesh_file+'.vtk',
             )

print('Loading mesh')
[balls_df, springs_df] = pickle.load( open(thick_mesh_file+'.pickle', 'rb') )

#GET RID OF THESE
#adding offset to the pole
#seed has been set before
pole_offset_theta = np.random.rand()*np.pi/(10000) #fixed offset for pole
pole_offset_phi = np.random.rand()*np.pi/(10000) #fixed offset for pole
#getting indices of balls on poles
indices = np.array(balls_df[(balls_df['x'] == 0) & (balls_df['y'] == 0)]['ID'].values)
#getting the radius of the balls on poles
radii = np.array(np.sqrt(balls_df.loc[indices, 'x']**2 + balls_df.loc[indices, 'y']**2 + balls_df.loc[indices, 'z']**2 ))
balls_df.loc[indices, 'x'] = radii*np.sin(pole_offset_theta)*np.cos(pole_offset_phi)
balls_df.loc[indices, 'y'] = radii*np.sin(pole_offset_theta)*np.sin(pole_offset_phi)
balls_df.loc[indices, 'z'] = radii*np.cos(pole_offset_theta)
springs_df = update_springs(springs_df, balls_df[['x', 'y', 'z']])

#Change the radius of the spherical cap

balls_df.to_csv(dirname + 'sim_output/orig_sized_balls.csv', index = False)
springs_df.to_csv(dirname + 'sim_output/orig_sized_springs.csv', index = False)
init_balls_df = balls_df.copy(deep = True)
init_springs_df = springs_df.copy(deep = True)

if k_type == 'k_DV':
    theta_min = min(np.arccos(-balls_df['x']/np.sqrt(balls_df['x']**2 + balls_df['y']**2 + balls_df['z']**2)))
    theta_max = max(np.arccos(-balls_df['x']/np.sqrt(balls_df['x']**2 + balls_df['y']**2 + balls_df['z']**2)))
elif k_type == 'k_c':
    theta_min = min(np.arccos(balls_df['z']/np.sqrt(balls_df['x']**2 + balls_df['y']**2 + balls_df['z']**2)))
    theta_max = max(np.arccos(balls_df['z']/np.sqrt(balls_df['x']**2 + balls_df['y']**2 + balls_df['z']**2)))


##################
# Reading stages #
##################



stages_df = pd.read_pickle('stages_df.pkl') #pd.read_csv("stages_df.csv")
#stages_df = stages_df.sort_values(by = 'stage')
stages = np.unique(stages_df["stage"])
#stage_names = np.unique(stages_df["stage_name"])

for j in range(len(stages)):
    stage = stages[j]
    temp_stages_df = stages_df[stages_df["stage"] == stage].reset_index(drop = True)

    print("reading " + temp_stages_df.iloc[0]["stage_name"])

    props = [
    "lambda_isotropic_coeffs", "lambda_anisotropic_coeffs",
    'inDV_lambda_isotropic_coeffs', 'inDV_lambda_anisotropic_coeffs',
    'lambda_height_coeffs', 'inDV_lambda_height_coeffs'
    ]
    values = [temp_stages_df[temp_stages_df["prop"] == prop]["value"].values[0] for prop in props]
    for prop,value in zip(props,values):
        print(prop + " : " + str(value))
        exec(prop+"=value")
        exec(prop.replace("_coeffs","_obj") + "=np.poly1d(value)")
    
    #get the target l0 for each spring
    #save it to to springs_dataframe
    #l0_stage_1

    #changing natural lengths of springs
    for index, row in springs_df.iterrows():
        if k_type == 'k_c':
            lambda_alpha = get_fit_lambda([row['x1'], row['y1'], row['z1']],
                                              theta_max = theta_max,
                                              DV_present = DV_present, theta_DV = theta_DV,
                                              volume_conservation = volume_conservation, 
                                              lambda_anisotropic_obj = lambda_anisotropic_obj, lambda_isotropic_obj = lambda_isotropic_obj,
                                              inDV_lambda_anisotropic_obj = inDV_lambda_anisotropic_obj, inDV_lambda_isotropic_obj = inDV_lambda_isotropic_obj,
                                              lambda_height_obj = lambda_height_obj,
                                              inDV_lambda_height_obj = inDV_lambda_height_obj,
                                              )
            lambda_beta = get_fit_lambda([row['x2'], row['y2'], row['z2']],
                                              theta_max = theta_max,
                                              DV_present = DV_present, theta_DV = theta_DV,
                                              volume_conservation = volume_conservation, 
                                              lambda_anisotropic_obj = lambda_anisotropic_obj, lambda_isotropic_obj = lambda_isotropic_obj,
                                              inDV_lambda_anisotropic_obj = inDV_lambda_anisotropic_obj, inDV_lambda_isotropic_obj = inDV_lambda_isotropic_obj,
                                              lambda_height_obj = lambda_height_obj,
                                              inDV_lambda_height_obj = inDV_lambda_height_obj,
                                              )
        lambda_alpha_beta = 0.5*(lambda_alpha + lambda_beta)
        spring_vector = np.array([ row['x1'] - row['x2'], row['y1'] - row['y2'], row['z1'] - row['z2']])
        virtual_spring_vector = np.matmul(lambda_alpha_beta, spring_vector)
        l0 = np.linalg.norm(virtual_spring_vector)
        springs_df.loc[index, 'l0_stage_'+str(stage)] = l0


springs_df['l0_target_final'] = springs_df['l0_stage_'+str(stage)]
springs_df['l0_target'] = springs_df['l0_target_final']
springs_df['l1_initial'] = springs_df['l1']

#making plots
title = k_type + ', thickness:' + str(thickness)
plot_shell(balls_df, springs_df, x = 'y', y = 'z', filename = dirname + 'sim_output/DV_parallel.pdf', cbar_name = r'$\frac{l_{o}}{l_{initial}}$', title = title)
plot_shell(balls_df, springs_df, x = 'x', y = 'z', filename = dirname + 'sim_output/DV_across.pdf', cbar_name = r'$\frac{l_{o}}{l_{initial}}$', title = title)

balls_df.to_csv(dirname + 'sim_output/init_balls.csv', index = False)
springs_df.to_csv(dirname + 'sim_output/init_springs.csv', index = False)

# doing simulation

vtk_filename = dirname + 'sim_output/' + k_type
vtk_filename = vtk_filename + '_thickness_' + str(thickness)
vtk_filename = vtk_filename + '_'
dfToVtk(balls_df,springs_df,filename=vtk_filename + '0.vtk', add_polygons = True)

springs_df['l0'] = springs_df['l1']


for i in range(len(stages)*nb_iterations):

    print('growth step : ' + str(i))

    l0_stage_2 = "l0_stage_" + str(int(i/nb_iterations))
    if int(i/nb_iterations) == 0:
        l0_stage_1 = 'l1_initial'
    else:
        l0_stage_1 = "l0_stage_" + str(int(i/nb_iterations)-1)

    springs_df["l0_target"] = springs_df[l0_stage_2]
    springs_df['l0'] = springs_df['l0'] + (springs_df[l0_stage_2] - springs_df[l0_stage_1])/nb_iterations
    tol = 1e-6

    [balls_df, springs_df] = initialize_cpp_simulation(balls_df, springs_df, dt = 0.1,
                                                       csv_t_save = 1, tol = tol, path = dirname)

    if os.path.exists('files/'):
        shutil.rmtree('files/')

    print('$$$$$$$ Running openfpm $$$$$$$')
    os.system("cd " + dirname + " && source ~/openfpm_vars && make && grid")
    print('$$$$ Exit OpenFPM $$$$')
    
    #rename the folder
    sim_folder = dirname + 'sim_output/growth_'+str(i+1)
    if os.path.exists(sim_folder):
        shutil.rmtree(sim_folder)
    shutil.move(dirname + 'files/', sim_folder)

    #delete everything that is not needed
    #import glob, os
    for f in glob.glob(sim_folder + "/*.vtk"):
        os.remove(f)
    for f in glob.glob(sim_folder + "/Spring_*.csv"):
        os.remove(f)

    #open the last csv file
    df = pd.read_csv(sim_folder + '/final_0_0.csv')

    #update the position of the balls and springs
    balls_df['x'] = df['x[0]']
    balls_df['y'] = df['x[1]']
    balls_df['z'] = df['x[2]']
    springs_df = update_springs(springs_df, balls_df[['x', 'y', 'z']]) #this line probably not required

    #dfToVtk(balls_df,springs_df,filename=vtk_filename + str(i+1) + '.vtk', add_polygons = True)

    #save teh csv file if last file of stage
    if (i+1)%nb_iterations == 0:
        temp_stages_df = stages_df[stages_df["stage"] == int(i/nb_iterations)]
        stage_name = temp_stages_df.iloc[0]["stage_name"]
        balls_df.to_csv(dirname + 'sim_output/' + stage_name + '.csv')
        dfToVtk(balls_df,springs_df,filename=dirname + 'sim_output/' + stage_name + '.vtk', add_polygons = True)




















