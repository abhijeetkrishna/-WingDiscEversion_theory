from curvature_functions import *
from wd_functions import *

map_index_dest=sys.argv[1]
task_id=int(sys.argv[2])

#reading the map_index csv
print('reading the map_index csv')
map_index=pd.read_csv(map_index_dest)
input_var=list(map_index.columns) #variables for which values are being imported
for i in range(len(input_var)):
    if input_var[i] == 'folder_name':
        dirname = map_index.loc[task_id,str(input_var[i])]
    if input_var[i] == 'thickness':
        thickness = map_index.loc[task_id,str(input_var[i])]
        init_mesh_path = 'gmsh_r_30_rotated_spherical_cap_theta_23.08_thickness_' + str(thickness) + '.pickle'
    #var_dict[input_var[i]]=map_index.loc[task_id,str(input_var[i])]


############
# Adding properties to edges
############

springs_df = pd.read_csv(dirname + 'sim_output/init_springs.csv')
balls_df = pd.read_csv(dirname + 'sim_output/init_balls.csv')



files = glob.glob(dirname + "sim_output/growth_*/")
times_str = [re.search('growth_(.*)/',x).group(1) for x in files]
times = []
for i in range(len(times_str)):
    if len(times_str[i]) == 0:
        continue
    times.append(int(times_str[i]))
times = np.sort(times)



for t in times:
    
    print('timepoint : ' + str(t))
    
    df_timepoint = pd.read_csv(dirname + 'sim_output/growth_' + str(t) + '/final_0_0.csv') 
    balls_df[['x', 'y', 'z']] = df_timepoint[['x[0]', 'x[1]', 'x[2]']]
    springs_df = update_springs(springs_df, balls_df[['x', 'y', 'z']])
    
    #calculating some properties that are not there already
    springs_df['l0_target/l1'] = springs_df['l0_target']/springs_df['l1']
    springs_df['l1/l0_target'] = springs_df['l1']/springs_df['l0_target']
    springs_df['l0/l1_initial'] = springs_df['l0']/springs_df['l1_initial']
    springs_df['l1/l1_initial'] = springs_df['l1']/springs_df['l1_initial']
    springs_df['l0/l1'] = springs_df['l0']/springs_df['l1']
    springs_df['l1/l0'] = springs_df['l0']/springs_df['l1']
    
    #getting vtk with properties on thick mesh
    dfToVtk(balls_df, springs_df, add_lines_properties=True, return_text=False,
            filename = dirname + 'sim_output/thick_wireframe_'+ str(t) + '.vtk',
           )
    
    #getting vtk with properties on thick mesh
    dfToVtk(balls_df, springs_df, only_top_surface=True, add_lines_properties=True, return_text=False,
            filename = dirname + 'sim_output/top_wireframe_'+ str(t) + '.vtk',
           )
    dfToVtk(balls_df, springs_df, only_bottom_surface=True, add_lines_properties=True, return_text=False,
            filename = dirname + 'sim_output/bottom_wireframe_'+ str(t) + '.vtk',
           )






#############
# Adding curvature
#############

#not every file needs to be computed because it takes long
skipby = 5


springs = pd.read_csv(dirname + 'sim_output/init_springs.csv')
balls = pd.read_csv(dirname + 'sim_output/init_balls.csv')

files = glob.glob(dirname + "sim_output/growth_*/")
times_str = [re.search('growth_(.*)/',x).group(1) for x in files]
times = []
for i in range(len(times_str)):
    if len(times_str[i]) == 0:
        continue
    times.append(int(times_str[i]))
times = np.sort(times)



#get the r theta and phi values
#def cart2sph(x, y, z):
#    hxy = np.hypot(x, y)
#    r = np.hypot(hxy, z)
#    el = np.arctan2(hxy, z)
#    az = np.arctan2(y, x)
#    return r, el, az

rs, thetas, phis = cart2sph(balls['x'].values, balls['y'].values, balls['z'].values)
balls['theta'] = thetas
balls['r'] = rs
balls['phi'] = phis

balls_init = balls.copy(deep = True)
springs_init = springs.copy(deep = True)

#########
# getting the top and bottom surfaces
########

print('getting top surface')
#get top surface
springs_top = springs[(springs['ball1'] >= len(balls)/2) & (springs['ball2'] >= len(balls)/2)]
balls_top = balls[balls['ID'] >= len(balls)/2]
top_id_array = balls_top['ID'].values
#reindex
[balls_top, springs_top] = reindex_balls_springs(balls_top, springs_top)
#get triangles
print('getting triangles')
triangles_top = get_oriented_triangles(balls_top, springs_top)
#get indices of vertices that are not on the boundary
#we need these vertices because the Gaussian curvature is not calculated on the boundary vertices
nonboundary_id_top = balls_top.loc[balls_top['theta'] < 0.9*max(balls_top['theta']),'ID']

print('getting bottom surface')
#get bottom surface
springs_bottom = springs[(springs['ball1'] < len(balls)/2) & (springs['ball2'] < len(balls)/2)]
balls_bottom = balls[balls['ID'] < len(balls)/2]
bottom_id_array = balls_bottom['ID'].values
#reindex
[balls_bottom, springs_bottom] = reindex_balls_springs(balls_bottom, springs_bottom)
#get triangles
print('getting triangles')
triangles_bottom = get_oriented_triangles(balls_bottom, springs_bottom)
#get indices of vertices that are not on the boundary
#we need these vertices because the Gaussian curvature is not calculated on the boundary vertices
nonboundary_id_bottom = balls_bottom.loc[balls_bottom['theta'] < 0.9*max(balls_bottom['theta']),'ID']

print('measuring curvature')

for t in times:

    if not(t%skipby == 0):
        if not(t == times[-1]):
            continue

    print('timepoint : ' + str(t))
    
    df_timepoint = pd.read_csv(dirname + 'sim_output/growth_' + str(t) + '/final_0_0.csv') 
    balls[['x', 'y', 'z']] = df_timepoint[['x[0]', 'x[1]', 'x[2]']]
    springs = update_springs(springs, balls[['x', 'y', 'z']])
    
    
    balls_bottom[['x','y','z']] = balls.loc[bottom_id_array, ['x','y','z']].values
    balls_top[['x','y','z']] = balls.loc[top_id_array, ['x','y','z']].values

    #measure curvature
    #if not(os.path.exists(dirname + 'sim_output/bottom_surface_'+ str(t) + '.vtk')):
    [gc_bottom, mc_bottom, triangles_db_bottom, vertices_db_bottom] = measure_integrated_curvature(balls_bottom, springs_bottom, triangles = triangles_bottom,
                                                                                                   filename = dirname + 'sim_output/bottom_surface_'+ str(t) + '.vtk',
                                                                                                   nonboundary_indices = nonboundary_id_bottom, write_vtk=True,
                                                                                                   z_offset = -0.0001
                                                                                                  )
    #if not(os.path.exists(dirname + 'sim_output/top_surface_'+ str(t) + '.vtk')):
    [gc_top, mc_top, triangles_db_top, vertices_db_top] = measure_integrated_curvature(balls_top, springs_top, triangles = triangles_top,
                                                                                       filename = dirname + 'sim_output/top_surface_'+ str(t) + '.vtk',
                                                                                       nonboundary_indices = nonboundary_id_top, write_vtk=True,
                                                                                       z_offset = 0.0001
                                                                                      )
    #mean_integrated_gc = 0.5*(gc_top + gc_bottom)
    #mean_integrated_mc = 0.5*(mc_top + mc_bottom)



######


###########################################
# Measuring curvature of 2D crosssections # 
###########################################

scale = 167.13
crosssections = ['Across_DV', 'Along_DV']
ncols = len(crosssections)
fig, axs = plt.subplots(2, ncols, figsize=(11*ncols, 16))
titles = [str(x) for x in crosssections]
projection_y = 'z'

for i in range(len(crosssections)):
    
    crosssection = crosssections[i]
    
    if crosssection == 'Along_DV':
        projection_x = 'y' #for along DV cross-section
    elif crosssection == 'Across_DV':
        projection_x = 'x' #for across DV cross-section

    #get initial balls and springs and curve
    #init_path = 'simulation_data/gmsh_r_20_rotated_spherical_cap_theta_45_thickness_0.1.pickle'
    [balls_df, springs_df] = pickle.load( open(init_mesh_path, 'rb') )
    springs_df['l1_initial'] = springs_df['l1']
    init_curve = get_2D_curve_from_simulation(springs_df, #indices = xz_plane_springs_id, 
                                        projection_x = projection_x, projection_y = projection_y,
                                              balls_initial = balls_df, springs_initial = springs_df
                                        )
    init_smooth_curve = compute_2D_curve_curvature(init_curve[[projection_x, projection_y]])
    
    #get final balls and springs
    #print(dirname)
    [balls_timepoint, springs_timepoint] = get_final_dataframe(path = dirname+'sim_output/', 
                                                               balls_initial=balls_df, 
                                                               springs_initial=springs_df
                                                              )
    

    # get smoothened final curve
    curve = get_2D_curve_from_simulation(springs_timepoint, #indices = xz_plane_springs_id, 
                                         projection_x = projection_x, projection_y = projection_y,
                                         balls_initial = balls_df, springs_initial = springs_df
                                        )
    smooth_curve = compute_2D_curve_curvature(curve[[projection_x, projection_y]])
    
    #scale things
    balls_timepoint[['x', 'y', 'z']] = scale*balls_timepoint[['x', 'y', 'z']]
    z_offset = max(balls_timepoint['z'])
    balls_timepoint['z'] = balls_timepoint['z'] - z_offset
    
    #springs_timepoint
    springs_timepoint[['x1','y1','z1','x2','y2','z2']] = scale*springs_timepoint[['x1','y1','z1','x2','y2','z2']]
    springs_timepoint['z1'] =springs_timepoint['z1'] - z_offset
    springs_timepoint['z2'] =springs_timepoint['z2'] - z_offset
    
    #smooth_curve
    smooth_curve[['x','y','arclength']] = scale*smooth_curve[['x','y','arclength']]
    smooth_curve['y'] = smooth_curve['y'] - max(smooth_curve['y'])
    smooth_curve['curvature'] = smooth_curve['curvature']/scale #here we divide by scale
    
    #init_smooth_curve
    init_smooth_curve[['x','y','arclength']] = scale*init_smooth_curve[['x','y','arclength']]
    init_smooth_curve['y'] = init_smooth_curve['y'] - max(init_smooth_curve['y']) 
    init_smooth_curve['curvature'] = init_smooth_curve['curvature']/scale #here we divide by scale
    
    smooth_curve.to_csv(dirname + 'sim_output/' + crosssection + '_curve.csv', index = False)
    init_smooth_curve.to_csv(dirname + 'sim_output/' + crosssection + '_init_curve.csv', index = False)
    
    #plot mesh and curve
    if ncols == 1:
        ax = axs[0]
    else:
        ax = axs[0,i]
    ax.axis('off')

    plot_shell_on_given_ax(balls_timepoint, springs_timepoint,
                           x = projection_x, y = projection_y,
                           ax = ax, fig = fig,
                           line_color_values = 'final_vs_initial',
                           cbar_name=r'$\frac{l_{final}}{l_{initial}}$'
                          )
    ax.set_title(crosssections[i], fontsize = 30, pad = 25)
    
    ax.plot(smooth_curve.x, smooth_curve.y, alpha = 0.5, linewidth = 10, color = 'red', linestyle = '--', label = 'final')
    ax.plot(init_smooth_curve.x, init_smooth_curve.y, alpha = 0.5, linewidth = 10, color = 'blue', linestyle = '--', label = 'initial')
    
    ax.legend(loc = 'upper left', fontsize = 25)
    ax.tick_params(axis='both', which='major', labelsize=20)

    
    # plot the curvature 
    if ncols == 1:
        ax = axs[1]
    else:
        ax = axs[1,i]
        
    ax.plot(smooth_curve['arclength'],smooth_curve['curvature'], 
            color = 'red', #alpha = 0.5,
              #label = 'individual discs'
              #color = single_color, 
              linewidth = 10, alpha = 0.5, #linestyle = single_linestyle,
            label = 'final',
             )
    #ax.set_ylim(-0.5,2)
    if i == 0:
        ylim_max = max(smooth_curve['curvature']) + 0.1*np.abs(max(smooth_curve['curvature']))
        ylim_min = min(smooth_curve['curvature']) - 0.1*np.abs(min(smooth_curve['curvature']))

    ax.set(ylim = (-0.005,0.05),)
    ax.set_ylabel('curvature', fontsize = 40)
    ax.set_xlabel('arclength', fontsize = 40)
    
    #plot initial curvature
    
    ax.plot(init_smooth_curve['arclength'],init_smooth_curve['curvature'], 
            color = 'blue', #alpha = 0.5,
              #label = 'individual discs'
              #color = single_color, 
              linewidth = 10, alpha = 0.5, #linestyle = single_linestyle,
            label = 'initial',
             )
    ax.legend(loc = 'upper left', fontsize = 25)
    ax.axhline(0, linestyle = '-', c = 'grey', linewidth = 0.5)
    ax.axvline(0, linestyle = '-', c = 'grey', linewidth = 0.5)
    #ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_yticks([ 0, 0.02, 0.04])
    #ax.set_yticks([-2, 0, 2])
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    
plt.savefig(dirname + 'sim_output/crosssection_plot.pdf', bbox_inches = 'tight')




