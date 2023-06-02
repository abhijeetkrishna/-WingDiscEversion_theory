from curvature_functions import *
from methods import *

map_index_dest= sys.argv[1] #"map_index_0.csv"
task_id= int(sys.argv[2]) #0

#reading the map_index csv
print('reading the map_index csv')
map_index=pd.read_csv(map_index_dest)
input_var=list(map_index.columns) #variables for which values are being imported
for i in range(len(input_var)):
    if input_var[i] == 'folder_name':
        dirname = map_index.loc[task_id,str(input_var[i])]
    if input_var[i] == 'thickness':
        thickness = map_index.loc[task_id,str(input_var[i])]
init_mesh_path = dirname + "runfiles/init_mesh.pickle"

stages_df = pd.read_pickle(dirname+'runfiles/fit_lambdas_df.pkl')
stages = np.append(np.array(stages_df.iloc[0]["stage_init"]), np.array(stages_df.iloc[0:(len(np.unique(stages_df["stage_final"])))]["stage_final"]))
stage_names = np.array(stages_df.iloc[0:(len(np.unique(stages_df["stage_name"])))]["stage_name"])

scale = 77.66
crosssections = ['Across_DV', 'Along_DV']
ncols = len(stages) #number of stages, read stages_df.pkl and get unique of stage
projection_y = 'z'

#taking crosssections and measuring curvature

#take one which has all info (except xyz) which stay same between stages and crosssections
balls_timepoint = pd.read_csv(dirname + "sim_output/init_balls.csv")
springs_timepoint = pd.read_csv(dirname + "sim_output/init_springs.csv")

for i in range(len(stages)):
    
    stage = stages[i]
    if i>0: 
        #for consecutive stages, update the xyz values
        timepoint_pos = pd.read_csv(dirname + "sim_output/"+stage_names[i-1]+".csv")
        balls_timepoint[['x', 'y', 'z']] =  timepoint_pos[['x', 'y', 'z']]
        springs_timepoint = update_springs(springs_timepoint, balls_timepoint[['x', 'y', 'z']])
        
    for crosssection in crosssections:
    
        #take the cross-section
        projection_x = 'x' if crosssection == 'Across_DV' else 'y'
        curve = get_2D_curve_from_simulation(balls_timepoint, springs_timepoint, projection_x = projection_x, projection_y = projection_y)

        #interpolate - take curvature - define a center
        curve = curve.rename(columns={projection_x:"x", projection_y:"y"})
        smooth_curve = compute_2D_curve_curvature(curve)

        #scale the curve
        for prop in ["x", "y", "arclength"]: smooth_curve[prop+"_scaled"] = scale*smooth_curve[prop]
        smooth_curve["curvature_scaled"] = smooth_curve["curvature"]/scale

        #translate the curve
        center_point = smooth_curve[smooth_curve["arclength_scaled"].abs() == smooth_curve["arclength_scaled"].abs().min()][["x_scaled","y_scaled"]].values[0]
        smooth_curve[["x_scaled","y_scaled"]] = smooth_curve[["x_scaled", "y_scaled"]] - center_point

        #save
        smooth_curve.to_csv(dirname + 'sim_output/' + stage + '_' + crosssection + '_curve.csv', index = False)

#plotting


fig, axs = plt.subplots(3, ncols, figsize=(11*ncols, 24))
titles = [str(x) for x in crosssections]

for i in range(len(crosssections)):
    
    crosssection = crosssections[i]
        
    for j in range(len(stages)):
        
        stage = stages[j]
        #read the xyz values for different stages
        if j == 0:
            balls_timepoint = pd.read_csv(dirname + "sim_output/init_balls.csv")
            springs_timepoint = pd.read_csv(dirname + "sim_output/init_springs.csv")
        else:
            timepoint_pos = pd.read_csv(dirname + "sim_output/"+stage_names[j-1]+".csv")
            balls_timepoint[['x', 'y', 'z']] =  timepoint_pos[['x', 'y', 'z']]
            springs_timepoint = update_springs(springs_timepoint, balls_timepoint[['x', 'y', 'z']])
        
        #plot mesh and curve
        ax = axs[i] if ncols == 1 else axs[i, j]
        
        #read the 2D crosssection curve
        projection_x = 'x' if crosssection == 'Across_DV' else 'y'
        smooth_curve = pd.read_csv(dirname + 'sim_output/' + stage + '_' + crosssection + '_curve.csv')
            
        ax.axis('off')
        plot_shell_on_given_ax(balls_timepoint, springs_timepoint,
                               x = projection_x, y = projection_y,
                               ax = ax, fig = fig,
                               #line_color_values = 'final_vs_initial',
                               #cbar_name=r'$\frac{l_{final}}{l_{initial}}$',
                               #cmap = "jet",
                               plot_only_top=True,
                               #line_color_values = np.array([1]*len(springs_timepoint)),
                               show_cbar = False,
                               line_color = "gray",
                               #color_min = 0.6,
                               #color_max = 1.4,
                               line_color_values = np.array([1]*len(springs_timepoint))
                              )
        
        ax.set_title(stage + '\n' + crosssections[i], fontsize = 30, pad = 25)
        
        label = "Across DV" if i == 0 else "Along DV"
        linestyle = "-" if i == 0 else "--"

        ax.plot(smooth_curve.x, smooth_curve.y, alpha = 0.5, linewidth = 10, color = 'red', linestyle = '--', label = 'final')

        ax.legend(loc = 'upper left', fontsize = 25)
        ax.tick_params(axis='both', which='major', labelsize=20)

        # plot the curvature 
        ax = axs[2] if ncols == 1 else axs[2,j]
            
        ax.plot(smooth_curve['arclength_scaled'],smooth_curve['curvature_scaled'], 
                color = 'red', #alpha = 0.5,
                  #label = 'individual discs'
                  #color = single_color, 
                  linewidth = 5, alpha = 0.5, #linestyle = single_linestyle,
                linestyle = linestyle,
                label = label,
                 )
        #ax.set_ylim(-0.5,4.5)
        #if i == 0:
        #    ylim_max = max(smooth_curve['curvature']) + 0.1*np.abs(max(smooth_curve['curvature']))
        #    ylim_min = min(smooth_curve['curvature']) - 0.1*np.abs(min(smooth_curve['curvature']))

        ax.set(ylim = (-0.005,0.07), xlim = (-120,120))
        ax.set_ylabel('curvature', fontsize = 40)
        ax.set_xlabel('arclength', fontsize = 40)
        ax.legend(loc = 'upper left', fontsize = 25)
        ax.axhline(0, linestyle = '-', c = 'grey', linewidth = 0.5)
        ax.axvline(0, linestyle = '-', c = 'grey', linewidth = 0.5)
        #ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.set_yticks([ 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        #ax.set_yticks([-2, 0, 2])
        ax.tick_params(axis='both', which='major', labelsize=20)

plt.savefig(dirname + 'sim_output/crosssection_stages_plot.pdf', bbox_inches = 'tight')



########

skipby = 5

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

    if not(t%skipby == 0):
        if not(t == times[-1]):
            continue

    print('timepoint : ' + str(t))
    
    df_timepoint = pd.read_csv(dirname + 'sim_output/growth_' + str(t) + '/final_0_0.csv') 
    balls_df[['x', 'y', 'z']] = df_timepoint[['x[0]', 'x[1]', 'x[2]']]
    springs_df = update_springs(springs_df, balls_df[['x', 'y', 'z']])
    
    #calculating some properties that are not there already
    springs_df['l0_target/l1'] = springs_df['l0_target']/springs_df['l1']
    springs_df['l1/l0_target'] = springs_df['l1']/springs_df['l0_target']
    springs_df['l0_target_final/l1'] = springs_df['l0_target_final']/springs_df['l1']
    springs_df['l1/l0_target_final'] = springs_df['l1']/springs_df['l0_target_final']
    springs_df['l0/l1_initial'] = springs_df['l0']/springs_df['l1_initial']
    springs_df['l1/l1_initial'] = springs_df['l1']/springs_df['l1_initial']
    springs_df['l0/l1'] = springs_df['l0']/springs_df['l1']
    springs_df['l1/l0'] = springs_df['l0']/springs_df['l1']
    
    
    dfToVtk(balls_df,springs_df,filename=dirname + 'sim_output/bulk_'+ str(t) + '.vtk',
        add_polygons = True)

    #getting vtk with properties on thick mesh
    dfToVtk(balls_df, springs_df, add_lines_properties=True, return_text=False,
            filename = dirname + 'sim_output/thick_wireframe_'+ str(t) + '.vtk',
           )
    
    #getting vtk with properties on thick mesh
    #dfToVtk(balls_df, springs_df, only_top_surface=True, add_lines_properties=True, return_text=False,
    #        filename = dirname + 'sim_output/top_wireframe_'+ str(t) + '.vtk',
    #       )
    #dfToVtk(balls_df, springs_df, only_bottom_surface=True, add_lines_properties=True, return_text=False,
    #        filename = dirname + 'sim_output/bottom_wireframe_'+ str(t) + '.vtk',
    #       )






#############
# Adding curvature
#############

#not every file needs to be computed because it takes long



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

#print('getting bottom surface')
#get bottom surface
#springs_bottom = springs[(springs['ball1'] < len(balls)/2) & (springs['ball2'] < len(balls)/2)]
#balls_bottom = balls[balls['ID'] < len(balls)/2]
#bottom_id_array = balls_bottom['ID'].values
#reindex
#[balls_bottom, springs_bottom] = reindex_balls_springs(balls_bottom, springs_bottom)
#get triangles
#print('getting triangles')
#triangles_bottom = get_oriented_triangles(balls_bottom, springs_bottom)
#get indices of vertices that are not on the boundary
#we need these vertices because the Gaussian curvature is not calculated on the boundary vertices
#nonboundary_id_bottom = balls_bottom.loc[balls_bottom['theta'] < 0.9*max(balls_bottom['theta']),'ID']

print('measuring curvature')

for t in times:

    if not(t%skipby == 0):
        if not(t == times[-1]):
            continue

    print('timepoint : ' + str(t))
    
    df_timepoint = pd.read_csv(dirname + 'sim_output/growth_' + str(t) + '/final_0_0.csv') 
    balls[['x', 'y', 'z']] = df_timepoint[['x[0]', 'x[1]', 'x[2]']]
    springs = update_springs(springs, balls[['x', 'y', 'z']])
    
    
    #balls_bottom[['x','y','z']] = balls.loc[bottom_id_array, ['x','y','z']].values
    balls_top[['x','y','z']] = balls.loc[top_id_array, ['x','y','z']].values

    #measure curvature
    #if not(os.path.exists(dirname + 'sim_output/bottom_surface_'+ str(t) + '.vtk')):
    #[gc_bottom, mc_bottom, triangles_db_bottom, vertices_db_bottom] = measure_integrated_curvature(balls_bottom, springs_bottom, triangles = triangles_bottom,
    #                                                                                               filename = dirname + 'sim_output/bottom_surface_'+ str(t) + '.vtk',
    #                                                                                               nonboundary_indices = nonboundary_id_bottom, write_vtk=True,
    #                                                                                               z_offset = -0.0001
    #                                                                                              )
    #if not(os.path.exists(dirname + 'sim_output/top_surface_'+ str(t) + '.vtk')):
    [gc_top, mc_top, triangles_db_top, vertices_db_top] = measure_integrated_curvature(balls_top, springs_top, triangles = triangles_top,
                                                                                       filename = dirname + 'sim_output/top_surface_'+ str(t) + '.vtk',
                                                                                       nonboundary_indices = nonboundary_id_top, write_vtk=True,
                                                                                       z_offset = 0.0001
                                                                                      )
    #mean_integrated_gc = 0.5*(gc_top + gc_bottom)
    #mean_integrated_mc = 0.5*(mc_top + mc_bottom)



######

