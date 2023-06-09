#!/usr/bin/env python
# coding: utf-8

# # Averaging simulated curves
# 
# Each sim has two cross-sections saved. For each parameter combination (PC), there are different discs (or seeds or some other alteration). I want to do the following
# 
# - for each PC each crosssection, combine all discs into one dataframe
# - for each PC each crosssection, get an average of all curves
# - repeat for each PC each crosssection
# - Get a parameter sweep for different PCs and diff crosssection
# 
# I want to do it in such a way that the code should work also in the following two conditions:
# - When there is just one disc
# - When there are spherical meshes instead of actual discs

# In[1]:


from wd_2D_functions import *


# In[2]:


def helper_combine_dfs(curves = None, crosssection="Along_DV", devstage="4hAPF", folder="/", discname = None, genotype = 'ecadGFPnbG4'):
    file = "sim_output/"+devstage+"_"+crosssection+"_curve.csv"
    #get the disc name
    if discname is None:
        discname = folder.replace("/","")#re.search("disc_name_(.+?)/", folder)
    #read the curve file
    try:
        single_curve = pd.read_csv(folder+file)
    except:
        return(curves)
    if devstage == "4hAPF" and crosssection == "Across_DV":
        #min_curvature = min(single_curve[(np.abs(single_curve["arclength_scaled"]) <= 50)]["curvature_scaled"].values)
        max_curvature = max(np.abs(single_curve[(np.abs(single_curve["arclength_scaled"]) <= 50)]["curvature_scaled"].values))
        arclength_offset = single_curve[(np.abs(single_curve["curvature_scaled"]) == max_curvature) & (np.abs(single_curve["arclength_scaled"]) <= 50)]["arclength_scaled"].values[0]
        single_curve["arclength_scaled"] = single_curve["arclength_scaled"] - arclength_offset
    single_curve = single_curve[["x_scaled", "y_scaled", "arclength_scaled"]]
    single_curve.columns = ["x", "y", "arclength_offset"]
    single_curve["disc"] = discname
    single_curve["genotype"] = genotype
    single_curve["devstage"] = devstage
    single_curve["crosssection"] = crosssection
    #add to the curves dataframe
    if curves is None: 
        curves = single_curve.copy(deep = True).reset_index(drop = True)
    else: 
        curves = pd.concat([curves, single_curve]).reset_index(drop = True)

    return(curves)


# In[6]:


def helper_plotter(fig,axs,
                   main_xlabel = "stages",main_ylabel = "thickness/R", tick_intervals_x_str = None, tick_intervals_y_str = None,
                   subplot_xlabel = None, subplot_ylabel = None,
                   set_axis_equal = False,x_name = "arclength",y_name = "curvature",
                   xlim = None,ylim = None, subplot_ticks_bool = True, sub_title = "",plot_data =True, plot_sim = True,
                   sim_data_file = "analysis/crosssections.csv",exp_data_path = "../exp_data/segmented_curves/",
                   Large_font_size = 40, Medium_font_size = 30, Small_font_size = 20, linewidth = 1,
                   layout_pad = 3, main_ax_labelpad = 20, spine_position = 100,
                  ):
    

    ######
    # Customizing main axis
    #####

    main_ax = fig.add_subplot(111, frameon=True, alpha = 0.5)
    main_ax.set_facecolor('none')
    # Hide the right and top spines
    main_ax.spines['right'].set_visible(False)
    main_ax.spines['top'].set_visible(False)
    #main_ax.spines['left'].set_position(('axes', -0.05))
    #main_ax.spines['bottom'].set_position(('axes', -0.05))
    main_ax.spines['left'].set_position(('outward', spine_position))
    main_ax.spines['bottom'].set_position(('outward', spine_position))
    main_ax.set_xlabel(main_xlabel, fontsize = Large_font_size, labelpad = main_ax_labelpad)
    main_ax.set_ylabel(main_ylabel, fontsize = Large_font_size, labelpad = main_ax_labelpad, rotation = 90)
    main_ax.set_xlim(-0.5,len(col_vals) -0.5)
    main_ax.set_ylim(-0.5, len(row_vals) -0.5)
    #tick_intervals_x = [round(x,2) for x in col_vals]
    tick_intervals_x = col_vals
    tick_intervals_y = np.sort(row_vals) #[round(x,2) for x in row_vals]
    #tick_intervals_x_str = [str(round(x,3)) for x in col_vals]
    #tick_intervals_x_str = ['%.1E' % Decimal(str(x)) for x in col_vals]
    if tick_intervals_x_str is None : tick_intervals_x_str = [str(round(x,2)) for x in tick_intervals_x]
    if tick_intervals_y_str is None : tick_intervals_y_str = [str(round(x,2)) for x in tick_intervals_y]
    main_ax.set_xticks(range(len(tick_intervals_x_str)), tick_intervals_x_str, fontsize = Medium_font_size)
    main_ax.set_yticks(range(len(tick_intervals_y_str)), tick_intervals_y_str, fontsize = Medium_font_size)
    main_ax.tick_params(axis=u'both', which=u'both',length=Medium_font_size/2)
    #main_ax.set_title(main_title, fontsize = 40, y = 1.1)

    #####
    # adding subplots
    #####

    for i in range(len(row_vals)):
        row_val = row_vals[i]
        thickness = row_val

        for j in range(len(col_vals)):
            col_val = col_vals[j]
            devstage = col_val

            ax = axs[i,j]
            if subplot_xlabel is not None: ax.set_xlabel(subplot_xlabel, fontsize = Medium_font_size)
            if subplot_ylabel is not None: ax.set_ylabel(subplot_ylabel, fontsize = Medium_font_size)
                
            for crosssection in crosssections:

                linestyle = "-" if crosssection == "Across_DV" else "--"

                if set_axis_equal : ax.axis("equal")
                if ylim is not None: ax.set_ylim(ylim[0], ylim[1])
                if xlim is not None: ax.set_xlim(xlim[0], xlim[1])
                if subplot_ticks_bool is not None: ax.tick_params(axis='both', which='major', labelsize=Small_font_size)

                #ax.tick_params(axis='both', which='minor', labelsize=8)
                #Comment following if you want ticks
                #ax.set_xticklabels([])
                #ax.set_yticklabels([])
                #ax.set_xticks([])
                #ax.set_yticks([])

                if i == 0: ax.set_title(sub_title, fontsize = Medium_font_size,)

                if plot_data:
                    #print("plotting data")
                    [x_name_data,y_name_data] = [x_name.replace("_scaled",""),y_name.replace("_scaled","")]
                    df_all = pd.read_csv(exp_data_path + crosssection + "_" + genotype + "_pouchShape_interpolated_all.csv")
                    df_mean = pd.read_csv(exp_data_path + crosssection + "_" + genotype + "_pouchShape_interpolated_mean.csv")
                    df = df_mean[((df_mean['genotype'] == genotype) & (df_mean['devstage'] == devstage)) & (df_mean['crosssection'] == crosssection)]
                    #get the arclength_threshold
                    max_pathlength_df = pd.read_csv(exp_data_path + "max_pathlength_df_"+genotype+".csv")
                    arclength_threshold = max_pathlength_df.query("devstage == @devstage and crosssection == @crosssection")["max_pathlength"].values[0]
                    df = df[np.abs(df["arclength"]) <= arclength_threshold]
                    #plot
                    ax.plot(df[x_name_data], df[y_name_data],color = 'red',linewidth = linewidth, label = 'Data ' + crosssection, zorder = 2, linestyle = linestyle)
                    x_name_data_sd = x_name_data+"_sd" if x_name_data=="x" else x_name_data
                    ax.errorbar(df[x_name_data], df[y_name_data], #xerr = df[x_name_data_sd], 
                                yerr = df[y_name_data+"_sd"], color = 'red', alpha = 0.5, zorder = 1)

                if plot_sim:
                    #print("plotting simulation results")
                    df_all = pd.read_csv(sim_data_file.replace(".csv", "_interpolated_all.csv"))
                    df_mean = pd.read_csv(sim_data_file.replace(".csv", "_interpolated_mean.csv"))
                    df = df_mean.query("genotype == @genotype and devstage == @devstage and crosssection == @crosssection and thickness == @thickness")
                    [((df_mean['genotype'] == genotype) & (df_mean['devstage'] == devstage)) & (df_mean['crosssection'] == crosssection)]
                    ax.plot(df[x_name], df[y_name],color = 'black',linewidth = linewidth, label = 'Model ' + crosssection, zorder = 2, linestyle = linestyle)
                    #x_name_sd = x_name+"_sd" if x_name=="x" else x_name
                    ax.errorbar(df[x_name], df[y_name], #xerr = df[x_name_sd], 
                                yerr = df[y_name+"_sd"], color = 'gray', alpha = 0.5, zorder = 1)
                    
                if i == 0 and j == 0:
                    ax.legend(fontsize = Small_font_size)
    
    fig.tight_layout(pad = layout_pad)

    return(fig,axs)


# # Average crosssections

# In[4]:


crosssections = ["Across_DV","Along_DV"]
param_comb_df = pd.read_csv(glob.glob('map_index_[0-9]*.csv')[0])
devstages = ["wL3", "0hAPF", "2hAPF", "4hAPF"]
thicknesses = np.sort(np.unique(param_comb_df["thickness"]))
genotype = 'ecadGFPnbG4'

curves_all = pd.DataFrame()
curves_mean = pd.DataFrame()

step_size_in_microns = 2

for thickness in thicknesses:
    
    thickness_param_comb_df = param_comb_df.query("thickness == @thickness")
    folders = thickness_param_comb_df["folder_name"].values
    
    #combine all discs to one dataframe
    curves = None
    for folder in folders:
        #Within each folder we have 4 devstages
        if "tol_by_dt_1e-06" in folder:
            continue
        for devstage in devstages:
            #For each devstage we have two crosssections
            for crosssection in crosssections:
                #discname = thickness_param_comb_df.query("folder_name == @folder").iloc[0]["disc_name"]
                curves = helper_combine_dfs(curves, crosssection=crosssection, devstage=devstage, folder=folder, 
                                               genotype = genotype, #discname = discname,
                                              )
    if curves is None: continue
        
    #take average within each cross-section
    for crosssection in crosssections:

        curves_filtered = curves.query("crosssection == @crosssection")
        #this function groups data by devstages so we don't need to loop here
        [df, df_mean] = interpolate_average_curves(curves_filtered, genotypes = [genotype], devstages = devstages,step_size_in_microns=step_size_in_microns)

        #save to curves_all - here each simulation is stored separately
        df["thickness"] = thickness
        df["crosssection"] = crosssection
        curves_all = pd.concat([curves_all, df]).reset_index(drop = True)
        #save to curves_mean - here we take a mean between all simulations
        df_mean["thickness"] = thickness
        df_mean["crosssection"] = crosssection
        curves_mean = pd.concat([curves_mean, df_mean]).reset_index(drop = True)
        
output_dir = "analysis/"
os.makedirs(output_dir, exist_ok = True)
output_file = output_dir+"crosssections.csv"
curves_all.to_csv(output_file.replace(".csv", "_interpolated_all.csv"), index = False)
curves_mean.to_csv(output_file.replace(".csv", "_interpolated_mean.csv"), index = False)


# # Plotting
# Next we do a parameter sweep kind of plot - for shapes and for curvature
# I want the following -
# - On y axis thickness and on x axis devstages
# - I want to plot exp data as well as simulation - but it should be optional
# - I want to plot individual traces or std dev or sem - it should be optional

# In[9]:

exp_data_path = "/projects/project-krishna/WingDiscEversion_theory/Experiments/data/segmented_curves/"
thicknesses = np.unique(curves_all["thickness"])
Large_font_size= 15
Medium_font_size = 12
Small_font_size = 8

row_vals = np.flip(np.sort(thicknesses)) #need to keep in descending order
col_vals = devstages
[nrows,ncols] = [len(row_vals), len(col_vals)]
fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))

fig,axs = helper_plotter(fig,axs, x_name="arclength", y_name = "curvature", set_axis_equal = False,
                         exp_data_path = exp_data_path, 
                         ylim = (-0.005,0.06), xlim = (-110, 110),
                         tick_intervals_x_str = devstages,
                         subplot_xlabel = "s", subplot_ylabel = r"$\kappa$",
                         Large_font_size= Large_font_size, Medium_font_size = Medium_font_size, Small_font_size = Small_font_size,
                         spine_position=50, layout_pad=1, linewidth = 1,
                        )
os.makedirs("plots/", exist_ok = True)
fig.savefig("plots/curvature_averaged_with_data.pdf", bbox_inches = "tight")


# In[11]:


thicknesses = np.unique(curves_all["thickness"])
#devstages = ["wL3", "4hAPF"]

row_vals = np.flip(np.sort(thicknesses)) #need to keep in descending order
col_vals = devstages
[nrows,ncols] = [len(row_vals), len(col_vals)]
fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))

fig,axs = helper_plotter(fig,axs, x_name="x", y_name = "y", set_axis_equal = True,
                         exp_data_path = exp_data_path, 
                         ylim = (-50,5),xlim = (-60,60),
                         tick_intervals_x_str = devstages,
                         #subplot_xlabel = "x", subplot_ylabel = "y",
                         Large_font_size= Large_font_size, Medium_font_size = Medium_font_size, Small_font_size = Small_font_size,
                         spine_position=50, layout_pad=1, linewidth = 1,
                         plot_data=True,
                        )
os.makedirs("plots/", exist_ok = True)
fig.savefig("plots/shape_averaged_with_data.pdf", bbox_inches = "tight")

