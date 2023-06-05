 
# for different thicknesses and extrapolation

# for different crosssections

# get data shape

# get simulation shape

# get data curvature

# get simulation curvature

from wd_2D_functions import *

def plotter(fig, axs, crosssections = ["Along_DV", "Across_DV"],
            main_xlabel = 'Thickness/R', main_ylabel = 'Cap-size\n' + r'$(\theta_{max}/\theta_{ref})$',
            tick_intervals_x_str = None, tick_intervals_y_str = None, tick_fontsize = None,
            row_vals = None, col_vals = None, set_axis_equal = True,
            ylim = (-0.9, 0.01), xlim = (-0.6, 0.6), scale = 77.66,
            plot_data = True, exp_data_path = "../exp_data/segmented_curves/",
            plot_initial = True,
            sub_title = "4hAPF", main_title = "", x_name = "x", y_name = "y",
            param_comb_df = None, query_str = "thickness == @col_val and theta_max == @row_val",
            ):

    main_ax = fig.add_subplot(111, frameon=True, alpha = 0.5)
    main_ax.set_facecolor('none')
    # Hide the right and top spines
    main_ax.spines['right'].set_visible(False)
    main_ax.spines['top'].set_visible(False)
    main_ax.spines['left'].set_position(('axes', -0.05))
    main_ax.spines['bottom'].set_position(('axes', -0.05))
    #ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    #ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    # Only show ticks on the left and bottom spines
    #main_ax.yaxis.set_ticks_position('left')
    #main_ax.xaxis.set_ticks_position('bottom')
    main_ax.set_xlabel(main_xlabel, fontsize = 40, labelpad = 30)
    main_ax.set_ylabel(main_ylabel, fontsize = 40, labelpad = 80, rotation = 0)
    main_ax.set_xlim(-0.5,len(col_vals) -0.5)
    main_ax.set_ylim(-0.5, len(row_vals) -0.5)
    #tick_intervals_x = [round(x,2) for x in col_vals]
    tick_intervals_x = col_vals
    tick_intervals_y = np.sort(row_vals) #[round(x,2) for x in row_vals]
    #tick_intervals_x_str = [str(round(x,3)) for x in col_vals]
    #tick_intervals_x_str = ['%.1E' % Decimal(str(x)) for x in col_vals]
    if tick_intervals_x_str is None : tick_intervals_x_str = [str(round(x,2)) for x in tick_intervals_x]
    if tick_intervals_y_str is None : tick_intervals_y_str = [str(round(x,2)) for x in tick_intervals_y]
    main_ax.set_xticks(range(len(tick_intervals_x_str)), tick_intervals_x_str, fontsize = 30)
    #plt.yticks(tick_intervals_x, [r'$8.5 \times 10^{-5}$', r'$1 \times 10^{-4}$', r'$1.15 \times 10^{-4}$'], fontsize = 16)
    main_ax.set_yticks(range(len(tick_intervals_y_str)), tick_intervals_y_str, fontsize = 30)
    main_ax.tick_params(axis=u'both', which=u'both',length=10)
    #main_ax.set_title(main_title, fontsize = 40, y = 1.1)

    for i in range(len(row_vals)):
        row_val = row_vals[i]
        for j in range(len(col_vals)):
            col_val = col_vals[j]

            ax = axs[i,j]

            #sim_dir = f"wd_thickness_{thickness}_nb_iterations_20_theta_max_{theta_max}/sim_output/"
            #read 

            for crosssection in crosssections:

                linestyle = "-" if crosssection == "Across_DV" else "--"

                #df_all = pd.read_csv(exp_data_path + crosssection + "_" + genotype + "_pouchShape_interpolated_all.csv")
                #df_mean = pd.read_csv(exp_data_path + crosssection + "_" + genotype + "_pouchShape_interpolated_mean.csv")
                #df = df_mean[((df_mean['genotype'] == genotype) & (df_mean['devstage'] == devstage)) & (df_mean['crosssection'] == crosssection)]

                #select axis
                ax = axs[i,j]
                if set_axis_equal : ax.axis("equal")
                if ylim is not None: ax.set_ylim(ylim[0], ylim[1])
                if xlim is not None: ax.set_xlim(xlim[0], xlim[1])
                if tick_fontsize is not None: ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                
                #ax.tick_params(axis='both', which='minor', labelsize=8)
                #Comment following if you want ticks
                #ax.set_xticklabels([])
                #ax.set_yticklabels([])
                #ax.set_xticks([])
                #ax.set_yticks([])
                if i == 0: ax.set_title(sub_title, fontsize = 30,)

                if plot_data:
                    print("plotting data")
                    [x_name_data,y_name_data] = [x_name.replace("_scaled",""),y_name.replace("_scaled","")]
                    df_all = pd.read_csv(exp_data_path + crosssection + "_" + genotype + "_pouchShape_interpolated_all.csv")
                    df_mean = pd.read_csv(exp_data_path + crosssection + "_" + genotype + "_pouchShape_interpolated_mean.csv")
                    df = df_mean[((df_mean['genotype'] == genotype) & (df_mean['devstage'] == devstage)) & (df_mean['crosssection'] == crosssection)]
                    if crosssection == "Along_DV": df = df[np.abs(df["arclength"]) <= 103.39] #this is missing DV width in 4hAPF
                    if crosssection == "Across_DV": df = df[np.abs(df["arclength"]) <= (82 + 15/2)] #this is missing DV width in 4hAPF
                    #df = df[np.abs(df["arclength"]) <= scale*row_val]
                    ax.plot(df[x_name_data], df[y_name_data],color = 'red',linewidth = 4, label = 'Mean', zorder = 2, linestyle = linestyle)
                    #x_name_data_sd = x_name_data+"_sd" if x_name_data=="x" else x_name_data
                    #ax.errorbar(df[x_name_data], df[y_name_data], xerr = df[x_name_data_sd], yerr = df[y_name_data+"_sd"], color = 'gray', alpha = 0.5, label = 'Std dev', zorder = 1)

                #read simulation data
                try:
                    sim_dir = param_comb_df.query(query_str)["folder_name"].values[0] + "sim_output/"
                    sim_df = pd.read_csv(sim_dir + devstage + "_" + crosssection + "_curve.csv")
                except:
                    print("could not read : ")
                    print(sim_dir + devstage + "_" + crosssection + "_curve.csv")
                    continue
                #plot simulation shape
                print("plotting simulation")
                ax.plot(sim_df[x_name], sim_df[y_name],color = 'black',linewidth = 4, label = 'thickness='+str(col_val), zorder = 2, linestyle = linestyle)

                #plot initial
                if plot_initial:
                    init_smooth_curve = pd.read_csv(sim_dir + crosssection + '_init_curve.csv')
                    ax.plot(init_smooth_curve[x_name], init_smooth_curve[y_name], color = "blue", linewidth = 2, label = "initial", linestyle = linestyle)


    return(fig,axs)

genotype = "ecadGFPnbG4"
exp_data_path = "../exp_data/segmented_curves/"
plot_folder = 'plots/'
os.makedirs(plot_folder,exist_ok=True)
theta_ref = 0.8662

crosssections = ["Across_DV",
                 "Along_DV", 
                ]
devstage_name_update_dict = {
    "120hAEL":"120hAEL",
    "96hAEL":"96hAEL",
    "upcrawling":"wL3",
    "whitePupa" : "0hAPF",
    "2hAPF": "2hAPF",
    "4hAPF": "4hAPF",
    "6hAPF": "6hAPF",
}
devstages = ['wL3', '0hAPF', '2hAPF', '4hAPF','6hAPF']
devstage = "4hAPF"

#read the map_index_[0-9].csv file
param_comb_df = pd.read_csv(glob.glob('map_index_[0-9]*.csv')[0])
#get the unique values of two parameters
thicknesses = np.sort(np.unique(param_comb_df['thickness']))#[0:7] #np.linspace(0.07, 0.27,11)
#thicknesses = thicknesses[[1,2,3,4]]
#second parameter needs to be in descending order
theta_maxes = np.sort(np.unique(param_comb_df['theta_max']))#[0:7] #0.4029*np.linspace(1,1.8,9)
#theta_maxes = theta_maxes[[0,3,6,9]]
theta_maxes = -np.sort(-theta_maxes)
tick_intervals_y_str = np.sort([str(round(x/theta_ref,2)) for x in theta_maxes])

[nrows,ncols] = [len(theta_maxes), len(thicknesses)]
print("plotting shapes - full size")
fig, axs = plt.subplots(nrows, ncols, figsize=(7*ncols, 8*nrows))
fig,axs = plotter(fig, axs, crosssections = ["Along_DV", "Across_DV"],
            main_xlabel = 'Thickness/R', main_ylabel = 'Cap-size\n' + r'$(\theta_{max}/\theta_{ref})$',
            row_vals = theta_maxes, col_vals = thicknesses, set_axis_equal = True,
            ylim = (-0.9, 0.01), xlim = (-0.6, 0.6), 
            plot_data = False, tick_intervals_y_str = tick_intervals_y_str, tick_fontsize = 20,
            plot_initial = True,
            sub_title = "4hAPF", main_title = "", x_name = "x", y_name = "y",
            param_comb_df = param_comb_df, query_str = "thickness == @col_val and theta_max == @row_val",
            )
fig.savefig(f"plots/shape_{devstage}.pdf", bbox_inches = "tight")


print("plotting scaled shapes")
fig_zoomed, axs_zoomed = plt.subplots(nrows, ncols, figsize=(7*ncols, 8*nrows))
fig_zoomed,axs_zoomed = plotter(fig_zoomed, axs_zoomed, crosssections = ["Along_DV", "Across_DV"],
            main_xlabel = 'Thickness/R', main_ylabel = 'Cap-size\n' + r'$(\theta_{max}/\theta_{ref})$',
            row_vals = theta_maxes, col_vals = thicknesses, set_axis_equal = True,
            ylim = (-140,10), xlim = (-75,75), 
            plot_data = True, tick_intervals_y_str = tick_intervals_y_str, tick_fontsize = 20,
            plot_initial = False, 
            sub_title = "4hAPF", main_title = "", x_name = "x_scaled", y_name = "y_scaled",
            param_comb_df = param_comb_df, query_str = "thickness == @col_val and theta_max == @row_val",
            )
fig_zoomed.savefig(f"plots/shape_{devstage}_scaled.pdf", bbox_inches = "tight")


print("plotting curvature")
fig_curv, axs_curv = plt.subplots(nrows, ncols, figsize=(7*ncols, 8*nrows))
fig_curv,axs_curv = plotter(fig_curv, axs_curv, crosssections = ["Along_DV", "Across_DV"],
            main_xlabel = 'Thickness/R', main_ylabel = 'Cap-size\n' + r'$(\theta_{max}/\theta_{ref})$',
            row_vals = theta_maxes, col_vals = thicknesses, set_axis_equal = False,
            ylim = (-1, 17), xlim = (-1, 1), 
            plot_data = False, tick_intervals_y_str = tick_intervals_y_str, tick_fontsize = 20,
            sub_title = "4hAPF", main_title = "", 
            plot_initial = True,
            x_name = "arclength", y_name = "curvature",
            param_comb_df = param_comb_df, query_str = "thickness == @col_val and theta_max == @row_val",
            )
fig_curv.savefig(f"plots/curvature_{devstage}.pdf", bbox_inches = "tight")

print("plotting scaled curvature")
fig_curv, axs_curv = plt.subplots(nrows, ncols, figsize=(7*ncols, 8*nrows))
fig_curv,axs_curv = plotter(fig_curv, axs_curv, crosssections = ["Along_DV", "Across_DV"],
            main_xlabel = 'Thickness/R', main_ylabel = 'Cap-size\n' + r'$(\theta_{max}/\theta_{ref})$',
            row_vals = theta_maxes, col_vals = thicknesses, set_axis_equal = False,
            ylim = (-0.005, 0.06), xlim = (-110, 110), 
            plot_data = True, tick_intervals_y_str = tick_intervals_y_str, tick_fontsize = 20,
            sub_title = "4hAPF", main_title = "", 
            plot_initial = False, 
            x_name = "arclength_scaled", y_name = "curvature_scaled",
            param_comb_df = param_comb_df, query_str = "thickness == @col_val and theta_max == @row_val",
            )
fig_curv.savefig(f"plots/curvature_{devstage}_scaled.pdf", bbox_inches = "tight")
