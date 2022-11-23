# for different thicknesses and extrapolation

# for different crosssections

# get data shape

# get simulation shape

# get data curvature

# get simulation curvature

from wd_2D_functions import *

genotype = "ecadGFPnbG4"
exp_data_path = "../exp_data/segmented_curves/"
plot_folder = 'plots/'
os.makedirs(plot_folder,exist_ok=True)

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
thicknesses = [0.05, 0.1, 0.15, 0.2, 0.25]
theta_maxes = [0.6043499999999999,0.4029,]

[nrows,ncols] = [len(theta_maxes), len(thicknesses)]
fig, axs = plt.subplots(nrows, ncols, figsize=(7*ncols, 8*nrows))
fig2, axs2 = plt.subplots(nrows, ncols, figsize=(7*ncols, 8*nrows))

for i in range(nrows):
    theta_max = theta_maxes[i]
    for j in range(ncols):
        thickness = thicknesses[j]

        ax = axs[i,j]
        ax2 = axs2[i,j]

        sim_dir = f"wd_thickness_{thickness}_nb_iterations_20_theta_max_{theta_max}/sim_output/"

        for crosssection in crosssections:

            linestyle = "-" if crosssection == "Across_DV" else "--"

            #df_all = pd.read_csv(exp_data_path + crosssection + "_" + genotype + "_pouchShape_interpolated_all.csv")
            df_mean = pd.read_csv(exp_data_path + crosssection + "_" + genotype + "_pouchShape_interpolated_mean.csv")
            df = df_mean[((df_mean['genotype'] == genotype) & (df_mean['devstage'] == devstage)) & (df_mean['crosssection'] == crosssection)]

            #select axis
            ax = axs[i,j]
            ax.axis("equal")
            ax.set_ylim(-80, 10)
            ax.set_xlim(-110, 110)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0: ax.set_title(devstage, fontsize = 30,)

            #plot data shape
            ax.plot(df['x'], df['y'],color = 'red',linewidth = 4, label = 'Mean', zorder = 2, linestyle = linestyle)
            ax.errorbar(df['x'], df['y'], xerr = df['x_sd'], yerr = df['y_sd'],ecolor = 'gray', alpha = 0.5, label = 'Std dev', zorder = 1)

            #read simulation data
            try:
                sim_df = pd.read_csv(sim_dir + devstage + "_" + crosssection + "_curve.csv")
            except:
                print(sim_dir + devstage + "_" + crosssection + "_curve.csv")
                continue
            #plot simulation shape
            ax.plot(sim_df['x'], sim_df['y'],color = 'black',linewidth = 4, label = 'thickness='+str(thickness), zorder = 2, linestyle = linestyle)

            #select axis 
            ax2 = axs2[i, j]
            ax2.set_ylim(-0.022, 0.022)
            ax2.set_xlim(-140, 140)
            ax2.set_xlabel('Arclength ' + r'$(\mu m)$', fontsize = 20)
            if i == 0: ax2.set_title('Curvature ' + r'$(\mu m^{-1})$', fontsize = 30)
            ax2.set_xticks([-100, 0, 100])
            ax2.tick_params(axis = 'both', labelsize = 20)
            ax2.grid()
            ax2.legend(fontsize = 15)

            #plot data curvature
            ax2.fill_between(x = df['arclength'].tolist(), y1 = (df['curvature'] - df['curvature_sd']).tolist(), y2 = (df['curvature'] + df['curvature_sd']).tolist(), color = 'gray', alpha = 0.2)
            ax2.plot(df['arclength'], df['curvature'], label = crosssection, linewidth = 4, color = 'red', linestyle = linestyle,
                   )
            #plot sim curvature
            ax2.plot(sim_df['arclength'], sim_df['curvature'], label = 'thickness='+str(thickness), linewidth = 4, color = 'black', linestyle = linestyle,
            )

fig.savefig(f"plots/shape_{devstage}.pdf", bbox_inches = "tight")
fig2.savefig(f"plots/curvature_{devstage}.pdf", bbox_inches = "tight")


#Archive
"""
scalebar_stage = "wL3"
scalebar_x_pos = 120
scalebar_y_pos = -450
scalebar_size = 100
scalebar_linewidth = 5
scalebar_fontsize = 20
scalebar_color = 'black'
"""










