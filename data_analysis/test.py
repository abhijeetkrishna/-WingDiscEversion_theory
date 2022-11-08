#files = ['../exp_data/segmented_curves/along_DV_pouchShape_ecadGFPnbG4.pkl'] #Along DV
#files = ['../exp_data/segmented_curves/ecadGFPnbG4pouchShape.pkl'] #Across DV
#files = ['shapes.csv']

allcurves = allcurves[allcurves["genotype"] == genotype]
devstages = ['upcrawling', 'whitePupa', '2hAPF', '4hAPF','6hAPF']
allcurves = import_raw_crosssection_points(files = [file], plot = False,)


#updating name of devstages
devstage_name_update_dict = {
    "120hAEL":"120hAEL",
    "96hAEL":"96hAEL",
    "upcrawling":"wL3",
    "whitePupa" : "0hAPF",
    "2hAPF": "2hAPF",
    "4hAPF": "4hAPF",
    "6hAPF": "6hAPF",
}

allcurves["devstage"] = allcurves.apply(lambda row: devstage_name_update_dict[row["devstage"]], axis = 1) #[devstage_name_update[x] for x in allcurves["devstage"].values]
devstages = ['wL3', '0hAPF', '2hAPF', '4hAPF','6hAPF']

fig, axs = plt.subplots(2, len(devstages), figsize=(7*len(devstages), 15))


# align
allcurves = align_curves(allcurves = allcurves,orientation = 'horizontal', devstages=devstages, filename  = plot_folder + genotype + "_" + crosssection + '_aligned.pdf'
                        )

#interpolate
[df_all, df_mean] = interpolate_average_curves(allcurves, devstages=devstages,)
df_all["crosssection"] = crosssection
df_mean["crosssection"] = crosssection


#save csv
df_all.to_csv(file.replace(".pkl", "_interpolated_all.csv"), index = False)
df_mean.to_csv(file.replace(".pkl", "_interpolated_mean.csv"), index = False)

#plotting
scalebar_stage = "wL3"
scalebar_x_pos = 120
scalebar_y_pos = -450
scalebar_size = 100
scalebar_linewidth = 5
scalebar_fontsize = 20
scalebar_color = 'black'

for j in range(len(devstages)):
    
    devstage = devstages[j]
    
    df = df_mean[((df_mean['genotype'] == genotype) & (df_mean['devstage'] == devstage)) & (df_mean['crosssection'] == crosssection)]
    #gen_dev = database[(database['genotype'] == genotype) & (database['devstage'] == devstage)]

    #plot mean curve
    
    #linestyle
    if crosssection == "Across_DV":
        linestyle = "-"
    else:
        linestyle = "--"
    

    ax = axs[0, j] #axs[j, 0]
    #ax.axis('equal')
    ax.set_ylim(-510, 10)
    ax.set_xlim(-260, 260)
    ax.plot(df['x'], df['y'],color = 'red',linewidth = 4, label = 'Mean', zorder = 2, linestyle = linestyle)
    ax.errorbar(df['x'], df['y'], xerr = df['x_sd'], yerr = df['y_sd'],ecolor = 'gray', alpha = 0.5, label = 'Std dev', zorder = 1)
    #ax.fill_between(df['x'], df['curvature'] - df['curvature_sd'], df['curvature'] + df['curvature_sd'], color = 'gray', alpha = 0.2)

    #ax.set_ylabel(devstage, fontsize = 30, rotation = 0, labelpad = 100)


    #adding scale bar
    ax.plot([scalebar_x_pos, scalebar_x_pos + scalebar_size],[scalebar_y_pos,scalebar_y_pos], lw = scalebar_linewidth, color = scalebar_color)
    if devstage == scalebar_stage:
        ax.text(x = scalebar_x_pos + 10, y = scalebar_y_pos + 20, s = str(scalebar_size) + ' ' + r'$\mu m$', fontsize = scalebar_fontsize)


    #switching off ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

    #if j == 0:
    ax.set_title(devstage, fontsize = 30,)


    #plot mean curvature
    ax = axs[1, j]
    ax.set_ylim(-0.022, 0.022)
    ax.set_xlim(-300, 300)
    #ax.plot(df['arclength'],df['curvature'], color = 'black', alpha = 1,)
    ax.set_xlabel('Arclength ' + r'$(\mu m)$', fontsize = 20)
    #ax.set_ylabel('Curvature ' + r'$(\mu m^{-1})$', fontsize = 20, labelpad = 10)
    #ax.fill_between(df['arclength'], df['curvature'] - df['curvature_sd'], df['curvature'] + df['curvature_sd'], color = 'gray', alpha = 0.2)
    ax.fill_between(x = df['arclength'].tolist(), y1 = (df['curvature'] - df['curvature_sd']).tolist(), y2 = (df['curvature'] + df['curvature_sd']).tolist(), color = 'gray', alpha = 0.2)
    ax.plot(df['arclength'], df['curvature'], label = genotype, linewidth = 4, color = 'red', linestyle = linestyle )


    #if j == 0:
    ax.set_title('Curvature ' + r'$(\mu m^{-1})$', fontsize = 30)
    ax.set_xticks([-150, 0, 150])
    if devstage == '6hAPF':
        if max(df['curvature']) > 0.02:
            ax.set_ylim(-1.1*max(df['curvature'] + df['curvature_sd']),1.1*max(df['curvature'] + df['curvature_sd']))
            ax.set_yticks([-0.03, 0, 0.03])
    else:
        ax.set_yticks([-0.02, -0.01, 0, 0.01, 0.02])
    #ax.tick_params(axis = 'both', which = 'both', fontsize = 20)
    ax.tick_params(axis = 'both', labelsize = 20)
    ax.grid()

