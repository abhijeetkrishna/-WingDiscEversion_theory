#!/usr/bin/env python
# coding: utf-8

# In[50]:


from methods import *


# In[51]:



def plot_shell(balls_df, springs_df, x = 'x', y = 'z', filename = None, title = '', line_color_values = None,
               color_min = None, color_max = None, cbar_labels = None, cbar_ticks = None, cmap = 'viridis',
               cbar_name = None,
               linewidth = 1,
               xlim_min = None, xlim_max = None, ylim_min = None, ylim_max = None,
               plot_only_top = False,
               fig = None, ax = None, return_ax = False, fontsize = None,
              ):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm

    #x = 'x'
    #y = 'z'
    if fontsize is None: fontsize = 20

    #springs_df_stack = springs_df
    if plot_only_top:
        springs_df_stack = springs_df[(springs_df['ball1'] >= len(balls_df)/2) & (springs_df['ball2'] >= len(balls_df)/2)]
    else:
        springs_df_stack = springs_df

    if ax is None:
        fig,ax = plt.subplots()
    #points_demo = np.array(balls_final[['x', 'y']]).reshape(-1, 1, 2)

    #plot the original network too
    #springs = update_springs(springs_final, balls_unstretched[['x', 'y', 'z']])
    points_1 = np.array(springs_df_stack[[x + '1', y + '1']]).reshape(-1, 1, 2)
    points_2 = np.array(springs_df_stack[[x + '2', y + '2']]).reshape(-1, 1, 2)

    #create a collection of lines
    segments_demo = np.concatenate([points_1, points_2], axis = 1)

    #value by which to color lines
    #if relative_to == 'initial':
    #    springs_final['l1_l0'] = springs_final['l1']/springs_final['l0']
    #elif relative_to == 'absolute':
    #    springs_final['l1_l0'] = springs_final['l1']

    if line_color_values is None:
        line_color_values = np.array(springs_df_stack['l0_target']/springs_df_stack['l1_initial'])
    #dydx = np.array(springs_final['l1_l0']) #just given a name from tutorial, dydx does not mean anything in my case

    # Create a continuous norm to map from data points to colors

    #color_min = 1.5
    #color_max = 1

    if color_min is None:
        color_min = line_color_values.min()
    if color_max is None:
        color_max = line_color_values.max()

    norm = plt.Normalize(color_min, color_max)
    lc = LineCollection(segments_demo, cmap=cmap, norm=norm)

    # Set the values used for colormapping
    lc.set_array(line_color_values)
    lc.set_linewidth(linewidth)
    line = ax.add_collection(lc)
    #fig.colorbar(line, ax=ax
    #            )

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    if cbar_ticks is None:
        cbar_ticks = [color_min,
                      1,#(color_max + color_min)/2,
                      color_max]
    if cbar_labels is None:
        cbar_labels = [str(round(tick,2)) for tick in cbar_ticks]

    cbar = fig.colorbar(line, ax = ax, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(labels = cbar_labels, fontsize = fontsize)  # vertically oriented colorbar
    if not(cbar_name is None):
        cbar.ax.set_ylabel(cbar_name, rotation=0, fontsize = fontsize)

    if xlim_min is None:
        [xlim_min,xlim_max] = [min(balls_df[x]), max(balls_df[x])]
        [ylim_min,ylim_max] = [min(balls_df[y]), max(balls_df[y])]

    plt.ylim(ylim_min, ylim_max)
    plt.xlim(xlim_min, xlim_max)
    plt.axis('equal')

    plt.title(title, fontsize = fontsize)

    if not(filename is None):
        plt.savefig(filename, bbox_inches = 'tight')
        
    if return_ax:
        return(ax)


# In[52]:


Large_font_size = 12
Medium_font_size = 7
Small_font_size = 5
lw = 1
text_kwargs = dict(ha='center', va='center', color='C1')
cm_ = 1/2.54  # centimeters in inches 


# In[53]:


#get_ipython().run_line_magic('matplotlib', 'inline')
dirname = ""

balls_df = pd.read_csv(dirname + "sim_output/init_balls.csv")
springs_df = pd.read_csv(dirname + "sim_output/init_springs.csv")
[top_balls, top_springs] = extract_thin_mesh(balls_df, springs_df, which = "top", reindex = True)

figsize = (8*cm_, 5*cm_)
fig,ax = plt.subplots(figsize = figsize)
title = ""
ax = plot_shell(top_balls, top_springs, x = 'y', y = 'x', 
           #filename = dirname + 'sim_output/top_view_' + temp_stages_df.iloc[0]["stage_name"] + '.pdf', 
           color_max=1.5, color_min=0.5, cmap = "bwr",
           cbar_name = r'$l_{t_f}/l_{t_o}$', title = title, linewidth=lw/2,
                fig = fig, ax = ax, return_ax=True, fontsize = Medium_font_size,
               )

ax.axis("off")

os.makedirs("sim_output/plots/", exist_ok = True)
fig.savefig("sim_output/plots/top_view.pdf", bbox_inches = "tight")


# In[54]:


figsize = (3*cm_, 3*cm_)
fig,ax = plt.subplots(figsize = figsize)

devstage = "4hAPF"
crosssections = ["Along_DV", "Across_DV"]

for crosssection in crosssections:
    linestyle = "-" if crosssection == "Across_DV" else "--"
    smooth_curve = pd.read_csv(f"{dirname}sim_output/{devstage}_{crosssection}_curve.csv")
    ax.plot(smooth_curve['arclength_scaled'],smooth_curve['curvature_scaled'], 
            color = 'black',linewidth = lw, linestyle = linestyle,
            label = crosssection,
             )
    
ax.axhline(y = 0.0128, linewidth = lw/2, linestyle = '-', color = "gray", zorder = 0)
ax.set_ylim(-0.005, 0.05)
ax.set_xlim(-110, 110)
ax.set_xlabel("Arclength " + r"$(\mu m)$", fontsize = Medium_font_size)
ax.set_ylabel("Curvature " + r"$(\mu m^{-1})$", fontsize = Medium_font_size)
ax.tick_params(axis='both', which='major', labelsize=Medium_font_size, pad = 0)
_ = ax.set_yticks([0,0.02, 0.04])
_ = ax.set_xticks([-100,0,100])

os.makedirs("sim_output/plots/", exist_ok = True)
fig.savefig(f"sim_output/plots/curvature_{devstage}.pdf", bbox_inches = "tight")

