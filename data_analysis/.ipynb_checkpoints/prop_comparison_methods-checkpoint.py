
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

"""
def piecewise_linear_interpolate(xs_interpolated, xs, ys):
    
    #assuming that xs are sorted
    #we want to know first the index i of xs for which xs[i + 1] >= x > xs[i]
    
    indexes = np.searchsorted(a = xs, v = xs_interpolated, side = 'right')  #function tells you the index where the element v should go to maintain the order of a. ##side does not matter if y(x) is continuous 
    ys_interpolated = []
    
    if indexes.shape == 1:
        indexes = [indexes]
        
    for ctr in range(len(indexes)):
        
        i = indexes[ctr]
        x_interpolated = xs_interpolated[ctr]
        i = i-1
        
        if (i >= len(xs) - 1) or (i < 0):
            #special boundary cases
            if x_interpolated == xs[0]:
                y_interpolated = ys[0]
                ys_interpolated.append(y_interpolated)
                continue
            elif x_interpolated == xs[len(xs)-1]:
                y_interpolated = ys[len(xs)-1]
                ys_interpolated.append(y_interpolated)
                continue 
            else:
                #print('cannot interpolate outside the domain of x values')
                #print(x_interpolated)
                y_interpolated = None
                ys_interpolated.append(y_interpolated)
                continue
            
        slope = (ys[i + 1] - ys[i])/(xs[i + 1] - xs[i])
        y_interpolated = ys[i] + (x_interpolated - xs[i])*slope
        ys_interpolated.append(y_interpolated)
        
    if len(ys_interpolated) == 1:
        return(ys_interpolated[0])

    return(np.array(ys_interpolated))
"""

def piecewise_linear_interpolate(xs_interpolated, xs, ys, manner = 'linear', include_zero = False):
    
    #assuming that xs are sorted
    #we want to know first the index i of xs for which xs[i + 1] >= x > xs[i]
    
    if include_zero:
        xs = np.array(xs)
        ys = np.array(ys)
        xs = xs.insert(xs, 0, 0) #0 included at beginning
        ys = ys.insert(ys, 0, 0) #0 included at beginning
    
    indexes = np.searchsorted(a = xs, v = xs_interpolated, side = 'right')  #function tells you the index where the element v should go to maintain the order of a. ##side does not matter if y(x) is continuous 
    #print('indexes')
    #print(indexes)
    ys_interpolated = []
    
    if indexes.shape == 1:
        indexes = [indexes]
        
    for ctr in range(len(indexes)):
        
        i = indexes[ctr]
        x_interpolated = xs_interpolated[ctr]
        if manner == 'linear':
        	i = i-1
        if manner == 'step-backward':
        	i = i-1
        # if manner is step-forward then i remains i
        
        if (i >= len(xs) - 1) or (i < 0):
            #special boundary cases
            if x_interpolated == xs[0]:
                y_interpolated = ys[0]
                ys_interpolated.append(y_interpolated)
                continue
            elif ((manner == 'linear') or (manner == 'step-backward')) and (x_interpolated == xs[len(xs)-1]):
                y_interpolated = ys[len(xs)-1]
                ys_interpolated.append(y_interpolated)
                continue 
            elif (manner == 'step-forward') and (x_interpolated == xs[len(xs)-2]):
                y_interpolated = ys[len(xs)-2]
                ys_interpolated.append(y_interpolated)
                continue 
            elif (manner == 'step-forward') and (x_interpolated <= xs[len(xs)-1]):
                y_interpolated = ys[len(xs)-1]
                ys_interpolated.append(y_interpolated)
                continue 
            else:
                #print('cannot interpolate outside the domain of x values')
                #print(x_interpolated)
                y_interpolated = None
                ys_interpolated.append(y_interpolated)
                continue
            
        if manner == 'linear':
            slope = (ys[i + 1] - ys[i])/(xs[i + 1] - xs[i])
        elif 'step' in manner:
            slope = 0
            
        y_interpolated = ys[i] + (x_interpolated - xs[i])*slope
        ys_interpolated.append(y_interpolated)
        
    if len(ys_interpolated) == 1:
        return(ys_interpolated[0])

    return(np.array(ys_interpolated))


def analyze_ring_cell_numbers_per_roi(df, roi = 'outDV', devstages = ['upcrawling','whitePupa','2hAPF','4hAPF']):
    
    df_k_N_alldiscs = pd.DataFrame(columns = ['devstage', 'disc', 'k_dist', 'N_cum'], dtype=float)

    df_k_N_mean = pd.DataFrame(columns=['devstage', 'k_dist', 'N_cum_mean', 'N_cum_std'])

    for devstage in devstages:

        df_devstage = df[df['devstage'] == devstage]
        discs = np.unique(df_devstage['discName'])
        #print(devstage + ' roi')
        #print(np.unique(df_devstage['roi']))
        #print('number of discs ' + str(len(discs)))

        for disc in discs:

            df_disc_devstage = df[(df['devstage'] == devstage) & (df['discName'] == disc)]

            #setting an offset in k_dists
            df_disc_devstage.loc[:,'k_dist'] = df_disc_devstage['k_dist'].values - min(df_disc_devstage['k_dist'])

            k_dists = np.sort(np.unique(df_disc_devstage['k_dist']))

            N = 0

            for k_dist in k_dists:

                if roi == 'outDV':
                    if len(df_disc_devstage.loc[(df_disc_devstage['roi'] == 'ventral') | (df_disc_devstage['roi'] == 'dorsal')]) == 0:
                        #print('no dorsal or ventral')
                        #print(k_dist)
                        break

                    cumcounts = np.unique(df_disc_devstage.loc[(df_disc_devstage['k_dist'] == k_dist) & ((df_disc_devstage['roi'] == 'ventral') | (df_disc_devstage['roi'] == 'dorsal')),'cumcount'])


                if roi == 'DV':
                    if len(df_disc_devstage.loc[(df_disc_devstage['roi'] == 'DV')]) == 0:
                        #print('no DV boundary')
                        #print(k_dist)
                        break

                    cumcounts = np.unique(df_disc_devstage.loc[(df_disc_devstage['k_dist'] == k_dist) & ((df_disc_devstage['roi'] == 'DV')),'cumcount'])

                cumcount= np.mean(cumcounts)

                row = pd.DataFrame([[devstage, disc, k_dist, cumcount]],
                                   columns = df_k_N_alldiscs.columns
                                  )
                df_k_N_alldiscs=pd.concat([df_k_N_alldiscs,row])

    df_k_N_alldiscs = df_k_N_alldiscs.reset_index(drop=True)
    df_k_N_alldiscs['roi'] = roi
    
    df_k_N_mean = df_k_N_alldiscs.groupby(['devstage', 'k_dist']).mean()
    df_k_N_mean.columns = ['N_cum_mean']
    df_k_N_std = df_k_N_alldiscs.groupby(['devstage', 'k_dist']).std()
    df_k_N_std.columns = ['N_cum_std']
    df_k_N_mean = pd.concat([df_k_N_mean,df_k_N_std ], axis = 1)
    df_k_N_mean = df_k_N_mean.reset_index()
    df_k_N_mean['roi'] = roi
    
    return([df_k_N_alldiscs, df_k_N_mean])

def analyze_ring_cell_numbers(df, devstages = ['upcrawling','whitePupa','2hAPF','4hAPF']):
    
    [outDV_k_N_alldiscs, outDV_k_N_mean] = analyze_ring_cell_numbers_per_roi(df, roi = 'outDV', devstages = devstages)
    [DV_k_N_alldiscs, DV_k_N_mean] = analyze_ring_cell_numbers_per_roi(df, roi = 'DV', devstages = devstages)
    
    k_N_alldiscs = pd.concat([outDV_k_N_alldiscs, DV_k_N_alldiscs]).reset_index(drop=True)
    k_N_mean = pd.concat([outDV_k_N_mean, DV_k_N_mean]).reset_index(drop=True)
    
    return([k_N_alldiscs, k_N_mean])

def get_k_differences_per_roi_per_stagePair(k_N_alldiscs, devstages = ['upcrawling', '4hAPF'], roi = 'outDV', fit_deg = 1, fit_param = 'k_beta', N_ref_pathlength_dict = None):
    
    roi_k_N_alldiscs = k_N_alldiscs[k_N_alldiscs['roi'] == roi]
    
    discs_alpha = np.unique(roi_k_N_alldiscs[roi_k_N_alldiscs['devstage'] == devstages[1]]['disc']) #discs for t+Delta_t stage
    discs_beta = np.unique(roi_k_N_alldiscs[roi_k_N_alldiscs['devstage'] == devstages[0]]['disc'])  #discs for t stage

    f_alpha_beta_df = pd.DataFrame(columns=['disc_combination','disc_alpha', 'disc_beta', 'N_beta', 'f_alpha_beta'])

    for alpha in range(len(discs_alpha)):

        disc_alpha = discs_alpha[alpha]
        k_alpha_df = roi_k_N_alldiscs[roi_k_N_alldiscs['disc'] == disc_alpha]

        for beta in range(len(discs_beta)):

            disc_beta = discs_beta[beta]
            k_beta_df = roi_k_N_alldiscs[roi_k_N_alldiscs['disc'] == disc_beta]
            k_beta_df = k_beta_df[k_beta_df['N_cum']<= max(k_alpha_df['N_cum'])]
            N_beta = k_beta_df['N_cum'].values
            k_beta = k_beta_df['k_dist'].values
            k_alpha_interpolated = piecewise_linear_interpolate(N_beta, xs = k_alpha_df['N_cum'].values, ys = k_alpha_df['k_dist'].values)
            non_None_indexes = np.invert(np.isnan(k_alpha_interpolated.astype(float)))
            k_diffs = k_alpha_interpolated[non_None_indexes] - k_beta[non_None_indexes]
            #ax.plot(N_beta[non_None_indexes], k_diffs, color = 'gray', alpha = 0.2)
            #ax.scatter(N_beta[non_None_indexes], f_alpha_beta, color = 'gray', alpha = 0.2)

            f_alpha_beta = pd.DataFrame({'disc_combination':[disc_alpha + disc_beta]*len(k_diffs),
                                         'disc_alpha':[disc_alpha]*len(k_diffs),
                                         'disc_beta':[disc_beta]*len(k_diffs),
                                         'N_beta':N_beta[non_None_indexes],
                                         'k_diff':k_diffs,
                                         'k_beta':k_beta[non_None_indexes], #np.arange(len(k_diffs)) + 1,
                                        })
            f_alpha_beta['k_diff'] = f_alpha_beta['k_diff'].astype(float)

            f_alpha_beta_df = pd.concat([f_alpha_beta_df,f_alpha_beta]).reset_index(drop=True)

    N_beta_stat = f_alpha_beta_df.groupby('k_beta')['N_beta'].describe()[['mean', 'std']].reset_index().rename(columns = {'mean':'N_beta_mean','std':'N_beta_std'})
    k_diff_stat = f_alpha_beta_df.groupby('k_beta')['k_diff'].describe()[['mean', 'std']].reset_index().rename(columns = {'mean':'k_diff_mean','std':'k_diff_std'})
    stat_df = pd.merge(N_beta_stat, k_diff_stat)

    f_alpha_beta_df['DDk/DNDt'] = 0
    f_alpha_beta_df['DDk/Dt'] = 0
    f_alpha_beta_df['lambda_rearrangement'] = 0

    for i in range(len(f_alpha_beta_df)):

        if i == len(f_alpha_beta_df)-1:
            f_alpha_beta_df.loc[i,'DDk/DNDt'] = None
            break

        if not( (f_alpha_beta_df.loc[i,'disc_alpha'] == f_alpha_beta_df.loc[i+1,'disc_alpha']) and (f_alpha_beta_df.loc[i,'disc_beta'] == f_alpha_beta_df.loc[i+1,'disc_beta']) ):
            #this is the last row of this combination of alpha and beta
            f_alpha_beta_df.loc[i,'DDk/DNDt'] = None
            f_alpha_beta_df.loc[i,'DDk/Dt'] = None
            f_alpha_beta_df.loc[i,'lambda_rearrangement'] = None
            continue

        f_alpha_beta_df.loc[i, 'DDk/DNDt'] = (f_alpha_beta_df.loc[i+1, 'k_diff'] - f_alpha_beta_df.loc[i, 'k_diff'])/(f_alpha_beta_df.loc[i+1, 'N_beta'] - f_alpha_beta_df.loc[i, 'N_beta'])
        f_alpha_beta_df.loc[i, 'DDk/Dt'] = (f_alpha_beta_df.loc[i+1, 'k_diff'] - f_alpha_beta_df.loc[i, 'k_diff'])
        f_alpha_beta_df.loc[i, 'lambda_rearrangement'] = 1 + f_alpha_beta_df.loc[i, 'DDk/Dt']

    #N_beta_stat = f_alpha_beta_df.groupby('k_beta')['N_beta'].describe()[['mean', 'std']].reset_index().rename(columns = {'mean':'N_beta_mean','std':'N_beta_std'})
    DDk_DNDt_stat = f_alpha_beta_df.groupby('k_beta')['DDk/DNDt'].describe()[['mean', 'std']].reset_index().rename(columns = {'mean':'DDk/DNDt_mean','std':'DDk/DNDt_std'})
    stat_df = pd.merge(stat_df, DDk_DNDt_stat)

    DDk_Dt_stat = f_alpha_beta_df.groupby('k_beta')['DDk/Dt'].describe()[['mean', 'std']].reset_index().rename(columns = {'mean':'DDk/Dt_mean','std':'DDk/Dt_std'})
    stat_df = pd.merge(stat_df, DDk_Dt_stat)
    
    lambda_rearrangement_stat = f_alpha_beta_df.groupby('k_beta')['lambda_rearrangement'].describe()[['mean', 'std']].reset_index().rename(columns = {'mean':'lambda_rearrangement_mean','std':'lambda_rearrangement_std'})
    stat_df = pd.merge(stat_df, lambda_rearrangement_stat)

    f_alpha_beta_df['disc_combination'] = [f_alpha_beta_df.loc[i,'disc_alpha'] + f_alpha_beta_df.loc[i,'disc_beta'] for i in range(len(f_alpha_beta_df))]
    
    f_alpha_beta_df['roi'] = roi
    stat_df['roi'] = roi
    
    f_alpha_beta_df['devstage_init'] = devstages[0]
    stat_df['devstage_init'] = devstages[0]
    
    f_alpha_beta_df['devstage_final'] = devstages[1]
    stat_df['devstage_final'] = devstages[1]

    #excluding boundary values
    stat_df = stat_df.iloc[1:-1,:]

    #adding ref pathlength scaled
    f_alpha_beta_df["ref_pathlength_scaled_beta"] = f_alpha_beta_df.apply(lambda row: N_ref_pathlength_dict[row['roi']](row["N_beta"]),axis=1)
    stat_df["ref_pathlength_scaled_beta_mean"] = stat_df.apply(lambda row: N_ref_pathlength_dict[row['roi']](row["N_beta_mean"]),axis=1)

    #fit lambda_rearrangement
    #get fit line
    coeffs = np.polyfit(stat_df[fit_param], stat_df['lambda_rearrangement_mean'], #excluding boundary values
        w = 1/stat_df['lambda_rearrangement_std'], #not adding weight beacause of outliears at max values of k
        deg = fit_deg)
    poly_obj = np.poly1d(coeffs)
    stat_df['fit_lambda_rearrangement'] = poly_obj(stat_df[fit_param])
    stat_df['fit_lambda_rearrangement_coeffs'] = [coeffs]*len(stat_df)

    return([f_alpha_beta_df, stat_df])

def get_k_differences(k_N_alldiscs, devstage_combinations = None, devstages = ['upcrawling','whitePupa','2hAPF','4hAPF'], rois = ['outDV', 'DV'], fit_param = 'k_beta', N_ref_pathlength_dict = None):

    if devstage_combinations is None:
        devstage_combinations = pd.DataFrame({'devstage_init':devstages[0:len(devstages)-1], 'devstage_final':devstages[1:]})
        
    k_diff = pd.DataFrame()
    k_diff_stat = pd.DataFrame()
    
    for i in range(len(devstage_combinations)):

        devstage_init = devstage_combinations.loc[i,'devstage_init']
        devstage_final = devstage_combinations.loc[i,'devstage_final']

        for roi in rois:
            
            [k_diff_stage_roi, k_diff_stat_stage_roi] =  get_k_differences_per_roi_per_stagePair(k_N_alldiscs, devstages = [devstage_init, devstage_final], roi = roi, fit_param = fit_param, N_ref_pathlength_dict = N_ref_pathlength_dict)
            k_diff = pd.concat([k_diff, k_diff_stage_roi]).reset_index(drop = True)
            k_diff_stat = pd.concat([k_diff_stat, k_diff_stat_stage_roi]).reset_index(drop = True)
            
    return([k_diff, k_diff_stat])

def get_subplot(ax, df = None, stat_df = None, query_str = '', df_cases_col = 'disc_combination',
                x_col = 'N_beta', y_col = 'k_diff', 
                x_col_mean = None, y_col_mean = None, x_col_std = None, y_col_std = None, 
                xlabel = 'N (cumulative)', ylabel = r'$\Delta_{N}\Delta_{T}k$', title = 'Stage1 to Stage2',
                individual_color = 'gray', mean_color = 'red', individual_linewidth = 1, mean_linewidth = 1.5,
                mean_linestyle = '-', ylabelpad = 50, mean_label = '', mean_alpha = 0.2,
                xticks = None, yticks = None, xlim = None, ylim = None,
                plot_type = 'linear', error_style = 'fill_between',
               ):
    
    #df contains data for every individual plot
    #df_cases_col is the column name used to filter data for single curves
    #x_col, y_col are the columns to be plotted in df
    #individual_color is the color of the line for individual plots
    
    #stat_df contains mean, std for each category
    #x_col_mean, y_col_mean, x_col_std, y_col_std are the names of the columns in stat_df
    #mean_color is the color of the line for mean plots
    
    #query_str is used to filter the data
    
    if df is not None:
        
        f_alpha_beta_df = df.query(query_str) #[(k_diff['devstage_init'] == devstage_init) & (k_diff['devstage_final'] == devstage_final) & (k_diff['roi'] == roi)]


        disc_combinations = np.unique(f_alpha_beta_df[df_cases_col].values)
        #fig, ax = plt.subplots()

        for disc_combination in disc_combinations:
            df = f_alpha_beta_df[f_alpha_beta_df[df_cases_col] == disc_combination]
            #ax.scatter(df['N_beta'], df['DDk/DNDt'])
            if plot_type == 'linear':
                ax.plot(df[x_col], df[y_col], color = individual_color, lw = individual_linewidth, alpha = 0.2)
            elif plot_type == 'step':
                ax.step(df[x_col], df[y_col], where = 'pre', color = individual_color, lw = individual_linewidth, alpha = 0.2)

        
    if stat_df is not None:
        stat_df = stat_df.query(query_str)

        if x_col_mean is None:
            #first we try to add mean in front of the name
            x_col_mean = x_col + '_mean'
            if x_col_mean not in stat_df.columns:
                #next we try the name without any addition
                x_col_mean = x_col
        if y_col_mean is None:
            y_col_mean = y_col + '_mean'
            if y_col_mean not in stat_df.columns:
                #next we try the name without any addition
                y_col_mean = y_col

        if x_col_std is None:
            x_col_std = x_col + '_std'
            if x_col_std in stat_df.columns:
                xerr = stat_df[x_col_std]
            else:
                xerr = 0
        if y_col_std is None:
            y_col_std = y_col + '_std'
            if y_col_std in stat_df.columns:
                yerr = stat_df[y_col_std]
            else:
                yerr = 0

        #ax.plot(stat_df[x_col_mean], stat_df[y_col_mean])
        if error_style == 'fill_between':
            ax.fill_between(stat_df[x_col_mean], stat_df[y_col_mean] - yerr, stat_df[y_col_mean] + yerr, color = mean_color, alpha = mean_alpha)            
        elif error_style == 'errorbar':
            ax.errorbar(x = stat_df[x_col_mean], y = stat_df[y_col_mean], xerr = xerr, yerr = yerr, color = mean_color, lw = mean_linewidth, label = mean_label)
        ax.plot(stat_df[x_col_mean], stat_df[y_col_mean], color = mean_color, lw = mean_linewidth, label = mean_label, linestyle = mean_linestyle)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.set_xlabel(xlabel, fontsize = 20)
    ax.set_ylabel(ylabel, fontsize = 20, rotation = 'horizontal', labelpad = ylabelpad)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_title(title, fontsize = 20, pad = 25)
    ax.grid()
    
    return(ax)    

def get_piecewise_average(prop_alpha, N_alpha, N_beta):
    
    #might break if piecewise_linear_interpolate returns None
    
    N_start = 0 #here we start from zero 
    prop_alpha_piecewise_averaged = np.array([])
    
    for i in range(len(N_beta)):
        
        N_end = N_beta[i]
        
        #find all N_alphas between N_start and N_end
        #append N_start and N_end
        #remove duplicates and remove N > max(N_alpha) 
        #sort
        N_alpha_mids = N_alpha[np.where((N_alpha > N_start) & (N_alpha<N_end)) ]
        N_alpha_mids = np.unique(np.append(N_alpha_mids, [N_start, N_end]))
        N_alpha_mids = N_alpha_mids[np.where(N_alpha_mids <= max(N_alpha))]
        N_alpha_mids = np.sort(N_alpha_mids)
        
        prop_alpha_mids = piecewise_linear_interpolate(N_alpha_mids[1:], xs = N_alpha, ys = prop_alpha, manner = 'step-forward')
        prop_N_weights = N_alpha_mids[1:] - N_alpha_mids[:-1]
        
        sum_piecewise = np.sum(prop_alpha_mids*prop_N_weights)
        #except:
        #    print(prop_alpha_mids)
        #    print('####')
        #    print('N_alpha_mids')
        #    print(N_alpha_mids)
        #    print('xs')
        #    print(N_alpha)
        #    print('ys')
        #    print(prop_alpha)
        try:
            avg_piecewise = sum_piecewise/(max(N_alpha_mids) - min(N_alpha_mids))
        except:
            print('prop_alpha')
            print(prop_alpha)
            print('N_start')
            print(N_start)
            print('N_end')
            print(N_end)
            print('N_alpha')
            print(N_alpha)
            
            avg_piecewise = sum_piecewise/(max(N_alpha_mids) - min(N_alpha_mids))
        
        prop_alpha_piecewise_averaged = np.append(prop_alpha_piecewise_averaged, avg_piecewise)
        
        #for next iteration
        N_start = N_end
    
    return(prop_alpha_piecewise_averaged)

def get_prop_differences_per_roi_per_stagePair(df, prop = 'area', operation = 'subtract', devstages = ['upcrawling', '4hAPF'], roi = 'outDV', fit_deg = 1, fit_param = 'k_beta'):
    
    #alpha corresponds to index for T+ Delta T stage
    #beta corresponds to index for T stage
    
    roi_df = df[df['region'] == roi]
    
    discs_alpha = np.unique(roi_df[roi_df['devstage'] == devstages[1]]['discName']) #discs for t+Delta_t stage
    discs_beta = np.unique(roi_df[roi_df['devstage'] == devstages[0]]['discName'])  #discs for t stage

    f_alpha_beta_df = pd.DataFrame()

    for alpha in range(len(discs_alpha)):

        disc_alpha = discs_alpha[alpha]
        alpha_df = roi_df[roi_df['discName'] == disc_alpha]
        N_alpha = np.array(alpha_df['cumcount'].values)
        prop_alpha = np.array(alpha_df[prop].values)

        for beta in range(len(discs_beta)):

            disc_beta = discs_beta[beta]
            beta_df = roi_df[roi_df['discName'] == disc_beta]
            beta_df = beta_df[beta_df['cumcount'] <= max(N_alpha)]
            N_beta = np.array(beta_df['cumcount'].values)
            prop_beta = np.array(beta_df[prop].values)
            k_beta = np.array(beta_df['k_dist'].values)
            ref_pathlength_scaled_beta = np.array(beta_df['ref_pathlength_scaled'].values)
            
            #here we take average of prop_alpha in steps of N_beta
            prop_alpha_piecewise_averaged = get_piecewise_average(prop_alpha, N_alpha, N_beta)

            #removing any values which may not have been interpolated or averaged
            #non_None_indexes = np.invert(np.isnan(prop_alpha_piecewise_averaged.astype(float)))
            #prop_beta = prop_beta[non_None_indexes]
            #prop_alpha_piecewise_averaged = prop_alpha_piecewise_averaged[non_None_indexes]
            #N_beta = N_beta[non_None_indexes]
            #k_beta = k_beta[non_None_indexes]
            
            try:
                if operation == 'subtract':
                    prop_diffs = prop_alpha_piecewise_averaged - prop_beta
                elif operation == 'divide':
                    prop_diffs = prop_alpha_piecewise_averaged/prop_beta
                elif operation == 'divide-sqrt':
                    prop_diffs = np.sqrt(prop_alpha_piecewise_averaged/prop_beta)
            except:
                print('N_beta')
                print(N_beta)
                print('N_alpha')
                print(N_alpha)
                print('prop_alpha')
                print(prop_alpha)
                print('prop_alpha_piecewise')
                print(prop_alpha_piecewise_averaged)
                print('prop_beta')
                print(prop_beta)

            f_alpha_beta = pd.DataFrame({'disc_combination':[disc_alpha + disc_beta]*len(prop_diffs),
                                         'disc_alpha':[disc_alpha]*len(prop_diffs),
                                         'disc_beta':[disc_beta]*len(prop_diffs),
                                         'N_beta':N_beta,
                                         'k_beta':k_beta,
                                         'ref_pathlength_scaled_beta':ref_pathlength_scaled_beta,
                                         prop+'_diff':prop_diffs,
                                         prop+'_beta':prop_beta,
                                        })
            
            #f_alpha_beta['prop_diff'] = f_alpha_beta['prop_diff'].astype(float)

            f_alpha_beta_df = pd.concat([f_alpha_beta_df,f_alpha_beta]).reset_index(drop=True)

    stat_df = f_alpha_beta_df.groupby('k_beta').agg(['mean', 'std']).reset_index()
    colnames = [x[0]+'_'+x[1] if x[0] != 'k_beta' else x[0] for x in stat_df.columns]
    stat_df.columns = colnames #removing multi-indexing

    f_alpha_beta_df['roi'] = roi
    stat_df['roi'] = roi
    
    f_alpha_beta_df['devstage_init'] = devstages[0]
    stat_df['devstage_init'] = devstages[0]
    
    f_alpha_beta_df['devstage_final'] = devstages[1]
    stat_df['devstage_final'] = devstages[1]

    #######
    #get fit line
    coeffs = np.polyfit(stat_df[fit_param], stat_df[prop + '_diff_mean'], w = 1/stat_df[prop + '_diff_std'], deg = fit_deg)
    poly_obj = np.poly1d(coeffs)
    stat_df['fit_'+prop+'_diff'] = poly_obj(stat_df[fit_param])
    stat_df['fit_'+prop+'_coeffs'] = [coeffs]*len(stat_df)


    return([f_alpha_beta_df, stat_df])

def get_prop_differences(df, prop = 'area', operation = 'subtract', devstage_combinations = None, devstages = ['upcrawling','whitePupa','2hAPF','4hAPF'], rois = ['outDV', 'DV'], fit_deg = 1, fit_param = 'k_beta'):

    if devstage_combinations is None:
        devstage_combinations = pd.DataFrame({'devstage_init':devstages[0:len(devstages)-1], 'devstage_final':devstages[1:]})
        
    
    prop_diff = pd.DataFrame()
    prop_diff_stat = pd.DataFrame()
    
    for i in range(len(devstage_combinations)):

        devstage_init = devstage_combinations.loc[i,'devstage_init']
        devstage_final = devstage_combinations.loc[i,'devstage_final']

        for roi in rois:
            
            [prop_diff_stage_roi, prop_diff_stat_stage_roi] =  get_prop_differences_per_roi_per_stagePair(df, prop = prop, operation = operation, devstages = [devstage_init, devstage_final], roi = roi, fit_deg = fit_deg, fit_param = fit_param)
            prop_diff = pd.concat([prop_diff, prop_diff_stage_roi]).reset_index(drop = True)
            prop_diff_stat = pd.concat([prop_diff_stat, prop_diff_stat_stage_roi]).reset_index(drop = True)
            
    return([prop_diff, prop_diff_stat])

def get_prop_diff_vs_dist_per_roi_per_stagePair(df, prop = 'area', operation = 'subtract', devstages = ['upcrawling', '4hAPF'], roi = 'outDV', fit_deg = 1, fit_param = 'dist_beta'):
    
    #alpha corresponds to index for T+ Delta T stage
    #beta corresponds to index for T stage
    
    roi_df = df[df['region'] == roi]
    
    discs_alpha = np.unique(roi_df[roi_df['devstage'] == devstages[1]]['discName']) #discs for t+Delta_t stage
    discs_beta = np.unique(roi_df[roi_df['devstage'] == devstages[0]]['discName'])  #discs for t stage

    f_alpha_beta_df = pd.DataFrame()
    
    dists_orig = np.linspace(0,1,100)

    for alpha in range(len(discs_alpha)):

        disc_alpha = discs_alpha[alpha]
        alpha_df = roi_df[roi_df['discName'] == disc_alpha]
        alpha_df = alpha_df.sort_values(by = ['distanceFraction'])
        dist_alpha = np.array(alpha_df['distanceFraction'].values)
        #N_alpha = np.array(alpha_df['cumcount'].values)
        prop_alpha = np.array(alpha_df[prop].values)

        for beta in range(len(discs_beta)):

            disc_beta = discs_beta[beta]
            beta_df = roi_df[roi_df['discName'] == disc_beta]
            beta_df = beta_df.sort_values(by = ['distanceFraction'])
            #beta_df = beta_df[(beta_df['distanceFraction'] >= min(dist_alpha)) & (beta_df['distanceFraction'] <= max(dist_alpha))]
            dist_beta = np.array(beta_df['distanceFraction'].values)
            prop_beta = np.array(beta_df[prop].values)
            #k_beta = np.array(beta_df['k_dist'].values)
            #ref_pathlength_scaled_beta = np.array(beta_df['ref_pathlength_scaled'].values)
            
            #here we take average of prop_alpha in steps of N_beta
            #prop_alpha_piecewise_averaged = get_piecewise_average(prop_alpha, N_alpha, N_beta)
            [min_lim, max_lim] = [max(min(dist_alpha), min(dist_beta)), min(max(dist_alpha),max(dist_beta))]
            dists = dists_orig[(dists_orig > min_lim) & (dists_orig < max_lim)]
            prop_alpha_interpolated = piecewise_linear_interpolate(dists, dist_alpha, prop_alpha, manner = "linear")
            prop_beta_interpolated = piecewise_linear_interpolate(dists, dist_beta, prop_beta, manner = "linear")

            #removing any values which may not have been interpolated or averaged
            #non_None_indexes = np.invert(np.isnan(prop_alpha_piecewise_averaged.astype(float)))
            #prop_beta = prop_beta[non_None_indexes]
            #prop_alpha_piecewise_averaged = prop_alpha_piecewise_averaged[non_None_indexes]
            #N_beta = N_beta[non_None_indexes]
            #k_beta = k_beta[non_None_indexes]
            
            if operation == 'subtract':
                prop_diffs = prop_alpha_interpolated - prop_beta_interpolated
            elif operation == 'divide':
                prop_diffs = prop_alpha_interpolated/prop_beta_interpolated
            elif operation == 'divide-sqrt':
                prop_diffs = np.sqrt(prop_alpha_interpolated/prop_beta_interpolated)

            f_alpha_beta = pd.DataFrame({'disc_combination':[disc_alpha + disc_beta]*len(prop_diffs),
                                         'disc_alpha':[disc_alpha]*len(prop_diffs),
                                         'disc_beta':[disc_beta]*len(prop_diffs),
                                         'dist_beta':dists,
                                         #'k_beta':k_beta,
                                         #'ref_pathlength_scaled_beta':ref_pathlength_scaled_beta,
                                         prop+'_diff':prop_diffs,
                                         prop+'_beta':prop_beta_interpolated,
                                        })
            
            #f_alpha_beta['prop_diff'] = f_alpha_beta['prop_diff'].astype(float)

            f_alpha_beta_df = pd.concat([f_alpha_beta_df,f_alpha_beta]).reset_index(drop=True)

    stat_df = f_alpha_beta_df.groupby('dist_beta').agg(['mean', 'std']).reset_index()
    colnames = [x[0]+'_'+x[1] if x[0] != 'dist_beta' else x[0] for x in stat_df.columns]
    stat_df.columns = colnames #removing multi-indexing

    f_alpha_beta_df['roi'] = roi
    stat_df['roi'] = roi
    
    f_alpha_beta_df['devstage_init'] = devstages[0]
    stat_df['devstage_init'] = devstages[0]
    
    f_alpha_beta_df['devstage_final'] = devstages[1]
    stat_df['devstage_final'] = devstages[1]

    #######
    #get fit line
    coeffs = np.polyfit(stat_df[fit_param], stat_df[prop + '_diff_mean'], 
                        #w = 1/stat_df[prop + '_diff_std'], 
                        deg = fit_deg)
    poly_obj = np.poly1d(coeffs)
    stat_df['fit_'+prop+'_diff'] = poly_obj(stat_df[fit_param])
    stat_df['fit_'+prop+'_coeffs'] = [coeffs]*len(stat_df)


    return([f_alpha_beta_df, stat_df])

def get_prop_diff_vs_dist(df, prop = 'area', operation = 'subtract', devstage_combinations = None, devstages = ['upcrawling','whitePupa','2hAPF','4hAPF'], rois = ['outDV', 'DV'], fit_deg = 1, fit_param = 'k_beta'):

    if devstage_combinations is None:
        devstage_combinations = pd.DataFrame({'devstage_init':devstages[0:len(devstages)-1], 'devstage_final':devstages[1:]})
        
    
    prop_diff = pd.DataFrame()
    prop_diff_stat = pd.DataFrame()
    
    for i in range(len(devstage_combinations)):

        devstage_init = devstage_combinations.loc[i,'devstage_init']
        devstage_final = devstage_combinations.loc[i,'devstage_final']

        for roi in rois:
            
            [prop_diff_stage_roi, prop_diff_stat_stage_roi] =  get_prop_diff_vs_dist_per_roi_per_stagePair(df, prop = prop, operation = operation, devstages = [devstage_init, devstage_final], roi = roi, fit_deg = fit_deg)
            prop_diff = pd.concat([prop_diff, prop_diff_stage_roi]).reset_index(drop = True)
            prop_diff_stat = pd.concat([prop_diff_stat, prop_diff_stat_stage_roi]).reset_index(drop = True)
            
    return([prop_diff, prop_diff_stat])






    