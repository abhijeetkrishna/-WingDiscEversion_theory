#this code computes the mean cross-sectional shapes and curvature along the arclenght of 2D crosssections


#importing packages 
from matplotlib.lines import Line2D
import glob
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#pd.set_option('display.max_rows', None)
from scipy.interpolate import CubicSpline
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')
from scipy.interpolate import BSpline
from scipy.interpolate import splrep
from scipy.interpolate import splev

#this code computes the mean cross-sectional shapes and curvature along the arclenght of 2D crosssections


#importing packages 
from matplotlib.lines import Line2D
import glob
import os
import warnings
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
#pd.set_option('display.max_rows', None)
from scipy.interpolate import CubicSpline
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')
from scipy.interpolate import BSpline
from scipy.interpolate import splrep
from scipy.interpolate import splev



def plot_montage(allcurves, genotypes = None, devstages = None, filename = None, orientation = 'horizontal', debug = False):
    
    if genotypes is None:
        genotypes = ['ecadGFPnbG4', 'ecadGFPnbG4myoVI']
        #genotypes = ['ecadGFPnbG4']
    if devstages is None:
        devstages = np.unique(allcurves['devstage'])
        #devstages = [ 'upcrawling','whitePupa', '2hAPF','4hAPF','6hAPF']
        #devstages = ['4hAPF']
    
    if orientation == 'horizontal':
        fig, axs = plt.subplots(len(genotypes), len(devstages), figsize=(7*len(devstages), 7*len(genotypes)))
    elif orientation == 'vertical':
        fig, axs = plt.subplots(len(devstages), len(genotypes), figsize=(7*len(genotypes), 7*len(devstages)))


    #fig = plt.figure(constrained_layout=True)
    #ax_array = fig.subplots(2, 2, squeeze=False)

    #ax_array[0, 0].bar(['a', 'b', 'c'], [5, 7, 9])
    #ax_array[0, 1].plot([1, 2, 3])
    #ax_array[1, 0].hist(hist_data, bins='auto')
    #ax_array[1, 1].imshow([[1, 2], [2, 1]])



    for i in range(len(genotypes)):

        genotype = genotypes[i]

        for j in range(len(devstages)):

            devstage = devstages[j]

            gen_dev = allcurves[(allcurves['genotype'] == genotype) & (allcurves['devstage'] == devstage)]

            disc_names = np.unique(gen_dev['disc'])

            #print(disc_names)
            #print(np.unique(gen_dev['devstage']))
            #print(np.unique(gen_dev['genotype']))

            if len(genotypes) == 1:
            	ax = axs[j] #not tested for vertical
            else:
            	ax = axs[i,j]

            ax.set_aspect('equal')
            ax.grid(False)
            #axs[i,j].set_xlim(-250, 250)
            #axs[i,j].set_ylim(-450, 50)
            if orientation == 'horizontal':
                ax.set_title(devstage, fontsize = 30)
            else:
                ax.set_ylabel(devstage, fontsize = 30)

            if j == 0:
                if i == 0:
                    if orientation == 'horizontal':
                        ax.set_ylabel('WT', fontsize = 30)
                    else:
                        ax.set_title('WT', fontsize = 30)
                else:
                    ax.set_ylabel('MyoVI', fontsize = 30)

            for disc_name in disc_names:

                #if disc_name in ['20200703_nubG4myoVI_upcrawling_disc2', '20200717_nubG4myoVI_upcrawling_disc2', '20201022_ecadGFPnbG4_2hAPF_disc3']:
                #    continue

                disc_curve = gen_dev[gen_dev['disc'] == disc_name]

                ax.scatter(disc_curve['x'], disc_curve['y'])
                #axs[i,j].plot(disc_curve['x'], disc_curve['y'])
                if debug:
                    [avg_x, avg_y] = [np.mean(disc_curve['x']),np.mean(disc_curve['y']) ]
                    ax.plot([0,np.mean(disc_curve['x'])],[0,np.mean(disc_curve['y'])], color = 'gray',)
                    ax.scatter([0,avg_x],[0,avg_y], color = 'black',)

                    arclength_threshold = min(np.abs(min(disc_curve['arclength_offset'])), max(disc_curve['arclength_offset']))
                    cropped_curve = disc_curve[(disc_curve['arclength_offset'] > -arclength_threshold) & (disc_curve['arclength_offset'] < arclength_threshold)]
                    cropped_curve_dorsal = cropped_curve[cropped_curve['region'] == 'dorsal']
                    [avg_x_dorsal, avg_y_dorsal] = [np.mean(cropped_curve_dorsal['x']), np.mean(cropped_curve_dorsal['y'])]
                    cropped_curve_ventral = cropped_curve[cropped_curve['region'] == 'ventral']
                    [avg_x_ventral, avg_y_ventral] = [np.mean(cropped_curve_ventral['x']), np.mean(cropped_curve_ventral['y'])]
                    [avg_x, avg_y] = [np.mean([avg_x_dorsal, avg_x_ventral]), np.mean([avg_y_dorsal, avg_y_ventral])]


                    ax.plot([0,avg_x],[0,avg_y], color = 'blue',)
                    ax.scatter([0,avg_x],[0,avg_y], color = 'red',)
                
        if not(filename is None):
            plt.savefig(filename, bbox_inches = 'tight')


def import_raw_crosssection_points(files = [],plot = False, crosssection = None, scale_file = "Pixelscales.pkl"):

    #this function imports the data from a pickle file 
    #checks the orientation of the curves
    #measures the arclength of the curves

    #files : an array of pickle files 
    #crosssection : 'Along_DV' for cases where there are no assigned DV boundary location,
    #				'Across_DV' when the curves are divided into 'dorsal' and 'ventral' region
    #				if None then code figures out the crosssection based on whether dorsal or ventral are present in data
    #plot : True if you want to print the plot of each curve, their x and y values as a function of arclength

    allcurves = pd.DataFrame()

    for file in files:

        #print('crosssection')
        #print(crosssection)

        print('file : ' + file)
        if not(os.path.exists(file)):
            print('***file does not exist***')

        if 'csv' in file:
            print('file is csv')
            df_all = pd.read_csv(file)
        else:
            df_all = pd.read_pickle(file)
        newnames = [i.replace('nubG4myoVI','ecadGFPnbG4myoVI') for i in df_all['disc']]
        df_all['disc'] = newnames

        with open(scale_file, "rb") as f:
            scalesDF = pickle.load(f)
        df_all = pd.merge(df_all, scalesDF)
        df_all['pixel to micron'] = [np.float(i) for i in df_all['pixel to micron']]

        df_all['x'] = df_all['x']*df_all['pixel to micron']
        df_all['y'] = df_all['y']*df_all['pixel to micron']

        #df_all = df_all.drop_duplicates()
        disc_names = np.unique(df_all['disc'])

        if crosssection is None:
            if len(np.unique(df_all['region'])) == 1:
                print('crosssection is Along DV')
                crosssection = 'Along_DV'
            elif 'dorsal' in df_all['region'].values and 'ventral' in df_all['region'].values:
                print('crosssection is Across DV')
                crosssection = 'Across_DV'
            else:
                print("Can't figure out which crosssection")

        for disc_name in disc_names:

            #print(disc_name)
            #disc_name = '20200703_nubG4myoVI_2hAPF_disc2'

            curve = df_all[df_all['disc'] == disc_name].drop_duplicates()

            ########
            # Make sure that Dorsal starts where ventral begins
            if crosssection == 'Across_DV':

                curve_dorsal = curve[curve['region'] == 'dorsal'].drop_duplicates()
                curve_ventral = curve[curve['region'] == 'ventral'].drop_duplicates()

                #Checking if the orientation of the dorsal and ventral curves are both anticlockwise
                dist1 = (curve_ventral['x'].iloc[len(curve_ventral) - 1] - curve_dorsal['x'].iloc[0])**2 + (curve_ventral['y'].iloc[len(curve_ventral) - 1] - curve_dorsal['y'].iloc[0])**2
                dist2 = (curve_ventral['x'].iloc[len(curve_ventral) - 1] - curve_dorsal['x'].iloc[len(curve_dorsal) - 1])**2 + (curve_ventral['y'].iloc[len(curve_ventral) - 1] - curve_dorsal['y'].iloc[len(curve_dorsal) - 1])**2

                if dist1>dist2:
                    #reverse the order of points based on which endpoint of dorsal is closer to the endpoint of ventral
                    curve_dorsal = curve_dorsal.reindex(index=curve_dorsal.index[::-1])

                curve = pd.concat( [curve_ventral, curve_dorsal], ignore_index = True)
                curve = curve.reset_index(drop = True)

            ###########
            # Determine left and right - determine counterclockwise direction of curve
            ##########

            #select the a point in the middle -> 
            # get the vector from one end to this point and this point to the other end ->
            # Take the cross product of these two vectors
            # If cross product faces in positive z direction then everything is fine 
            # Else reverse the order of points

            init_point = np.array([curve.loc[ curve.index[0], 'x'], curve.loc[ curve.index[0], 'y']])
            mid_point = np.array([curve.loc[ curve.index[int(len(curve)/2)], 'x'], curve.loc[ curve.index[int(len(curve)/2)], 'y']])
            end_point = np.array([curve.loc[ curve.index[-1], 'x'], curve.loc[ curve.index[-1], 'y']])

            init_mid_vector = mid_point - init_point
            mid_end_vector = end_point - mid_point

            cross_z= np.cross(init_mid_vector, mid_end_vector)

            if cross_z < 0:
                curve['y'] = -1 * curve['y']
                #filcurve = curve.reindex(index=curve.index[::-1])
            
            ##############
            #getting the arclength of along the curve
            ##############

            orig_id = curve.index
            curve = curve.reset_index(drop = True)

            length = 0
            #tot_curvature = 0

            if len(curve)<3:
                print(disc_name, ' has less than 3 points ')
                continue

            for i in range(curve.index[1], curve.index[-1]):

                xi = curve.loc[i,'x']
                yi = curve.loc[i, 'y']
                xi0 = curve.loc[i-1,'x']
                yi0 = curve.loc[i-1, 'y']
                xi1 = curve.loc[i+1,'x']
                yi1 = curve.loc[i+1, 'y']

                #calculating. length. of curve
                if i == curve.index[1]:
                    length = np.sqrt( (xi - xi0)**2 + (yi - yi0)**2 )
                    curve.loc[i-1, 'arclength'] = 0

                length = length + np.sqrt( (xi1 - xi)**2 + (yi1- yi)**2 )
                curve.loc[i, 'arclength'] = curve.loc[i-1, 'arclength'] + np.sqrt( (xi - xi0)**2 + (yi - yi0)**2 )

                if i == curve.index[-2]:
                    curve.loc[i + 1, 'arclength'] = curve.loc[i, 'arclength'] + np.sqrt( (xi1 - xi)**2 + (yi1 - yi)**2 )

            ##########
            #setting the midpoint or the DV boundary as the midpoint of the curve
            ##########

            if crosssection == 'Along_DV':
                curve['region'] = ''
                curve.loc[curve['arclength'] <= length/2, 'region'] = 'ventral'
                curve.loc[curve['arclength'] > length/2, 'region'] = 'dorsal'

            curve_dorsal = curve[curve['region'] == 'dorsal']
            curve_ventral = curve[curve['region'] == 'ventral']
            #arclength of the midpoint is the average of the last point of curve_ventral and the first point of curve_dorsal
            arclength_DV =  (curve_ventral.iloc[-1]['arclength'] + curve_dorsal.iloc[0]['arclength'])/2
            curve['arclength_offset'] = curve['arclength'] - arclength_DV
            allcurves = pd.concat([allcurves, curve], ignore_index = True)

            if plot:
                curve_dorsal = curve[curve['region'] == 'dorsal']
                curve_ventral = curve[curve['region'] == 'ventral']

                fig, axs = plt.subplots(1, 3, figsize=(20, 5))                

                axs[0].plot(curve['arclength_offset'], curve['x'], marker = 'o', )
                axs[0].set(title = disc_name)
                axs[0].set_xlabel('arclength_offset, s', fontsize = 20)
                axs[0].set_ylabel('x(s)', fontsize = 20)
                axs[1].plot(curve['arclength_offset'], curve['y'], marker = 'o', )
                axs[1].set(title = disc_name)
                axs[1].set_xlabel('arclength_offset, s', fontsize = 20)
                axs[1].set_ylabel('y(s)', fontsize = 20)
                #axs[2] = fig.gca()
                #axs[2].plot(curve['x'], curve['y'], marker = 'o', label = 'rotated')
                axs[2].scatter(curve_ventral['x'], curve_ventral['y'], label = 'pre-mid', c = 'red')
                axs[2].scatter(curve_dorsal['x'], curve_dorsal['y'], label = 'post-mid', c = 'blue')
                axs[2].plot([(curve_ventral.iloc[-1]['x'] + curve_dorsal.iloc[0]['x'])/2,np.mean(curve['x'])],[(curve_ventral.iloc[-1]['y'] + curve_dorsal.iloc[0]['y'])/2,np.mean(curve['y'])],color = 'gray')
                axs[2].scatter([(curve_ventral.iloc[-1]['x'] + curve_dorsal.iloc[0]['x'])/2,np.mean(curve['x'])],[(curve_ventral.iloc[-1]['y'] + curve_dorsal.iloc[0]['y'])/2,np.mean(curve['y'])],color = 'black')
                axs[2].legend()
                #axs[2].scatter(end_x, end_y)
                axs[2].set(title = disc_name)
                axs[2].set_aspect('equal')
                axs[2].set_xlabel('x(s)', fontsize = 20)
                axs[2].set_ylabel('y(s)', fontsize = 20)
                axs[2].legend()

    return(allcurves)

def align_curves(allcurves, 
                 genotypes = ['ecadGFPnbG4'], 
                 devstages = None,
                 filename = None,
                 plot = True,
                 orientation = 'horizontal'
                ):
    #This function takes different curves and aligns
    #To align, we need to have curves which have a dorsal and ventral part
    #The midpoint between the dorsal and ventral sides are made the origin
    #Then we rotate the disc so that the line from midpoint to center of mass is the y axis
    
    if devstages is None:
        devstages = np.unique(allcurves['devstage'])


    for i in range(len(genotypes)):

        genotype = genotypes[i]

        for j in range(len(devstages)):

            devstage = devstages[j]

            gen_dev = allcurves[(allcurves['genotype'] == genotype) & (allcurves['devstage'] == devstage)]
            #gen_dev = database[(database['genotype'] == genotype) & (database['devstage'] == devstage)]

            disc_names = np.unique(gen_dev['disc'])

            #fitting a cubic spline to each disc
            for disc_name in disc_names:

                #if disc_name in ['20200703_nubG4myoVI_upcrawling_disc2', '20200717_nubG4myoVI_upcrawling_disc2', '20201022_ecadGFPnbG4_2hAPF_disc3']:
                #    continue

                #print(disc_name)

                curve = gen_dev[gen_dev['disc'] == disc_name]

                curve_dorsal = curve[curve['region'] == 'dorsal'].drop_duplicates()
                curve_ventral = curve[curve['region'] == 'ventral'].drop_duplicates()

                orig_id = curve.index
                curve = curve.reset_index(drop = True)

                #setting origin of curve - translation
                [origin_x, origin_y] = [(curve_ventral.iloc[-1]['x'] + curve_dorsal.iloc[0]['x'])/2, (curve_ventral.iloc[-1]['y']  + curve_dorsal.iloc[0]['y'])/2]
                curve['x'] = curve['x'] - origin_x
                curve['y'] = curve['y'] - origin_y

                #reorienting the curve - rotation

                #cropping the curve equal amounts of dorsal and ventral
                arclength_threshold = min(np.abs(min(curve['arclength_offset'])), max(curve['arclength_offset']))
                cropped_curve = curve[(curve['arclength_offset'] > -arclength_threshold) & (curve['arclength_offset'] < arclength_threshold)]
                cropped_curve_dorsal = cropped_curve[cropped_curve['region'] == 'dorsal']
                [avg_x_dorsal, avg_y_dorsal] = [np.mean(cropped_curve_dorsal['x']), np.mean(cropped_curve_dorsal['y'])]
                cropped_curve_ventral = cropped_curve[cropped_curve['region'] == 'ventral']
                [avg_x_ventral, avg_y_ventral] = [np.mean(cropped_curve_ventral['x']), np.mean(cropped_curve_ventral['y'])]
                [avg_x, avg_y] = [np.mean([avg_x_dorsal, avg_x_ventral]), np.mean([avg_y_dorsal, avg_y_ventral])]

                #[end_x, end_y] = [ (curve_ventral.iloc[0]['x'] + curve_dorsal.iloc[-1]['x'])/2 , (curve_ventral.iloc[0]['y'] + curve_dorsal.iloc[-1]['y'])/2 ]
                #[avg_x, avg_y] = [ np.mean(cropped_curve['x']), np.mean(cropped_curve['y']) ]
                [cos_t, sin_t] = [ - avg_y/(np.sqrt(avg_x**2 + avg_y**2)) , avg_x/(np.sqrt(avg_x**2 + avg_y**2))  ]
                [xs, ys] = [np.array(curve['x']), np.array(curve['y'])]
                curve['x'] = cos_t*xs + sin_t*ys
                curve['y'] = -sin_t*xs + cos_t*ys

                allcurves.loc[orig_id, 'x'] = np.array(curve['x'])
                allcurves.loc[orig_id, 'y'] = np.array(curve['y'])


    plot_montage(allcurves, genotypes = genotypes, devstages = devstages, filename=filename, orientation = orientation)
    
    return(allcurves)

def get_interpolated_curve_from_points(curve, disc_name = 'curve', arclengths = None, max_negative_s = -10, min_positive_s = 10,
                                       return_spline = False):

    degree = 5
    #bspline_x = Bspline(curve['arclength_offset'], curve['x'], k = degree)
    #bspline_y = Bspline(curve['arclength_offset'], curve['y'], k = degree)
    
    # t -> knots 
    # the function splrep will add k+1 knots as the initial point and k+1 knots as the last point
    # the size of t has to be n + k + 1 where n is the number of points
    # we have to add n + k + 1 - 2(k + 1)  = n - k - 1 knots ourselves
    # we can do this by dropping the first (k + 1)/2 points and the last (k+1)/2 points
    
    t = np.array(curve['arclength_offset'])[int((degree + 1)/2):-int((degree + 1)/2)]
    
    spl_x = splrep(curve['arclength_offset'], curve['x'], k = degree, w = [1]*len(curve),
                           #task = -1, # to find the least square solution to the B spline
                           t = [3*min(curve['arclength_offset'])/4, min(curve['arclength_offset'])/2,min(curve['arclength_offset'])/4,0,max(curve['arclength_offset'])/4,max(curve['arclength_offset'])/2,3*max(curve['arclength_offset'])/4,]
                           #t = t
                          )
    spl_y = splrep(curve['arclength_offset'], curve['y'], k = degree, w = [1]*len(curve),
                           #task = -1, # to find the least square solution to the B spline
                           t = [3*min(curve['arclength_offset'])/4, min(curve['arclength_offset'])/2,min(curve['arclength_offset'])/4,0,max(curve['arclength_offset'])/4,max(curve['arclength_offset'])/2,3*max(curve['arclength_offset'])/4,]
                           #t = t
                          )
    
    if arclengths is None:
        arclengths = np.linspace(max_negative_s, min_positive_s, 100)
        
    x = splev(arclengths, spl_x)
    y = splev(arclengths, spl_y)
    
    if return_spline:
        return([spl_x, spl_y])

    x_1 = splev(arclengths, spl_x, der = 1)
    x_2 = splev(arclengths, spl_x, der = 2)
    y_1 = splev(arclengths, spl_y, der = 1)
    y_2 = splev(arclengths, spl_y, der = 2)
    curvatures = (y_2*x_1 - x_2*y_1)/(x_1**2 + y_1**2)**(3/2)

    df = pd.DataFrame({
        'disc':[disc_name]*len(arclengths),
        'arclength':arclengths,
        'x':x,
        'y':y,
        'curvature':curvatures
                      })
    
    return(df)

def interpolate_average_curves(allcurves,
                       genotypes = ['ecadGFPnbG4'], 
                       devstages = [ 'upcrawling','whitePupa',  '2hAPF','4hAPF','6hAPF'],
                      ):
    
    #

    if devstages is None:
        devstages = np.unique(allcurves['devstage'])

    df_all = pd.DataFrame()
    #columns = ['genotype', 'devstage', 'arclength','x','y','x_sd','y_sd','curvature','curvature_sd']
    #Modified by Jana
    columns = ['genotype', 'devstage', 'arclength','x','y','x_sd','y_sd','x_sem','y_sem','curvature','curvature_sd','curvature_sem']
    df_mean = pd.DataFrame(columns = columns)


    for i in range(len(genotypes)):

        genotype = genotypes[i]

        for j in range(len(devstages)):

            devstage = devstages[j]
            gen_dev = allcurves[(allcurves['genotype'] == genotype) & (allcurves['devstage'] == devstage)]
            disc_names = np.unique(gen_dev['disc'])

            #getting the max and min s for which all discs are present
            min_positive_s = 999
            max_negative_s = -999
            for disc_name in disc_names:
                #if disc_name in ['20200703_nubG4myoVI_upcrawling_disc2', '20200717_nubG4myoVI_upcrawling_disc2', '20201022_ecadGFPnbG4_2hAPF_disc3']:
                #    continue
                curve = gen_dev[gen_dev['disc'] == disc_name]
                if min(curve['arclength_offset']) > max_negative_s:
                    max_negative_s = min(curve['arclength_offset'])
                if max(curve['arclength_offset']) < min_positive_s:
                    min_positive_s = max(curve['arclength_offset'])

            #points on which we will find the coordinates of interpolated curve
            #we will make a mean curve out of the curve positions on these arclength locations
            #arclengths = np.linspace(max_negative_s, min_positive_s, 100)

            ## intoducing a stepsize makes sure that spaing is the same in differnet length curves
            #Added by Jana
            step_size_in_microns = 5
            arclengths = np.linspace(max_negative_s, min_positive_s, int(np.abs((min_positive_s - max_negative_s)/ step_size_in_microns)))

            #get interpolated curves with curvature as discrete points on 'arclengths' positions
            for disc_name in disc_names:
                #if disc_name in ['20200703_nubG4myoVI_upcrawling_disc2', '20200717_nubG4myoVI_upcrawling_disc2', '20201022_ecadGFPnbG4_2hAPF_disc3']:
                #    continue
                curve = gen_dev[gen_dev['disc'] == disc_name]
                df = get_interpolated_curve_from_points(curve, disc_name = disc_name, arclengths = arclengths)
                df['devstage'] = devstage
                df['genotype'] = genotype
                df_all = pd.concat([df_all,df], ignore_index = True)

            #get the mean curve
            for arclength in arclengths:
                df_arclength = df_all[df_all['arclength'] == arclength]
                #columns = ['genotype', 'devstage', 'arclength','x','y','x_sd','y_sd','curvature','curvature_sd']
                #row = pd.DataFrame([[genotype, devstage,arclength,np.mean(df_arclength['x']), np.mean(df_arclength['y']), np.std(df_arclength['x']), np.std(df_arclength['y']), np.mean(df_arclength['curvature']), np.std(df_arclength['curvature'])]], columns=columns)
                #Modified by Jana
                row = pd.DataFrame([[genotype, devstage,arclength,np.mean(df_arclength['x']), np.mean(df_arclength['y']), np.std(df_arclength['x']), np.std(df_arclength['y']), np.std(df_arclength['x'])/ np.sqrt(len(df_arclength)), np.std(df_arclength['y'])/ np.sqrt(len(df_arclength)), np.mean(df_arclength['curvature']), np.std(df_arclength['curvature']), np.std(df_arclength['curvature']) / np.sqrt(len(df_arclength['curvature']))]], columns=columns)
                df_mean = pd.concat([df_mean, row], ignore_index = True)

    return([df_all, df_mean])


#only plot 80 % of the arclength
def plot_mean_shapes_w_individual_traces(
                                  df_all,
                                  df_mean,
                                  genotypes = None, 
                                  devstages = None,
                                  orientation = 'vertical',
                                  filename = None,
                                  scalebar_stage = None, scalebar_x_pos = 120, scalebar_y_pos = -450, scalebar_size = 100,
                                  scalebar_linewidth = 5, scalebar_fontsize = 20, scalebar_color = 'black',
                                  ):

    if genotypes is None:
        genotypes = ['ecadGFPnbG4',]
        #genotypes = ['ecadGFPnbG4']
    if devstages is None:
        #devstages = [ 'upcrawling','whitePupa', '2hAPF','4hAPF','6hAPF']
        devstages = np.unique(df_all['devstage'])

    for i in range(len(genotypes)):
        genotype = genotypes[i]
        fig, axs = plt.subplots(len(devstages), 2, figsize=(15, 7*len(devstages)))
        for j in range(len(devstages)):
            #axs[j,0].axis('equal')
            axs[j,0].set_ylim(-510, 10)
            axs[j,0].set_xlim(-260, 260)

        for j in range(len(devstages)):
            
            devstage = devstages[j]
            gen_dev = df_all[(df_all['genotype'] == genotype) & (df_all['devstage'] == devstage)]
            #gen_dev = database[(database['genotype'] == genotype) & (database['devstage'] == devstage)]
            disc_names = np.unique(gen_dev['disc'])
            #fitting a cubic spline to each disc
            for disc_name in disc_names:
                
                df = df_all[df_all['disc'] == disc_name]
                #plot curve
                ax = axs[j, 0]
                #ax.axis('equal')
                ax.set_ylim(-510, 10)
                ax.set_xlim(-260, 260)
                ax.plot(df['x'],df['y'], color = 'gray', alpha = 0.5,)
                
                #plot curvature
                #first we crop to not include the ends
                df_cropped = df[(df['arclength']>0.8*min(df['arclength'])) & (df['arclength']<0.8*max(df['arclength']))]
                ax = axs[j, 1]
                #ax = axs[j, 1]
                ax.set_ylim(-0.022, 0.022)
                ax.set_xlim(-300, 300)
                ax.plot(df_cropped['arclength'],df_cropped['curvature'], color = 'gray', alpha = 0.5,)
                
            #plot mean curve
                
            df = df_mean[(df_mean['genotype'] == genotype) & (df_mean['devstage'] == devstage) ]
            ax = axs[j, 0]
            #ax.axis('equal')
            ax.set_ylim(-510, 10)
            ax.set_xlim(-260, 260)
            ax.plot(df['x'],df['y'], color = 'red', alpha = 1,linewidth = 4)
            ax.set_ylabel(devstage, fontsize = 30, rotation = 0, labelpad = 100)
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

            if j == 0:
                ax.set_title('Shape',  fontsize = 30)
            
            
            #plot mean curvature
            #first we crop to not include the ends
            df_cropped = df[(df['arclength']>0.8*min(df['arclength'])) & (df['arclength']<0.8*max(df['arclength']))]
            ax = axs[j, 1]
            ax.set_ylim(-0.022, 0.022)
            ax.set_xlim(-300, 300)
            ax.plot(df_cropped['arclength'],df_cropped['curvature'], color = 'red', alpha = 1,linewidth = 4)
            ax.set_xlabel('Arclength ' + r'$(\mu m)$', fontsize = 20)
            #ax.set_ylabel('Curvature ' + r'$(\mu m^{-1})$', fontsize = 20, labelpad = 10)
            
            
            if j == 0:
                ax.set_title('Curvature ' + r'$(\mu m^{-1})$', fontsize = 30)
            ax.set_xticks([-150, 0, 150])
            if devstage == '6hAPF':
                if max(df_cropped['curvature']) > 0.02:
                    ax.set_ylim(-1.1*max(df_cropped['curvature']),1.1*max(df_cropped['curvature']))
                    ax.set_yticks([-0.03, 0, 0.03])
            else:
                ax.set_yticks([-0.02, -0.01, 0, 0.01, 0.02])
            #ax.tick_params(axis = 'both', which = 'both', fontsize = 20)
            ax.tick_params(axis = 'both', labelsize = 20)
            ax.grid()
            
            
    if filename is not None:
        plt.savefig(filename, bbox_inches = 'tight')


def plot_mean_shapes_w_error_bars(
                                  df_all,
                                  df_mean,
                                  genotypes = None, 
                                  devstages = None,
                                  orientation = 'vertical',
                                  scalebar_stage = None, scalebar_x_pos = 120, scalebar_y_pos = -450, scalebar_size = 100,
                                  scalebar_linewidth = 5, scalebar_fontsize = 20, scalebar_color = 'black',
                                  filename = None,
                                  ):

    if genotypes is None:
        genotypes = ['ecadGFPnbG4', ]
        #genotypes = ['ecadGFPnbG4']
    if devstages is None:
        devstages = np.unique(df_mean['devstage'])
        #devstages = [ 'upcrawling','whitePupa', '2hAPF','4hAPF','6hAPF']
        #devstages = ['4hAPF']


    for i in range(len(genotypes)):
        genotype = genotypes[i]
        fig, axs = plt.subplots(len(devstages), 2, figsize=(15, 7*len(devstages)))
        for j in range(len(devstages)):
            #axs[j,0].axis('equal')
            axs[j,0].set_ylim(-510, 10)
            axs[j,0].set_xlim(-260, 260)

        for j in range(len(devstages)):
            
            devstage = devstages[j]
            df = df_mean[(df_mean['genotype'] == genotype) & (df_mean['devstage'] == devstage)]
            #gen_dev = database[(database['genotype'] == genotype) & (database['devstage'] == devstage)]
                
            #plot mean curve

                
            ax = axs[j, 0]
            #ax.axis('equal')
            ax.set_ylim(-510, 10)
            ax.set_xlim(-260, 260)
            ax.plot(df['x'], df['y'],color = 'red',linewidth = 4, label = 'Mean', zorder = 2, )
            ax.errorbar(df['x'], df['y'], xerr = df['x_sd'], yerr = df['y_sd'],ecolor = 'gray', alpha = 0.5, label = 'Std dev', zorder = 1)
            #ax.fill_between(df['x'], df['curvature'] - df['curvature_sd'], df['curvature'] + df['curvature_sd'], color = 'gray', alpha = 0.2)
            
            ax.set_ylabel(devstage, fontsize = 30, rotation = 0, labelpad = 100)


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

            if j == 0:
                ax.set_title('Shape',  fontsize = 30)
            
            
            #plot mean curvature
            ax = axs[j, 1]
            ax.set_ylim(-0.022, 0.022)
            ax.set_xlim(-300, 300)
            #ax.plot(df['arclength'],df['curvature'], color = 'black', alpha = 1,)
            ax.set_xlabel('Arclength ' + r'$(\mu m)$', fontsize = 20)
            #ax.set_ylabel('Curvature ' + r'$(\mu m^{-1})$', fontsize = 20, labelpad = 10)
            #ax.fill_between(df['arclength'], df['curvature'] - df['curvature_sd'], df['curvature'] + df['curvature_sd'], color = 'gray', alpha = 0.2)
            ax.fill_between(x = df['arclength'].tolist(), y1 = (df['curvature'] - df['curvature_sd']).tolist(), y2 = (df['curvature'] + df['curvature_sd']).tolist(), color = 'gray', alpha = 0.2)
            ax.plot(df['arclength'], df['curvature'], label = genotype, linewidth = 4, color = 'red', )
            
            
            if j == 0:
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
            
            
    if filename is not None:
        plt.savefig(filename, bbox_inches = 'tight')

