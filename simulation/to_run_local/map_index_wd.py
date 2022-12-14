# var1 goes from value1 to value2 with intervals of interval1

import itertools
import pandas as pd
import os
import sys
import shutil
import numpy as np

def main():
    job_id=sys.argv[1]
    var_dict={
    'thickness':[0.2],#[0.1,0.15,0.25],
    'nb_iterations':[10],
    #'lambda_anisotropic':[10**(-1), 10**(-0.5), 10**(-0.2), 10**(0), 10**(0.2), 10**(0.5), 10**(1)]
    #'lambda_anisotropic':[0.8, 0.85,0.9,0.95,1, 1.05, 1.1, 1.15, 1.2]
    #'lambda_anisotropic_inDV':[0.96, 0.98, 1, 1.01, 1.02, 1.03,1.04,1.05],
    #'lambda_anisotropic':[0.96, 0.98, 1, 1.01, 1.02, 1.03,1.04,1.05]
    }

    dirname = ''
    a= list(var_dict.values())
    combinations=list(itertools.product(*a))
    comb_df=pd.DataFrame(combinations, columns=var_dict.keys())
    
    input_var=list(comb_df.columns)

    filelist = ['Makefile', 'main.cpp', 'functions.py']

    for i in range(len(comb_df)):

        folder = dirname + "wd_"+'_'.join([var+ '_' +str(comb_df.loc[i,var]) for var in input_var])+'/'
        comb_df.loc[i, 'folder_name'] = folder

        os.makedirs(folder, exist_ok=True)    
        os.makedirs(folder+'runfiles/', exist_ok=True)
        os.makedirs(folder+'sim_output/', exist_ok=True)
    
        comb_df[comb_df.index == i].to_csv(folder+'runfiles/simulation_parameters.csv', index=True)
    
        for file in filelist:
            shutil.copy(file, folder)

    comb_df.to_csv(dirname+'map_index_'+job_id+'.csv', index=False)


if __name__ == '__main__':
    main()