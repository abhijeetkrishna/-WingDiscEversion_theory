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
    'thickness': [0.1],#np.around(np.linspace(0.03, 0.3,10),2),#np.linspace(0.07, 0.17, 11),
    'theta_max': [0.8662],#np.around(0.4029*np.linspace(1,2,11),4),
    'mesh_refine_factor':[30,50,80],#[50, 80],
    'angle_of_rotation':[0.1, 0.2, 0.3, 0.4, 0.5],
    'theta_DV':[0, 0.1931, 3.14]
    }

    a= list(var_dict.values())
    combinations=list(itertools.product(*a))
    comb_df=pd.DataFrame(combinations, columns=var_dict.keys())
    comb_df.to_csv('map_index_'+job_id+'.csv', index=False)
    os.makedirs("pickle", exist_ok=True)  
    os.makedirs("vtk", exist_ok=True)  


if __name__ == '__main__':
    main()