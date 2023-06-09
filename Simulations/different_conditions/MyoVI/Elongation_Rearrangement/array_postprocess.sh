#!/bin/bash

#SBATCH -J wd
#SBATCH -o log_files/pst_out_%a
#SBATCH -e log_files/pst_err_%a
#SBATCH -t 24:00:00
#SBATCH -a 0-6

module load python/3.10.7
source ../../dependencies/env_WDProject/bin/activate

py_map_creator=map_index_wd.py
simulation=array_wd.py
postprocess=postprocess.py
map_file=map_index_47160557.csv

#export OPENBLAS_NUM_THREADS=1

#if [ ! -f $map_file ]
#then
#	python $py_map_creator $job_id #$SLURM_ARRAY_JOB_ID
#	#can we give flag kind of options while running a python script?
#	#get output from this python file which teells it the map_index file and the directory where things have to be stored
#	#give that as input to the other file.
#fi

#python $simulation $map_file $SLURM_ARRAY_TASK_ID
python $postprocess $map_file $SLURM_ARRAY_TASK_ID