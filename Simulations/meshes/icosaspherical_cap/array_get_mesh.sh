#!/bin/bash

#SBATCH -J wd
#SBATCH -o log_files/out_%A_%a
#SBATCH -e log_files/err_%A_%a
#SBATCH -t 24:00:00
#SBATCH -a 0-45

module load python/3.10.7
source /projects/project-krishna/WingDiscEversion_theory/Environment/env_WDTheory/bin/activate

map_file=map_index_$SLURM_ARRAY_JOB_ID.csv

python3 map_index.py $SLURM_ARRAY_JOB_ID
python3 array_get_mesh.py $map_file $SLURM_ARRAY_TASK_ID