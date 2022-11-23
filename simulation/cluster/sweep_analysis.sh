#!/bin/bash
#SBATCH -J sweep_analysis     # the job's name
#SBATCH -t 00:01:00       # max. wall clock time 5s
#SBATCH -n 1              # number of tasks
#SBATCH -o log_files/sweep_out  # output file
#SBATCH -e log_files/sweep_err  # output file
#SBATCH --partition=batch

module load python/3.10.7
source ../../dependencies/env_WDProject/bin/activate

python sweep_analysis.py