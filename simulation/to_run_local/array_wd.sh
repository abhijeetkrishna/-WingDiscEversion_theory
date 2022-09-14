#!/bin/bash

#COMMENTED - SBATCH -J wd
#COMMENTED - SBATCH -o log_files/wd_out_%A_%a
#COMMENTED - SBATCH -e log_files/wd_err_%A_%a
#COMMENTED - SBATCH -t 3:00:00
#COMMENTED - SBATCH -a 0-7


py_map_creator=map_index_wd.py
simulation=array_wd.py
postprocess=postprocess.py
job_id=0
map_file=map_index_$job_id.csv


python3 $py_map_creator $job_id

for i in {0..10}
do
	echo "Running task : $i"
	python3 $simulation $map_file $i
	python3 $postprocess $map_file $i
done