#!/bin/bash

num_images=$1
num_runs=$2

for ((i=0; i<num_runs; i+=1))
do
    nohup srun -w newton5 python3 ./my_test.py 0 $num_images > ./output_$i & disown
done
