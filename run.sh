#!/bin/bash

num_images=$1
num_runs=$2
bit_flips=$3

for ((i=0; i<num_runs; i+=1))
do
    nohup srun python3 ./my_test.py 0 $num_images $bit_flips > ./output_$i & disown
done
