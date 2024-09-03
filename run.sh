#!/bin/bash

step=1

for ((i=100; i<150; i+=step))
do
    nohup srun python3 ./my_test.py $i $step > ./output_$i & disown
done