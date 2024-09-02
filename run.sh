#!/bin/bash

step=2

for ((i=0; i<100; i+=step))
do
    nohup srun python3 ./my_test.py $i $step > ./output_$i & disown
done