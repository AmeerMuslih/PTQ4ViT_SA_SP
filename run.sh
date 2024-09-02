#!/bin/bash

step=2

for ((i=0; i<50; i+=step))
do
    nohup srun --gres=gpu:1 --mem=32G python3 ./my_test.py $((i * step)) $step & disown
done