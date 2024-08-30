#!/bin/bash

for ((i=0; i<2; i++))
do
    nohup srun --gres=gpu:1 --mem=32G python3 ./my_test.py $i & disown
done