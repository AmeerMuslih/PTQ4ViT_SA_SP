#!/bin/bash

for ((i=0; i<1; i++))
do
    nohup srun --gres=gpu:1 --mem=32G python3 ./my_test.py $i 1 & disown
done