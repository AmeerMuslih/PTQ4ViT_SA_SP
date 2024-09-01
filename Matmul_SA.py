import torch
import c_smt_sa
import sys
import csv
import pandas as pd
import numpy as ny
import pickle
import argparse
import math
import os
type_bits=33
#parser = argparse.ArgumentParser(description='Firas Ramadan, firasramadan@campus.technion.ac.il')
#parser.add_argument('--dimension', type=int, required=True, help='choose the dimension of the systolic array')
#parser.add_argument('--GroupID', type=int, required=True, help='choose which group of the generated pickle files to load- all are in size of 10')

dim = 128 #dim of the SA

all_util = torch.zeros(dim,dim,17)
Accumulator_TOT = torch.zeros(dim,dim,32)
InputA_TOT = torch.zeros(dim,dim,8)
InputB_TOT = torch.zeros(dim,dim,8)
totalCycles=torch.zeros(1)
		

def percentage_of_zeros(tensor):
    total_elements = tensor.numel()  # Total number of elements in the tensor
    zero_elements = torch.sum(tensor == 0).item()  # Number of zeros in the tensor
    percentage = (zero_elements / total_elements) * 100
    return percentage

def matmul_sa(tensor_a, tensor_b):
	start_idx = None
	if 'start_idx' in os.environ:
		start_idx = int(os.environ['start_idx'])
	assert start_idx >= 0 and start_idx < 1000
	global all_util
	global Accumulator_TOT
	global InputA_TOT
	global InputB_TOT
	global totalCycles

	a = tensor_a
	b = tensor_b
	#print("Percentage of zeroes A = ", percentage_of_zeros(a))
	#print("Percentage of zeroes B = ", percentage_of_zeros(b))
	dut, util, cycles, PUs_access_count,AccumulatorBitsCount,Input_A_BitsCount,Input_B_BitsCount,_,_,_,_ = c_smt_sa.exec(a[None,:,:].detach().cpu(),b[:,:].detach().cpu(), dim, 1, 1024)
	all_util += PUs_access_count
	Accumulator_TOT += AccumulatorBitsCount
	InputA_TOT += Input_A_BitsCount
	InputB_TOT += Input_B_BitsCount
	totalCycles += cycles
	checkpoint = {
		'all_util': all_util,
		'Accumulator_TOT': Accumulator_TOT,
		'InputA_TOT': InputA_TOT,
		'InputB_TOT': InputB_TOT,
		'totalCycles': totalCycles
	}

	# Save checkpoint to file
	checkpoint_file = f'/home/firasramadan/PTQ4ViT_Firas/OutputFiles/image_{start_idx}_results.txt'
	with open(checkpoint_file, 'w') as f:
		f.write(str(checkpoint))

	#print("Cycles = ", cycles)
	#print(dut.shape)
	return dut