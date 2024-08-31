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
ToggleCount_MultiplierBits = torch.zeros(dim,dim,16,2)
ToggleCount_InputA_Bits = torch.zeros(dim,dim,8,2)
ToggleCount_InputB_Bits = torch.zeros(dim,dim,8,2)
ToggleCount_Accumulator_Bits = torch.zeros(dim,dim,32,2)
totalCycles=torch.zeros(1)


def matmul_sa(tensor_a, tensor_b):
	global all_util
	global Accumulator_TOT
	global InputA_TOT
	global InputB_TOT
	global ToggleCount_MultiplierBits
	global ToggleCount_InputA_Bits
	global ToggleCount_InputB_Bits
	global ToggleCount_Accumulator_Bits
	global totalCycles

	a = tensor_a
	b = tensor_b
	#print("Percentage of zeroes A = ", percentage_of_zeros(a))
	#print("Percentage of zeroes B = ", percentage_of_zeros(b))
	dut, util, cycles, PUs_access_count,AccumulatorBitsCount,Input_A_BitsCount,Input_B_BitsCount,MultiplierToggle,AccumulatorToggle,InputAToggle,InputBToggle = c_smt_sa.exec(a[None,:,:].detach().cpu(),b[:,:].detach().cpu(), dim, 1, 1024)
	all_util += PUs_access_count
	Accumulator_TOT += AccumulatorBitsCount
	InputA_TOT += Input_A_BitsCount
	InputB_TOT += Input_B_BitsCount
	ToggleCount_MultiplierBits += MultiplierToggle
	ToggleCount_Accumulator_Bits += AccumulatorToggle
	ToggleCount_InputA_Bits += InputAToggle
	ToggleCount_InputB_Bits += InputBToggle
	totalCycles += cycles
	#print("Cycles = ", cycles)
	#print(dut.shape)
	return dut
		

if __name__=='__main__':
	a = torch.randint(10000,(128,128))
	b = torch.randint(100000,(128,128))
	matmul_sa(a,b)
	assert torch.allclose(matmul_sa(a,b), torch.matmul(a,b))