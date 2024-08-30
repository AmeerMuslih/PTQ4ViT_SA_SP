import torch
import pandas as pd

def csvFilesMaker(all_util, Accumulator_TOT, InputA_TOT, InputB_TOT, ToggleCount_MultiplierBits, ToggleCount_InputA_Bits, ToggleCount_InputB_Bits, ToggleCount_Accumulator_Bits, totalCycles, dim):
		#normalize the results by number of cycles
		all_util = all_util/totalCycles
		Accumulator_TOT = Accumulator_TOT/totalCycles
		InputA_TOT = InputA_TOT/totalCycles
		InputB_TOT = InputB_TOT/totalCycles
		ToggleCount_MultiplierBits = ToggleCount_MultiplierBits/totalCycles
		ToggleCount_InputA_Bits = ToggleCount_InputA_Bits/totalCycles
		ToggleCount_InputB_Bits = ToggleCount_InputB_Bits/totalCycles
		ToggleCount_Accumulator_Bits = ToggleCount_Accumulator_Bits/totalCycles

		filenamek_csv = "UtilityFor"+str(dim)+"X"+str(dim)+"Dim.csv"
		Destination_path = "/home/firasramadan/PTQ4ViT_SA_SP/OutputFiles/" #"/home/firasramadan/miniconda3/project_quantization_8bit/c_smt_sa/OutputFiles/Group-" + str(GroupID)+ "/"
		all_util_df = pd.DataFrame(all_util[:,:,0].numpy())
		all_util_df.to_csv(Destination_path + filenamek_csv,header = False, index = False)
    
		for bits in range(1,33): # Making files for each bit separately
			if bits < 9:
				filename_csv_InA = "InputA-SP-Bit"+str(9-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
				InA_df = pd.DataFrame(InputA_TOT[:,:,bits-1].numpy())
				InA_df.to_csv(Destination_path + filename_csv_InA,header = False, index = False)
				filename_csv_InB = "InputB-SP-Bit"+str(9-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
				InB_df = pd.DataFrame(InputB_TOT[:,:,bits-1].numpy())
				InB_df.to_csv(Destination_path + filename_csv_InB,header = False, index = False)
				filename_csv_InA_toggle = "InputA-TR-Bit"+str(9-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
				InA_toggle_df = pd.DataFrame(ToggleCount_InputA_Bits[:,:,bits-1,0].numpy())
				InA_toggle_df.to_csv(Destination_path + filename_csv_InA_toggle,header = False, index = False)
				filename_csv_InB_toggle = "InputB-TR-Bit"+str(9-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
				InB_toggle_df = pd.DataFrame(ToggleCount_InputB_Bits[:,:,bits-1,0].numpy())
				InB_toggle_df.to_csv(Destination_path + filename_csv_InB_toggle,header = False, index = False)
			if bits < 17:	
				filename_csv_mult = "Multiplier-SP-Bit"+str(17-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 16 is the MSB
				mult_df = pd.DataFrame(all_util[:,:,bits].numpy())
				mult_df.to_csv(Destination_path + filename_csv_mult,header = False, index = False)
				filename_csv_mult_toggle = "Multiplier-TR-Bit"+str(17-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 16 is the MSB
				mult_toggle_df = pd.DataFrame(ToggleCount_MultiplierBits[:,:,bits-1,0].numpy())
				mult_toggle_df.to_csv(Destination_path + filename_csv_mult_toggle,header = False, index = False)
			filename_csv_accum_toggle = "Accumulator-TR-Bit"+str(33-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
			accum_toggle_df = pd.DataFrame(ToggleCount_Accumulator_Bits[:,:,bits-1,0].numpy())
			accum_toggle_df.to_csv(Destination_path + filename_csv_accum_toggle,header = False, index = False)
			filename_csv_accum = "Accumulator-SP-Bit"+str(33-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
			accum_df = pd.DataFrame(Accumulator_TOT[:,:,bits-1].numpy())
			accum_df.to_csv(Destination_path + filename_csv_accum,header = False, index = False)

def load_checkpoint(file_path):
		checkpoint = {}
		with open(file_path, 'r') as f:
			data = f.read()
			checkpoint = eval(data)
		return checkpoint

def load_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    tensor = torch.tensor(df.values)
    return tensor

def main():
    dim = 128 #dim of the SA

    all_util = torch.zeros(dim,dim,17)
    Accumulator_TOT = torch.zeros(dim,dim,32)
    InputA_TOT = torch.zeros(dim,dim,8)
    InputB_TOT = torch.zeros(dim,dim,8)
    ToggleCount_MultiplierBits = torch.zeros(dim,dim,16,2)
    ToggleCount_InputA_Bits = torch.zeros(dim,dim,8,2)
    ToggleCount_InputB_Bits = torch.zeros(dim,dim,8,2)
    ToggleCount_Accumulator_Bits = torch.zeros(dim,dim,32,2)
    totalCycles=0
    for i in range(50):
        # Specify the file path of the checkpoint
        checkpoint_file = f'/home/<user>/PTQ4ViT_SA_SP/OutputFiles/image_{i}_results.txt'

        # Load the checkpoint
        checkpoint = load_checkpoint(checkpoint_file)

        # Load the tensors from the checkpoint
        all_util += torch.tensor(checkpoint['all_util'])
        Accumulator_TOT += torch.tensor(checkpoint['Accumulator_TOT'])
        InputA_TOT += torch.tensor(checkpoint['InputA_TOT'])
        InputB_TOT += torch.tensor(checkpoint['InputB_TOT'])
        ToggleCount_MultiplierBits += torch.tensor(checkpoint['ToggleCount_MultiplierBits'])
        ToggleCount_InputA_Bits += torch.tensor(checkpoint['ToggleCount_InputA_Bits'])
        ToggleCount_InputB_Bits += torch.tensor(checkpoint['ToggleCount_InputB_Bits'])
        ToggleCount_Accumulator_Bits += torch.tensor(checkpoint['ToggleCount_Accumulator_Bits'])
        totalCycles += checkpoint['totalCycles']
	
    csvFilesMaker(all_util, Accumulator_TOT, InputA_TOT, InputB_TOT, ToggleCount_MultiplierBits, ToggleCount_InputA_Bits, ToggleCount_InputB_Bits, ToggleCount_Accumulator_Bits, totalCycles, dim)
	

if __name__ == "__main__":
    main()
	