import csv
import os
import sys

# Directory where the files are located
directory = '/home/firasramadan/miniconda3/Ameer_Project_Transformers/PTQ4ViT_FI'

# List to store the results
results = []
bit_flips = int(sys.argv[1])
# Iterate over the files in the directory
for filename in os.listdir(directory):
    if filename.startswith(f'output_{bit_flips}_'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            lines = file.readlines()
            accuracy = lines[-2].split(':')[1].strip()
            runtime = lines[-1].split(':')[1].strip()
            results.append([accuracy, runtime])

# Write the results to a CSV file
output_file = 'FI_results_output.csv'
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Accuracy', 'Run Time'])
    writer.writerows(results)

print('CSV file created successfully.')