import csv
import os

# Directory where the files are located
directory = '/home/a.mosa/Ameer/PTQ4ViT_Firas'

# List to store the results
results = []

# Iterate over the files in the directory
for filename in os.listdir(directory):
    if filename.startswith('output_'):
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