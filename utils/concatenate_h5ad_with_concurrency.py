
import os
import anndata as ad
from concurrent.futures import ThreadPoolExecutor

# Path to the folder containing h5ad files
folder_path = "/home/Samir.Aisin/geneformer_project/AgeAnno_data"

# Get a list of all h5ad files in the directory
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5ad")]

# Function to read an h5ad file
def read_h5ad(file_path):
    return ad.read_h5ad(file_path)

# Load the h5ad files concurrently
with ThreadPoolExecutor() as executor:
    ann_datas = list(executor.map(read_h5ad, file_paths))

# Concatenate by observations (adding rows)
combined_adata = ad.concat(ann_datas, axis=0)

# Save the concatenated dataset to a new h5ad file
output_file = "/home/Samir.Aisin/geneformer_project/combined_data.h5ad"  # Output path
combined_adata.write(output_file)

print("Concatenated dataset saved to:", output_file)

