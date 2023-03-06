import torch

#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

# Supported edge types
SUPPORTED_EDGES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# Supported atoms 
SUPPORTED_ATOMS = ["C", "N", "O", "F"]
ATOMIC_NUMBERS =  [6, 7, 8, 9]

# Dataset (if you change this, delete the processed files to run again)
MAX_MOLECULE_SIZE = 10  

# To remove valence errors ect.
DISABLE_RDKIT_WARNINGS = True