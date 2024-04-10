#%%
import pandas as pd
import numpy as np
import torch
from pathlib import Path
#%%
# Load the CSV file
traj_names_df = pd.read_csv('extended_mars_transfers/64/csv/traj_names_df.csv')
sampled_files = traj_names_df.sample(n=1000, random_state=42)
# Initialize a list to store the data
traj_data_list = []

for file_name in sampled_files['traj_file_name']:
    file_path = Path(file_name)
    if file_path.is_file():
        with open(file_path, 'rb') as file:
            # Load the trajectory data with torch
            traj_data = torch.load(file)
            # Convert to numpy (if necessary) and append to the list
            traj_data_list.append(traj_data.numpy())  # Use .numpy() if data is in torch tensor format

# Convert the list of arrays to a single numpy array
# Use np.vstack or np.array depending on your data shape and requirements
test = np.array(traj_data_list)
# Save the big numpy array to a file
np.save('all_traj_data.npy', test)

# %%
