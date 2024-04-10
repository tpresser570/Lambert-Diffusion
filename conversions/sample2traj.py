#%% load in samples
import torch
import numpy as np 
from helpers import img2traj
#ranges from csv
ranges = np.genfromtxt("/Users/tylerpresser/Documents/GitHub/diffusion_model_analysis/V2_samples/16/8/ranges.csv",delimiter=',')
ranges = ranges[1:]
#image samples from models
image_samples = torch.load("/Users/tylerpresser/Documents/GitHub/diffusion_model_analysis/V2_samples/16/8/samples_400000.pth")
# %% convert to trajectories
traj_array = []
for sample in image_samples:
    single_trajectory = img2traj(sample,ranges)
    traj_array.append(single_trajectory)

vectors_array = np.stack(traj_array)
np.save("traj_16_8_v2.npy",vectors_array)


# %%
