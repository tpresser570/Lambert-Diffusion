#%%
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from plot_helpers import load_vectors,img2traj

samples = torch.load("/Users/tylerpresser/Documents/GitHub/LambAIrt/model_analysis/samples_295000.pth")
tmax = 86400*270
ranges = np.genfromtxt("/Users/tylerpresser/Documents/GitHub/LambAIrt/model_analysis/earth_mars/csv/ranges.csv",delimiter=',')
test = samples[10,:,:,:]
traj = img2traj(test,tmax,ranges)
#%%
plt.plot(traj[4,:],traj[5,:])

# %%
