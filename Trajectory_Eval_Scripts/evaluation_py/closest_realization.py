#%%
import torch 
import matplotlib.pyplot as plt
import numpy as np 
from helpers import find_min_distance_single,load_vectors
import pandas as pd
dataset= pd.read_csv('/Users/tylerpresser/Documents/GitHub/ScoreNet/earth_mars/csv/sample_initial_conditions_scaled.csv')
file_list = dataset['traj_file_name']
file_list_random = np.random.choice(file_list,10000)
test = torch.load("/Users/tylerpresser/Documents/GitHub/ScoreNet/samples_295000.pth")
vectors = load_vectors(file_list_random)

min_list=[]
#%%
for i in range(100):
    target_state = test[i,0,0:7,[0,-1]]
    target_state = np.array(target_state)

    min = find_min_distance_single(target_state,vectors)
    min_list.append(min)

# %%
