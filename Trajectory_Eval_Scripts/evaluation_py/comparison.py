#%%
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from prop_helpers import img2traj, solve_lambert, model_2BP
import pykep as pk
from scipy.integrate import odeint

samples = np.load("/Users/tylerpresser/Documents/GitHub/LambAIrt/model_analysis/samples_np.npy")
ranges = np.genfromtxt("/Users/tylerpresser/Documents/GitHub/LambAIrt/model_analysis/ranges.csv",delimiter=',')
tmax = 86400*270

#%% solve lambert
test = samples[5,:,:,:]
nn_traj = img2traj(test,tmax,ranges)

r1 = nn_traj[1:4,0]
r2 = nn_traj[1:4,-1]
tof = nn_traj[0,-1] - nn_traj[0,0]
mu = pk.MU_SUN / 1e9
N = 0
tof,a, ecc, V1, V2 = solve_lambert(r1,r2,tof,mu,N,which=0)
# %% propagate
state_0 = [r1[0],r1[1],r1[2],V1[0],V1[1],V1[2]]
t = nn_traj[0,:]
sol = odeint(model_2BP,state_0,t)
sol = np.transpose(sol)
# %%
fig = plt.figure()
axis_limits = [-2e8, 2e8, -2e8,2e8,-1e8,1e8]

ax = plt.axes(projection='3d')
ax.plot3D (nn_traj[1,:], nn_traj[2,:], nn_traj[3,:], 'blue')
ax.plot3D (sol[0,:], sol[1,:], sol[2,:], 'red')
ax.set_title('nn vs lambert')
plt.axis(axis_limits)

# %% regular norm
sol_2d = sol[[0,1,2,3,4],:]
nn_traj_2d = nn_traj[[1,2,4,5],:]
norm_2d = np.linalg.norm(sol_2d-nn_traj_2d)


#%% 2d norm
sol_2d = sol[[0,1,3,4],:]
nn_traj_2d = nn_traj[[1,2,4,5],:]
norm_2d = np.linalg.norm(sol_2d-nn_traj_2d)

