#%%
import numpy as np
import matplotlib.pyplot as plt

# Load the combined error and standard deviation data
batch_IC_errors = np.load('trajectories/shooting_error_trends_64.npy')  # Assuming this is the correct path

# Separate the errors and standard deviations
errors = batch_IC_errors[0:4, :]  # Rows 0:3 for errors
std_devs = batch_IC_errors[4:8, :]  # Rows 4:7 for standard deviations
#%%
# Generate node numbers (1 to 16 assuming each trajectory has 16 nodes)
time_steps = np.arange(1, errors.shape[1] + 1)  # Adjust based on the actual number of nodes
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})
# Set up the subplot: 1 row, 2 columns
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# Plotting
# Left plot: Position Errors with error bars
axs[0].errorbar(time_steps, errors[0, :], yerr=std_devs[0, :], label='X Position Error', fmt='-o', capsize=3)
axs[0].errorbar(time_steps, errors[1, :], yerr=std_devs[1, :], label='Y Position Error', fmt='-o', capsize=3)
axs[0].set_xlabel('Node Number')
axs[0].set_ylabel('Scaled Position Error')
axs[0].legend(loc='upper right')
#axs[0].set_aspect('equal', adjustable='box')  # Enforce square aspect ratio for the subplot

# Right plot: Velocity Errors with error bars
axs[1].errorbar(time_steps, errors[2, :], yerr=std_devs[2, :], label='X Velocity Error', fmt='-o', capsize=3)
axs[1].errorbar(time_steps, errors[3, :], yerr=std_devs[3, :], label='Y Velocity Error', fmt='-o', capsize=3)

axs[1].set_xlabel('Node Number')
axs[1].set_ylabel('Scaled Velocity Error')
axs[1].legend(loc='upper right')
#axs[1].set_aspect('equal', adjustable='box')  # Enforce square aspect ratio for the subplot

# Adjust layout to prevent overlap
plt.tight_layout()  # Adjust layout to not overlap
#plt.yscale("log")

#plt.subplots_adjust(wspace=0.2)  # Increase wspace as needed to prevent label overlap
plt.savefig('shooting_error_trends.png',format='png',bbox_inches='tight')  # Save the figure as an SVG
plt.show()















# %%
