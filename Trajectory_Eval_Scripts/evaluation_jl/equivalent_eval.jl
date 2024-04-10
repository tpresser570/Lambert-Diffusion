#load packages
using NPZ
using Plots
using Statistics
include("eval_helpers.jl")

#load trajectories from numpy
trajectories = npzread("trajectories/traj_256.npy")

#define odefunction
scale_r = 1.495978707*10^8 #km
scale_v = 30 #km/s
MU = 1.32712440018e11
odefun = R2BPdynamics_2d
scale_vec = [scale_r,scale_r,scale_v,scale_v]

equivalent_trajectories = condense_batch(trajectories)

batch_defect,traj_rms =  eval_batch(equivalent_trajectories, MU, odefun,scale_vec)

