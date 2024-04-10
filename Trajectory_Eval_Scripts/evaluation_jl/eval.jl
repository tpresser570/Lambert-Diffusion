#load packages
using NPZ
using Plots
using Statistics
include("eval_helpers.jl")

#load trajectories from numpy
trajectories = npzread("trajectories/traj_16_8_v2.npy")
eval_trajs= condense_batch(trajectories)

#condense_batch if necessary

#define odefunction
scale_r = 1.495978707*10^8 #km
scale_v = 30 #km/s
MU = 1.32712440018e11
odefun = R2BPdynamics_2d
scale_vec = [scale_r,scale_r,scale_v,scale_v]

batch_defect,traj_rms =  eval_batch(trajectories, MU, odefun,scale_vec)

