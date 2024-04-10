#load packages
using NPZ
using Plots
using Statistics 
using LinearAlgebra
using DifferentialEquations
include("eval_helpers.jl")


# Example usage:
MU = 1.32712440018e11  # Adjust as necessary
trajectories = npzread("trajectories/traj_16.npy")  # Load your trajectories
odefun = R2BPdynamics_2d  # Your dynamics function

#condensed_batch = condense_batch(trajectories)

all_defects, difference_rms = forward_process_batch_trajectories(trajectories, odefun, MU)

difference_trend = zeros(4,16)

for i = 1:16
    temp_diff = all_defects[:,:,i]
    averages = [mean(abs.(temp_diff[:,1]));mean(abs.(temp_diff[:,2]));mean(abs.(temp_diff[:,3]));mean(abs.(temp_diff[:,4]))]
    difference_trend[:,i] = averages
end 


plot(abs.(difference_trend[1,:]))
plot!(abs.(difference_trend[2,:]))