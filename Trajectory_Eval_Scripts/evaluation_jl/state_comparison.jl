#load packages
using NPZ
using Plots
using Statistics 
using LinearAlgebra
using DifferentialEquations
include("eval_helpers.jl")


# Example usage:
trajectories = npzread("trajectories/traj_16.npy")  # Load your trajectories
scale_r = 1.495978707*10^8 #km
scale_v = 30 #km/s
MU = 1.32712440018e11
odefun = R2BPdynamics_2d
scale_vec = [scale_r,scale_r,scale_v,scale_v]


all_defects, difference_scaled_rms = forward_process_batch_trajectories(trajectories, odefun, MU, scale_vec)

difference_trend = zeros(4,16)
difference_variance = zeros(4,16)
for i = 1:16
    temp_diff = all_defects[:,:,i]
    averages = [mean(abs.(temp_diff[:,1]));mean(abs.(temp_diff[:,2]));mean(abs.(temp_diff[:,3]));mean(abs.(temp_diff[:,4]))]
    std_devs = [std(abs.(temp_diff[:,1])); std(abs.(temp_diff[:,2])); std(abs.(temp_diff[:,3])); std(abs.(temp_diff[:,4]))]


    difference_trend[:,i] = averages
    difference_variance[:,i] = std_devs
end 

error_trends = vcat(difference_trend,difference_variance)

npzwrite("error_trends.npy", error_trends)
