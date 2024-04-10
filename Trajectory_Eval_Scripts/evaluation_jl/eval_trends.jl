#load packages
using NPZ
using Plots
using Statistics
include("eval_helpers.jl")

#load trajectories from numpy
trajectories = npzread("trajectories/traj_64_v2.npy")

#define odefunction
scale_r = 1.495978707*10^8 #km
scale_v = 30 #km/s
MU = 1.32712440018e11
odefun = R2BPdynamics_2d
scale_vec = [scale_r,scale_r,scale_v,scale_v]
batch_defect,traj_rms =  eval_batch(trajectories, MU, odefun,scale_vec)

difference_trend = zeros(4,size(trajectories)[3]-1)
difference_variance = zeros(4,size(trajectories)[3]-1)
for i = 1:size(trajectories)[3]-1
    temp_diff = batch_defect[:,:,i] 

    for j = 1:size(temp_diff)[1]
        temp_diff[j,:] = temp_diff[j,:]./scale_vec
    end 

    averages = [mean(abs.(temp_diff[:,1]));mean(abs.(temp_diff[:,2]));mean(abs.(temp_diff[:,3]));mean(abs.(temp_diff[:,4]))]
    std_devs = [std(abs.(temp_diff[:,1])); std(abs.(temp_diff[:,2])); std(abs.(temp_diff[:,3])); std(abs.(temp_diff[:,4]))]


    difference_trend[:,i] = averages
    difference_variance[:,i] = std_devs
end 

error_trends = vcat(difference_trend,difference_variance)

npzwrite("shooting_error_trends_64.npy", error_trends)