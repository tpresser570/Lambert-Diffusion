#load packages
using NPZ
using Plots
using Statistics
include("eval_helpers.jl")

#load trajectories from numpy
trajectories = npzread("trajectories/traj_16.npy")

#define odefunction
scale_r = 1.496*10^8 #km
scale_v = 30 #km/s
MU = 1.327e11
odefun = R2BPdynamics_2d
single_traj = Matrix(trajectories[1,:,:])
single_traj = convert(Matrix{Float64},single_traj)
t_TU = single_traj[1,:]
X_all = single_traj[2:end,:]
plot()
plot!(X_all[1,:],X_all[2,:])
n_nodes = size(X_all)[2]
n_state = size(X_all)[1]


scale_vec = [scale_r,scale_r,scale_v,scale_v]    
    
    
#pre-allocate the defect array
defect1 = zeros(n_state, n_nodes-1)

#loop over all nodes
i = 1

#find midpoint between nodes
t_mid_TU = (t_TU[15] - t_TU[i]) 

###prop forward
x0 = X_all[:,i]
tspan = (0, t_TU[16])
prob = ODEProblem(odefun,x0,tspan,MU)
sol = solve(prob, Vern9(),reltol=1e-12, abstol=1e-12)
forward_states = sol.u
stateF_forward = forward_states[end]
plot!(sol,vars=(1,2))

###prop backwards
x0_back = X_all[:,i+1]
tspan_back = (0,-t_mid_TU)
prob = ODEProblem(odefun,x0_back,tspan_back,MU)
sol_back = solve(prob, Vern9(),reltol=1e-12, abstol=1e-12)
back_states = sol_back.u
stateF_back = back_states[end]
plot!(sol_back,vars=(1,2))



### defect between forward and backward results:
defect1[:,i] = stateF_forward - stateF_back

