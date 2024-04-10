using NPZ
using Plots
using Statistics
include("eval_helpers.jl")

# Load trajectories from numpy
trajectories = npzread("trajectories/traj_64_v2.npy")
trajectories= condense_batch(trajectories)

# Define odefunction
scale_r = 1.495978707*10^8 #km
scale_v = 30 #km/s
MU = 1.32712440018e11
odefun = R2BPdynamics_2d
scale_vec = [scale_r, scale_r, scale_v, scale_v]

# Function to calculate Lambert defects and save propagated states
function lambert_defectCalc_save(X_all, t_TU, n_state, n_nodes, MU, odefun)

    forward_states_all = zeros(n_nodes-1,n_state,32)
    backward_states_all = zeros(n_nodes-1,n_state,32)

    #pre-allocate the defect array
    defect1 = zeros(n_state, n_nodes-1)

    #loop over all nodes
    for i = 1:n_nodes-1

        #find midpoint between nodes
        t_mid_TU = (t_TU[i+1] - t_TU[i]) / 2

        ###prop forward
        x0 = X_all[:,i]
        tspan = (0, t_mid_TU)
        prob = ODEProblem(odefun,x0,tspan,MU)
        sol = solve(prob, Vern9(),reltol=1e-12, abstol=1e-12)
        forward_states = sol.u

        t_test = range(0,t_mid_TU,32)
        temp_test_states= sol(t_test)
        print(size(temp_test_states))
        forward_states_all[i,:,:] .= temp_test_states
        stateF_forward = forward_states[end]
    

        ###prop backwards
        x0_back = X_all[:,i+1]
        tspan_back = (0,-t_mid_TU)
        prob = ODEProblem(odefun,x0_back,tspan_back,MU)
        sol_back = solve(prob, Vern9(),reltol=1e-12, abstol=1e-12)
        back_states = sol_back.u
        stateF_back = back_states[end]

        t_test_back = range(0,-t_mid_TU,32)
        temp_test_states_back= sol_back(t_test_back)
        backward_states_all[i,:,:] .= temp_test_states_back


        ### defect between forward and backward results:
        defect1[:,i] = stateF_forward - stateF_back

    end 


    return defect1, forward_states_all, backward_states_all
end

i = 1
single_traj = Matrix(trajectories[i,:,:])
single_traj = convert(Matrix{Float64},single_traj)

t_TU = single_traj[1,:]
X_all = single_traj[2:end,:]
n_nodes = size(X_all)[2]
n_state = size(X_all)[1]
defect1, forward_states_all, backward_states_all = lambert_defectCalc_save(X_all, t_TU, n_state, n_nodes, MU, odefun)


npzwrite("condense_16.npy", single_traj)
npzwrite("forward_16.npy", forward_states_all)
npzwrite("prop_backward_16.npy", backward_states_all)
