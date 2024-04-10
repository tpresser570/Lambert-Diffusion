using LinearAlgebra
using DifferentialEquations
using Statistics


"""
    R2BPdynamics(rv, μ, t)
Compute time derivative of state vector in the restricted two-body system. `rv` is the state
vector `[r; v]` {km; km/s}, `μ` is the gravitational parameter {km³/s²}, and `t` is time
{s}.
"""
function R2BPdynamics(rv, mu, t)  #make sure rv and μ are in km and km³/s²
    r,v = rv[1:3], rv[4:6]
    rvdot = zeros(6)
    rvdot[1:3] = v
    rvdot[4:6] = -mu/norm(r)^3 * r   #acceleration [km/s]
    return rvdot
end

"""
In-place version of `R2BPdynamics(rvdot, rv, μ, t)`.
"""
function R2BPdynamics!(rvdot, rv, mu, t)  #make sure rv and μ are in km and km³/s²
    rvdot[:] = R2BPdynamics(rv,mu,t)
    return nothing
end


function R2BPdynamics_2d(rv, mu, t)  #make sure rv and μ are in km and km³/s²
    r,v = rv[1:2], rv[3:4]
    rvdot = zeros(4)
    rvdot[1:2] = v
    rvdot[3:4] = -mu/norm(r)^3 * r   #acceleration [km/s]
    return rvdot
end

"""
In-place version of `R2BPdynamics(rvdot, rv, μ, t)`.
"""
function R2BPdynamics_2d!(rvdot, rv, mu, t)  #make sure rv and μ are in km and km³/s²
    rvdot[:] = R2BPdynamics_2d(rv,mu,t)
    return nothing
end


"""
function that computes the defects of nodes in a lambert trajectory at 
the midpoint between nodes 
"""
function lambert_defectCalc(X_all, t_TU, n_state, n_nodes, MU, odefun)
    
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
        stateF_forward = forward_states[end]

        ###prop backwards
        x0_back = X_all[:,i+1]
        tspan_back = (0,-t_mid_TU)
        prob = ODEProblem(odefun,x0_back,tspan_back,MU)
        sol_back = solve(prob, Vern9(),reltol=1e-12, abstol=1e-12)
        back_states = sol_back.u
        stateF_back = back_states[end]


        ### defect between forward and backward results:
        defect1[:,i] = stateF_forward - stateF_back

    end 

    return defect1
end 


"""
function that computes the defects of nodes in all trajectories 
in a sample batch from the model - uses midpoint multiple shooting method
"""

function eval_batch(trajectories,MU,odefun,scale_vec)
    batch_defects = zeros(size(trajectories)[1],size(trajectories)[2]-1,size(trajectories)[3]-1)
    traj_rms = zeros(size(trajectories)[1])
    for i = 1:size(trajectories)[1]
        single_traj = Matrix(trajectories[i,:,:])
        single_traj = convert(Matrix{Float64},single_traj)

        t_TU = single_traj[1,:]
        X_all = single_traj[2:end,:]
        n_nodes = size(X_all)[2]
        n_state = size(X_all)[1]

        defect1 =  lambert_defectCalc(X_all, t_TU, n_state, n_nodes, MU, odefun)
        defect_scaled = defect1 ./ scale_vec
        defect_scaled = convert(Matrix{Float64},defect_scaled)
        traj_rms[i] = sqrt(mean((defect_scaled.^2)))
        batch_defects[i,:,:] = defect1
        

    end 

    return batch_defects,traj_rms

end 




"""
Uses intial condition from the model to propagate forward
computes the defect at all the points using model states
"""
function forward_process_batch_trajectories(trajectories, odefun, MU, scale_vec)
    n_trajectories = size(trajectories, 1)
    n_state = size(trajectories, 2) - 1  # Assuming the first row is time
    n_nodes = size(trajectories, 3)

    # Initialize an array to hold the defects for all trajectories
    all_defects = zeros(n_trajectories, n_state, n_nodes)

    for i in 1:n_trajectories
        single_traj = convert(Matrix{Float64}, trajectories[i, :, :])
        t_TU = single_traj[1, :]
        X_all = single_traj[2:end, :]

        x0 = X_all[:, 1]
        tspan = (0.0, t_TU[end])
        prob = ODEProblem(odefun, x0, tspan, MU)
        sol = solve(prob, Vern9(), reltol=1e-12, abstol=1e-12)

        for j in 1:n_nodes
            t = t_TU[j]
            propagated_state_at_t = sol(t)
            difference = X_all[:, j] - propagated_state_at_t
            all_defects[i, :, j] = difference
        end
    end

    difference_scaled_rms = zeros(size(all_defects)[1])


    for i = 1:size(all_defects)[1]
        scaled_difference = all_defects[i,:,:] ./ scale_vec
        scaled_difference = convert(Matrix{Float64},scaled_difference)
        difference_scaled_rms[i] = sqrt(mean((scaled_difference.^2)))
    end 

    return all_defects, difference_scaled_rms
end





"""
Condenses a trajectory from N timesteps to 16 using evenly spaced points.
"""
function condense_trajectory(trajectory)
    # Input trajectory is assumed to be of shape [5,64]
    # Output trajectory will be of shape [5,16]
    
    n_rows, n_cols = size(trajectory) # Get the dimensions of the trajectory matrix
    output_cols = 16 # Define the number of columns for the output matrix
    
    # Calculate step size for picking points. We subtract 2 to account for the first and last points,
    # then divide by the number of intervals (15) to get the step size between points.
    step = floor(Int, (n_cols - 2) / (output_cols - 1))
    
    # Initialize an array to hold the selected indices
    selected_indices = [1] # Always include the first index
    
    # Add indices for the intermediate points
    for i = 1:output_cols-2 # Subtract 2 because we manually add the first and last points
        push!(selected_indices, 1 + i*step)
    end
    
    # Ensure the last point is always included
    push!(selected_indices, n_cols)
    
    # Select the columns from the original trajectory based on the indices
    condensed_trajectory = trajectory[:, selected_indices]
    
    return condensed_trajectory
end

"""
Condenses a batch
"""
function condense_batch(batch_trajectories)
    # Input batch_trajectories is assumed to be of shape [100, 5, 64]
    # Output will be of shape [100, 5, 16]
    
    # Get the size of the batch
    n_trajectories, n_rows, n_cols = size(batch_trajectories)
    
    # Define the size of the output trajectories
    output_cols = 16
    
    # Initialize the output array with zeros
    condensed_batch = Array{Float64}(undef, n_trajectories, n_rows, output_cols)
    
    # Loop over each trajectory in the batch
    for i = 1:n_trajectories
        # Condense the current trajectory
        condensed_trajectory = condense_trajectory(batch_trajectories[i, :, :])
        
        # Store the condensed trajectory in the output array
        condensed_batch[i, :, :] = condensed_trajectory
    end
    
    return condensed_batch
end