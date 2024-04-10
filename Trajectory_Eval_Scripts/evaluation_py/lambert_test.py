#%%
import numpy as np
import pykep as pk

def compute_velocity_differences(trajectories):
    # Constants
    mu_sun = pk.MU_SUN  # Gravitational parameter of the Earth
    
    # Initialize an array to hold the velocity differences
    velocity_differences_v1 = []
    velocity_differences_v2 = []

    
    # Iterate over the trajectories
    for i in range(trajectories.shape[0]):
        # Extract the initial and final states
        initial_state = trajectories[i, :, 0]
        final_state = trajectories[i, :, -1]
        print(final_state)
        # Ensure positions and tof are properly formatted
        r1 = [float(initial_state[1]) * 1000, float(initial_state[2]) * 1000, 0.0]  # Convert km to m
        r2 = [float(final_state[1]) * 1000, float(final_state[2]) * 1000, 0.0]
        tof = float(final_state[0])   # Convert days to seconds
        print(tof)
        # Solve Lambert's problem
        l = pk.lambert_problem(r1, r2, tof, mu_sun)
        v1_lambert, v2_lambert = l.get_v1()[0], l.get_v2()[0]

        # Initial velocity from the trajectory
        initial_velocity = [initial_state[3] * 1000, initial_state[4] * 1000, 0.0]  # Convert km/s to m/s
        final_velocity = [final_state[3] * 1000, final_state[4] * 1000, 0.0]  # Convert km/s to m/s
        # Compute the velocity difference (magnitude)
        velocity_diff_v1 = np.linalg.norm(np.array(v1_lambert) - np.array(initial_velocity))
        velocity_diff_v2 = np.linalg.norm(np.array(v2_lambert) - np.array(final_velocity))

        # Store the velocity difference
        velocity_differences_v1.append(velocity_diff_v1)
        velocity_differences_v2.append(velocity_diff_v2)

    
    return np.array(velocity_differences_v1),np.array(velocity_differences_v2)



#%%
# Example usage
trajectories = np.load('trajectories/traj_64_v2.npy')  # Load your trajectories data
differences_v1,differences_v2 = compute_velocity_differences(trajectories)
# Now 'differences' holds the computed differences for each trajectory

# %%
