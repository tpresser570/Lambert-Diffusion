using LinearAlgebra
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