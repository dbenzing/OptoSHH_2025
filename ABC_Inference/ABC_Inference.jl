ENV["PYTHON"]=""; import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
using GpABC, OrdinaryDiffEq, Distances, Distributions, Plots, StatsBase, Printf, DelimitedFiles
pyplot()

# Define your parameters and priors
true_params =  [0.0, 0.0, 0.35, 1.5, 0.5, 0.0, 10.0, 1] # nominal parameter values

priors = [Uniform(0.01, 5.), Uniform(0.1, 2.), Uniform(log10(0.1), log10(3.)),
          Uniform(log10(1.), log10(3.)), Uniform(log10(0.1), log10(5.)), Uniform(log10(0.1), log10(10.)),
          Uniform(log10(0.1), log10(10.)), Uniform(log10(0.1),log10(10.))]


param_indices = [3,4,5,7,8]  #indices of the parameters to be estimated
priors = priors[param_indices]

# ODE solver settings
Tspan = (0.0, 8.0)
solver = Tsit5()
saveat = [0, 1, 2, 3, 4, 5, 6, 8] * 1.0
data = readdlm("clearance_data.csv", ',', Float64)

# Define the ODE system
function ODE_SHH_simple(dx, x, par, t)
    dx[1] = par[1] * par[2] * (1 - x[1]) - par[3] * x[1]
    dx[2] = x[1] ^ par[4] / (x[1] ^ par[4] + par[5] ^ par[4]) - (par[6] + par[7]) * x[2]
    dx[3] = par[7] * x[2] - par[8] * x[3]
end

# Function to solve the ODE system with non-negative constraints
function GeneReg(params::AbstractArray{Float64,1},
                 Tspan::Tuple{Float64,Float64}, solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm,
                 saveat::AbstractArray{Float64,1})

    Hill0 = 1. / (1. + params[5] ^ params[4])
    x0 = [1., Hill0 / (params[6] + params[7]), params[7] * Hill0 / (params[8] * (params[6] + params[7]))]

    prob = ODEProblem(ODE_SHH_simple, x0, Tspan, params)
    
    # Define a custom ContinuousCallback for non-negativity
    function condition(u, t, integrator)
        minimum(u) < 0
    end

    function affect!(integrator)
        integrator.u .= max.(integrator.u, 0.0)
    end

    cb = ContinuousCallback(condition, affect!)

    # Solve the problem with the callback
    sol = solve(prob, solver, abstol=1e-15, reltol=1e-15, saveat=saveat, callback=cb)
    
    # Only return species 3 normalized to its initial value
    res3 = hcat(sol.u...)[3,:]
    return reshape((res3) ./ (res3[1]), (:,1))
end

# Function that simulates the model
function simulator_function(log_params)
    var_params = 10.0 .^ log_params  # convert from log10-space to linear
    params = copy(true_params)
    params[param_indices] .= var_params
    GeneReg(params, Tspan, solver, saveat)
end


#
# Simulation
#
n_particles = 2000
threshold = .175
sim_result = SimulatedABCRejection(data, simulator_function, priors, threshold, n_particles;
    max_iter=convert(Int, 1e8),
    write_progress=true)

writedlm("particles.csv", sim_result.population, ',')   
writedlm("distances.csv", sim_result.distances, ',')