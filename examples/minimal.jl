using Gen
using Gen2DAgentMotion
import Distributions

const scene = Gen2DAgentMotion.example_apartment_floorplan()
const couch = Point(0.15, 0.85)

make_times(dt, T::Int) = collect(range(0.0, step=dt, length=T))

@gen function model(T::Int)
    planner_params = PlannerParams(2000, 3.0, 5000, 1.)
    noise = 0.05
    destination ~ uniform_coord()
    start = couch
    path = Point[start]
    (path_rest, failed, tree) = plan_and_optimize_path(scene, start, Point(destination), planner_params)
    append!(path, path_rest)
    obs_times = make_times(0.01, T)
    nominal_speed = 0.1
    prob_lag = 0.2
    prob_normal = 0.6
    prob_skip = 0.2
    obs_params = ObsModelParams(nominal_speed, prob_lag, prob_normal, prob_skip, noise)
    observations ~ path_observation_model(path, obs_times, obs_params)
    return (scene, start, path, failed) 
end

function generate_synthetic_data()
    T = 10
    trace = simulate(model, (T,))
    observations = Vector{Point}(undef, T)
    for t in 1:T
        observations[t] = Point(trace[:observations => (:x, t)], trace[:observations => (:y, t)])
    end
    return observations
end

function infer(
        observations::Vector{Point}, num_particles::Int, num_samples::Int)
    T = length(observations)
    constraints = choicemap()
    for t in 1:T
        constraints[(:x, t)] = observations[t].x
        constraints[(:y, t)] = observations[t].y
    end
    (traces, log_weights, _) = importance_sampling(model, (T,), constraints, num_particles)
    weights = exp.(log_weights)
    idx = Vector{Int}(undef, num_samples)
    Distributions.rand!(Distributions.Categorical(weights / sum(weights)), idx)
    return traces[idx]
end

observations = generate_synthetic_data()
traces = infer(observations, 10000, 100)
display(get_choices(traces[1]))
