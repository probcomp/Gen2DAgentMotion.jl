using Gen
using Gen2DAgentMotion
import Distributions
using Plots
import Random

const scene = Gen2DAgentMotion.example_apartment_floorplan()
const couch = Point(0.15, 0.85)

make_times(dt, T::Int) = collect(range(0.0, step=dt, length=T))

@gen function model(T::Int)
    planner_params = PlannerParams(400, 3.0, 200, 0.02)
    noise = 0.02
    destination ~ uniform_coord()
    start = couch
    path = Point[start]
    (path_rest, failed, tree) = plan_and_optimize_path(scene, start, Point(destination), planner_params)
    append!(path, path_rest)
    obs_times = make_times(0.01, T)
    nominal_speed = 5.0
    walk_noise = 0.2
    obs_params = ObsModelParams(nominal_speed, walk_noise, noise)
    observations ~ path_observation_model(path, obs_times, obs_params)
    return (scene, start, path, failed) 
end

get_path(trace) = get_retval(trace)[3]
get_T(trace) = get_args(trace)[1]

function generate_synthetic_data()
    T = 10
    trace = simulate(model, (T,))
    (_, _, path, failed) = get_retval(trace)
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
        constraints[:observations => (:x, t)] = observations[t].x
        constraints[:observations => (:y, t)] = observations[t].y
    end
    (traces, log_weights, _) = importance_sampling(model, (T,), constraints, num_particles)
    weights = exp.(log_weights)
    idx = Vector{Int}(undef, num_samples)
    Distributions.rand!(Distributions.Categorical(weights / sum(weights)), idx)
    return traces[idx]
end

function draw_scene!()
    for wall in scene.walls
        plot!([wall.a.x, wall.b.x], [wall.a.y, wall.b.y], color="black", label=nothing, aspect_ratio=:equal)
    end
end

function draw_paths(traces)
    for trace in traces
        path = get_path(trace)
        for i in 1:(length(path)-1)
            plot!([path[i].x, path[i+1].x], [path[i].y, path[i+1].y], color="gray", label=nothing)
        end
    end
end

function draw_trace(trace)
    p = scatter([couch.x], [couch.y], color="blue", label=nothing, markerstrokewidth=0)
    scatter!([trace[:destination][1]], [trace[:destination][2]], color="red", label=nothing, markerstrokewidth=0)
    draw_scene!()
    path = get_path(trace)
    for i in 1:(length(path)-1)
        plot!([path[i].x, path[i+1].x], [path[i].y, path[i+1].y], color="gray", label=nothing)
    end
    xs = [trace[:observations => (:x, t)] for t in 1:get_T(trace)]
    ys = [trace[:observations => (:y, t)] for t in 1:get_T(trace)]
    scatter!(xs, ys, color="black", label=nothing, markerstrokewidth=0)
    return p
end

function draw_simulated_traces()
    Random.seed!(2)
    n = 10
    plots = []
    for i in 1:n
        trace = simulate(model, (10,))
        push!(plots, draw_trace(trace))
    end
    plot(plots...)
    savefig("simulated.png")
end

draw_simulated_traces()

function draw_inferences()
    Random.seed!(2)
    plots = []
    for i in 1:4
        observations = generate_synthetic_data()
        traces = infer(observations, 5000, 100)
        dests = [tr[:destination] for tr in traces]
        p = scatter([couch.x], [couch.y], color="blue", label=nothing, markerstrokewidth=0)
        draw_paths(traces)
        xs = [dest[1] for dest in dests]
        ys = [dest[2] for dest in dests]
        scatter!(xs, ys, color="red", label=nothing, markerstrokewidth=0)
        xs = [obs.x for obs in observations]
        ys = [obs.y for obs in observations]
        scatter!(xs, ys, color="black", label=nothing, markerstrokewidth=0)
        draw_scene!()
        push!(plots, p)
    end
    plot(plots...)
    savefig("inferences.png")
end

draw_inferences()
