using Gen
using Gen2DAgentMotion
using PyPlot
using Statistics: median
using LaTeXStrings
import Random
using Statistics: mean, median, std

start = Point(0.1, 0.9)
#scene = Gen2DAgentMotion.example_apartment_floorplan()
scene = Gen2DAgentMotion.Scene(Bounds(0.0, 1.0, 0.0, 1.0), Wall[
    Wall(Point(0.0, 0.5), Point(0.47, 0.5)),
    Wall(Point(0.53, 0.5), Point(1.0, 0.5))])
planner_params = PlannerParams(3000, 1.0, 0.2, 10000, 0.02, 0.01)
obs_params = ObsModelParams(0.005, 0.66, 0.01)

macro make_model(name, motion_model)
    return esc(quote
        @gen (static) function $(name)(T::Int)
            destination ~ uniform_coord()
            (path_rest, failed, tree) = plan_and_optimize_path(scene, start, Point(destination), planner_params)
            path = [start, path_rest...]
            {:measurements} ~ $(motion_model)(path, obs_params, T)
            return (failed, path)
        end
    end)
end

get_failed(trace) = get_retval(trace)[1]
get_path(trace) = get_retval(trace)[2]

@make_model(uncollapsed, motion_and_measurement_model_uncollapsed)
@make_model(uncollapsed_incremental, motion_and_measurement_model_uncollapsed_incremental)
@make_model(pseudomarginal_1, make_motion_and_measurement_model_smc_pseudomarginal(1))
@make_model(pseudomarginal_1_optimal, make_motion_and_measurement_model_smc_pseudomarginal_optimal_local_proposal(1))
@make_model(pseudomarginal_3, make_motion_and_measurement_model_smc_pseudomarginal(3))
@make_model(pseudomarginal_3_optimal, make_motion_and_measurement_model_smc_pseudomarginal_optimal_local_proposal(3))
@make_model(pseudomarginal_10, make_motion_and_measurement_model_smc_pseudomarginal(10))
@make_model(pseudomarginal_10_optimal, make_motion_and_measurement_model_smc_pseudomarginal_optimal_local_proposal(10))
@make_model(pseudomarginal_100, make_motion_and_measurement_model_smc_pseudomarginal(100))
@make_model(pseudomarginal_100_optimal, make_motion_and_measurement_model_smc_pseudomarginal_optimal_local_proposal(100))
@make_model(pseudomarginal_200, make_motion_and_measurement_model_smc_pseudomarginal(200))
@make_model(collapsed, motion_and_measurement_model_collapsed)
@make_model(collapsed_incremental, motion_and_measurement_model_collapsed_incremental)

@gen (static) function prior_proposal(trace)
    destination ~ uniform_coord()
end

@gen (static) function bottom_half_proposal(trace)
    destination ~ uniform_coord_rect(0.0, 1.0, 0.0, 0.5)
end


@load_generated_functions()

function get_apartment_dataset()
    failed = true
    T = 50
    attempt() = generate(uncollapsed, (T,), choicemap((:destination, [0.5, 0.1])))[1]
    failed(trace) = get_failed(trace)
    trace = attempt()
    i = 0
    while failed(trace)
        println("attempt $i")
        trace = attempt()
    end
    observations = Point[]
    for t in 1:T
        push!(observations, Point(trace[:measurements => obs_addr(motion_and_measurement_model_uncollapsed, t)]))
    end
    return observations
end

function evaluate_lml_given_path(motion_model, observations, path::Vector{Point}, reps)
    T = length(observations)
    estimates = Float64[]
    for rep in 1:reps
        # obtain initial trace, starting from inferred destination
        obs_choices = choicemap()
        (trace, _) = generate(motion_model, (path, obs_params, 1), choicemap(
                (obs_addr(motion_model, 1), [observations[1].x, observations[1].y])))
        for t in 2:T
            trace, = update(
                trace,
                (path, obs_params, t), (NoChange(), NoChange(), UnknownChange(),),
                choicemap((obs_addr(motion_model, t), [observations[t].x, observations[t].y])))
        end
        push!(estimates, get_score(trace))
    end
    return estimates
end

function evaluate_lml(model, motion_model, observations, destination, reps)
    T = length(observations)
    estimates = Float64[]
    for rep in 1:reps
        # obtain initial trace, starting from inferred destination
        obs_choices = choicemap()
        (trace, _) = generate(model, (1,), choicemap(
                (:destination, destination),
                (:measurements => obs_addr(motion_model, 1), [observations[1].x, observations[1].y])))
        for t in 2:T
            trace, = update(
                trace,
                (t,), (UnknownChange(),),
                choicemap((:measurements => obs_addr(motion_model, t), [observations[t].x, observations[t].y])))
        end
        push!(estimates, get_score(trace))
    end
    return estimates
end

function mh_with_debug(
        trace, proposal::GenerativeFunction, proposal_args::Tuple)
    model_args = get_args(trace)
    argdiffs = map((_) -> NoChange(), model_args)
    proposal_args_forward = (trace, proposal_args...,)
    (fwd_choices, fwd_weight, _) = propose(proposal, proposal_args_forward)
    (new_trace, weight, _, discard) = update(trace,
        model_args, argdiffs, fwd_choices)
    proposal_args_backward = (new_trace, proposal_args...,)
    (bwd_weight, _) = assess(proposal, proposal_args_backward, discard)
    alpha = weight - fwd_weight + bwd_weight
    if log(rand()) < alpha
        # accept
        return (new_trace, true, new_trace)
    else
        # reject
        return (trace, false, new_trace)
    end
end

function evaluate_acceptance_rate(model, motion_model, observations, n, reps; burn_in=1000)
    T = length(observations)
    accepted = zeros(reps)

    all_traces = []

    for rep in 1:reps

        # obtain initial trace, starting from destination in the bottom half
        obs_choices = choicemap()
        (trace, _) = generate(model, (1,), choicemap(
                (:destination, [0.5, 0.1]),
                (:measurements => obs_addr(motion_model, 1), [observations[1].x, observations[1].y])))
        for t in 2:T
            trace, = update(
                trace,
                (t,), (UnknownChange(),),
                choicemap((:measurements => obs_addr(motion_model, t), [observations[t].x, observations[t].y])))
        end

        # do burn-in mcmc moves over the encapsulated random choices
        for iter in 1:burn_in
            trace, acc = mh(trace, bottom_half_proposal, ())
        end

        init_trace = trace
        accepted_traces = []
        rejected_traces = []

        # TODO visualize the trace, and the accepted / rejected traces on top...

        # do n MCMC moves from the same initial trace
        for i in 1:n
            trace, acc, proposed = mh_with_debug(init_trace, bottom_half_proposal, ())
            if acc
                push!(accepted_traces, trace)
            else
                push!(rejected_traces, proposed)
            end
            accepted[rep] += acc
        end
        println("rep $rep; acceptance rate: $(accepted[rep]) / $n = $(accepted[rep] / n)")

        push!(all_traces, (init_trace, accepted_traces, rejected_traces))
    end
    rates = accepted / n
    return rates, all_traces
end

function jitter(ax, x, data)
    x_jitter = x .+ (rand(length(data)) .- 0.5)  * 0.25
    ax.scatter(x_jitter, data, marker=".", s=2, alpha=0.5, color="black")
    xmin = x - 0.25
    xmax = x + 0.25
    return (xmin, xmax)
end

import JSON

function draw_trace(trace)
    scatter([trace[:destination][1]], [trace[:destination][2]], color="red", s=50, alpha=0.5)
    path = get_path(trace)
    for i in 1:(length(path)-1)
        plot([path[i].x, path[i+1].x], [path[i].y, path[i+1].y], color="gray")
    end
end

function draw_scene()
    ax = gca()
    for wall in scene.walls
        plot([wall.a.x, wall.b.x], [wall.a.y, wall.b.y], color="black")
    end
    gca().set_xlim(0, 1)
    gca().set_ylim(0, 1)
    tight_layout()
end

function plot_jitter!(ax, labels, label, data)
    push!(labels, label)
    (xmin, xmax) = jitter(ax, length(labels), data)
    n = length(data)
    plot([xmin, xmax], [mean(data), mean(data)], linestyle="-", color="red")
end

function generate_acceptance_rate_results()
    Random.seed!(1) 
    observations = get_apartment_dataset()
    T = length(observations)
    n = 100 # 100 200?
    reps = 25 # 25

    results = Dict()
    debug_traces = Dict()

    println("collapsed (inc)...")
    results["collapsed-inc"], debug_traces["collapsed-inc"] = evaluate_acceptance_rate(
        collapsed_incremental,
        motion_and_measurement_model_collapsed_incremental,
        observations, n, reps)

    println("pseudomarginal(1)...")
    results["pseudomarginal-1"], debug_traces["pseudomarginal-1"] = evaluate_acceptance_rate(
        pseudomarginal_1,
        make_motion_and_measurement_model_smc_pseudomarginal(1),
        observations, n, reps)

    println("pseudomarginal(1) optimal...")
    results["pseudomarginal-1-optimal"], debug_traces["pseudomarginal-1-optimal"] = evaluate_acceptance_rate(
        pseudomarginal_1_optimal,
        make_motion_and_measurement_model_smc_pseudomarginal_optimal_local_proposal(1),
        observations, n, reps)

    println("pseudomarginal(3)...")
    results["pseudomarginal-3"], debug_traces["pseudomarginal-3"] = evaluate_acceptance_rate(
        pseudomarginal_3,
        make_motion_and_measurement_model_smc_pseudomarginal(3),
        observations, n, reps)

    println("pseudomarginal(3) optimal...")
    results["pseudomarginal-3-optimal"], debug_traces["pseudomarginal-3-optimal"] = evaluate_acceptance_rate(
        pseudomarginal_3_optimal,
        make_motion_and_measurement_model_smc_pseudomarginal_optimal_local_proposal(3),
        observations, n, reps)

    println("pseudomarginal(10)...")
    results["pseudomarginal-10"], debug_traces["pseudomarginal-10"] = evaluate_acceptance_rate(
        pseudomarginal_10,
        make_motion_and_measurement_model_smc_pseudomarginal(10),
        observations, n, reps)

    println("pseudomarginal(10) optimal...")
    results["pseudomarginal-10-optimal"], debug_traces["pseudomarginal-10-optimal"] = evaluate_acceptance_rate(
        pseudomarginal_10_optimal,
        make_motion_and_measurement_model_smc_pseudomarginal_optimal_local_proposal(10),
        observations, n, reps)

    #println("pseudomarginal(100)...")
    #results["pseudomarginal-100"], debug_traces["pseudomarginal-100"] = evaluate_acceptance_rate(
        #pseudomarginal_100,
        #make_motion_and_measurement_model_smc_pseudomarginal(100),
        #observations, n, reps)

    #println("pseudomarginal(100) optimal...")
    #results["pseudomarginal-100-optimal"], debug_traces["pseudomarginal-100-optimal"] = evaluate_acceptance_rate(
        #pseudomarginal_100_optimal,
        #make_motion_and_measurement_model_smc_pseudomarginal_optimal_local_proposal(100),
        #observations, n, reps)

    #println("pseudomarginal(200)...")
    #results["pseudomarginal-200"], debug_traces["pseudomarginal-200"] = evaluate_acceptance_rate(
        #pseudomarginal_200,
        #make_motion_and_measurement_model_smc_pseudomarginal(200),
        #observations, n, reps)

    return (results, debug_traces)
end

function acceptance_rate_plots()
    close("all")

    ## get approximate posterior samples
    #num_posterior_samples = 0
    #posterior_num_particles = 1000
    #println("obtaining initial destination samples")
    #obs_choices = choicemap()
    #for t in 1:T
        #obs_choices[:measurements => obs_addr(motion_and_measurement_model_uncollapsed, t)] = [observations[t].x, observations[t].y]
    #end
    #posterior_dests = []
    #for rep in 1:num_posterior_samples
        #@time trace, = importance_resampling(uncollapsed, (T,), obs_choices, posterior_num_particles)
        #push!(posterior_dests, trace[:destination])
    #end
#
    ## show data set and posterior
    #println(observations)
    #figure(figsize=(6,6))
    #scatter([pt.x for pt in observations], [pt.y for pt in observations], color="black", s=5)
    #scatter([vec[1] for vec in posterior_dests], [vec[2] for vec in posterior_dests], color="red", s=5)
    #ax = gca()
    #for wall in scene.walls
        #plot([wall.a.x, wall.b.x], [wall.a.y, wall.b.y], color="black")
    #end
    #gca().set_xlim(0, 1)
    #gca().set_ylim(0, 1)
    #tight_layout()
    #savefig("dataset.png")

    (results, debug_traces) = generate_acceptance_rate_results()

    # generate results
    open("acceptance_rate_results.json", "w") do f
       JSON.print(f, results)
    end

    results = JSON.parsefile("acceptance_rate_results.json")

    ## show debug traces
    ##for method in ["uncollapsed-inc", "collapsed-inc", "pseudomarginal-1"]
    #for method in ["pseudomarginal-1"]
        #println("plotting results for method: $method")
        #figure(figsize=(3 * 4, reps * 4))
        #for rep in 1:reps
            #(init_trace, accepted_traces, rejected_traces) = debug_traces[method][rep]
            #subplot(reps, 3, (rep-1)*3 + 1)
            #title("rep: $rep, init trace")
            #scatter([pt.x for pt in observations], [pt.y for pt in observations], color="black", s=5)
            #draw_trace(init_trace)
            #draw_scene()
            #subplot(reps, 3, (rep-1)*3 + 2)
            #title("rep: $rep, accepted")
            #scatter([pt.x for pt in observations], [pt.y for pt in observations], color="black", s=5)
            #for trace in accepted_traces
                #draw_trace(trace)
            #end
            #draw_scene()
            #subplot(reps, 3, (rep-1)*3 + 3)
            #title("rep: $rep, rejected")
            #scatter([pt.x for pt in observations], [pt.y for pt in observations], color="black", s=5)
            #for trace in rejected_traces
                #draw_trace(trace)
            #end
            #draw_scene()
        #end
        #tight_layout()
        #savefig("debug_traces_$method.png")
    #end
    
    figure()
    ax = gca()
    labels = String[]
    plot_jitter!(ax, labels, "exact", results["collapsed-inc"])
    plot_jitter!(ax, labels, "PM (1)", results["pseudomarginal-1"])
    plot_jitter!(ax, labels, "PM-opt (1)", results["pseudomarginal-1-optimal"])
    plot_jitter!(ax, labels, "PM (3)", results["pseudomarginal-3"])
    plot_jitter!(ax, labels, "PM-opt (3)", results["pseudomarginal-3-optimal"])
    plot_jitter!(ax, labels, "PM (10)", results["pseudomarginal-10"])
    plot_jitter!(ax, labels, "PM-opt (10)", results["pseudomarginal-10-optimal"])
    #plot_jitter!(ax, labels, "PM (100)", results["pseudomarginal-100"])
    #plot_jitter!(ax, labels, "PM-opt (100)", results["pseudomarginal-100-optimal"])
    gca().set_ylim(0, 1)
    gca().set_xticks(collect(1:length(labels)))
    gca().set_xticklabels(labels)
    ylabel("acceptance rate")
    savefig("acceptance_rates.png")

end

function generate_lml_results(reps)
    Random.seed!(1)
    observations = get_apartment_dataset()
    path = Point[Point(0.1, 0.9), Point(0.5, 0.5)] # actually path
    T = length(observations)

    results = Dict()

    println("collapsed (inc)...")
    results["collapsed-inc"] = evaluate_lml_given_path(
        motion_and_measurement_model_collapsed_incremental,
        observations, path, reps)

    println("pseudomarginal(1)...")
    results["pseudomarginal-1"] = evaluate_lml_given_path(
        make_motion_and_measurement_model_smc_pseudomarginal(1),
        observations, path, reps)

    println("pseudomarginal(3)...")
    results["pseudomarginal-3"] = evaluate_lml_given_path(
        make_motion_and_measurement_model_smc_pseudomarginal(3),
        observations, path, reps)

    println("pseudomarginal(3) optimal...")
    results["pseudomarginal-3-optimal"] = evaluate_lml_given_path(
        make_motion_and_measurement_model_smc_pseudomarginal_optimal_local_proposal(3),
        observations, path, reps)

    println("pseudomarginal(10)...")
    results["pseudomarginal-10"] = evaluate_lml_given_path(
        make_motion_and_measurement_model_smc_pseudomarginal(10),
        observations, path, reps)

    println("pseudomarginal(10) optimal...")
    results["pseudomarginal-10-optimal"] = evaluate_lml_given_path(
        make_motion_and_measurement_model_smc_pseudomarginal_optimal_local_proposal(10),
        observations, path, reps)

    println("pseudomarginal(100)...")
    results["pseudomarginal-100"] = evaluate_lml_given_path(
        make_motion_and_measurement_model_smc_pseudomarginal(100),
        observations, path, reps)

    println("pseudomarginal(100) optimal...")
    results["pseudomarginal-100-optimal"] = evaluate_lml_given_path(
        make_motion_and_measurement_model_smc_pseudomarginal_optimal_local_proposal(100),
        observations, path, reps)

    return results
end

function lml_estimate_plots()

    close("all")

    reps = 100

    #results = generate_lml_results(reps)
    #open("lml_estimate_results.json", "w") do f
        #JSON.print(f, results)
    #end

    results = JSON.parsefile("lml_estimate_results.json")

    figure()
    ax = gca()
    labels = String[]
    plot_jitter!(labels, "exact", results["collapsed-inc"])
    plot_jitter!(labels, "PM (1)", results["pseudomarginal-1"])
    plot_jitter!(labels, "PM (3)", results["pseudomarginal-3"])
    plot_jitter!(labels, "PM-opt (3)", results["pseudomarginal-3-optimal"])
    plot_jitter!(labels, "PM (10)", results["pseudomarginal-10"])
    plot_jitter!(labels, "PM-opt (10)", results["pseudomarginal-10-optimal"])
    plot_jitter!(labels, "PM (100)", results["pseudomarginal-100"])
    plot_jitter!(labels, "PM-opt (100)", results["pseudomarginal-100-optimal"])
    gca().set_ylim(280, 310)
    gca().set_xticks(collect(1:length(labels)))
    gca().set_xticklabels(labels)
    ylabel("log marginal likelihood estimates")
    savefig("lml_estimates.png")
end

close("all")
#lml_estimate_plots()
acceptance_rate_plots()
