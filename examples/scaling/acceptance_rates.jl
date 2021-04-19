using Gen
using Gen2DAgentMotion
using PyPlot
using Statistics: median
using LaTeXStrings
import Random

start = Point(0.1, 0.9)
#scene = Gen2DAgentMotion.example_apartment_floorplan()
scene = Gen2DAgentMotion.Scene(Bounds(0.0, 1.0, 0.0, 1.0), Wall[Wall(Point(0.0, 0.5), Point(0.4, 0.5)), Wall(Point(0.6, 0.5), Point(1.0, 0.5))])
planner_params = PlannerParams(1000, 1.0, 0.2, 1000, 0.02, 0.05)
obs_params = ObsModelParams(0.005, 0.5, 0.01) # TODO

macro make_model(name, motion_model)
    return esc(quote
        @gen (static) function $(name)(T::Int)
            destination ~ uniform_coord()
            (path_rest, failed, tree) = plan_and_optimize_path(scene, start, Point(destination), planner_params)
            path = [start, path_rest...]
            {:measurements} ~ $(motion_model)(path, obs_params, T)
            return failed
        end
    end)
end

@make_model(uncollapsed, motion_and_measurement_model_uncollapsed)
@make_model(uncollapsed_incremental, motion_and_measurement_model_uncollapsed_incremental)
@make_model(pseudomarginal_1, make_motion_and_measurement_model_smc_pseudomarginal(1))
@make_model(pseudomarginal_3, make_motion_and_measurement_model_smc_pseudomarginal(3))
@make_model(pseudomarginal_10, make_motion_and_measurement_model_smc_pseudomarginal(10))
@make_model(pseudomarginal_100, make_motion_and_measurement_model_smc_pseudomarginal(100))
@make_model(collapsed, motion_and_measurement_model_collapsed)
@make_model(collapsed_incremental, motion_and_measurement_model_collapsed_incremental)

@gen (static) function prior_proposal(trace)
    destination ~ uniform_coord()
end

@load_generated_functions()

function get_apartment_dataset()
    failed = true
    T = 100
    attempt() = generate(uncollapsed, (T,), choicemap((:destination, [0.9, 0.1])))[1]
    failed(trace) = get_retval(trace)
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

function evaluate_acceptance_rate(model, motion_model, observations, n, reps; burn_in=100)
    T = length(observations)
    accepted = zeros(reps)

    for rep in 1:reps

        # obtain initial trace, starting from inferred destination
        obs_choices = choicemap()
        (trace, _) = generate(model, (1,), choicemap(
                (:measurements => obs_addr(motion_model, 1), [observations[1].x, observations[1].y])))
        for t in 2:T
            trace, = update(
                trace,
                (t,), (UnknownChange(),),
                choicemap((:measurements => obs_addr(motion_model, t), [observations[t].x, observations[t].y])))
        end

        # do burn-in mcmc moves over the encapsulated random choices
        println("burn-in..")
        @time for iter in 1:burn_in
            trace, acc = mh(trace, prior_proposal, ())
        end

        # do n MCMC moves from the same initial trace
        println("testing..")
        @time for i in 1:n
            _, acc = mh(trace, prior_proposal, ())
            accepted[rep] += acc
        end
        println("rep $rep; acceptance rate: $(accepted[rep]) / $n = $(accepted[rep] / n)")
    end
    rates = accepted / n
    return rates
end

function jitter(ax, x, data)
    x_jitter = x .+ (rand(length(data)) .- 0.5)  * 0.25
    ax.scatter(x_jitter, data, marker=".", s=20, alpha=0.5, color="black")
end

import JSON

function acceptance_rate_plots()
    Random.seed!(1)
    close("all")
    observations = get_apartment_dataset()
    T = length(observations)
    n = 200 # 100
    reps = 500 # 100

    # show the posterior
    posterior_num_particles = 1000
    println("obtaining initial destination samples")
    obs_choices = choicemap()
    for t in 1:T
        obs_choices[:measurements => obs_addr(motion_and_measurement_model_uncollapsed, t)] = [observations[t].x, observations[t].y]
    end
    posterior_dests = []
    for rep in 1:200
        @time trace, = importance_resampling(uncollapsed, (T,), obs_choices, posterior_num_particles)
        push!(posterior_dests, trace[:destination])
    end

    # show data set and posterior
    println(observations)
    figure(figsize=(6,6))
    scatter([pt.x for pt in observations], [pt.y for pt in observations], color="black", s=5)
    scatter([vec[1] for vec in posterior_dests], [vec[2] for vec in posterior_dests], color="red", s=5)
    ax = gca()
    for wall in scene.walls
        plot([wall.a.x, wall.b.x], [wall.a.y, wall.b.y], color="black")
    end
    gca().set_xlim(0, 1)
    gca().set_ylim(0, 1)
    tight_layout()
    savefig("dataset.png")

    # generate results
    results = Dict()

    println("collapsed (inc)...")
    results["collapsed-inc"] = evaluate_acceptance_rate(
        collapsed_incremental,
        motion_and_measurement_model_collapsed_incremental,
        observations, n, reps)

    println("uncollapsed (inc)...")
    results["uncollapsed-inc"] = evaluate_acceptance_rate(
        uncollapsed_incremental,
        motion_and_measurement_model_uncollapsed_incremental,
        observations, n, reps)

    println("pseudomarginal(1)...")
    results["pseudomarginal-1"] = evaluate_acceptance_rate(
        pseudomarginal_1,
        make_motion_and_measurement_model_smc_pseudomarginal(1),
        observations, n, reps)

    println("pseudomarginal(3)...")
    results["pseudomarginal-3"] = evaluate_acceptance_rate(
        pseudomarginal_3,
        make_motion_and_measurement_model_smc_pseudomarginal(3),
        observations, n, reps)

    println("pseudomarginal(10)...")
    results["pseudomarginal-10"] = evaluate_acceptance_rate(
        pseudomarginal_10,
        make_motion_and_measurement_model_smc_pseudomarginal(10),
        observations, n, reps)

    open("acceptance_rate_results.json", "w") do f
        JSON.print(f, results)
    end

    results = JSON.parsefile("acceptance_rate_results.json")
    

    figure()
    ax = gca()

    labels = String[]
    push!(labels, "collapsed (inc)")
    push!(labels, "uncollapsed (inc)")
    push!(labels, "pseudomarginal (10)")
    jitter(ax, 1, results["collapsed-inc"])
    jitter(ax, 2, results["uncollapsed-inc"])
    jitter(ax, 3, results["pseudomarginal-10"])

    #println("collapsed...")
    #results = evaluate_acceptance_rate(
        #collapsed,
        #motion_and_measurement_model_collapsed,
        #observations, n, reps)
    #@show push!(labels, "collapsed")
    #jitter(ax, length(labels), results)

    #println("pseudomarginal(100)...")
    #results = evaluate_acceptance_rate(
        #pseudomarginal_100,
        #make_motion_and_measurement_model_smc_pseudomarginal(100),
        #observations, n, reps)
    #@show push!(labels, "pseudomarginal (100)")
    #jitter(ax, length(labels), results)

    #println("uncollapsed...")
    #results = evaluate_acceptance_rate(
        #uncollapsed,
        #motion_and_measurement_model_uncollapsed,
        #observations, n, reps)
    #@show push!(labels, "uncollapsed")
    #jitter(ax, length(labels), results)


    gca().set_xticks(collect(1:length(labels)))
    gca().set_xticklabels(labels)
    savefig("acceptance_rates.png")
end

function lml_estimate_plots()

    Random.seed!(1)
    close("all")
    observations = get_apartment_dataset()
    #destination = [0.5, 0.5]
    path = Point[Point(0.1, 0.9), Point(0.5, 0.5)] # actually path
    T = length(observations)
    reps = 100

    # generate results
    results = Dict()

    println("collapsed (inc)...")
    results["collapsed-inc"] = evaluate_lml_given_path(
        motion_and_measurement_model_collapsed_incremental,
        observations, path, reps)

    #println("uncollapsed (inc)...")
    #results["uncollapsed-inc"] = evaluate_lml_given_path(
        #motion_and_measurement_model_uncollapsed_incremental,
        #observations, path, reps)

    println("pseudomarginal(1)...")
    results["pseudomarginal-1"] = evaluate_lml_given_path(
        make_motion_and_measurement_model_smc_pseudomarginal(1),
        observations, path, reps)

    println("pseudomarginal(3)...")
    results["pseudomarginal-3"] = evaluate_lml_given_path(
        make_motion_and_measurement_model_smc_pseudomarginal(3),
        observations, path, reps)

    println("pseudomarginal(10)...")
    results["pseudomarginal-10"] = evaluate_lml_given_path(
        make_motion_and_measurement_model_smc_pseudomarginal(10),
        observations, path, reps)

    println("pseudomarginal(100)...")
    results["pseudomarginal-100"] = evaluate_lml_given_path(
        make_motion_and_measurement_model_smc_pseudomarginal(100),
        observations, path, reps)

    open("lml_estimate_results.json", "w") do f
        JSON.print(f, results)
    end

    results = JSON.parsefile("lml_estimate_results.json")

    figure()
    ax = gca()

    labels = String[]
    push!(labels, "exact")
    #push!(labels, "uncollapsed (inc)")
    push!(labels, "PM (1)")
    push!(labels, "PM (3)")
    push!(labels, "PM (10)")
    push!(labels, "PM (100)")
    jitter(ax, 1, results["collapsed-inc"])
    #jitter(ax, 2, results["uncollapsed-inc"])
    jitter(ax, 2, results["pseudomarginal-1"])
    jitter(ax, 3, results["pseudomarginal-3"])
    jitter(ax, 4, results["pseudomarginal-10"])
    jitter(ax, 5, results["pseudomarginal-100"])
    gca().set_ylim(250, 700)
    gca().set_xticks(collect(1:length(labels)))
    gca().set_xticklabels(labels)
    ylabel("log marginal likelihood estimates")
    savefig("lml_estimates.png")
end

lml_estimate_plots()
#acceptance_rate_plots()
