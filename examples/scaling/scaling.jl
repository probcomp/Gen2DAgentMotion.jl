using Gen
using Gen2DAgentMotion
using PyPlot
using Statistics: median

function evaluate_update_scaling(motion_model, path, params, observations, reps)
    elapsed = Matrix{Float64}(undef, length(observations), reps)
    for rep in 1:reps
        println("$motion_model; rep: $rep")
        obs = choicemap((obs_addr(motion_model, 1), [observations[1].x, observations[1].y]))
        start = time_ns()
        (trace, _) = generate(motion_model, (path, params, 1), obs)
        elapsed[1, rep] = (time_ns() - start)/1e9
        for t in 2:length(observations)
            obs = choicemap((obs_addr(motion_model, t), [observations[t].x, observations[t].y]))
            start = time_ns()
            (trace, _) = update(
                trace, (path, params, t), (NoChange(), NoChange(), UnknownChange()), obs)
            _elapsed = (time_ns() - start)/1e9
            elapsed[t, rep] = elapsed[t-1, rep] + _elapsed
        end
    end
    return median(elapsed; dims=2)[:]
end

function plots()
    path = Point[Point(0, 0), Point(1, 0), Point(1, 1)]
    params = ObsModelParams(0.05, 0.3, 1.0)
    observations = Point[Point(i, i) for i in 1:500]
    reps = 5
    figure()
    pseudomarginal = evaluate_update_scaling(
        make_motion_and_measurement_model_smc_pseudomarginal(10),
        path, params, observations, reps)
    uncollapsed = evaluate_update_scaling(
        motion_and_measurement_model_uncollapsed,
        path, params, observations, reps)
    uncollapsed_incremental = evaluate_update_scaling(
        motion_and_measurement_model_uncollapsed_incremental,
        path, params, observations, reps)
    collapsed = evaluate_update_scaling(
        motion_and_measurement_model_collapsed,
        path, params, observations, reps)
    collapsed_incremental = evaluate_update_scaling(
        motion_and_measurement_model_collapsed_incremental,
        path, params, observations, reps)
    plot(pseudomarginal, label="pseudomarginal")
    plot(uncollapsed, label="uncollapsed")
    plot(uncollapsed_incremental, label="uncollapsed, incremental")
    plot(collapsed, label="collapsed")
    plot(collapsed_incremental, label="collapsed, incremental")
    legend()
    gca().set_ylim((0, 1))
    savefig("scaling.png")
end

plots()
