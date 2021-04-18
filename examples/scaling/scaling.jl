using Gen
using Gen2DAgentMotion
using PyPlot
using Statistics: median
using LaTeXStrings

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
    T = 1000
    observations = Point[Point(i, i) for i in 1:T]
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
    plot(log.(1:T), log.(pseudomarginal), label="pseudomarginal")
    plot(log.(1:T), log.(uncollapsed), label="uncollapsed")
    plot(log.(1:T), log.(uncollapsed_incremental), label="uncollapsed, incremental")
    plot(log.(1:T), log.(collapsed), label="collapsed")
    plot(log.(1:T), log.(collapsed_incremental), label="collapsed, incremental")
    xlim = gca().get_xlim()
    linear = collect(1:T)
    quadratic = collect(1:T).^2
    cubic = collect(1:T).^3
    plot(log.(1:T), log.(linear) .- 10, color="black", linestyle="--")
    plot(log.(1:T), log.(quadratic) .- 10, color="black", linestyle="--")
    plot(log.(1:T), log.(cubic) .- 10, color="black", linestyle="--")
    annotate(L"$O(T)$", (log(T), log(linear[end])))
    annotate(L"$O(T^2)$", (log(T), log(quadratic[end])))
    annotate(L"$O(T^3)$", (log(T), log(cubic[end])))
    legend()
    #gca().set_ylim((0, 1))
    savefig("scaling.png")
end

plots()
