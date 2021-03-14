function get_gf_test_args()
    path = Point[Point(0, 0), Point(1, 0), Point(1, 1)]
    params = ObsModelParams(0.05, 0.2, 1.0)
    obs = Point[Point(0.1, 0.1), Point(0.5, 0.1), Point(0.7, 0.8)]
    (points, _, _) = Gen2DAgentMotion.walk_path(path, params.nominal_speed, 3)
    return (path, params, obs, points)
end

@testset "simulate" begin
    (path, params, _, _) = get_gf_test_args()
    T = 3
    for (gen_fn, test_score) in [
            (Gen2DAgentMotion.motion_and_measurement_model_uncollapsed, false),
            (Gen2DAgentMotion.motion_and_measurement_model_collapsed, true),
            (Gen2DAgentMotion.motion_and_measurement_model_collapsed_incremental, true)]
        trace = simulate(gen_fn, (path, params, T))
        obs = Point[]
        for t in 1:T
            push!(obs, Point(trace[(:x, t)], trace[(:y, t)]))
        end
        (points, _, _) = Gen2DAgentMotion.walk_path(path, params.nominal_speed, T)
        @test get_args(trace) == (path, params, T)
        retval = get_retval(trace)
        @test length(retval) == 4
        @test length(retval[1]) == T
        @test length(retval[2]) == T
        if test_score
            exact_lml = Gen2DAgentMotion.log_marginal_likelihood(params, points, obs)
            @test isapprox(get_score(trace), exact_lml)
        end
    end
end

@testset "generate" begin
    (path, params, obs, points) = get_gf_test_args()
    T = length(obs)

    constraints = choicemap()
    for t in 1:T
        constraints[(:x, t)] = obs[t].x
        constraints[(:y, t)] = obs[t].y
    end

    exact_lml = Gen2DAgentMotion.log_marginal_likelihood(params, points, obs)

    for gen_fn in [
            #Gen2DAgentMotion.motion_and_measurement_model_uncollapsed,
            Gen2DAgentMotion.motion_and_measurement_model_collapsed,
            Gen2DAgentMotion.motion_and_measurement_model_collapsed_incremental]
        trace, log_weight = generate(gen_fn, (path, params, T), constraints)
        @test isapprox(log_weight, exact_lml)
        @test isapprox(get_score(trace), exact_lml)
        # check args
        # check the retval
        # TODO check other elements: prev_pt_idx, dist_past_prev_pt
    end

    # test that if we estimate the log marginal likelihood on the uncollapsed model 
    # using importance sampling, we get approximately the right answer
    # TODO there is a small discrepancy, perhaps due to how the end of the sequence is handled
    # XXX Resolve this..
    # it's possible there's a bug in all of (i) the manual test case above,
    # (ii) the core log marignal likelihood calculation
    for num_particles in [1, 10, 100, 1000, 10000]
        (_, _, lml_estimate) = importance_sampling(
            Gen2DAgentMotion.motion_and_measurement_model_uncollapsed,
            (path, params, T), constraints, num_particles)
        println("$num_particles lml_estimate: $lml_estimate, actual: $exact_lml")
    end
end

@testset "update change path" begin
    (path, params, obs, points) = get_gf_test_args()
    old_lml = Gen2DAgentMotion.log_marginal_likelihood(params, points, obs)
    new_path = Point[Point(0.5, 0.5), Point(1.5, 0.5), Point(1.5, 1.5)]
    (new_points, _, _) = Gen2DAgentMotion.walk_path(new_path, params.nominal_speed, length(obs))
    new_lml = Gen2DAgentMotion.log_marginal_likelihood(params, new_points, obs)

    init_constraints = choicemap(
        ((:x, 1), obs[1].x), 
        ((:y, 1), obs[1].y),
        ((:x, 2), obs[2].x),
        ((:y, 2), obs[2].y),
        ((:x, 3), obs[3].x),
        ((:y, 3), obs[3].y)
    )

    for gen_fn in [
            Gen2DAgentMotion.motion_and_measurement_model_collapsed,
            Gen2DAgentMotion.motion_and_measurement_model_collapsed_incremental]

        trace, = generate(gen_fn, (path, params, 3), init_constraints)
        new_trace, log_weight, retdiff = update(
                trace, (new_path, params, 3), (UnknownChange(), UnknownChange(), NoChange()),
                choicemap())
    
        @test retdiff == UnknownChange()
    
        # check that new trace has the right values in it
        for t in 1:length(obs)
            @test new_trace[(:x, t)] == obs[t].x
            @test new_trace[(:y, t)] == obs[t].y
        end
    
        # check the retval
        retval = get_retval(trace)
        @test length(retval) == 4
        @test length(retval[1]) == 3
        @test length(retval[2]) == 3
        # TODO check other elements: prev_pt_idx, dist_past_prev_pt
    
        # check args in the trace
        @test get_args(new_trace) == (new_path, params, length(obs))
    
        # check the score
        @test isapprox(get_score(new_trace), new_lml)
    
        # check the weight
        @test isapprox(log_weight, new_lml - old_lml)
    end
end

@testset "update extension" begin

    (path, params, obs, points) = get_gf_test_args()
    T = length(obs)

    # compute the log marginal likelhoods for all prefixes of the data set
    lmls = [
        Gen2DAgentMotion.log_marginal_likelihood(params, points[1:t], obs[1:t])
        for t in 1:T
    ]
    
    # initial traces will have the first T1 observations
    # we will use update to extend to all T observations

    for T1 in 1:T

        init_constraints = choicemap()
        for t in 1:T1
            init_constraints[(:x, t)] = obs[t].x
            init_constraints[(:y, t)] = obs[t].y
        end

        update_constraints = choicemap()
        for t in (T1+1):T
            update_constraints[(:x, t)] = obs[t].x
            update_constraints[(:y, t)] = obs[t].y
        end

        for gen_fn in [
                Gen2DAgentMotion.motion_and_measurement_model_collapsed,
                Gen2DAgentMotion.motion_and_measurement_model_collapsed_incremental]

            trace, = generate(gen_fn, (path, params, T1), init_constraints)
    
            # do the update
            new_trace, log_weight, retdiff = update(
                trace, (path, params, T), (NoChange(), NoChange(), UnknownChange()),
                update_constraints)
    
            @test retdiff == UnknownChange()
    
            # check that new trace has the right values in it
            for t in 1:T
                @test new_trace[(:x, t)] == obs[t].x
                @test new_trace[(:y, t)] == obs[t].y
            end
    
            # check the retval
            retval = get_retval(new_trace)
            @test length(retval) == 4
            @test length(retval[1]) == T
            @test length(retval[2]) == T
            # TODO check other elements
            #(trace.points, trace.obs, trace.prev_pt_idx, trace.dist_past_prev_pt)
    
            # check args in the trace
            @test get_args(new_trace) == (path, params, T)
    
            # check the score
            @test isapprox(get_score(new_trace), lmls[T])
    
            # check the weight
            @test isapprox(log_weight, lmls[T] - lmls[T1])
        end
    end
end

# TODO test sample_alignment
