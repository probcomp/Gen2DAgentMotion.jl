function get_gf_test_args()
    path = Point[Point(0, 0), Point(1, 0), Point(1, 1)]
    params = ObsModelParams(0.05, 0.3, 1.0)
    measurements = Point[Point(0.1, 0.1), Point(0.5, 0.1), Point(0.7, 0.8)]
    return (path, params, measurements)
end

@testset "simulate" begin
    (path, params, _) = get_gf_test_args()
    T = 3

    for (gen_fn, test_score) in [
            (Gen2DAgentMotion.motion_and_measurement_model_uncollapsed, false),
            (Gen2DAgentMotion.motion_and_measurement_model_uncollapsed_incremental, false),
            (Gen2DAgentMotion.motion_and_measurement_model_collapsed, true),
            (Gen2DAgentMotion.motion_and_measurement_model_collapsed_incremental, true)]

        # simulate
        trace = simulate(gen_fn, (path, params, T))

        # get exact log marginal likelhood of simulated data in trace
        measurements = Point[]
        for t in 1:T
            push!(measurements, Point(trace[Gen2DAgentMotion.obs_addr(gen_fn, t)]))
        end
        exact_lml = Gen2DAgentMotion.log_marginal_likelihood(
            params, path, measurements)

        @test get_args(trace) == (path, params, T)
        retval = get_retval(trace)
        @test length(retval) == T
        if test_score
            @test isapprox(get_score(trace), exact_lml)
        end
    end
end

function make_obs_choicemap(observations::Vector{Point}, gen_fn)
    constraints = choicemap()
    for t in 1:length(observations)
        constraints[obs_addr(gen_fn, t)] = [observations[t].x, observations[t].y]
    end
    return constraints
end

@testset "pseudomarginal" begin

    path, params, obs = get_gf_test_args()

    num_particles = 10000
    pseudomarginal_gen_fn = make_motion_and_measurement_model_smc_pseudomarginal(num_particles)

    # test that if we use the pseudomarginal SMC version, we get approximately
    # the right log marginal likelihoods
    obs1 = choicemap((obs_addr(pseudomarginal_gen_fn, 1), [obs[1].x, obs[1].y]))
    obs2 = choicemap((obs_addr(pseudomarginal_gen_fn, 2), [obs[2].x, obs[2].y]))
    obs3 = choicemap((obs_addr(pseudomarginal_gen_fn, 3), [obs[3].x, obs[3].y]))
    trace, log_weight_1 = generate(pseudomarginal_gen_fn, (path, params, 1), obs1)
    exact_lml = Gen2DAgentMotion.log_marginal_likelihood(params, path, obs[1:1])
    @test isapprox(log_weight_1, exact_lml, rtol=1e-2)

    argdiffs = (NoChange(), NoChange(), UnknownChange())
    trace, log_weight_2 = update(trace, (path, params, 2), argdiffs, obs2)
    exact_lml = Gen2DAgentMotion.log_marginal_likelihood(params, path, obs[1:2])
    @test isapprox(log_weight_1 + log_weight_2, exact_lml, rtol=1e-2)

    trace, log_weight_3 = update(trace, (path, params, 3), argdiffs, obs3)
    exact_lml = Gen2DAgentMotion.log_marginal_likelihood(params, path, obs[1:3])
    @test isapprox(log_weight_1 + log_weight_2 + log_weight_3, exact_lml, rtol=1e-2)
end

@testset "generate" begin
    path, params, obs = get_gf_test_args()
    T = length(obs)

    exact_lml = Gen2DAgentMotion.log_marginal_likelihood(params, path, obs)

    for gen_fn in [
            #Gen2DAgentMotion.motion_and_measurement_model_uncollapsed,
            Gen2DAgentMotion.motion_and_measurement_model_collapsed,
            Gen2DAgentMotion.motion_and_measurement_model_collapsed_incremental]

        constraints = make_obs_choicemap(obs, gen_fn)
        trace, log_weight = generate(gen_fn, (path, params, T), constraints)
        @test isapprox(log_weight, exact_lml)
        @test isapprox(get_score(trace), exact_lml)
        # check args
        # check the retval
        # TODO check other elements: prev_pt_idx, dist_past_prev_pt
    end

    # test that if we estimate the log marginal likelihood on the uncollapsed model 
    # using importance sampling, we get approximately the right answer
    num_particles = 10000
    constraints = make_obs_choicemap(obs, Gen2DAgentMotion.motion_and_measurement_model_uncollapsed)
    (_, _, lml_estimate) = importance_sampling(
        Gen2DAgentMotion.motion_and_measurement_model_uncollapsed,
        (path, params, T), constraints, num_particles)
    println("DML version: $num_particles lml_estimate: $lml_estimate, actual: $exact_lml")
    @test isapprox(lml_estimate, exact_lml, rtol=1e-2)
    constraints = make_obs_choicemap(obs, Gen2DAgentMotion.motion_and_measurement_model_uncollapsed_incremental)
    (_, _, lml_estimate) = importance_sampling(
        Gen2DAgentMotion.motion_and_measurement_model_uncollapsed_incremental,
        (path, params, T), constraints, num_particles)
    println("SML version: $num_particles lml_estimate: $lml_estimate, actual: $exact_lml")
    @test isapprox(lml_estimate, exact_lml, rtol=1e-2)
end

@testset "update change path" begin
    # old and new paths
    old_path, params, observations = get_gf_test_args()
    new_path = Point[Point(0.5, 0.5), Point(1.5, 0.5), Point(1.5, 1.5)]

    # compute log marginal likelihood for previous path
    old_lml = Gen2DAgentMotion.log_marginal_likelihood(params, old_path, observations)

    # compute log marginal likelihood for new path
    new_lml = Gen2DAgentMotion.log_marginal_likelihood(params, new_path, observations)

    init_constraints = choicemap(
        (:meas => 1, [observations[1].x, observations[1].y]), 
        (:meas => 2, [observations[2].x, observations[2].y]),
        (:meas => 3, [observations[3].x, observations[3].y])
    )

    for gen_fn in [
            Gen2DAgentMotion.motion_and_measurement_model_collapsed,
            Gen2DAgentMotion.motion_and_measurement_model_collapsed_incremental]
        println("gen_fn: $gen_fn")

        # obtain trace with old path in it
        trace, = generate(gen_fn, (old_path, params, 3), init_constraints)

        # update the path to the new path
        new_trace, log_weight, retdiff = update(
                trace, (new_path, params, 3),
                (UnknownChange(), UnknownChange(), NoChange()),
                choicemap())
    
        @test retdiff == UnknownChange()
    
        # check that new trace has the right values in it
        for t in 1:length(observations)
            @test new_trace[:meas => t] == [observations[t].x, observations[t].y]
        end
    
        # check the retval
        retval = get_retval(trace)
        @test length(retval) == length(observations)
    
        # check args in the trace
        @test get_args(new_trace) == (new_path, params, length(observations))
    
        # check the score
        @test isapprox(get_score(new_trace), new_lml)
    
        # check the weight
        @test isapprox(log_weight, new_lml - old_lml)
    end
end

@testset "update extension" begin

    path, params, obs = get_gf_test_args()
    T = length(obs)

    # compute the log marginal likelhoods for all prefixes of the data set
    lmls = [
        Gen2DAgentMotion.log_marginal_likelihood(params, path, obs[1:t])
        for t in 1:T
    ]
    
    # initial traces will have the first T1 observations
    # we will use update to extend to all T observations

    for T1 in 1:T

        init_constraints = choicemap()
        for t in 1:T1
            init_constraints[:meas => t] = [obs[t].x, obs[t].y]
        end

        update_constraints = choicemap()
        for t in (T1+1):T
            update_constraints[:meas => t] = [obs[t].x, obs[t].y]
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
                @test new_trace[:meas => t] == [obs[t].x, obs[t].y]
            end
    
            # check the retval
            retval = get_retval(new_trace)
            @test length(retval) == T
    
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
