@testset "dynamic programming" begin

    nominal_speed = 0.1
    walk_noise = 0.2
    prob_lag = 0.2
    prob_skip = 0.2
    prob_normal = 0.6
    walk_noise = prob_lag + prob_skip
    noise = 1.0
    obs_params = ObsModelParams(nominal_speed, walk_noise, noise)
    likelihood(a, b) = exp(Gen2DAgentMotion.noise_log_likelihood(obs_params, a, b))
    A = Point(0,0)
    B = Point(1,1)
    C = Point(2,2)
    obs1 = Point(0,0)
    obs2 = Point(1,1)
    actual = Gen2DAgentMotion.log_marginal_likelihood(obs_params, [A, B, C], [obs1, obs2])
    first_prob_normal = prob_lag + prob_normal
    first_prob_skip = prob_skip
    expected = 0.0
    # 1-A, 2-A
    expected += first_prob_normal * prob_lag * likelihood(A, obs1) * likelihood(A, obs2)
    # 1-A, 2-B
    expected += first_prob_normal * prob_normal * likelihood(A, obs1) * likelihood(B, obs2)
    # 1-A, 2-C
    expected += first_prob_normal * prob_skip * likelihood(A, obs1) * likelihood(C, obs2)
    # 1-B, 2-B
    expected += first_prob_skip * prob_lag * likelihood(B, obs1) * likelihood(B, obs2)
    # 1-B, 2-C
    expected += first_prob_skip * prob_normal * likelihood(B, obs1) * likelihood(C, obs2)
    @test isapprox(actual, log(expected))

end
