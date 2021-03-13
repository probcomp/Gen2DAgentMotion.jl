@testset "walk path" begin
    
    # start at first point
    (prev_pt_idx, dist_past_prev_pt, cur) = Gen2DAgentMotion.walk_path_step(
        Point[Point(0, 0), Point(0, 1)], 1, 0.0, 0.1)
    @test prev_pt_idx == 1
    @test isapprox(dist_past_prev_pt, 0.1)
    @test isapprox(cur, Point(0, 0.1))

    # start after first point
    (prev_pt_idx, dist_past_prev_pt, cur) = Gen2DAgentMotion.walk_path_step(
        Point[Point(0, 0), Point(0, 1)], 1, 0.1, 0.2)
    @test prev_pt_idx == 1
    @test isapprox(dist_past_prev_pt, 0.3)
    @test isapprox(cur, Point(0, 0.3))

    # start after first point and walk past a point
    (prev_pt_idx, dist_past_prev_pt, cur) = Gen2DAgentMotion.walk_path_step(
        Point[Point(0, 0), Point(0, 1), Point(1, 1)], 1, 0.1, 1.1)
    @test prev_pt_idx == 2
    @test isapprox(dist_past_prev_pt, 0.2)
    @test isapprox(cur, Point(0.2, 1))

    # walk to the end
    (prev_pt_idx, dist_past_prev_pt, cur) = Gen2DAgentMotion.walk_path_step(
        Point[Point(0, 0), Point(0, 1), Point(1, 1)], 1, 0.1, 3.0)
    @test prev_pt_idx == 3
    @test isapprox(dist_past_prev_pt, 0.0)
    @test isapprox(cur, Point(1, 1))

    # walk exactly to a point (edge case) and then again exactly to the next point (edge case)
    path = Point[Point(0, 0), Point(0, 1), Point(1, 1)]
    (prev_pt_idx, dist_past_prev_pt, cur) = Gen2DAgentMotion.walk_path_step(
        path, 1, 0.1, 0.9)
    @test isapprox(cur, Point(0, 1))
    (prev_pt_idx, dist_past_prev_pt, cur) = Gen2DAgentMotion.walk_path_step(
        path, prev_pt_idx, dist_past_prev_pt, 1.0)
    @test isapprox(cur, Point(1, 1))

    function test_walk_path(path, speed, expected_points)
        for T in 0:length(expected_points)
            (points, _, _) = Gen2DAgentMotion.walk_path(path, speed, T)
            @test length(points) == T
            for t in 1:T
                @test isapprox(points[t], expected_points[t])
            end
        end
    end

    #function test_walk_path_to_end(path, speed, expected_points)
        #points = Gen2DAgentMotion.walk_path(path, speed)
        #@test length(points) == length(expected_points)
        #for t in 1:length(expected_points)
            #@test isapprox(points[t], expected_points[t])
        #end
    #end

    function test_walk_path_incremental(path, speed, expected_points)
        T = length(expected_points)
        # T1 is number of initial steps
        for T1 in 0:T

            # T2 is number of incremental steps
            for T2 in 1:(length(expected_points)-T1)
                (points, prev_pt_idx, dist_past_prev_pt) = Gen2DAgentMotion.walk_path(path, speed, T1)
                @test length(points) == T1
            
                for i in 1:T2
                    (points, prev_pt_idx, dist_past_prev_pt) = Gen2DAgentMotion.walk_path_incremental(
                        path, speed, points, 1, prev_pt_idx, dist_past_prev_pt)
                end
                @test length(points) == T1 + T2

                for t in 1:(T1 + T2)
                    @test isapprox(points[t], expected_points[t])
                end
            end
        end
    end
    
    # non-edge case
    path = Point[Point(0, 0), Point(0, 1), Point(1, 1)]
    speed = 0.3
    expected_points = Point[
        Point(0, 0), Point(0, 0.3), Point(0, 0.6), Point(0, 0.9),
        Point(0.2, 1), Point(0.5, 1), Point(0.8, 1),
        Point(1, 1), Point(1, 1), Point(1, 1)]
    test_walk_path(path, speed, expected_points)
    test_walk_path_incremental(path, speed, expected_points)
    #test_walk_path_to_end(path, speed, expected_points)

    # case where the distance to walk brings you exactly to a path point
    path = Point[Point(0, 0), Point(0, 1), Point(1, 1)]
    speed = 0.5
    expected_points = Point[
        Point(0, 0), Point(0, 0.5), Point(0, 1), Point(0.5, 1),
        Point(1, 1), Point(1, 1), Point(1, 1)]
    test_walk_path(path, speed, expected_points)
    test_walk_path_incremental(path, speed, expected_points)
    #test_walk_path_to_end(path, speed, expected_points)

    # case where the input path has some consecutive points
    path = Point[Point(0, 0), Point(0, 1), Point(0, 1), Point(0, 1), Point(1, 1), Point(1, 1)]
    speed = 0.5
    expected_points = Point[
        Point(0, 0), Point(0, 0.5), Point(0, 1), Point(0.5, 1),
        Point(1, 1), Point(1, 1), Point(1, 1)]
    test_walk_path(path, speed, expected_points)
    test_walk_path_incremental(path, speed, expected_points)
    #test_walk_path_to_end(path, speed, expected_points)
end

@testset "forward filtering" begin

    nominal_speed = 0.1
    walk_noise = 0.2
    prob_lag = 0.2
    prob_skip = 0.2
    prob_normal = 0.6
    walk_noise = prob_lag + prob_skip
    noise = 1.0
    params = ObsModelParams(nominal_speed, walk_noise, noise)

    # test the log marginal likelihood against hand-computed value

    likelihood(a, b) = exp(Gen2DAgentMotion.noise_log_likelihood(params, a, b))
    A = Point(0,0)
    B = Point(1,1)
    C = Point(2,2)
    obs1 = Point(0,0)
    obs2 = Point(1,1)
    (alphas, actual) = Gen2DAgentMotion.forward_filtering(params, [A, B, C], [obs1, obs2])
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

@testset "incremental forward filtering" begin

    nominal_speed = 0.1
    walk_noise = 0.2
    prob_lag = 0.2
    prob_skip = 0.2
    prob_normal = 0.6
    walk_noise = prob_lag + prob_skip
    noise = 1.0
    params = ObsModelParams(nominal_speed, walk_noise, noise)

    A = Point(0,0)
    B = Point(1,1)
    C = Point(2,2)
    D = Point(2,3)
    obs1 = Point(0,0)
    obs2 = Point(1,1)
    obs3 = Point(1,2)
    obs4 = Point(2,3)

    # batch
    (_, expected) = Gen2DAgentMotion.forward_filtering(params, [A, B, C, D], [obs1, obs2, obs3, obs4])

    # check that we get the same lml if we compute it in batch versus incremetnally
    (alphas, lml1) = Gen2DAgentMotion.forward_filtering(params, [A], [obs1])
    (alphas, lml2) = Gen2DAgentMotion.forward_filtering_incremental(params, [A, B], [obs1, obs2], alphas)
    (alphas, lml3) = Gen2DAgentMotion.forward_filtering_incremental(params, [A, B, C], [obs1, obs2, obs3], alphas)
    (alphas, lml4) = Gen2DAgentMotion.forward_filtering_incremental(params, [A, B, C, D], [obs1, obs2, obs3, obs4], alphas)
    @test isapprox(lml1 + lml2 + lml3 + lml4, expected)
end
