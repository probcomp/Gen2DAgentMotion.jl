@testset "walk path" begin
    
    # start at first point
    (progress, cur) = Gen2DAgentMotion.walk_path_step(
        Point[Point(0, 0), Point(0, 1)], PathProgress(1, 0.0), 0.1)
    @test isapprox(progress, PathProgress(1, 0.1))
    @test isapprox(cur, Point(0, 0.1))

    # start after first point
    (progress, cur) = Gen2DAgentMotion.walk_path_step(
        Point[Point(0, 0), Point(0, 1)], PathProgress(1, 0.1), 0.2)
    @test isapprox(progress, PathProgress(1, 0.3))
    @test isapprox(cur, Point(0, 0.3))

    # start after first point and walk past a point
    (progress, cur) = Gen2DAgentMotion.walk_path_step(
        Point[Point(0, 0), Point(0, 1), Point(1, 1)], PathProgress(1, 0.1), 1.1)
    @test isapprox(progress, PathProgress(2, 0.2))
    @test isapprox(cur, Point(0.2, 1))

    # walk to the end
    (progress, cur) = Gen2DAgentMotion.walk_path_step(
        Point[Point(0, 0), Point(0, 1), Point(1, 1)], PathProgress(1, 0.1), 3.0)
    @test isapprox(progress, PathProgress(3, 0.0))
    @test isapprox(cur, Point(1, 1))

    # walk exactly to a point (edge case) and then again exactly to the next point (edge case)
    path = Point[Point(0, 0), Point(0, 1), Point(1, 1)]
    (progress, cur) = Gen2DAgentMotion.walk_path_step(path, PathProgress(1, 0.1), 0.9)
    @test isapprox(cur, Point(0, 1))
    (progress, cur) = Gen2DAgentMotion.walk_path_step(path, progress, 1.0)
    @test isapprox(cur, Point(1, 1))

    function test_walk_path(path, speed, expected_points)
        for T in 0:length(expected_points)
            (points, progress) = Gen2DAgentMotion.walk_path(path, speed, T)
            @test length(points) == T
            for t in 1:T
                @test isapprox(points[t], expected_points[t])
            end
        end
    end

    function test_walk_path_incremental(path, speed, expected_points)
        T = length(expected_points)
        # T1 is number of initial steps
        for T1 in 0:T

            # T2 is number of incremental steps
            for T2 in 1:(length(expected_points)-T1)
                (points, progress) = Gen2DAgentMotion.walk_path(path, speed, T1)
                @test length(points) == T1
            
                for i in 1:T2
                    (points, progress) = Gen2DAgentMotion.walk_path_incremental(
                        path, speed, points, 1, progress)
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

    # case where the distance to walk brings you exactly to a path point
    path = Point[Point(0, 0), Point(0, 1), Point(1, 1)]
    speed = 0.5
    expected_points = Point[
        Point(0, 0), Point(0, 0.5), Point(0, 1), Point(0.5, 1),
        Point(1, 1), Point(1, 1), Point(1, 1)]
    test_walk_path(path, speed, expected_points)
    test_walk_path_incremental(path, speed, expected_points)

    # case where the input path has some consecutive points
    path = Point[Point(0, 0), Point(0, 1), Point(0, 1), Point(0, 1), Point(1, 1), Point(1, 1)]
    speed = 0.5
    expected_points = Point[
        Point(0, 0), Point(0, 0.5), Point(0, 1), Point(0.5, 1),
        Point(1, 1), Point(1, 1), Point(1, 1)]
    test_walk_path(path, speed, expected_points)
    test_walk_path_incremental(path, speed, expected_points)
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
    expected = 0.0
    # 1-A, 2-A
    expected += prob_lag * likelihood(A, obs1) * likelihood(A, obs2)
    # 1-A, 2-B
    expected += prob_normal * likelihood(A, obs1) * likelihood(B, obs2)
    # 1-A, 2-C
    expected += prob_skip * likelihood(A, obs1) * likelihood(C, obs2)
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
    (alphas, _) = Gen2DAgentMotion.forward_filtering(params, [A], [obs1])
    (alphas, _) = Gen2DAgentMotion.forward_filtering_incremental(params, [A, B], [obs1, obs2], alphas)
    (alphas, _) = Gen2DAgentMotion.forward_filtering_incremental(params, [A, B, C], [obs1, obs2, obs3], alphas)
    (alphas, actual) = Gen2DAgentMotion.forward_filtering_incremental(params, [A, B, C, D], [obs1, obs2, obs3, obs4], alphas)
    @test isapprox(actual, expected)
end

@testset "sample_locations_exact_conditional" begin
    # confirm that a single sample importance sampling using this as the proposal returns the exact log marginal likelihood
    # (the samples look reasonable, but would be good to test)
    # TODO
end
