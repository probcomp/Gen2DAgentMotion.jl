using Gen
using FunctionalCollections: PersistentVector, push, assoc

export PathProgress
export walk_path_step
export walk_path
export walk_path_incremental

export ObsModelParams
export motion_and_measurement_model_uncollapsed
export motion_and_measurement_model_uncollapsed_incremental
export obs_addr

function path_length(start::Point, path::Vector{Point})
    @assert length(path) > 0
    len = dist(start, path[1])
    for i in 1:length(path)-1
        len += dist(path[i], path[i+1])
    end
    return len
end

################
# walking path #
################

struct PathProgress
    prev_pt_idx::Int # index of path point
    dist_past_prev_pt::Float64 # distance traveled past previous path point
end

function Base.isapprox(a::PathProgress, b::PathProgress; kwargs...)
    if a.prev_pt_idx != b.prev_pt_idx
        return false
    end
    return isapprox(a.dist_past_prev_pt, b.dist_past_prev_pt; kwargs...)
end

PathProgress() = PathProgress(1, 0.0)

"""
    (progress::PathProgress, location::Point) = walk_path_step(
        path::AbstractVector{Point}, progress::PathProgress, distance::Float64)

Walk path for one time step.
"""
function walk_path_step(
        path::AbstractVector{Point},
        progress::PathProgress,
        distance::Float64)

    prev_pt_idx = progress.prev_pt_idx
    dist_past_prev_pt = progress.dist_past_prev_pt

    @assert 1 <= prev_pt_idx <= length(path)
    prev::Point = path[prev_pt_idx]
    local next::Point
    local dist_prev_to_next::Float64

    dist_remaining = distance 
    while true

        # if we have are at the last point and we still have distance to
        # travel, then stay at the last point
        if prev_pt_idx == length(path)
            return (PathProgress(prev_pt_idx, 0.0), prev)
        end

        # walk to the next point
        next = path[prev_pt_idx+1]
        dist_prev_to_next = dist(prev, next)
        dist_to_next = dist_prev_to_next - dist_past_prev_pt

        # break out if destination on the segment between prev and next
        if dist_to_next >= dist_remaining
            break
        end

        # set the next point as the previous point
        prev_pt_idx += 1
        dist_remaining -= dist_to_next
        dist_past_prev_pt = 0.0
        prev = next
    end

    # the next point overshoots our target distance so the desired point is
    # somewhere on the segment b/w prev and next.  get the point on the segment
    dist_past_prev_pt += dist_remaining
    @assert dist_past_prev_pt <= dist_prev_to_next
    frac = dist_past_prev_pt / dist_prev_to_next
    cur = Point(
        prev.x * (1.0 - frac) + next.x * frac,
        prev.y * (1.0 - frac) + next.y * frac)

    return (PathProgress(prev_pt_idx, dist_past_prev_pt), cur)
end

# TODO document me
function walk_path_incremental(
        path::Vector{Point}, speed::Float64,
        points::PersistentVector{Point}, num_additional_time_steps::Int,
        progress::PathProgress)::Tuple{PersistentVector{Point},PathProgress}

    prev_T = length(points)
    new_T = prev_T + num_additional_time_steps
    if prev_T == 0
        return walk_path(path, speed, new_T)
    end
    for t in 1:num_additional_time_steps
        (progress, point) = walk_path_step(
            path, progress, speed)
        points = push(points, point)
    end
    @assert length(points) == new_T
    return (points, progress)
end

"""
    walk_path(path::Vector{Point}, speed::Float64, T::Int)

Walk path for T time steps, recording the point at each time step
"""
function walk_path(path::Vector{Point}, speed::Float64, T::Int)
    points = Vector{Point}(undef, T)
    if T > 0
        points[1] = path[1]
    end
    progress = PathProgress()
    for t in 2:T    
        (progress, points[t]) = walk_path_step(path, progress, speed)
    end
    return (PersistentVector{Point}(points), progress)
end


############################################
# uncollapsed motion and measurement model #
############################################

struct ObsModelParams
    nominal_speed::Float64 # distance per time step
    walk_noise::Float64
    noise::Float64
end

prob_lag(params::ObsModelParams) = params.walk_noise/2
prob_skip(params::ObsModelParams) = params.walk_noise/2
prob_normal(params::ObsModelParams) = 1-params.walk_noise

function steps_prior_probs(params, t)
    if t == 1
        return [1.0, 0.0, 0.0]
    else
        return [prob_lag(params), prob_normal(params), prob_skip(params)]
    end
end

@dist function steps_prior(probs)
	index = categorical(probs)
    labels = [0, 1, 2]
    labels[index]
end

@gen (static) function record_locations(locations)
    return locations
end

@gen function motion_and_measurement_model_uncollapsed(
        path::AbstractVector{Point}, params::ObsModelParams, T::Int)

    T < 1 && error("expected T >= 1, got T = $T")
    cov = (params.noise^2) * [1.0 0.0; 0.0 1.0]

    _locations = PersistentVector{Point}() # ground truth locations
    measurements = PersistentVector{Point}() # measured locations

    # for each time step, advance some number of steps
    progress = PathProgress()
    for t in 1:T
        steps = ({(:steps => t)} ~ steps_prior(steps_prior_probs(params, t)))
        distance = steps * params.nominal_speed
        (progress, location) = walk_path_step(path, progress, distance)
        measurement = Point(({:meas => t} ~ mvnormal([location.x, location.y], cov)))
        _locations = push(_locations, location)
        measurements = push(measurements, measurement)
    end
    locations ~ record_locations(_locations)
    return measurements
end

obs_addr(::Any, t::Int) = (:meas => t)

##################################################
# uncollapsed motion and measurement model (SML) #
##################################################

struct MotionMeasurementState
    progress::PathProgress
    locations::PersistentVector{Point}
    measurements::PersistentVector{Point}
end

@gen (static) function step(
        t_minus_one::Int, prev_state::MotionMeasurementState,
        path::AbstractVector{Point}, params::ObsModelParams)
    cov = (params.noise^2) * [1.0 0.0; 0.0 1.0]
    t = t_minus_one + 1
    steps ~ steps_prior(steps_prior_probs(params, t))
    (progress, location) = walk_path_step(path, prev_state.progress, steps * params.nominal_speed)
    locations = push(prev_state.locations, location)
    measurement ~ mvnormal([location.x, location.y], cov)
    measurements = push(prev_state.measurements, Point(measurement))
    return MotionMeasurementState(progress, locations, measurements)
end

@gen (static) function motion_and_measurement_model_uncollapsed_incremental(
        path::AbstractVector{Point}, params::ObsModelParams, T::Int)
    cov = (params.noise^2) * [1.0 0.0; 0.0 1.0]

    # extra tracing (for visualization)
    locations = PersistentVector{Point}() # ground truth locations
    measurements = PersistentVector{Point}() # measured locations

    # initial location
    progress = PathProgress()
    location = path[1]
    locations = push(locations, location)

    # initial measurement
    init_steps ~ steps_prior(steps_prior_probs(params, 1))
    init_measurement ~ mvnormal([location.x, location.y], cov)
    measurements = push(measurements, Point(init_measurement))

    # initial state
    init_state = MotionMeasurementState(progress, locations, measurements)

    # remaining time steps
    rest_states ~ (Unfold(step))(T-1, init_state, path, params)

    # final state
    final_state = (T == 1) ? init_state : rest_states[T-1]

    locations ~ record_locations(final_state.locations)

    return measurements
end

@load_generated_functions()

function obs_addr(::typeof(motion_and_measurement_model_uncollapsed_incremental), t::Int)
    if t == 1
        return :init_measurement
    else
        return :rest_states => (t-1) => :measurement
    end
end
