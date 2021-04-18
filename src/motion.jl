using Gen
using FunctionalCollections: PersistentVector, push, assoc

export PathProgress
export walk_path_step
export walk_path
export walk_path_incremental

export ObsModelParams
export motion_and_measurement_model_uncollapsed
export motion_and_measurement_model_uncollapsed_incremental
export motion_and_measurement_model_collapsed
export motion_and_measurement_model_collapsed_incremental
export obs_addr

export log_marginal_likelihood
export sample_alignment

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

# TODO remove locations and progress from the return value spec!
# we could always implement getters for get_locations()
# for the collapsed ones it would need to sample..

struct ObsModelParams
    nominal_speed::Float64 # distance per time step
    walk_noise::Float64
    noise::Float64
end

prob_lag(params::ObsModelParams) = params.walk_noise/2
prob_skip(params::ObsModelParams) = params.walk_noise/2
prob_normal(params::ObsModelParams) = 1-params.walk_noise

function steps_prior(params, t)
    if t == 1
        probs = [1.0, 0.0, 0.0]
    else
        probs = [prob_lag(params), prob_normal(params), prob_skip(params)]
    end
end

@dist function sample_steps(probs)
	index = categorical(probs)
    labels = [0, 1, 2]
    labels[index]
end

@gen function motion_and_measurement_model_uncollapsed(
        path::AbstractVector{Point}, params::ObsModelParams, T::Int)

    T < 1 && error("expected T >= 1, got T = $T")
    cov = (params.noise^2) * [1.0 0.0; 0.0 1.0]

    locations = PersistentVector{Point}() # ground truth locations
    measurements = PersistentVector{Point}() # measured locations

    # for each time step, advance some number of steps
    progress = PathProgress()
    for t in 1:T
        steps = ({(:steps => t)} ~ sample_steps(steps_prior(params, t)))
        distance = steps * params.nominal_speed
        (progress, location) = walk_path_step(path, progress, distance)
        measurement = Point(({:meas => t} ~ mvnormal([location.x, location.y], cov)))
        locations = push(locations, location)
        measurements = push(measurements, measurement)
    end
    return measurements
end

# TODO

# Q1: how can we attach custom getters to traces of specific generative functions, including DML functions?
# example: getters that define what addresses are used for certain data
# A: this requires a code change to Gen. we should be able to dispatch on the
# julia function inside.. the DML function?

# Q2: how can we attach getters that read from subtraces..
# A: we can use auxiliary state

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
    steps ~ sample_steps(steps_prior(params, t))
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
    init_steps ~ sample_steps(steps_prior(params, 1))
    init_measurement ~ mvnormal([location.x, location.y], cov)
    measurements = push(measurements, Point(init_measurement))

    # initial state
    init_state = MotionMeasurementState(progress, locations, measurements)

    # remaining time steps
    rest_states ~ (Unfold(step))(T-1, init_state, path, params)

    # final state
    final_state = (T == 1) ? init_state : rest_states[T-1]

    return (final_state.locations, final_state.measurements, final_state.progress)
end

@load_generated_functions()

function obs_addr(::typeof(motion_and_measurement_model_uncollapsed_incremental), t::Int)
    if t == 1
        return :init_measurement
    else
        return :rest_states => (t-1) => :measurement
    end
end

############################################
# hand-coded dynamic programming inference #
############################################

# the observed points are generated by walking along the 
# path at a nominal speed, with random pauses and skips

@inline function noise_log_likelihood(params::ObsModelParams, location::Point, obs_pt::Point)
    ll = logpdf(normal, obs_pt.x, location.x, params.noise)
    ll += logpdf(normal, obs_pt.y, location.y, params.noise)
    return ll
end

@inline function compute_alpha_entry(params::ObsModelParams, location::Point, obs::Point, prev_alpha, k::Int)
    # observation model: obs corresponds with points_along_path
    noise_ll = noise_log_likelihood(params, location, obs)

    # dynamics model: given that we are k, the previous could have been
    # (i) k - 2 (skip one), (ii) k - 1 (advance as usual), or (iii) k
    # (don't advance, lag)
    if k > 2
        val1 = prev_alpha[k-2] + log(prob_skip(params))
    else
        val1 = -Inf
    end
    if k > 1
        val2 = prev_alpha[k-1] + log(prob_normal(params))
    else
        val2 = -Inf
    end
    val3 = prev_alpha[k] + log(prob_lag(params))
    if val1 == -Inf && val2 == -Inf && val3 == -Inf
        return -Inf
    end
    max_val = max(val1, val2, val3)

    # combine observation and dynamics terms
    return noise_ll + max_val + log(exp(val1 - max_val) + exp(val2 - max_val) + exp(val3 - max_val))
end

@inline function populate_new_alpha!(
        new_alpha, prev_alpha, points_along_path::AbstractVector{Point}, obs_t::Point, params::ObsModelParams)
    @assert length(new_alpha) == length(points_along_path)
    @assert length(prev_alpha) == length(points_along_path)
    for k in 1:length(points_along_path)
        new_alpha[k] = compute_alpha_entry(params, points_along_path[k], obs_t, prev_alpha, k)
    end
    return nothing
end

@inline function forward_transition_log_probs(next_z::Int, params::ObsModelParams, K::Int)
    log_probs = fill(-Inf, K)
    if next_z == 1
        log_probs[1] = log(prob_lag(params))
    elseif next_z == 2
        log_probs[1] = log(prob_normal(params))
        log_probs[2] = log(prob_lag(params))
    else
        log_probs[next_z] = log(prob_lag(params))
        log_probs[next_z-1] = log(prob_normal(params))
        log_probs[next_z-2] = log(prob_skip(params))
    end
    return log_probs
end

@inline function populate_initial_alpha!(
        alpha, params::ObsModelParams, points_along_path::AbstractVector{Point}, obs1::Point)
    @assert length(alpha) == length(points_along_path)
    fill!(alpha, -Inf)
    # always starts at the first location
    alpha[1] = 0.0 + noise_log_likelihood(params, points_along_path[1], obs1)
    return nothing
end

# non-incremental forward algorithm
function forward_filtering(
        params::ObsModelParams, points_along_path::AbstractVector{Point}, obs::AbstractVector{Point})

    K = length(points_along_path)
    T = length(obs)

    # allocate memory
    alphas = Vector{Vector{Float64}}(undef, T)
    for t in 1:T
        alphas[t] = Vector{Float64}(undef, K)
    end

    populate_initial_alpha!(alphas[1], params, points_along_path, obs[1])
    for t in 2:T
        populate_new_alpha!(alphas[t], alphas[t-1], points_along_path, obs[t], params)
    end
    log_marginal_likelihood = logsumexp(alphas[T])

    # make persistent
    alphas_persistent_vec = Vector{PersistentVector{Float64}}(undef, T)
    for t in 1:T
        alphas_persistent_vec[t] = PersistentVector{Float64}(alphas[t])
    end
    alphas_persistent = PersistentVector{PersistentVector{Float64}}(alphas_persistent_vec)
    return (alphas_persistent, log_marginal_likelihood)
end

# backwards sampling, given alphas from forward algorithm
function backwards_sampling(
        params::ObsModelParams, points_along_path::Vector{Point}, obs::Vector{Point},
        alphas)

    K = length(points_along_path)
    T = length(obs)

    alignment = Vector{Int}(undef, length(points_along_path))
    ldist = alphas[T]
    dist = exp.(ldist .- logsumexp(ldist))
    alignment[T] = categorical(dist)
    for t in T-1:-1:1
        ldist = alphas[t] .+ forward_transition_log_probs(alignment[t+1], params, K)
        dist = exp.(ldist .- logsumexp(ldist))
        alignment[t] = categorical(dist)
    end
    return alignment
end

# forward computation that does not retain alphas
# (and therefore cannot be used for backwards samping
# or incrementally doing forward algorithm)
function log_marginal_likelihood(
        params::ObsModelParams, points_along_path::AbstractVector{Point}, measurements::AbstractVector{Point})
    K = length(points_along_path)
    T = length(measurements)
    alpha = Vector{Float64}(undef, K)
    new_alpha = Vector{Float64}(undef, K)
    populate_initial_alpha!(alpha, params, points_along_path, measurements[1])
    for t in 2:T
        populate_new_alpha!(new_alpha, alpha, points_along_path, measurements[t], params)
        tmp = alpha; alpha = new_alpha; new_alpha = tmp
    end
    return logsumexp(alpha)
end

# incremental forward algorithm
function forward_filtering_incremental(
        params::ObsModelParams,
        points_along_path::AbstractVector{Point}, obs::AbstractVector{Point},
        alphas::PersistentVector{PersistentVector{Float64}})

    @assert length(alphas) > 0

    # this would require a different enough code path that I am just using the
    # non-incremental version; since there is one time step, there is not likely
    # to be any speedup due to incrementalization anyways
    if length(alphas) == 1
        return forward_filtering(params, points_along_path, obs)
    end

    # new observation; we will increment number of columns by one
    T = length(obs)
    new_obs = obs[T]

    # new point along path; we will increment the number of rows by one
    K_prev = length(alphas[1])
    K_new = length(points_along_path)
    new_alphas = alphas
    for K in (K_prev+1):K_new
        new_point_along_path = points_along_path[K]
        # first, compute new row for the new points along the path (i.e. new hidden states)
        new_alphas = assoc(new_alphas, 1, push(new_alphas[1], -Inf))
        for t in 2:(T-1)
            new_alphas_t = push(new_alphas[t], compute_alpha_entry(params, new_point_along_path, obs[t], new_alphas[t-1], K))
            new_alphas = assoc(new_alphas, t, new_alphas_t)
        end
    end

    # next, compute new column for the new observation
    new_alpha_T = Vector{Float64}(undef, K_new)
    populate_new_alpha!(
        new_alpha_T, new_alphas[T-1], points_along_path, new_obs, params)
    new_alphas = push(new_alphas, PersistentVector{Float64}(new_alpha_T))

    # could be incrementalized, but is already just O(K)
    log_marginal_likelihood = logsumexp(new_alphas[T])

    return (new_alphas, log_marginal_likelihood)
end

# TODO change to sample_steps
@gen function sample_alignment(
        params::ObsModelParams, points_along_path::AbstractVector{Point}, obs::AbstractVector{Point})

    (alphas, _) = forward_filtering(params, points_along_path, obs)
    T = length(obs)
    K = length(points_along_path)

    alignment = Vector{Int}(undef, length(points_along_path))
    ldist = alphas[T]
    dist = exp.(ldist .- logsumexp(ldist))
    alignment[T] = ({(:alignment, T)} ~ categorical(dist))
    for t in T-1:-1:1
        ldist = alphas[t] .+ forward_transition_log_probs(alignment[t+1], params, K)
        dist = exp.(ldist .- logsumexp(ldist))
        alignment[t] = ({(:alignment, t)} ~ categorical(dist))
    end
    return alignment
end

###########################################
# collapsed motion and measurement model  #
#    (incremental and non-incremental)    #
###########################################

struct ObsModelTrace <: Trace
    gen_fn::GenerativeFunction

    # input path
    path::AbstractVector{Point}

    # input parameters
    params::ObsModelParams

    # maximum possible progress, for incremental dynamic programming
    dp_progress::PathProgress

    # all possible points along path, for incremental dynamic programming
    # (each point corresponds to a hidden state)
    dp_points_along_path::PersistentVector{Point}
        
    # observations
    measurements::PersistentVector{Point}

    # log marginal likelihood of observations
    lml::Float64

    # cached state of forward algorithm
    alphas::PersistentVector{PersistentVector{Float64}}
end

Gen.get_gen_fn(trace::ObsModelTrace) = trace.gen_fn
Gen.get_retval(trace::ObsModelTrace) = trace.measurements
Gen.get_args(trace::ObsModelTrace) = (trace.path, trace.params, length(trace.measurements))
Gen.get_score(trace::ObsModelTrace) = trace.lml
Gen.project(trace::ObsModelTrace, ::EmptySelection) = 0.0
Gen.project(trace::ObsModelTrace, ::AllSelection) = trace.lml

function Gen.get_choices(trace::ObsModelTrace)
    cm = choicemap()
    for (t, pt) in enumerate(trace.measurements)
        cm[:meas => t] = [pt.x, pt.y]
    end
    return cm
end

struct ObsModel <: GenerativeFunction{
        Tuple{PersistentVector{Point},PersistentVector{Point},PathProgress},
        ObsModelTrace}
    incremental::Bool
end

const motion_and_measurement_model_collapsed = ObsModel(false)
const motion_and_measurement_model_collapsed_incremental = ObsModel(true)

num_hidden_states(T::Int) = 2*(T-1)+1

function Gen.simulate(gen_fn::ObsModel, args::Tuple)
    path, params, T = args

    # sample
    measurements = motion_and_measurement_model_uncollapsed(path, params, T)

    # compute marginal likelihood
    (dp_points_along_path, dp_progress) = walk_path(path, params.nominal_speed, num_hidden_states(T))
    (alphas, lml) = forward_filtering(params, dp_points_along_path, measurements)
    @assert !isnan(lml)

    return ObsModelTrace(gen_fn, path, params, dp_progress, dp_points_along_path, measurements, lml, alphas)
end

function Gen.generate(gen_fn::ObsModel, args::Tuple, constraints::ChoiceMap)
    if isempty(constraints)
        trace = simulate(gen_fn, args)
        return (trace, 0.0, nothing)
    end
    (path, params, T) = args

    # read observations from constraints
    measurements = PersistentVector{Point}()
    for t in 1:T
        measurements = push(measurements, Point(constraints[:meas => t]))
    end

    # walk the path
    (dp_points_along_path, dp_progress) = walk_path(path, params.nominal_speed, num_hidden_states(T))

    # compute log marginal likelihood
    (alphas, lml) = forward_filtering(params, dp_points_along_path, measurements)
    @assert !isnan(lml)
    
    # construct trace
    trace = ObsModelTrace(gen_fn, path, params, dp_progress, dp_points_along_path, measurements, lml, alphas)
    retval = get_retval(trace)
    return (trace, lml, retval)
end

# handles changes to path and/or params, but not number of time steps
function Gen.update(
        trace::ObsModelTrace, args::Tuple,
        argdiffs::Tuple{U,V,NoChange},
        constraints::ChoiceMap) where {U,V}

    (new_path, new_params, T) = args 

    # T did not change
    @assert T == length(trace.measurements)

    if !isempty(constraints)
        error("not implemented")
    end

    # check invariant
    @assert num_hidden_states(T) == length(trace.dp_points_along_path)

    # TODO: if we only changed the parameters and not the path, then we don't need to re-walk it
    (dp_points_along_path, dp_progress) = walk_path(new_path, new_params.nominal_speed, T)

    # run forward-backward from scratch
    measurements = trace.measurements
    @assert length(measurements) == T
    (alphas, lml) = forward_filtering(new_params, dp_points_along_path, measurements)
    @assert !isnan(lml)

    new_trace = ObsModelTrace(
        get_gen_fn(trace), new_path, new_params, dp_progress, dp_points_along_path, measurements, lml, alphas)
    return (new_trace, lml - trace.lml, UnknownChange(), EmptyChoiceMap())
end

# handles extensions to time steps, but not changes to path or params
function Gen.update(
        trace::ObsModelTrace, args::Tuple,
        argdiffs::Tuple{NoChange,NoChange,UnknownChange},
        constraints::ChoiceMap)

    (path, params, old_T) = get_args(trace)
    (_, _, new_T) = args

    # params did not change
    @assert params == trace.params

    # path did not change
    @assert path == trace.path
    
    # we only handle extensions in time
    if new_T < old_T
        error("not implemented")
    end

    # TODO also check that the earlier time steps aren't constrained..
    # because support for this is not currently implemented here

    # check invariant
    @assert num_hidden_states(old_T) == length(trace.dp_points_along_path)

    local alphas::PersistentVector{PersistentVector{Float64}}
    local dp_progress::PathProgress
    local dp_points_along_path::PersistentVector{Point}
    local measurements::PersistentVector{Point} = trace.measurements
    local lml::Float64

    if get_gen_fn(trace).incremental

        # incrementally walk path and 
        # incremental forward filtering
        alphas = trace.alphas 
        dp_progress = trace.dp_progress
        dp_points_along_path = trace.dp_points_along_path
        lml = trace.lml
        for T in (old_T+1):new_T
            (dp_points_along_path, dp_progress) = walk_path_incremental(
                path, params.nominal_speed, dp_points_along_path, 2, dp_progress) # extend with two hidden states
            @assert length(dp_points_along_path) == num_hidden_states(T)
            measurements = push(measurements, Point(constraints[:meas => T]))
            (alphas, lml) = forward_filtering_incremental(params, dp_points_along_path, measurements, alphas)
            @assert !isnan(lml)
        end

    else

        # non-incremental walk path
        # non-incremental forward filtering
        (dp_points_along_path, dp_progress) = walk_path(path, params.nominal_speed, num_hidden_states(new_T))
        for T in (old_T+1):new_T
            measurements = push(measurements, Point(constraints[:meas => T]))
        end
        (alphas, lml) = forward_filtering(params, dp_points_along_path, measurements)
        @assert !isnan(lml)
    end

    # check invariant
    @assert num_hidden_states(new_T) == length(dp_points_along_path)

    new_trace = ObsModelTrace(
        get_gen_fn(trace), path, params, dp_progress, dp_points_along_path, measurements, lml, alphas)

    return (new_trace, lml - trace.lml, UnknownChange(), EmptyChoiceMap())
end

function Gen.regenerate(trace::ObsModelTrace, args::Tuple, argdiffs::Tuple, selection::Selection)
    if !isempty(selection)
        error("not implemented")
    end
    (new_trace, weight, retdiff, _) = update(trace, args, argdiffs, EmptyChoiceMap())
    return (new_trace, weight, retdiff)
end

##################################
# SMC-based pseudomarginal model #
##################################

import GenSMC

export make_motion_and_measurement_model_smc_pseudomarginal

function make_motion_and_measurement_model_smc_pseudomarginal(num_particles::Int)
    return GenSMC.PFPseudoMarginalGF(
        model = motion_and_measurement_model_uncollapsed_incremental,
        data_addrs = (T) -> [obs_addr(motion_and_measurement_model_uncollapsed_incremental, T)],
        get_step_args = (args, T) -> (args[1], args[2], T),
        num_particles = num_particles,
        get_T = (args) -> args[3],
        reuse_particle_system = (args1, args2, argdiffs) -> (
            argdiffs[1] == NoChange() && argdiffs[2] == NoChange()))
end

# TODO need to be able to attach methods to this specific instance
function obs_addr(::GenSMC.PFPseudoMarginalGF, T::Int)
    return obs_addr(motion_and_measurement_model_uncollapsed_incremental, T)
end

# TODO
# Q: does our model actually give us this during a call to update?
# do we need to foramlize how update handeles hte internal choices inside the path planner???
# (otherwise the plan is actually different on every call to update..)
# A: it should work; the plan should be NoChange(), but this isn't formalized yet...
