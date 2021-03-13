using Gen
using FunctionalCollections: PersistentVector, push

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

function walk_path_step(
        path::Vector{Point},
        prev_pt_idx::Int, dist_past_prev_pt::Float64,
        distance_to_walk::Float64)

    @assert 1 <= prev_pt_idx <= length(path)
    prev::Point = path[prev_pt_idx]
    local next::Point
    local dist_prev_to_next::Float64

    dist_remaining = distance_to_walk 
    while true

        # if we have are at the last point and we still have distance to
        # travel, then stay at the last point
        if prev_pt_idx == length(path)
            return (prev_pt_idx, 0.0, prev)
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

    return (prev_pt_idx, dist_past_prev_pt, cur)
end


function walk_path(path::Vector{Point}, speed::Float64, T::Int)
    points = Vector{Point}(undef, T)
    if T > 0
        points[1] = path[1]
    end
    prev_pt_idx = 1
    dist_past_prev_pt = 0.0
    for t in 2:T    
        (prev_pt_idx, dist_past_prev_pt, points[t]) = walk_path_step(
            path, prev_pt_idx, dist_past_prev_pt, speed)
    end
    return (PersistentVector{Point}(points), prev_pt_idx, dist_past_prev_pt)
end

function walk_path_incremental(
        path::Vector{Point}, speed::Float64,
        points::PersistentVector{Point}, num_additional_time_steps::Int,
        prev_pt_idx::Int, dist_past_prev_pt::Float64)::Tuple{PersistentVector{Point},Int,Float64}

    prev_T = length(points)
    new_T = prev_T + num_additional_time_steps
    if prev_T == 0
        return walk_path(path, speed, new_T)
    end
    for t in 1:num_additional_time_steps
        (prev_pt_idx, dist_past_prev_pt, point) = walk_path_step(
            path, prev_pt_idx, dist_past_prev_pt, speed)
        points = push(points, point)
    end
    @assert length(points) == new_T
    return (points, prev_pt_idx, dist_past_prev_pt)
end

###################
# alignment model #
###################

struct ObsModelParams
    nominal_speed::Float64 # distance per time step
    walk_noise::Float64
    noise::Float64
end

prob_lag(params::ObsModelParams) = params.walk_noise/2
prob_skip(params::ObsModelParams) = params.walk_noise/2
prob_normal(params::ObsModelParams) = 1-params.walk_noise

@inline function noise_log_likelihood(params::ObsModelParams, points_along_path_pt::Point, obs_pt::Point)
    ll = logpdf(normal, obs_pt.x, points_along_path_pt.x, params.noise)
    ll += logpdf(normal, obs_pt.y, points_along_path_pt.y, params.noise)
    return ll
end

@inline function populate_new_alpha!(new_alpha, prev_alpha, points_along_path::Vector{Point}, obs_t::Point, params::ObsModelParams)
    for k in 1:length(points_along_path)
        # observation model: obs[t] corresponds with points_along_path[k]
        noise_ll = noise_log_likelihood(params, points_along_path[k], obs_t)
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
            new_alpha[k] = -Inf
            continue
        end
        max_val = max(val1, val2, val3)
        new_alpha[k] = noise_ll + max_val + log(exp(val1 - max_val) + exp(val2 - max_val) + exp(val3 - max_val))
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

@inline function populate_initial_alpha!(alpha, params::ObsModelParams, points_along_path::Vector{Point}, obs1::Point)
    if length(alpha) == 1
        alpha[1] = noise_log_likelihood(params, points_along_path[1], obs1)
        return
    end
    fill!(alpha, -Inf)
    first_prob_normal = prob_lag(params) + prob_normal(params)
    first_prob_skip = prob_skip(params)
    alpha[1] = log(first_prob_normal) + noise_log_likelihood(params, points_along_path[1], obs1) # p(z1) is deterministic at start of points_along_path
    alpha[2] = log(first_prob_skip) + noise_log_likelihood(params, points_along_path[2], obs1) # p(z1) is deterministic at start of points_along_path
    println((points_along_path[1], points_along_path[2], obs1))
    println((noise_log_likelihood(params, points_along_path[1], obs1), noise_log_likelihood(params, points_along_path[2], obs1)))
    println((log(first_prob_normal), log(first_prob_skip)))
    println("initial alpha: $alpha")
    return nothing
end

function run_forward_backward(
        params::ObsModelParams, points_along_path::Vector{Point}, obs::Vector{Point})

    K = length(points_along_path)
    T = length(obs)

    # run forward filtering backward sampling

    # forward filtering
    alphas = Matrix{Float64}(undef, T, K) # optimize memory access?
    populate_initial_alpha!(view(alphas, 1, :), params, points_along_path, obs[1])
    for t in 2:T
        populate_new_alpha!(view(alphas, t, :), view(alphas, t-1, :), points_along_path, obs[t], params)
    end
    log_marginal_likelihood = logsumexp(alphas[end,:])

    # backward sampling
    alignment = Vector{Int}(undef, length(points_along_path))
    ldist = alphas[T,:]
    println(alphas)
    dist = exp.(ldist .- logsumexp(ldist))
    println(dist)
    alignment[T] = categorical(dist)
    for t in T-1:-1:1
        ldist = alphas[t,:] .+ forward_transition_log_probs(alignment[t+1], params, K)
        dist = exp.(ldist .- logsumexp(ldist))
        println(dist)
        alignment[t] = categorical(dist)
    end
    return (log_marginal_likelihood, alignment)
end

function log_marginal_likelihood(params::ObsModelParams, points_along_path::Vector{Point}, obs::Vector{Point})
    K = length(points_along_path)
    T = length(obs)
    alpha = Vector{Float64}(undef, K)
    new_alpha = Vector{Float64}(undef, K)
    populate_initial_alpha!(alpha, params, points_along_path, obs[1])
    for t in 2:T
        populate_new_alpha!(new_alpha, alpha, points_along_path, obs[t], params)
        tmp = alpha; alpha = new_alpha; new_alpha = tmp
    end
    return logsumexp(alpha)
end

###############
# DML version #
###############

@gen function gaussian_noise(std::Real, pt::Point, t::Int)
    x = ({(:x, t)} ~ normal(pt.x, std))
    y = ({(:y, t)} ~ normal(pt.y, std))
    return Point(x, y)
end

@gen function dml_path_observation_model(path::AbstractVector{Point}, params::ObsModelParams, T::Int)

    # initialize
    obs = Vector{Point}(undef, T) # measurements
    points_along_path = walk_path(path, params.nominal_speed, T)
    k = length(points_along_path)
    alignment = Vector{Int}(undef, T)
    alignment[1] = 1

    # initial time step; advance by 0 or 1 steps
    if k > 1
        probs = [prob_lag(params) + prob_normal(params), prob_skip(params)]
        alignment[1] += ({:steps => 1} ~ categorical(probs / sum(probs))) - 1
    end
    obs[1] = ({*} ~ gaussian_noise(params.noise, points_along_path[alignment[1]], 1))

    # for each remaining time step, advance by 0, 1, or 2 steps
    for t in 2:T
        _prob_skip = (alignment[t-1] < k - 1) ? prob_skip(params) : 0.0
        _prob_normal = (alignment[t-1] < k) ? prob_normal(params) : 0.0
        probs = [prob_lag(params), _prob_normal, _prob_skip]
        alignment[t] = alignment[t-1] + ({(:steps => t)} ~ categorical(probs / sum(probs))) - 1
        obs[t] = ({*} ~ gaussian_noise(params.noise, points_along_path[alignment[t]], t))
    end
    return (points_along_path, alignment, obs)
end

export dml_path_observation_model


##############################
# custom generative function #
##############################

using FunctionalCollections: PersistentVector

struct ObsModelTrace <: Trace
    gen_fn::GenerativeFunction

    # input path
    path::Vector{Point}

    # input parameters
    params::ObsModelParams

    # for walking path incrementally
    prev_pt_idx::Int
    dist_past_prev_pt::Float64
    points_along_path::PersistentVector{Point}
        
    # observations
    obs::PersistentVector{Point}

    # log marginal likelihood of observations
    lml::Float64
end

Gen.get_gen_fn(tr::ObsModelTrace) = tr.gen_fn
Gen.get_retval(trace::ObsModelTrace) = (trace.points_along_path, trace.obs)
Gen.get_args(tr::ObsModelTrace) = (tr.path, tr.params, length(tr.obs))
Gen.get_score(tr::ObsModelTrace) = tr.lml
Gen.project(tr::ObsModelTrace, ::EmptySelection) = 0.0

function Gen.get_choices(tr::ObsModelTrace)
    cm = choicemap()
    for (i, pt) in enumerate(tr.obs)
        cm[(:x, i)] = pt.x
        cm[(:y, i)] = pt.y
    end
    return cm
end

struct ObsModel <: GenerativeFunction{Tuple{Vector{Point},Vector{Int}},ObsModelTrace} end

const path_observation_model = ObsModel()

function Gen.simulate(gen_fn::ObsModel, args::Tuple)
    path, params, T = args
    (points_along_path, _, obs) = dml_path_observation_model(points_along_path, params, T)
    (lml, _) = run_forward_backward(params, points_along_path, obs)
    @assert !isnan(lml)
    return ObsModelTrace(gen_fn, path, params, points_along_path, obs, lml)
end

function Gen.generate(gen_fn::ObsModel, args::Tuple, constraints::ChoiceMap)
    path, params, T = args
    points_along_path = walk_path(path, params.nominal_speed, T)
    obs = Vector{Point}(undef, T)
    if isempty(constraints)
        trace = simulate(gen_fn, args)
        return (trace, 0.0, nothing)
    end
    for t in 1:T
        obs[t] = Point(constraints[(:x, t)], constraints[(:y, t)])
    end
    (lml, _) = run_forward_backward(params, points_along_path, obs)
    @assert !isnan(lml)
    trace = ObsModelTrace(gen_fn, path, params, points_along_path, obs, lml)
    retval = get_retval(trace)
    return (trace, lml, retval)
end

# OPTION 1: if we didn't keep the alignment in the trace, then we could
# continue on the forward algorithm.  this would result in a Rao-Blackwellized
# particle filter, and an O(1) update. but if we want to sample the alignment
# from the conditional distribution, this takes O(T).

# OPTION 2: if we do keep the alignment in the trace, then we have to sample
# the new alignment from the local conditional distribution.

# default:
# this handles changes to path and/or params
function Gen.update(
        tr::ObsModelTrace, args::Tuple,
        argdiffs::Tuple{T,U,NoChange},
        constraints::ChoiceMap) where {T,U}

    if !isempty(constraints)
        error("not implemented")
    end

    # NOTE: if we only changed the parametes and not the path, then we wouldn't
    # need to re-walk it..
    new_points_along_path = walk_path(new_path, new_params.nominal_speed, new_T)

    # need to run forward-backward from scratch
    obs = tr.obs
    (lml, _) = run_forward_backward(new_params, new_points_along_path, obs)
    @assert !isnan(lml)

    new_trace = ObsModelTrace(get_gen_fn(tr), new_path, new_params, new_points_along_path, obs, lml)
    return (new_trace, lml - tr.lml, UnknownChange(), EmptyChoiceMap())
end

# handles extensions to time steps, but not changes to path or params
function Gen.update(
        tr::ObsModelTrace, args::Tuple,
        argdiffs::Tuple{NoChange,NoChange,UnknownChange},
        constraints::ChoiceMap)

    (path, params, old_T) = get_args(tr)
    (_, _, new_T) = args

    if new_T < old_T
        error("not implemented")
    end

    # TODO also check that the earlier time steps aren't constrained..
    # because this is not implemented either

    # the number of time steps changes, incrementalize this, so that update can
    # be O(1) and not O(T)
    (points, prev_pt_idx, dist_past_prev_pt) = walk_path_incremental(
        path, params.nominal_speed, tr.points, new_T - old_T,
        tr.prev_pt_idx, tr.dist_past_prev_pt)

    # TODO make it handle more time steps
    if (new_T == old_T + 1) && has_value(constraints, (:x, new_T)) && has_value(constraints, (:y, new_T))

        obs = copy(tr.obs) # TODO this means we need to use a PersistentVector internally
        push!(obs, Point(constraints[(:x, new_T)], constraints[(:y, new_T)]))

    end

    @assert !isnan(lml)
    return (new_trace, lml - tr.lml, UnknownChange(), EmptyChoiceMap())
end


function Gen.update(tr::ObsModelTrace, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap)
    old_path, old_params, old_T = get_args(tr)
    new_path, new_params, new_T = args
    old_points_along_path = tr.points_along_path

    # TODO for the case where the path does not change, and only
    # the number of time steps changes, incrementalize this, so that update can
    # be O(1) and not O(T)
    new_points_along_path = walk_path(new_path, new_params.nominal_speed, new_T)

    # CASE 1
    # extend by one or more time steps
    # TODO make it handle more...
    if (new_T == old_T + 1) && has_value(constraints, (:x, new_T)) && has_value(constraints, (:y, new_T))

        obs = copy(tr.obs)
        push!(obs, Point(constraints[(:x, new_T)], constraints[(:y, new_T)]))

    # CASE 2
    # changing path and/or parameters
    elseif new_T == old_T && isempty(constraints)

        # need to run forward-backward from scratch
        obs = tr.obs
        (lml, _) = run_forward_backward(new_params, new_points_along_path, obs)
        new_trace = ObsModelTrace(get_gen_fn(tr), new_path, new_params, new_points_along_path, obs, lml)

    # CASE 3
    # any other cases not handled
    else
        error("not implemented")
    end

    @assert !isnan(lml)
    return (new_trace, lml - tr.lml, UnknownChange(), EmptyChoiceMap())
end

function Gen.regenerate(tr::ObsModelTrace, args::Tuple, argdiffs::Tuple, selection::Selection)
    if !isempty(selection)
        error("not implemented")
    end
    (new_trace, weight, retdiff, _) = update(tr, args, argdiffs, EmptyChoiceMap())
    return (new_trace, weight, retdiff)
end

export ObsModelParams, path_observation_model
