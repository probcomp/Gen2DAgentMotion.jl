using Gen
using FunctionalCollections: PersistentVector, push, assoc

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

@inline function walk_path_step(
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

#"""
    #walk_path(path::Vector{Point}, speed::Float64)
#
#Walk path until it ends recording the point at each time step.
#"""
#function walk_path(path::Vector{Point}, speed::Float64)
    #points = Point[path[1]]
    #prev_pt_idx = 1
    #dist_past_prev_pt = 0.0
    #t = 1
    #while prev_pt_idx < length(path)
        #t += 1
        #(prev_pt_idx, dist_past_prev_pt, point) = walk_path_step(
            #path, prev_pt_idx, dist_past_prev_pt, speed)
        #push!(points, point)
    #end
    ## TODO revert to using Vector{Point} in the trace?
    #return PersistentVector{Point}(points)
#end

"""
    walk_path(path::Vector{Point}, speed::Float64, T::Int)

Walk path for T time steps, recording the point at each time step
"""
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



##############################################
# probabilistic motion and measurement model #
##############################################

# the observed points are generated by walking along the 
# path at a nominal speed, with random pauses and skips

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

@inline function compute_alpha_entry(params::ObsModelParams, point_along_path::Point, obs::Point, prev_alpha, k::Int)

    # observation model: obs corresponds with points_along_path
    noise_ll = noise_log_likelihood(params, point_along_path, obs)
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

@inline function populate_initial_alpha!(alpha, params::ObsModelParams, points_along_path::AbstractVector{Point}, obs1::Point)
    @assert length(alpha) == length(points_along_path)
    if length(alpha) == 1
        alpha[1] = noise_log_likelihood(params, points_along_path[1], obs1)
        return
    end
    fill!(alpha, -Inf)
    first_prob_normal = prob_lag(params) + prob_normal(params)
    first_prob_skip = prob_skip(params)
    alpha[1] = log(first_prob_normal) + noise_log_likelihood(params, points_along_path[1], obs1) # p(z1) is deterministic at start of points_along_path
    alpha[2] = log(first_prob_skip) + noise_log_likelihood(params, points_along_path[2], obs1) # p(z1) is deterministic at start of points_along_path
    return nothing
end

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
        params::ObsModelParams, points_along_path::AbstractVector{Point}, obs::AbstractVector{Point})
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

#################################
# incremental forward algorithm #
#################################

# the state is alphas::PersistentVector{PersistentVector{Float64}} which is a set of T vectors, each of length K
# (actually it will always be K = T, but this code does not use that information) 

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
    @assert T == length(alphas) + 1
    new_obs = obs[T]

    # new point along path; we will increment the number of rows by one
    K = length(points_along_path)
    @assert K == length(alphas[1]) + 1
    new_point_along_path = points_along_path[K]

    # first, compute new row for the new point along the path (i.e. new hidden state)
    new_alphas = assoc(alphas, 1, push(alphas[1], -Inf))
    for t in 2:(T-1)
        new_alphas_t = push(alphas[t], compute_alpha_entry(params, new_point_along_path, obs[t], new_alphas[t-1], K))
        new_alphas = assoc(new_alphas, t, new_alphas_t)
    end

    # then compute new column for the new observation
    new_alpha_T = Vector{Float64}(undef, K)
    populate_new_alpha!(
        new_alpha_T, new_alphas[T-1], points_along_path, new_obs, params)
    new_alphas = push(new_alphas, PersistentVector{Float64}(new_alpha_T))

    # could be incrementalized, but is already just O(K)
    log_marginal_likelihood = logsumexp(new_alphas[T])

    return (new_alphas, log_marginal_likelihood)
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
    (points, prev_pt_idx, dist_past_prev_pt) = walk_path(path, params.nominal_speed, T)
    k = length(points)
    alignment = Vector{Int}(undef, T)
    alignment[1] = 1

    # initial time step; advance by 0 or 1 steps
    if k > 1
        probs = [prob_lag(params) + prob_normal(params), prob_skip(params)]
        alignment[1] += ({:steps => 1} ~ categorical(probs / sum(probs))) - 1
    end
    obs[1] = ({*} ~ gaussian_noise(params.noise, points[alignment[1]], 1))

    # for each remaining time step, advance by 0, 1, or 2 steps
    for t in 2:T
        _prob_skip = (alignment[t-1] < k - 1) ? prob_skip(params) : 0.0
        _prob_normal = (alignment[t-1] < k) ? prob_normal(params) : 0.0
        probs = [prob_lag(params), _prob_normal, _prob_skip]
        alignment[t] = alignment[t-1] + ({(:steps => t)} ~ categorical(probs / sum(probs))) - 1
        obs[t] = ({*} ~ gaussian_noise(params.noise, points[alignment[t]], t))
    end
    return (points, obs, prev_pt_idx, dist_past_prev_pt)
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
    points::PersistentVector{Point}
        
    # observations
    obs::PersistentVector{Point}

    # log marginal likelihood of observations
    lml::Float64

    # cached state of forward algorithm
    # TODO this is probably only with it for very large T (e.g. in hundreds or more)
    # do a benchmark where we set the speed to very small, and do sequential importance sampling
    alphas::PersistentVector{PersistentVector{Float64}}
end

Gen.get_gen_fn(tr::ObsModelTrace) = tr.gen_fn
Gen.get_retval(tr::ObsModelTrace) = (tr.points, tr.obs, tr.prev_pt_idx, tr.dist_past_prev_pt) # TODO XXX breaking interface change
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
    (points, obs, prev_pt_idx, dist_past_prev_pt) = dml_path_observation_model(path, params, T)
    (alphas, lml) = forward_filtering(params, points, obs)
    @assert !isnan(lml)
    return ObsModelTrace(gen_fn, path, params, prev_pt_idx, dist_past_prev_pt, points, obs, lml, alphas)
end

function Gen.generate(gen_fn::ObsModel, args::Tuple, constraints::ChoiceMap)
    if isempty(constraints)
        trace = simulate(gen_fn, args)
        return (trace, 0.0, nothing)
    end
    path, params, T = args

    # walk the path
    (points, prev_pt_idx, dist_past_prev_pt) = walk_path(path, params.nominal_speed, T)

    # read observations from constraints
    _obs = Vector{Point}(undef, T)
    for t in 1:T
        _obs[t] = Point(constraints[(:x, t)], constraints[(:y, t)])
    end
    obs = PersistentVector{Point}(_obs)

    # compute log marginal likelihood
    (alphas, lml) = forward_filtering(params, points, obs)
    @assert !isnan(lml)
    
    # construct trace
    trace = ObsModelTrace(gen_fn, path, params, prev_pt_idx, dist_past_prev_pt, points, obs, lml, alphas)
    retval = get_retval(trace)
    return (trace, lml, retval)
end

# handles changes to path and/or params
function Gen.update(
        tr::ObsModelTrace, args::Tuple,
        argdiffs::Tuple{T,U,NoChange},
        constraints::ChoiceMap) where {T,U}

    if !isempty(constraints)
        error("not implemented")
    end

    # TODO: if we only changed the parameters and not the path, then we wouldn't
    # need to re-walk it..
    (new_points, prev_pt_idx, dist_past_prev_pt) = walk_path(new_path, new_params.nominal_speed, new_T)

    # need to run forward-backward from scratch
    obs = tr.obs
    (alphas, lml) = forward_filtering(new_params, new_points, obs)
    @assert !isnan(lml)

    new_trace = ObsModelTrace(
        get_gen_fn(tr), new_path, new_params, prev_pt_idx, dist_past_prev_pt, new_points, obs, lml, alphas)
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

    # TODO it can be faster to do the increment in batch, instead of one at a time
    # but this code is simpler, so we start with this.

    # NOTE: we guarantee that we don't change the previous entries, so as long as 
    # the other trace doesn't determine T or K by the bounds of alphas, we are okay
    # (we are the only code that reads from alphas..; so this is a contract
    # with ourselves). therefore we don't need to copy any data here.
    # the only problem is if this code gets called twice on the same trace, but with different observations
    # then it breaks...
    # TODO.. we could use a linked list? the full PersistentVector of PersistentVectors is overkill 
    # since we guarantee that the old elements are never modified.

    alphas = tr.alphas 
    prev_pt_idx = tr.prev_pt_idx
    dist_past_prev_pt = tr.dist_past_prev_pt
    points = tr.points
    obs = tr.obs
    for T in old_T+1:new_T
        (points, prev_pt_idx, dist_past_prev_pt) = walk_path_incremental(
            path, params.nominal_speed, points, 1,
            prev_pt_idx, dist_past_prev_pt)
        @assert length(points) == T

        obs = push(obs, Point(constraints[(:x, T)], constraints[(:y, T)]))
        (alphas, lml) = forward_filtering_incremental(params, points, obs, alphas)
        @assert !isnan(lml)
    end

    new_trace = ObsModelTrace(
        get_gen_fn(tr), path, params, prev_pt_idx, dist_past_prev_pt, points, obs, lml, alphas)
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
