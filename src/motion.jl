using Gen

function path_length(start::Point, path::Vector{Point})
    @assert length(path) > 0
    len = dist(start, path[1])
    for i in 1:length(path)-1
        len += dist(path[i], path[i+1])
    end
    return len
end

function compute_distances_from_start(path::Vector{Point})
    distances_from_start = Vector{Float64}(undef, length(path))
    distances_from_start[1] = 0.0
    for i=2:length(path)
        distances_from_start[i] = distances_from_start[i-1] + dist(path[i-1], path[i])
    end
    return distances_from_start
end

function walk_path(path::Vector{Point}, speed::Float64, T::Int)
    distances_from_start = compute_distances_from_start(path)
    times = collect(0.0:1.0:(T-1))
    locations = Vector{Point}(undef, T)
    for (time_idx, t) in enumerate(times)
        desired_distance = t * speed
        used_up_time = false
        # NOTE: can be improved (iterate through path points along with times)
        for i=2:length(path)
            prev = path[i-1]
            cur = path[i]
            dist_to_prev = dist(prev, cur)
            if distances_from_start[i] >= desired_distance
                # we overshot, the location is between i-1 and i
                overshoot = distances_from_start[i] - desired_distance
                @assert overshoot <= dist_to_prev
                past_prev = dist_to_prev - overshoot
                frac = past_prev / dist_to_prev
                locations[time_idx] = Point(prev.x * (1. - frac) + cur.x * frac,
                                     prev.y * (1. - frac) + cur.y * frac)
                used_up_time = true
                break
            end
        end
        if !used_up_time
            # sit at the goal indefinitely
            locations[time_idx] = path[end] 
        end
    end
    return locations
end

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
    return nothing
end

function run_forward_backward(params::ObsModelParams, points_along_path::Vector{Point}, obs::Vector{Point})
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
    dist = exp.(ldist .- logsumexp(ldist))
    alignment[T] = categorical(dist)
    for t in T-1:-1:1
        ldist = alphas[t,:] .+ forward_transition_log_probs(alignment[t+1], params, K)
        dist = exp.(ldist .- logsumexp(ldist))
        alignment[t] = categorical(dist)
    end
    return (log_marginal_likelihood, alignment)
end

#function log_marginal_likelihood(params::ObsModelParams, points_along_path::Vector{Point}, obs::Vector{Point})
    #K = length(points_along_path)
    #T = length(obs)
    #alpha = Vector{Float64}(undef, K)
    #new_alpha = Vector{Float64}(undef, K)
    #populate_initial_alpha!(alpha, params, points_along_path, obs[1])
    #for t in 2:T
        #populate_new_alpha!(new_alpha, alpha, points_along_path, obs[t], params)
        #tmp = alpha; alpha = new_alpha; new_alpha = tmp
    #end
    #return logsumexp(alpha)
#end

#######################
# generative function #
#######################

struct ObsModelTrace <: Trace
    gen_fn::GenerativeFunction
    path::Vector{Point}
    params::ObsModelParams
    points_along_path::Vector{Point}
    obs::Vector{Point}
    lml::Float64
    alignment::Vector{Int}
end

Gen.get_gen_fn(tr::ObsModelTrace) = tr.gen_fn
Gen.get_retval(trace::ObsModelTrace) = (trace.points_along_path, trace.alignment)
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

function noise_sample(noise, pt::Point)
    x = normal(pt.x, noise)
    y = normal(pt.y, noise)
    return Point(x, y)
end

function sample_from_obs_model(params::ObsModelParams, points_along_path::Vector{Point}, T::Int)
    obs = Vector{Point}(undef, T)
    if length(points_along_path) == 1
        z_cur = 1
    else
        z_cur = bernoulli(prob_lag(params) + prob_normal(params)) ? 1 : 2
    end
    obs[1] = noise_sample(params.noise, points_along_path[z_cur])
    for t in 2:T
        _prob_skip = (z_cur < length(points_along_path) - 1) ? prob_skip(params) : 0.0
        _prob_normal = (z_cur < length(points_along_path)) ? prob_normal(params) : 0.0
        probs = [prob_lag(params), _prob_normal, _prob_skip]
        probs /= sum(probs)
        dowhat = categorical(probs)
        if dowhat == 1
            # lag
        elseif dowhat == 2
            # normal
            z_cur += 1
        else
            # skip
            z_cur += 2
        end
        obs[t] = noise_sample(params.noise, points_along_path[z_cur])
    end
    return obs
end

function Gen.simulate(gen_fn::ObsModel, args::Tuple)
    path, params, T = args
    points_along_path = walk_path(path, params.nominal_speed, T)
    obs = sample_from_obs_model(params, points_along_path, T)
    (lml, alignment) = run_forward_backward(params, points_along_path, obs)
    @assert !isnan(lml)
    return ObsModelTrace(gen_fn, path, params, points_along_path, obs, lml, alignment)
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
    (lml, alignment) = run_forward_backward(params, points_along_path, obs)
    @assert !isnan(lml)
    trace = ObsModelTrace(gen_fn, path, params, points_along_path, obs, lml, alignment)
    retval = get_retval(trace)
    return (trace, lml, retval)
end

function Gen.update(tr::ObsModelTrace, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap)
    old_path, old_params, old_T = get_args(tr)
    new_path, new_params, new_T = args
    old_points_along_path = tr.points_along_path
    new_points_along_path = walk_path(new_path, new_params.nominal_speed, new_T)
    if (new_T == old_T + 1) && has_value(constraints, (:x, new_T)) && has_value(constraints, (:y, new_T))
        obs = copy(tr.obs)
        push!(obs, Point(constraints[(:x, new_T)], constraints[(:y, new_T)]))
    elseif new_T == old_T && isempty(constraints)
        obs = tr.obs
    else
        error("not implemented")
    end
    (lml, alignment) = run_forward_backward(new_params, new_points_along_path, obs)
    @assert !isnan(lml)
    new_trace = ObsModelTrace(get_gen_fn(tr), new_path, new_params, new_points_along_path, obs, lml, alignment)
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
