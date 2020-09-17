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

function walk_path(path::Vector{Point}, speed::Float64, times::Vector{Float64})
    distances_from_start = compute_distances_from_start(path)
    locations = Vector{Point}(undef, length(times))
    for (time_idx, t) in enumerate(times)
        if t < 0.0
            error("times must be positive")
        end
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
    nominal_speed::Float64
    prob_lag::Float64
    prob_normal::Float64
    prob_skip::Float64
    noise::Float64
end

struct ObsModelTrace <: Trace
    gen_fn::GenerativeFunction
    path::Path
    obs_times::Vector{Float64}
    params::ObsModelParams
    traj::Vector{Point}
    obs::Vector{Point}
    lml::Float64
end

Gen.get_gen_fn(tr::ObsModelTrace) = tr.gen_fn
Gen.get_retval(::ObsModelTrace) = nothing
Gen.get_args(tr::ObsModelTrace) = (tr.path, tr.obs_times, tr.params)
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

struct ObsModel <: GenerativeFunction{Nothing,ObsModelTrace} end

const path_observation_model = ObsModel()

function noise_sample(noise, pt::Point)
    x = normal(pt.x, noise)
    y = normal(pt.y, noise)
    return Point(x, y)
end

function sample_from_obs_model(params::ObsModelParams, traj::Vector{Point}, T::Int)
    obs = Vector{Point}(undef, T)
    if length(traj) == 1
        z_cur = 1
    else
        z_cur = bernoulli(params.prob_lag + params.prob_normal) ? 1 : 2
    end
    obs[1] = noise_sample(params.noise, traj[z_cur])
    for t in 2:T
        prob_skip = (z_cur < length(traj) - 1) ? params.prob_skip : 0.0
        prob_normal = (z_cur < length(traj)) ? params.prob_normal : 0.0
        probs = [params.prob_lag, prob_normal, prob_skip]
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
        obs[t] = noise_sample(params.noise, traj[z_cur])
    end
    return obs
end

function Gen.simulate(gen_fn::ObsModel, args::Tuple)
    path, obs_times, params = args
    trajectory = walk_path(params.nominal_speed, obs_times)
    T = length(obs_times)
    obs = sample_from_obs_model(obs_params, traj, T)
    lml = log_marginal_likelihood(obs_params, traj, obs)
    @assert !isnan(lml)
    return ObsModelTrace(gen_fn, path, obs_times, params, trajectory, obs, lml)
end

function Gen.generate(gen_fn::ObsModel, args::Tuple, constraints::ChoiceMap)
    path, obs_times, params = args
    T = length(obs_times)
    trajectory = walk_path(params.nominal_speed, obs_times)
    obs = Vector{Point}(undef, T)
    if isempty(constraints)
        trace = simulate(gen_fn, args)
        return (trace, 0.0, nothing)
    end
    for t in 1:T
        obs[t] = Point(constraints[(:x, t)], constraints[(:y, t)])
    end
    lml = log_marginal_likelihood(obs_params, traj, obs)
    @assert !isnan(lml)
    return (ObsModelTrace(gen_fn, args, obs, lml), lml, nothing)
end

function Gen.update(tr::ObsModelTrace, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap)
    old_path, old_obs_times, old_params = args
    new_path, new_obs_times, new_params = args
    old_traj = tr.traj
    new_traj = walk_path(new_params.nomimal_speed, new_obs_times)
    old_T = length(old_obs_times)
    new_T = length(new_obs_times)
    if (new_T == old_T + 1) && has_value(constraints, (:x, new_T)) && has_value(constraints, (:y, new_T))
        obs = copy(tr.obs)
        push!(obs, Point(constraints[(:x, new_T)], constraints[(:y, new_T)]))
    elseif new_T == old_T && isempty(constraints)
        obs = tr.obs
    else
        error("not implemented")
    end
    lml = log_marginal_likelihood(new_params, new_traj, obs)
    @assert !isnan(lml)
    new_trace = ObsModelTrace(tr.gen_fn, args, obs, lml)
    return (new_trace, lml - tr.lml, NoChange(), EmptyChoiceMap())
end

function Gen.regenerate(tr::ObsModelTrace, args::Tuple, argdiffs::Tuple, selection::Selection)
    if !isempty(selection)
        error("not implemented")
    end
    (new_trace, weight, retdiff, _) = update(tr, args, argdiffs, EmptyChoiceMap())
    return (new_trace, weight, retdiff)
end




@inline function noise_log_likelihood(params::ObsModelParams, traj_pt::Point, obs_pt::Point)
    ll = logpdf(normal, obs_pt.x, traj_pt.x, params.noise)
    ll += logpdf(normal, obs_pt.y, traj_pt.y, params.noise)
    return ll
end

@inline function populate_new_alpha!(new_alpha, prev_alpha, traj::Vector{Point}, obs_t::Point, params::ObsModelParams)
    for k in 1:length(traj)
        # observation model: obs[t] corresponds with traj[k]
        noise_ll = noise_log_likelihood(params, traj[k], obs_t)
        # dynamics model: given that we are k, the previous could have been
        # (i) k - 2 (skip one), (ii) k - 1 (advance as usual), or (iii) k
        # (don't advance, lag)
        if k > 2
            val1 = prev_alpha[k-2] + log(params.prob_skip)
        else
            val1 = -Inf
        end
        if k > 1
            val2 = prev_alpha[k-1] + log(params.prob_normal)
        else
            val2 = -Inf
        end
        val3 = prev_alpha[k] + log(params.prob_lag)
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
        log_probs[1] = log(params.prob_lag)
    elseif next_z == 2
        log_probs[1] = log(params.prob_normal)
        log_probs[2] = log(params.prob_lag)
    else
        log_probs[next_z] = log(params.prob_lag)
        log_probs[next_z-1] = log(params.prob_normal)
        log_probs[next_z-2] = log(params.prob_skip)
    end
    return log_probs
end

@inline function populate_initial_alpha!(alpha, params::ObsModelParams, trajectory::Vector{Point}, obs1::Point)
    if length(alpha) == 1
        alpha[1] = noise_log_likelihood(params, trajectory[1], obs1)
        return
    end
    fill!(alpha, -Inf)
    first_prob_normal = params.prob_lag + params.prob_normal
    first_prob_skip = params.prob_skip
    alpha[1] = log(first_prob_normal) + noise_log_likelihood(params, trajectory[1], obs1) # p(z1) is deterministic at start of traj
    alpha[2] = log(first_prob_skip) + noise_log_likelihood(params, trajectory[2], obs1) # p(z1) is deterministic at start of traj
    return nothing
end

function log_marginal_likelihood(params::ObsModelParams, trajectory::Vector{Point}, obs::Vector{Point})
    K = length(trajectory)
    T = length(obs)
    alpha = Vector{Float64}(undef, K)
    new_alpha = Vector{Float64}(undef, K)
    populate_initial_alpha!(alpha, params, trajectory, obs[1])
    for t in 2:T
        populate_new_alpha!(new_alpha, alpha, trajectory, obs[t], params)
        tmp = alpha; alpha = new_alpha; new_alpha = tmp
    end
    return logsumexp(alpha)
end

function get_best_alignment(params::ObsModelParams, trajectory::Vector{Point}, obs::Vector{Point})
    K = length(trajectory)
    T = length(obs)

    # run forward filtering backward sampling

    # forward filtering
    alphas = Matrix{Float64}(undef, T, K) # optimize memory access?
    populate_initial_alpha!(view(alphas, 1, :), params, trajectory, obs[1])
    for t in 2:T
        populate_new_alpha!(view(alphas, t, :), view(alphas, t-1, :), trajectory, obs[t], params)
    end

    # backward sampling
    alignment = Vector{Int}(undef, length(trajectory))
    ldist = alphas[T,:]
    dist = exp.(ldist .- logsumexp(ldist))
    alignment[T] = categorical(dist)
    for t in T-1:-1:1
        ldist = alphas[t,:] .+ forward_transition_log_probs(alignment[t+1], params, K)
        dist = exp.(ldist .- logsumexp(ldist))
        alignment[t] = categorical(dist)
    end
    return alignment
end

function test_log_marginal_likelihood()
    prob_lag = 0.1
    prob_normal = 0.6
    prob_skip = 0.3
    noise = 1.0
    obs_params = ObsModelParams(prob_lag, prob_normal, prob_skip, noise)
    likelihood(a, b) = exp(noise_log_likelihood(obs_params, a, b))
    A = Point(0,0)
    B = Point(1,1)
    C = Point(2,2)
    obs1 = Point(0,0)
    obs2 = Point(1,1)
    actual = log_marginal_likelihood(obs_params, [A, B, C], [obs1, obs2])
    first_prob_normal = obs_params.prob_lag + obs_params.prob_normal
    first_prob_skip = obs_params.prob_skip
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
    println(expected)
    println(log(expected))
    println(actual)
    @assert isapprox(actual, log(expected))
end

test_log_marginal_likelihood()
