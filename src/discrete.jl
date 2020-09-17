using JLD
using Gen
using Parameters: @with_kw

include("../trace-translator/hmm.jl")

abstract type Discretization end

@with_kw mutable struct CustomDiscretization <: Discretization
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    e::Float64
    f::Float64
    g::Float64
    h::Float64
    i::Float64
    j::Float64
    k::Float64
    l::Float64
    bounds::Bounds
    bin_bounds::Vector{Bounds}
end

function set_bin_bounds!(disc::CustomDiscretization)
    bin_bounds = Vector{Bounds}(undef, get_num_bins(disc))

    # above a
    bin_bounds[1] = Bounds(0.0, disc.d, disc.a, 1.0)
    bin_bounds[2] = Bounds(disc.d, disc.f, disc.a, 1.0)
    bin_bounds[3] = Bounds(disc.f, 1.0, disc.a, 1.0)

    # above b
    bin_bounds[4] = Bounds(0.0, disc.d, disc.b, disc.a)
    bin_bounds[5] = Bounds(disc.d, disc.f, disc.b, disc.a)
    bin_bounds[6] = Bounds(disc.f, disc.g, disc.b, disc.a)
    bin_bounds[7] = Bounds(disc.g, 1.0, disc.b, disc.a)

    # above e, left of g
    bin_bounds[8] = Bounds(0.0, disc.d, disc.e, disc.b)
    bin_bounds[9] = Bounds(disc.d, disc.h, disc.e, disc.b)
    bin_bounds[10] = Bounds(disc.h, disc.g, disc.e, disc.b)

    # above c, left of g
    bin_bounds[11] = Bounds(0.0, disc.d, disc.c, disc.e)
    bin_bounds[12] = Bounds(disc.d, disc.h, disc.c, disc.e)
    bin_bounds[13] = Bounds(disc.h, disc.g, disc.c, disc.e)

    # above i, right of g
    bin_bounds[14] = Bounds(disc.g, 1.0, disc.i, disc.b)

    # below i, right of g
    bin_bounds[15] = Bounds(disc.g, disc.k, disc.c, disc.i)
    bin_bounds[16] = Bounds(disc.k, 1.0, disc.c, disc.i)

    # below c
    bin_bounds[17] = Bounds(0.0, disc.d, 0.0, disc.c)
    bin_bounds[18] = Bounds(disc.d, disc.j, 0.0, disc.c)
    bin_bounds[19] = Bounds(disc.j, disc.l, 0.0, disc.c)
    bin_bounds[20] = Bounds(disc.l, 1.0, 0.0, disc.c)
    
    disc.bin_bounds = bin_bounds
end

get_num_bins(disc::CustomDiscretization) = 20

function fix(disc::CustomDiscretization, pt::Point)
    return Point(
        max(disc.bounds.xmin + 1e-10, min(disc.bounds.xmax, pt.x) - 1e-10),
        max(disc.bounds.ymin + 1e-10, min(disc.bounds.ymax, pt.y) - 1e-10))
end

function get_bin(disc::CustomDiscretization, pt::Point)
    if pt.y > disc.b
        if pt.y > disc.a
            if pt.x < disc.d
                return 1
            elseif pt.x < disc.f
                return 2
            else
                return 3
            end
        else
            if pt.x < disc.f
                if pt.x < disc.d
                    return 4
                else
                    return 5
                end
            else
                if pt.x < disc.g
                    return 6
                else
                    return 7
                end
            end
        end
    else

        if pt.y < disc.c
            if pt.x < disc.j
                if pt.x < disc.d
                    return 17
                else
                    return 18
                end
            else
                if pt.x < disc.l
                    return 19
                else
                    return 20
                end
            end
        else
            if pt.x < disc.g
#Point(0.20696641618060418, 0.3581357819832387)
                if pt.y < disc.e
                    if pt.x < disc.d
                        return 11
                    else
                        if pt.x < disc.h
                            return 12
                        else
                            return 13
                        end
                    end
                else
                    if pt.x < disc.d
                        return 8
                    else
                        if pt.x < disc.h
                            return 9
                        else
                            return 10
                        end
                    end
                end
            else
                if pt.y < disc.i
                    if pt.x < disc.k
                        return 15
                    else
                        return 16
                    end
                else
                    return 14
                end
            end
        end
    end
    println(pt)
    @assert false
end

function get_bounds(disc::CustomDiscretization, bin_idx::Int)
    bnds = disc.bin_bounds[bin_idx]
    return (bnds.xmin, bnds.xmax, bnds.ymin, bnds.ymax)
end

const custom_disc = CustomDiscretization(
    a = 0.85,
    b = 0.59,
    c = 0.185,
    d = 0.242,
    e = 0.385,
    f = 0.435,
    g = 0.662,
    h = 0.540,
    i = 0.342,
    j = 0.61,
    k = 0.731,
    l = 0.845,
    bounds = Bounds(0.0, 1.0, 0.0, 1.0),
    bin_bounds = Vector{Bounds}())

set_bin_bounds!(custom_disc)

function load_hmms(start::Point)
    d = load("../trace-translator/trained_hmms_new.jld")
    ps = d["prior"]
    ts = d["transition"]
    es = d["emission"]
    num_waypoints = length(ts)
    hmms = Vector{HiddenMarkovModel}()
    start_bin = get_bin(custom_disc, start)
    for i in 1:num_waypoints
        prior_vec = 1e-6 * ones(get_num_bins(custom_disc))
        prior_vec[start_bin] = 1.0
        prior_vec = prior_vec / sum(prior_vec)
        push!(hmms, HiddenMarkovModel(
            prior_vec, ts[i], es[i]))
    end
    return hmms
end

struct CollapsedHMMTrace <: Trace
    obs::Vector{Int}
    score::Float64
end

Gen.get_score(trace::CollapsedHMMTrace) = trace.score
Gen.project(trace::CollapsedHMMTrace, ::EmptySelection) = 0.0
function Gen.get_choices(trace::CollapsedHMMTrace)
    choices = choicemap()
    for t in 1:length(trace.obs)
        choices[t] = trace.obs[t]
    end
    return choices
end
Gen.get_retval(trace::CollapsedHMMTrace) = nothing

struct CollapsedHMM <: GenerativeFunction{Nothing,CollapsedHMMTrace}
end

const lfprobs = Matrix{Float64}(undef, 10, get_num_bins(custom_disc)) # 10 is the number of time steps
const arr = Vector{Float64}(undef, get_num_bins(custom_disc))

function Gen.generate(::CollapsedHMM, args::Tuple, constraints::ChoiceMap)
    hmm, T = args
    obs = Vector{Int}(undef, T)
    for t in 1:T
        obs[t] = constraints[t]
    end
    score = hmm_log_marginal_likelihood(hmm, obs)
    trace = CollapsedHMMTrace(obs, score)
    return (trace, score)
end

const collapsed_hmm = CollapsedHMM()

@gen function discrete_single_waypoint_model(disc::Discretization, start_idx::Int, T::Int)
    waypoint_idx ~ uniform_discrete(1, get_num_bins(disc))
    observations ~ collapsed_hmm(hmms[waypoint_idx], T) # :observations => 1, ... :observations => T
end

function discrete_single_waypoint_posterior(disc::Discretization, start_idx::Int, obs::Vector{Int})
    choices = choicemap()
    for (i, v) in enumerate(obs)
        choices[:observations => i] = v
    end
    probs = Vector{Float64}(undef, get_num_bins(disc))
    for waypoint_idx in 1:get_num_bins(disc)
        choices[:waypoint_idx] = waypoint_idx
        _, weight = generate(discrete_single_waypoint_model, (disc, start_idx, length(obs)), choices)
        probs[waypoint_idx] = weight
    end
    probs = exp.(probs .- logsumexp(probs))
    @assert isapprox(sum(probs), 1.0, atol=1e-2)
    probs = probs / sum(probs)
    return probs
end

function load_hmms(start::Point)
    d = load("../trace-translator/trained_hmms_new.jld")
    ps = d["prior"]
    ts = d["transition"]
    es = d["emission"]
    num_waypoints = length(ts)
    hmms = Vector{HiddenMarkovModel}()
    start_bin = get_bin(custom_disc, start)
    for i in 1:num_waypoints
        prior_vec = 1e-6 * ones(get_num_bins(custom_disc))
        prior_vec[start_bin] = 1.0
        prior_vec = prior_vec / sum(prior_vec)
        push!(hmms, HiddenMarkovModel(
            prior_vec, ts[i], es[i]))
    end
    return hmms
end

