export motion_and_measurement_model_collapsed
export motion_and_measurement_model_collapsed_incremental
export log_marginal_likelihood

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
Gen.get_args(trace::ObsModelTrace) = (trace.path, trace.params, length(trace.measurements))
Gen.get_retval(trace::ObsModelTrace) = trace.measurements
Gen.get_score(trace::ObsModelTrace) = trace.lml
Gen.project(trace::ObsModelTrace, ::EmptySelection) = 0.0
Gen.project(trace::ObsModelTrace, ::AllSelection) = trace.lml

function Base.getindex(trace::ObsModelTrace, addr)
    # auxiliary data
    if addr == :locations
        params = trace.params
        path = trace.path
        measurements = trace.measurements
        return sample_locations_exact_conditional(params, path, measurements)
    else
        return get_choices(trace)[addr]
    end
end

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

# the maximum number of steps taken per time step is 2
num_hidden_states(T::Int) = 2*(T-1)+1

function Gen.simulate(gen_fn::ObsModel, args::Tuple)
    path, params, T = args

    # sample
    measurements = motion_and_measurement_model_uncollapsed(path, params, T)

    # compute marginal likelihood
    (dp_points_along_path, dp_progress) = walk_path(path, params.nominal_speed, num_hidden_states(T))
    (alphas, lml) = forward_filtering(params, dp_points_along_path, measurements)
    @assert !isnan(lml)

    trace = ObsModelTrace(gen_fn, path, params, dp_progress, dp_points_along_path, measurements, lml, alphas)

    # check invariant
    @assert num_hidden_states(T) == length(trace.dp_points_along_path)

    return trace
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

    # check invariant
    @assert num_hidden_states(T) == length(dp_points_along_path)

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
    (dp_points_along_path, dp_progress) = walk_path(new_path, new_params.nominal_speed, num_hidden_states(T))

    # check invariant
    @assert num_hidden_states(T) == length(dp_points_along_path)

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
