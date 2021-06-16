##################################
# SMC-based pseudomarginal model #
##################################

import GenSMC

export make_motion_and_measurement_model_smc_pseudomarginal
export make_motion_and_measurement_model_smc_pseudomarginal_optimal_local_proposal

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

function get_steps_addr(::typeof(motion_and_measurement_model_uncollapsed_incremental), t::Int)
    if t == 1
        return :init_steps
    else
        return :rest_states => (t-1) => :steps
    end
end

@gen function optimal_local_proposal(trace, data)
    T = get_args(trace)[3]
    y = data[obs_addr(motion_and_measurement_model_uncollapsed_incremental, T+1)]
    steps_addr = get_steps_addr(motion_and_measurement_model_uncollapsed_incremental, T+1)
    log_probs = Vector{Float64}(undef, 3)
    for i in 0:2 # 0, 1, 2 steps
        (_, log_probs[i+1], _, _) = update(
            trace, (get_args(trace)[1], get_args(trace)[2], T+1),
            (NoChange(), NoChange(), UnknownChange()),
            merge(data, choicemap((steps_addr, i))))
    end
    probs = exp.(log_probs .- logsumexp(log_probs))
    probs = probs / sum(probs)
    {steps_addr} ~ steps_prior(probs)
    return nothing
end

@gen function optimal_local_proposal_init(model, args, data)
    y = data[obs_addr(motion_and_measurement_model_uncollapsed_incremental, 1)]
    steps_addr = get_steps_addr(motion_and_measurement_model_uncollapsed_incremental, 1)
    log_probs = Vector{Float64}(undef, 3)
    for i in 0:2 # 0, 1, 2 steps
        (_, log_probs[i+1]) = generate(
            model, args,
            merge(data, choicemap((steps_addr, i))))
    end
    probs = exp.(log_probs .- logsumexp(log_probs))
    probs = probs / sum(probs)
    {steps_addr} ~ steps_prior(probs)
    return nothing
end

function make_motion_and_measurement_model_smc_pseudomarginal_optimal_local_proposal(num_particles::Int)
    return GenSMC.PFPseudoMarginalGF(
        model = motion_and_measurement_model_uncollapsed_incremental,
        data_addrs = (T) -> [obs_addr(motion_and_measurement_model_uncollapsed_incremental, T)],
        get_step_args = (args, T) -> (args[1], args[2], T),
        num_particles = num_particles,
        get_T = (args) -> args[3],
        reuse_particle_system = (args1, args2, argdiffs) -> (
            argdiffs[1] == NoChange() && argdiffs[2] == NoChange()),
        get_proposal = (T) -> (T == 1 ? optimal_local_proposal_init : optimal_local_proposal))
end

# TODO need to be able to attach methods to this specific instance
function obs_addr(::GenSMC.PFPseudoMarginalGF, T::Int)
    return obs_addr(motion_and_measurement_model_uncollapsed_incremental, T)
end
