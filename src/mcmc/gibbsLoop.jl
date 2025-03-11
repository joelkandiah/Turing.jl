
"""
    Gibbs

A type representing a Gibbs sampler.

# Constructors

`Gibbs` needs to be given a set of pairs of variable names and samplers. Instead of a single
variable name per sampler, one can also give an iterable of variables, all of which are
sampled by the same component sampler.

Each variable name can be given as either a `Symbol` or a `VarName`.

Some examples of valid constructors are:
```julia
Gibbs(:x => NUTS(), :y => MH())
Gibbs(@varname(x) => NUTS(), @varname(y) => MH())
Gibbs((@varname(x), :y) => NUTS(), :z => MH())
```

Currently only variable names without indexing are supported, so for instance
`Gibbs(@varname(x[1]) => NUTS())` does not work. This will hopefully change in the future.

# Fields
$(TYPEDFIELDS)
"""
struct GibbsLoop{N,V<:NTuple{N,AbstractVector{<:VarName}},A<:NTuple{N,Any}} <:
       InferenceAlgorithm
    # TODO(mhauru) Revisit whether A should have a fixed element type once
    # InferenceAlgorithm/Sampler types have been cleaned up.
    "varnames representing variables for each sampler"
    varnames::V
    "samplers for each entry in `varnames`"
    samplers::A

    function GibbsLoop(varnames, samplers)
        if length(varnames) != length(samplers)
            throw(ArgumentError("Number of varnames and samplers must match."))
        end

        for spl in samplers
            if !isgibbscomponent(spl)
                msg = "All samplers must be valid Gibbs components, $(spl) is not."
                throw(ArgumentError(msg))
            end
        end

        # Ensure that samplers have the same selector, and that varnames are lists of
        # VarNames.
        samplers = tuple(map(set_selector ∘ drop_space, samplers)...)
        varnames = tuple(map(to_varname_list, varnames)...)
        return new{length(samplers),typeof(varnames),typeof(samplers)}(varnames, samplers)
    end
end

function GibbsLoop(algs::Pair...)
    return GibbsLoop(map(first, algs), map(last, algs))
end

# The below two constructors only provide backwards compatibility with the constructor of
# the old Gibbs sampler. They are deprecated and will be removed in the future.
function GibbsLoop(alg1::InferenceAlgorithm, other_algs::InferenceAlgorithm...)
    algs = [alg1, other_algs...]
    varnames = map(algs) do alg
        space = getspace(alg)
        if (space isa VarName)
            space
        elseif (space isa Symbol)
            VarName{space}()
        else
            tuple((s isa Symbol ? VarName{s}() : s for s in space)...)
        end
    end
    msg = (
        "Specifying which sampler to use with which variable using syntax like " *
        "`Gibbs(NUTS(:x), MH(:y))` is deprecated and will be removed in the future. " *
        "Please use `Gibbs(; x=NUTS(), y=MH())` instead. If you want different iteration " *
        "counts for different subsamplers, use e.g. " *
        "`Gibbs(@varname(x) => RepeatSampler(NUTS(), 2), @varname(y) => MH())`"
    )
    Base.depwarn(msg, :Gibbs)
    return Gibbs(varnames, map(set_selector ∘ drop_space, algs))
end

function GibbsLoop(
    alg_with_iters1::Tuple{<:InferenceAlgorithm,Int},
    other_algs_with_iters::Tuple{<:InferenceAlgorithm,Int}...,
)
    algs_with_iters = [alg_with_iters1, other_algs_with_iters...]
    algs = Iterators.map(first, algs_with_iters)
    iters = Iterators.map(last, algs_with_iters)
    algs_duplicated = Iterators.flatten((
        Iterators.repeated(alg, iter) for (alg, iter) in zip(algs, iters)
    ))
    # This calls the other deprecated constructor from above, hence no need for a depwarn
    # here.
    return GibbsLoop(algs_duplicated...)
end

# TODO: Remove when no longer needed.
DynamicPPL.getspace(::GibbsLoop) = ()

"""
Initialise a VarInfo for the Gibbs sampler.

This is straight up copypasta from DynamicPPL's src/sampler.jl. It is repeated here to
support calling both step and step_warmup as the initial step. DynamicPPL initialstep is
incompatible with step_warmup.
"""
function initial_varinfo(rng, model, spl, initial_params)
    vi = DynamicPPL.default_varinfo(rng, model, spl)

    # Update the parameters if provided.
    if initial_params !== nothing
        vi = DynamicPPL.initialize_parameters!!(vi, initial_params, spl, model)

        # Update joint log probability.
        # This is a quick fix for https://github.com/TuringLang/Turing.jl/issues/1588
        # and https://github.com/TuringLang/Turing.jl/issues/1563
        # to avoid that existing variables are resampled
        vi = last(DynamicPPL.evaluate!!(model, vi, DynamicPPL.DefaultContext()))
    end
    return vi
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::DynamicPPL.Sampler{<:GibbsLoop};
    initial_params=nothing,
    kwargs...,
)
    alg = spl.alg
    varnames = alg.varnames
    samplers = alg.samplers
    vi = initial_varinfo(rng, model, spl, initial_params)

    vi, states = gibbs_initialstep_loop(
        rng,
        model,
        AbstractMCMC.step,
        varnames,
        samplers,
        vi;
        initial_params=initial_params,
        kwargs...,
    )
    return Transition(model, vi), GibbsState(vi, states)
end

function AbstractMCMC.step_warmup(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::DynamicPPL.Sampler{<:GibbsLoop};
    initial_params=nothing,
    kwargs...,
)
    alg = spl.alg
    varnames = alg.varnames
    samplers = alg.samplers
    vi = initial_varinfo(rng, model, spl, initial_params)

    vi, states = gibbs_initialstep_loop(
        rng,
        model,
        AbstractMCMC.step_warmup,
        varnames,
        samplers,
        vi;
        initial_params=initial_params,
        kwargs...,
    )
    return Transition(model, vi), GibbsState(vi, states)
end

"""
Take the first step of MCMC for the first component sampler, and call the same function
recursively on the remaining samplers, until no samplers remain. Return the global VarInfo
and a tuple of initial states for all component samplers.

The `step_function` argument should always be either AbstractMCMC.step or
AbstractMCMC.step_warmup.
"""
function gibbs_initialstep_loop(
    rng,
    model,
    step_function::Function,
    varname_vecs,
    samplers,
    vi,
    states=();
    initial_params=nothing,
    kwargs...,
)
    # End recursion
    # if isempty(varname_vecs) && isempty(samplers)
    #     return vi, states
    # end

    # varnames, varname_vecs_tail... = varname_vecs
    # sampler, samplers_tail... = samplers
    for varnames, sampler in zip(varname_vecs, samplers)
        # Get the initial values for this component sampler.
        initial_params_local = if initial_params === nothing
            nothing
        else
            DynamicPPL.subset(vi, varnames)[:]
        end

        # Construct the conditioned model.
        conditioned_model, context = make_conditional(model, varnames, vi)

        # Take initial step with the current sampler.
        _, new_state = step_function(
            rng,
            conditioned_model,
            sampler;
            # FIXME: This will cause issues if the sampler expects initial params in unconstrained space.
            # This is not the case for any samplers in Turing.jl, but will be for external samplers, etc.
            initial_params=initial_params_local,
            kwargs...,
        )
        new_vi_local = varinfo(new_state)
        # Merge in any new variables that were introduced during the step, but that
        # were not in the domain of the current sampler.
        vi = merge(vi, get_global_varinfo(context))
        # Merge the new values for all the variables sampled by the current sampler.
        vi = merge(vi, new_vi_local)

        states = (states..., new_state)
        # return gibbs_initialstep_recursive(
        #     rng,
        #     model,
        #     step_function,
        #     varname_vecs_tail,
        #     samplers_tail,
        #     vi,
        #     states;
        #     initial_params=initial_params,
        #     kwargs...,
        # )
    end

    return vi, states
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::DynamicPPL.Sampler{<:GibbsLoop},
    state::GibbsState;
    kwargs...,
)
    vi = varinfo(state)
    alg = spl.alg
    varnames = alg.varnames
    samplers = alg.samplers
    states = state.states
    @assert length(samplers) == length(state.states)

    vi, states = gibbs_step_loop(
        rng, model, AbstractMCMC.step, varnames, samplers, states, vi; kwargs...
    )
    return Transition(model, vi), GibbsState(vi, states)
end

function AbstractMCMC.step_warmup(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::DynamicPPL.Sampler{<:GibbsLoop},
    state::GibbsState;
    kwargs...,
)
    vi = varinfo(state)
    alg = spl.alg
    varnames = alg.varnames
    samplers = alg.samplers
    states = state.states
    @assert length(samplers) == length(state.states)

    vi, states = gibbs_step_loop(
        rng, model, AbstractMCMC.step_warmup, varnames, samplers, states, vi; kwargs...
    )
    return Transition(model, vi), GibbsState(vi, states)
end

"""
    setparams_varinfo!!(model, sampler::Sampler, state, params::AbstractVarInfo)

A lot like AbstractMCMC.setparams!!, but instead of taking a vector of parameters, takes an
`AbstractVarInfo` object. Also takes the `sampler` as an argument. By default, falls back to
`AbstractMCMC.setparams!!(model, state, params[:])`.

`model` is typically a `DynamicPPL.Model`, but can also be e.g. an
`AbstractMCMC.LogDensityModel`.
"""
# function setparams_varinfo!!(model, ::Sampler, state, params::AbstractVarInfo)
#     return AbstractMCMC.setparams!!(model, state, params[:])
# end

# function setparams_varinfo!!(
#     model::DynamicPPL.Model,
#     sampler::Sampler{<:MH},
#     state::AbstractVarInfo,
#     params::AbstractVarInfo,
# )
#     # The state is already a VarInfo, so we can just return `params`, but first we need to
#     # update its logprob.
#     # NOTE: Using `leafcontext(model.context)` here is a no-op, as it will be concatenated
#     # with `model.context` before hitting `model.f`.
#     return last(DynamicPPL.evaluate!!(model, params, DynamicPPL.leafcontext(model.context)))
# end

# function setparams_varinfo!!(
#     model::DynamicPPL.Model,
#     sampler::Sampler{<:ESS},
#     state::AbstractVarInfo,
#     params::AbstractVarInfo,
# )
#     # The state is already a VarInfo, so we can just return `params`, but first we need to
#     # update its logprob. To do this, we have to call evaluate!! with the sampler, rather
#     # than just a context, because ESS is peculiar in how it uses LikelihoodContext for
#     # some variables and DefaultContext for others.
#     return last(DynamicPPL.evaluate!!(model, params, SamplingContext(sampler)))
# end

# function setparams_varinfo!!(
#     model::DynamicPPL.Model,
#     sampler::Sampler{<:ExternalSampler},
#     state::TuringState,
#     params::AbstractVarInfo,
# )
#     logdensity = DynamicPPL.setmodel(state.logdensity, model, sampler.alg.adtype)
#     new_inner_state = setparams_varinfo!!(
#         AbstractMCMC.LogDensityModel(logdensity), sampler, state.state, params
#     )
#     return TuringState(new_inner_state, logdensity)
# end

# function setparams_varinfo!!(
#     model::DynamicPPL.Model,
#     sampler::Sampler{<:Hamiltonian},
#     state::HMCState,
#     params::AbstractVarInfo,
# )
#     θ_new = params[:]
#     hamiltonian = get_hamiltonian(model, sampler, params, state, length(θ_new))

#     # Update the parameter values in `state.z`.
#     # TODO: Avoid mutation
#     z = state.z
#     resize!(z.θ, length(θ_new))
#     z.θ .= θ_new
#     return HMCState(params, state.i, state.kernel, hamiltonian, z, state.adaptor)
# end

# function setparams_varinfo!!(
#     model::DynamicPPL.Model, sampler::Sampler{<:PG}, state::PGState, params::AbstractVarInfo
# )
#     return PGState(params, state.rng)
# end

"""
    match_linking!!(varinfo_local, prev_state_local, model)

Make sure the linked/invlinked status of varinfo_local matches that of the previous
state for this sampler. This is relevant when multilple samplers are sampling the same
variables, and one might need it to be linked while the other doesn't.
"""
# function match_linking!!(varinfo_local, prev_state_local, model)
#     prev_varinfo_local = varinfo(prev_state_local)
#     was_linked = DynamicPPL.istrans(prev_varinfo_local)
#     is_linked = DynamicPPL.istrans(varinfo_local)
#     if was_linked && !is_linked
#         varinfo_local = DynamicPPL.link!!(varinfo_local, model)
#     elseif !was_linked && is_linked
#         varinfo_local = DynamicPPL.invlink!!(varinfo_local, model)
#     end
#     # TODO(mhauru) The above might run into trouble if some variables are linked and others
#     # are not. `istrans(varinfo)` returns an `all` over the individual variables. This could
#     # especially be a problem with dynamic models, where new variables may get introduced,
#     # but also in cases where component samplers have partial overlap in their target
#     # variables. The below is how I would like to implement this, but DynamicPPL at this
#     # time does not support linking individual variables selected by `VarName`. It soon
#     # should though, so come back to this.
#     # Issue ref: https://github.com/TuringLang/Turing.jl/issues/2401
#     # prev_links_dict = Dict(vn => DynamicPPL.istrans(prev_varinfo_local, vn) for vn in keys(prev_varinfo_local))
#     # any_linked = any(values(prev_links_dict))
#     # for vn in keys(varinfo_local)
#     #     was_linked = if haskey(prev_varinfo_local, vn)
#     #         prev_links_dict[vn]
#     #     else
#     #         # If the old state didn't have this variable, we assume it was linked if _any_
#     #         # of the variables of the old state were linked.
#     #         any_linked
#     #     end
#     #     is_linked = DynamicPPL.istrans(varinfo_local, vn)
#     #     if was_linked && !is_linked
#     #         varinfo_local = DynamicPPL.invlink!!(varinfo_local, vn)
#     #     elseif !was_linked && is_linked
#     #         varinfo_local = DynamicPPL.link!!(varinfo_local, vn)
#     #     end
#     # end
#     return varinfo_local
# end

"""
Run a Gibbs step for the first varname/sampler/state tuple, and recursively call the same
function on the tail, until there are no more samplers left.

The `step_function` argument should always be either AbstractMCMC.step or
AbstractMCMC.step_warmup.
"""
function gibbs_step_loop(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    step_function::Function,
    varname_vecs,
    samplers,
    states,
    global_vi,
    new_states=();
    kwargs...,
)
    # End recursion.
    # if isempty(varname_vecs) && isempty(samplers) && isempty(states)
    #     return global_vi, new_states
    # end

    # varnames, varname_vecs_tail... = varname_vecs
    # sampler, samplers_tail... = samplers
    # state, states_tail... = states

    for varnames, sampler, state in zip(varname_vecs, samplers, states)

        # Construct the conditional model and the varinfo that this sampler should use.
        conditioned_model, context = make_conditional(model, varnames, global_vi)
        vi = subset(global_vi, varnames)
        vi = match_linking!!(vi, state, model)

        # TODO(mhauru) The below may be overkill. If the varnames for this sampler are not
        # sampled by other samplers, we don't need to `setparams`, but could rather simply
        # recompute the log probability. More over, in some cases the recomputation could also
        # be avoided, if e.g. the previous sampler has done all the necessary work already.
        # However, we've judged that doing any caching or other tricks to avoid this now would
        # be premature optimization. In most use cases of Gibbs a single model call here is not
        # going to be a significant expense anyway.
        # Set the state of the current sampler, accounting for any changes made by other
        # samplers.
        state = setparams_varinfo!!(conditioned_model, sampler, state, vi)

        # Take a step with the local sampler.
        new_state = last(step_function(rng, conditioned_model, sampler, state; kwargs...))

        new_vi_local = varinfo(new_state)
        # Merge the latest values for all the variables in the current sampler.
        new_global_vi = merge(get_global_varinfo(context), new_vi_local)
        new_global_vi = setlogp!!(new_global_vi, getlogp(new_vi_local))

        new_states = (new_states..., new_state)
        # return gibbs_step_recursive(
        #     rng,
        #     model,
        #     step_function,
        #     varname_vecs_tail,
        #     samplers_tail,
        #     states_tail,
        #     new_global_vi,
        #     new_states;
        #     kwargs...,
        # )

        global_vi = new_global_vi
    end
    return global_vi, new_states
end
