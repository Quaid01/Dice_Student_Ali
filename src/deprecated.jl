# Script deprecated.jl

# Collection of deprecated methods in a random order
# Can be moved to obsoletes.jl at any instance, do not use!

function get_initial_2(Nvert::Int, (vmin, vmax), p=0.5)
    # Generate random configuration of length `Nvert` in the separated
    # representation with the continuous component uniformly distributed
    # in the (vmin, vmax) interval
    #
    # NOTE: OBSOLETE, replaced by get_random_hybrid

    return realign_2((get_random_configuration(Nvert, p),
        get_initial(Nvert, (vmin, vmax))))
end

function realign_2(conf::Hybrid, r::Float64 = 0.0)
    # Changes the reference point for the separated representation by `r`
    # according to xi - r = sigma(r) + X(r)
    # INPUT & OUTPUT:
    #     conf = (sigma, X)
    #
    # NOTE: Obsolete, replaced by realign_hybrid
    V = Dice.hybrid_to_cont(conf)
    return Dice.cont_to_hybrid(V, r)
end

function propagate_2(model::Model, tmax, Sstart::SpinConf, Xstart::FVector)
    # Advances the model in the initial state (Sstart, Xstart)
    # for tmax time steps
    #
    # NOTE: Legacy version, use dispatched propagate instead
    X = Xstart
    S = Sstart
    graph = model.graph
    scale = model.scale
    mtd = model.coupling
    # Ns = model.Ns
    # noise = model.noise
    for _ = 1:(tmax-1)
        #        DX = step_rate_hybrid(graph, mtd, S, X, Ns, noise).*scale
        DX = step_rate_hybrid(graph, mtd, S, X) .* scale
        update_2!(S, X, DX)
    end
    return (S, X)
end

function trajectories_2(model::Model, tmax, Sstart::SpinConf,
    Xstart::FVector)
    # Advances the model in the initial state (Sstart, Xstart)
    # for tmax time steps
    # Keeps the full history of progression
    #
    # NOTE: Legacy version, use dispatched trajectories instead
    X = Xstart
    S = Sstart

    XFull = Xstart
    SFull = Sstart

    graph = model.graph
    scale = model.scale
    mtd = model.coupling
    # Ns = model.Ns
    # noise = model.noise
    for _ = 1:(tmax-1)
        #        DX = step_rate_hybrid(graph, mtd, S, X, Ns, noise).*scale
        DX = step_rate_hybrid(graph, mtd, S, X) .* scale
        update_2!(S, X, DX)
        XFull = [XFull X]
        SFull = [SFull S]
    end
    return (SFull, XFull)
end
