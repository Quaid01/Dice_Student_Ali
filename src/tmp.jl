# Temporary cache awainting complete deletion

"""
    energy (model, V)

Evaluate the coupling energy given distribution `V` in `model`.

The function uses the `energy` method of the `model` for finding the
coupling energy between pairs of nodes.

NOTE: This feature makes it ill-suited for treating local contributions
(anisotropy, field-like and such).
"""
function energy(model::Model, V::Array)
    # Evaluates the coupling energy corresponding to V
    # Note: this is essential that this energy evaluates only
    # coupling energy without any anisotropy
    #
    # INPUT:

    en = 0
    for edge in edges(model.graph)
        en += model.energy(V[edge.src], V[edge.dst])
    end

    return en
end

"""
    H_Ising(graph::SimpleGraph, conf::Array{Int, 1})

Evaluate the Ising energy for the provided `graph` and configuration `conf`.

INPUT:
      graph - Graphs object
      conf - configuration array from {-1, 1}^N
    
OUTPUT:
      energy = conf * A * conf /2 
"""
function H_Ising(graph::SimpleGraph, conf::Array{Integer, 1})
 
    en = 0
    for edge in edges(graph)
        en += conf[edge.src]*conf[edge.dst]
    end
    return en
end

function step_rate(graph::SimpleGraph, method::Function,
                   V::Array{Float64, 1}, Ks::Float64,
                   Ns::Float64, noise::Function)::Array{Float64, 1}
    # Evaluate ΔV for a single step
    # This version presumes that there is a single method for coupling and
    # anisotropy and that there is only easy-axis anisotropy
    #
    # This is a default function (notice how the anisotropy is evaluated
    # using method(V, -V)). It should be used with care as the
    # implemented idea of a "single method" is flawed. A true single
    # method would imply the presence of a field (a fixed
    # vertex). Here, however, the "single method" is used for
    # anisotropy, which presumes a tensor.
    #
    # INPUT:
    #    graph - unweighted graph carrying V's 
    #    method - method to evaluate different contributions
    #   V(1:|graph|) - current distribution of dynamical variables
    #    Ks - anisotropy constant
    #
    # OUTPUT:
    #   ΔV(1:|graph|) - array of increments

    #    out = Ks .* method(V, -V) # refactoring to scalars
    out  = zeros(Float64, size(V))
    for node in vertices(graph)
        Vnode = V[node]
        out[node] =  Ks .* method(Vnode, -Vnode)
        
        for neib in neighbors(graph, node)
            out[node] += method(Vnode, V[neib])
        end
    end
    noise_add::Array{Float64, 1} = Ns.*noise(length(out))
    return out + noise_add
end

function trajectories(graph::ModelGraph, duration, scale, method::Function,
                      Ks, Vini, Ns, noise::Function)
    # Advances the graph duration - 1 steps forward
    # This is the verbose version, which returns the full dynamics
    #
    # scale - parameter to tweak the dynamics
    # duration - how many time points to evaluate
    # V0 - the initial conditions
    #
    # OUTPUT:
    #   VFull = [V(0) V(1) ... V(duration-1)]

    VFull = Vini
    V = Vini

    for _ = 1:(duration - 1)
        ΔV = scale .* step_rate(graph, method, V, Ks, Ns, noise)
        V += ΔV
        VFull = [VFull V]
    end

    return VFull
end

function propagate(graph, duration, scale, method::Function, Ks,
                   Vini::Array{Float64, 1}, Ns, noise::Function)
    # Advances the graph duration - 1 steps forward
    # This is the short version, which returns only the final state vector
    #
    # scale - parameter to tweak the dynamics
    # duration - how many time points to evaluate
    # Vini - the initial conditions
    #
    # OUTPUT:
    #   [V[1] .. V[nv(graph)] at t = duration - 1
    #
    # NOTE: the order of parameters changed in 0.2.0

    V = Vini

    for _ = 1:(duration - 1)
        ΔV = scale .* step_rate(graph, method, V, Ks, Ns, noise)
        V += ΔV
    end

    return V
end

function step_rate_2(graph::SimpleGraph, coupling::Function,
                     s::Array{Int8, 1}, x::Array{Float64, 1},
                     Ns::Float64, Noise::Function)::Array{Float64, 1}
    out = zeros(Float64, size(x))
    for node in vertices(graph)
        xnode = x[node]
        for neib in neighbors(graph, node)
            out[node] += s[neib]*coupling(xnode, x[neib])
        end
        out[node] *= s[node]
    end
    noise_add::Array{Float64, 1} = Ns.*Noise(length(out))
    return out + noise_add
end


# """
#     cut(graph::SimpleGraph, conf::Array{Int, 1})
#
# Evaluate the cut value for the given graph and binary configuration
#
# INPUT:
#     graph - Graphs object
#     conf - binary configuration array {-1,1}^N
# OUTPUT:
#     sum_e (1 - e1. e.2)/2
#
# NOTE: This function is supposed to be generic and, therefore, is not
# restricted to integer cuts.
# """
# function cut(graph::SimpleGraph, conf::Array{Int, 1})
#
#     if nv(graph) != length(conf)
#         println("ERROR: The configuration size $(length(conf)) and the graph size $(nv(graph)) do not match")
#     end # side note: turned out to be useful
#
#     out = 0
#     for edge in edges(graph)
#         out += (1 - conf[edge.src] * conf[edge.dst]) / 2
#     end
#     return out
# end

# """
#     cut(graph::SimpleGraph, conf::Array{Int8, 1})
#
# Evaluate the cut value for the given graph and binary configuration.
# This version is mostly intended for internal purposes as it presumes
# that the configuration is an Int8 array.
#
# INPUT:
#     graph - Graphs object
#     conf - binary configuration Int8-array {-1,1}^N
# OUTPUT:
#     sum_e (1 - e1. e.2)/2
#
# NOTE: This function is supposed to be generic and, therefore, is not
# restricted to integer cuts.
# """
# function cut(graph::SimpleGraph, conf::Array{Int8, 1})
#
#     if nv(graph) != length(conf)
#         println("ERROR: The configuration size $(length(conf)) and the graph size $(nv(graph)) do not match")
#     end # side note: turned out to be useful
#
#     out = 0
#     for edge in edges(graph)
#         out += (1 - conf[edge.src] * conf[edge.dst]) / 2
#     end
#     return out
# end

# """
#     cut(model, conf, isbinary)
#
# Evaluate the cut value for the given model and binary configuration
#
# INPUT:
#     model - Dice's model
#     conf - binary configuration array with elemnts ± 1
#     isbinary - (true, false) If true (default), evaluate non-weighted cut
#
# OUTPUT:
#     (currently) sum_e (1 - e1. e.2)/2
#     (in perspective) sum_e w(e) (1 - e1. e.2)/2
#
# TODO:
#     1. Implement supporting model's weights
# """
# function cut(model::Model, conf::Array; isbinary::Bool)
#     # TODO
#     # What this implementation of the cut function should do
#     # cut(model, configuration, isbinary)
#     # if isbinary is set, treat conf as a binary array and use
#     # the "standard" cut
#     # if isbinary is false, treat conf as a continuous array
#     # and use the model's energy method for evaluating cut
#     if isbinary # TODO: integer cut
#         return cut(model.graph, conf)
#     else # TODO: weighted cut
#         return cut(model.graph, conf)
#     end
# end
