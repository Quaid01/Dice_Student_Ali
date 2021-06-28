# Dynamical Ising solver 
#
# This is an abridged version with basic functionality for the triangular paper
# The full development version is available at
# https://github.com/merement/Dice
#
# Misha Erementchouk, University of Michigan, 2021

module DiceBasic

using Base: Integer
using LightGraphs

export Model, sine, cosine, triangular, roundup, cut, get_best_cut, get_best_configuration, 
        get_best_rounding, get_initial, local_search, local_twosearch, extract_configuration,
        propagate

# The main type describing the Model
# For a compact description of simulation scenarios and controlling the module behavior
const silence_default = 3
mutable struct Model
    graph::SimpleGraph  # The graph carrying spins
    method::Function    # The coupling function
    scale::Float64      # Defines the magnitude of the exchange terms
    Ks::Float64         # The magnitude of the anisotropy term in scale's (not used)
    silence::Integer    # the inverse level of verbosity
    Model(graph, method, scale) = new(graph, method, scale, 0, silence_default)
    Model(graph, method, scale, Ks) = new(graph, method, scale, Ks, silence_default)
    Model(graph, method, scale, Ks, silence) = new(graph, method, scale, Ks, silence)
end

##############################
#
# Common methods for using in evaluating the change rate
#
##############################

const Pi2 = pi/2
const Pi4 = pi/4

function cosine(v) # this is the coupling energy inucing the sine model
    return cos.(Pi2.*v)
end

function sine(v)
    return Pi4.*sin.(Pi2.*v)
end

function sine(v1, v2)
    return sine(v1 - v2)
end

function triangular(v)
    p = Int.(floor.(v./2 .+ 1/2))
    return (v .- p.*2).*dtri(v)
end
    
function triangular(v1, v2)
    return triangular(v1-v2)
end

###############################
#
# Configuration analysis methods
#
###############################

function roundup(V)
    # returns V rounded into the interval [-2, 2]
    
    return mod.(V .+ 2, 4) .- 2
end

function cut(graph, conf)
    # Evaluates the cut value for the given graph and binary configuration
    #
    # INPUT:
    #   graph - LightGraphs object
    #   conf - binary configuration array with elemnts \pm 1
    #
    # OUTPUT: 
    #   sum_e (1 - e1. e.2)/2

    if nv(graph) != length(conf)
        num = conf_to_number(conf)
        println("ERROR: The configuration $num and the vertex set have different size")
    end
    
    out = 0
    for edge in edges(graph)
        out += (1 - conf[edge.src]*conf[edge.dst])/2
    end
    return out
end

function get_best_rounding(graph, V)
    # Finds the best cut following the CirCut algorithm
    # Returns the cut, the configuration, and the threshold (t_c)
    #
    # NOTE:
    #   The algorithm operates with the left boundary of the identifying interval.
    #   The function extract_configuration, in turn, asks for the rounding center (bad design?).
    #
    # INPUT:
    #   graph
    #   V - array with the voltage distribution assumed to be within [-2, 2]
    #
    # OUTPUT:
    #   (bestcut, bestbnd)
    #           where
    #               bestcut - the best cut found
    #               bestconf - rounded configuration
    #               bestbnd - the position of the respective rounding center (t_c)

    Nvert = nv(graph)

    bestcut = -1
    bestth = -2
    bestconf = zeros(Nvert)

    d = 2 # half-width of the interval
    
    vvalues = sort(V)
    push!(vvalues, 2)
    threshold = -2

    start = 1
    stop = findfirst(t -> t > 0, vvalues)
    if isnothing(stop) # there's no such element
        stop = Nvert + 1
    end

    while threshold < 0
        # here, we convert the tracked left boundary to the rounding center
        conf = extract_configuration(V, threshold + 1)
        cucu = cut(graph, conf)
        if cucu > bestcut
            bestcut = cucu
            bestth = threshold
            bestconf = conf
        end
        if vvalues[start] <= vvalues[stop] - d
            threshold = vvalues[start]
            start += 1
        else
            threshold = vvalues[stop] - d
            stop += 1
        end
    end

    return (bestcut, bestconf, bestth + 1)  # again, the conversion since the center is expected
end

function get_best_configuration(graph, V)
    # Finds the best cut following the CirCut algorithm
    # Returns the cut and the configuration
    #
    # INPUT:
    #   graph
    #   V - array with the voltage distribution assumed to be within [-2, 2]
    #
    # OUTPUT:
    #   (bestcut, bestbnd)
    #           where
    #               bestcut - the best cut found
    #               bestconf - rounded configuration
    
    (becu, beco, beth) = get_best_rounding(graph, V)
    return (becu, beco)
end

function get_best_cut(graph, V)
    # Finds the best cut following the CirCut algorithm
    # Returns the cut
    #
    # INPUT:
    #   graph
    #   V - array with the voltage distribution assumed to be within [-2, 2]
    #
    # OUTPUT:
    #   bestcut - the best cut found

    (becu, beco, beth) = get_best_rounding(graph, V)
    return becu
end

#######################
#
# Configuration processing methods
#
#######################

function majority_flip!(graph, conf, node)
    # Flips conf[node] to be of the opposite sign to the majority of its neighbors

    flip_flag = false
    tot = 0
    for j in neighbors(graph, node)
        tot += conf[node]*conf[j]
    end
    if tot > 0
        conf[node] *= -1
        flip_flag = true
    end
    return flip_flag
end

function majority_twoflip!(graph, conf, cut_edge)
    # Flips a cut pair if the neighborhood of the pair has the wrong kind of majority
    flip_flag = false
    tot = 0
    for i in neighbors(graph, cut_edge.src)
        tot += conf[cut_edge.src]*conf[i]
    end
    for i in neighbors(graph, cut_edge.dst)
        tot += conf[cut_edge.dst]*conf[i]
    end

    if tot > -2
        conf[cut_edge.src] *= -1
        conf[cut_edge.dst] *= -1
        flip_flag = true
    end
    return flip_flag
end

function local_search(graph, conf)
    # Eliminates vertices breaking the majority rule
    nonstop = true
    while nonstop
        nonstop = false
        for node in vertices(graph)
            nonstop |= Dice.majority_flip!(graph, conf, node)
        end
    end
    return conf
end

function local_twosearch(graph, conf)
    # Eliminates pairs breaking the majority rule
    nonstop = true
    while nonstop
        nonstop = false
        for link in edges(graph)
            if conf[link.src]*conf[link.dst] < 1
                nonstop |= Dice.majority_twoflip!(graph, conf, link)
            end
        end
    end
    return conf
end

function extract_configuration(V::Array, threshold)
    # Binarizes V according to the threshold
    # In the modular form, the mapping looks like
    #
    # V ∈ [left, left + 2] -> C_1
    # V ∈ [left - 2, left] -> C_2
    #
    # INPUT:
    #   V - data array (is presumed to be rounded and within [-2, 2])
    #   threshold - the rounding center
    #
    # OUTPUT:
    #   size(V) array with elements + 1 and -1 depending on the relation of 
    #           the respective V elements with threshold

    width = 1 # half-width of the rounding interval

    if abs(threshold) <= 1
        inds = threshold - width .<= V .< threshold + width 
    else
        return -extract_configuration(V, threshold - 2*sign(threshold))
    end
    out = 2 .* inds .- 1
    
    return out
end

####################
#
#   Dynamics methods
#
#####################

function get_initial(Nvert::Integer, (vmin, vmax))
    # Generate random vector with Nvert components uniformly distributed in the (vmin, vmax) intervals

    bot, top = minmax(vmin, vmax)
    mag = top - bot

    return mag.*rand(Float64, Nvert) .+ bot
end

function step_rate(graph::SimpleGraph, methods::Array{Function}, V::Array, Ks::Float64)
    # Evaluates ΔV for a single step
    # This version takes an array of methods for exploratory simulations
    #
    # INPUT:
    #    graph - unweighted graph carrying V's (TODO: allow for weights)
    #
    #    methods - array of methods to evaluate different contributions
    #               The anisotropy method takes only one variable
    #
    #   V(1:|graph|) - current distribution of dynamical variables
    #
    #    Ks - anisotropy constant
    #
    # OUTPUT:
    #   ΔV(1:|graph|) - array of increments

    out = Ks.*methods[2](2.0.*V)
    for node in vertices(graph)
        Vnode = V[node]

        for neib in neighbors(graph, node)
            out[node] += methods[1](Vnode, V[neib])
        end
    end
    return out
end

function step_rate(graph::SimpleGraph, method::Function, V::Array, Ks::Float64)
    # Evaluates ΔV for a single step
    # This version presumes that there is a single method for coupling and anisotropy
    # and that there is only easy-axis anisotropy
    #
    # INPUT:
    #    graph - unweighted graph carrying V's (TODO: allow for weights)
    #    method - method to evaluate different contributions
    #   V(1:|graph|) - current distribution of dynamical variables
    #    Ks - anisotropy constant
    #
    # OUTPUT:
    #   ΔV(1:|graph|) - array of increments

    out = Ks.*method(V, -V)
    for node in vertices(graph)
        Vnode = V[node]

        for neib in neighbors(graph, node)
            out[node] += method(Vnode, V[neib])
        end
    end
    return out
end

function propagate(model::Model, duration, Vini)
    # Advances the model::Model duration - 1 steps forward
    # Returns only the final state vector
    #
    # model - the model description
    # duration - how many time points to evaluate
    # Vini - the initial conditions
    #
    # OUTPUT:
    #   [V[1] .. V[nv(graph)] at t = duration - 1
       
    V = Vini
    graph = model.graph
    method = model.method
    Ks = model.Ks
    scale = model.scale
    
    for tau in 1:(duration - 1)
        ΔV = scale.*step_rate(graph, method, V, Ks)
        V += ΔV
    end
            
    return V
end

end # end of module DiceBasic
