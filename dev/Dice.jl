# Dynamical Ising solver
#
# Here, the dynamical variables are regarded from the perspective of the
# feasible configurations. Therefore, the Ising states are {-1, 1}^N,
# and the period of the main functions is [-2, 2).

# The main functions are F(v) and f = F'(v)
#
# F(v) is defined by
#
# C_G(v) = \sum_{(m,n) \in E} F(v_m - v_n), when v_m ∈ {-1, 1}
#
# So that F(0) = 0 (the same partition), F(±2) = -1. Hence, the period above.

# TODO:
#   0. [DONE] Fix rounding
#   1. [DONE] Make energy methods for correct energy evaluations
#   2. Enable LUT-defined methods
#   3. Add the weight support
#   4. Implement logging
#   5. Fix the chain of types
#   6. Admit differenttial equations for continuous time machines
#   7. Redesign: implement consistently the notion of functions on the graph
#   8. Redesign: incorporate linear terms
#
# NOTE:
# This is the version with explicitly defined functions for supporting
# the separated representation of Model II. The introduced functions are
#     separate
#     combine
#     update_2!
#     step_rate_2
#     propagate_2
#     trajectories_2
#     coupling_model_2
#     cut_2
#     integer_distribution

module Dice

using Base:Integer
using Arpack
using Graphs
using SparseArrays

export Model,
    get_connected, get_initial,
    get_random_configuration,
    sine, triangular,
    cut, get_best_cut, get_best_configuration, extract_configuration,
    H_Ising, energy,
    number_to_conf, 
    propagate, roundup,
    test_branch, scan_vicinity, scan_for_best_configuration,
    conf_decay,
    local_search, local_twosearch

# The main type describing the Model
# For a compact description of simulation scenarios and controlling the
# module behavior
const silence_default = 3
mutable struct Model
    graph::SimpleGraph
    # weights::Array
    method::Function # obsolete, phase out
    # dynamical coupling function
    coupling::Function
    # energy kernel function
    energy::Function
    # Defines the magnitude of the exchange terms
    # default = 1/Nvertices
    scale::Float64
    # The magnitude of the anisotropy term
    # default = 0
    Ks::Float64
    # anisotropy function
    # default = coupling(x, -x)
    anisotropy::Function
    extended::Bool # Whether there is an extension (remove?)
    # the inverse level of verbosity
    # default = silence_default
    silence::Integer 
    Model() = new() 
end

# Several explicit (legacy?) constructors
Model(graph, coupling) =
    begin
        M = Model()
        M.graph = graph
        M.coupling = coupling
        M.method = M.coupling
        # temporary placeholder
        M.energy = M.coupling
        # The default interpretation of anisotropy
        M.anisotropy = (x) -> M.coupling(x, -x) 
        M.scale = 1/nv(graph)
        M.Ks = 0
        M.silence = silence_default
        M.extended = false
        return M;
    end

Model(graph, coupling, scale) =
    begin
        M = Model(graph, coupling)
        M.scale = scale
        M.Ks = 0
        M.silence = silence_default
        M.extended = false
        return M;
    end

Model(graph, coupling, scale, Ks) =
    begin
        M = Model(graph, coupling, scale)
        M.Ks = Ks
        return M;
    end
        
############################################################
#
### Internal service functions
#
############################################################

function message(model::Model, out, importance=1)
    # must be replaced by a proper logging functionality
    if importance > model.silence
        println("$out ($importance)")
    end
end

const DEBUG = 1
function debug_msg(out)
    if DEBUG > 0
        println("$out (DEBUG = $DEBUG)")
    end
end

############################################################
#
### File support
#
############################################################

"""
Read graph in the "reduced" Matrixmarket format.

Usage:
    G = loadDumpedGraph(filename::String)::SimpleGraph

INPUT:
    filename::String name of the file

OUTPUT:
    SimpleGraph object

The reduced format:
    Header: lines starting with %
    |V| |E|
    u v w_{u,v}
"""
function loadDumpedGraph(filename::String)::SimpleGraph
    # TODO simple validation

    mtxFile = open(filename, "r")
    tokens = split(lowercase(readline(mtxFile)))
    # Skip commenting header
    while tokens[1][1] == '%'
        global tokens
        tokens = split(lowercase(readline(mtxFile)))
    end

    Nodes, Edges = map((x) -> parse(Int64, x), tokens)
    G = SimpleGraph(Nodes)

    for line in readlines(mtxFile)
        global tokens
        global countEdges
        tokens = split(lowercase(line))
        # we drop the weight for now
        u, v = map((x) -> parse(Int64, x), tokens[1:2])
        add_edge!(G, u, v)
    end
    close(mtxFile)

    return G
end

"""
    dumpGraph(Graph::SimpleGraph, filename::String)

Writes the sparce adjacency matrix of `Graph` to file `filename`.

NOTE: the current version writes the (0,1) matrix only.

The adjacency matrix is written in the reduced `MatrixMarket` format
format. The first line contains two numbers: `|V| |E|`
Next lines contain three numbers: `u v A_{u, v}`
where `u` and `v` are the nodes numbers and `A_{u, v}` is the edge weight.
"""
function dumpGraph(G, filename)
    weight = 1
    open(filename, "w") do out
        println(out, nv(G), " ", ne(G))
        for edge in edges(G)
            println(out, edge.src, " ", edge.dst, " ", weight)
        end
    end
end

############################################################
#
### A library of coupling functions (TODO: detach?)
#
############################################################

function none_method(v)
    return 0 * v  # for the type stability thing
end

function none_method(v1, v2)
    return 0 * v1  # for the type stability thing
end

const Pi2 = pi / 2
const Pi4 = pi / 4

"""
    cosine(v)

The coupling energy inducing the XY-model (rank-2 relaxation).
"""
function cosine(v)
    return cos.(Pi2 .* v)
end

function sine(v)
    return sin.(Pi2 .* v)
end

function sine(v1, v2)
    return sine(v1 - v2)
end

function linear(v1, v2)
    return v1 - v2
end

function dtri(v)
    # the derivative of the triangular function
    p = Int.(floor.(v ./ 2 .+ 1 / 2))
    parity = rem.(abs.(p), 2)
    return -2 .* parity .+ 1
end

function triangular(v)
    # Odd piece-wise linear defined within the period [-2, 2] by
    # phi(-2) = phi(0) = phi(2) = 0
    # - phi(-1) = phi(1) = 1
    # This works for both scalars and vectors 
    p = Int.(floor.(v ./ 2 .+ 1 / 2))
    parity = rem.(abs.(p), 2)
    return (v .- p .* 2) .* (-2 .* parity .+ 1)
end

function triangular_mod(v)
    # Odd piece-wise linear defined within the period [-2, 2] by
    # phi(-2) = phi(0) = phi(2) = 0
    # - phi(-1) = phi(1) = 1
    # this is slightly slower than triangular(v) above because of the mod
    # but looks easier to understand
    vr = mod.(v .+ 1, 4) .- 1
    return vr .+ min.(0 .* vr, -vr .+ 1).*2
end

function triangular(v1, v2)
    return triangular(v1 - v2)
end

const pwDELTA = 0.1
# const pwPERIOD = 4.0

"""
    piecewise_generic(v, Delta = 0.1)

Evaluate piece-wise linear odd periodic ([-2, 2]) skewed triangular
function of the vector argument `v` that equals to 0 at 0 and 2 and
to 1 at 1 + Delta.
"""
function piecewise_generic(v, Delta = 0.1)
    # This version is slow but generic
    vbar = mod.(v .+ 2, 4) .- 2
    s = sign.(vbar)
    vbabs = s.*vbar

    ind = vbabs .> 1 + Delta
    out = vbabs./(1 + Delta) - 2 .* ind.*(vbabs .- (1 + Delta))./(1 - Delta^2)

    return s .* out
end

function piecewise_local(v, Delta = 0.1)
    # this version relies on the small variation of v
    # TODO: make it decent
    vabs = abs(v)
    out =
        if vabs < 1 + Delta
            sign(v)/(1+Delta)*vabs
        else
            if vabs < 3 - Delta
                -sign(v)/(1 - Delta)*(vabs - 2)
            else
                if vabs < 5 + Delta
                    sign(v)/(1+Delta)*(vabs - 4)
                else
                    # add one more step?
                    piecewise_generic(v, Delta)
                end
            end
        end
    return out
 end

function piecewise(v1, v2, Delta = 0.1)
    return piecewise_local(v1 - v2, Delta)
end

function square(v)
    return dtri(v .+ 1)
end

function square(v1, v2)
    return square(v1 - v2)
end

function bilinear(v1, v2)
    return -dtri(v1) .* triangular(v2)
end

function bisine(v1, v2)
    return -Pi2 .* cos(Pi2 .* v1) .* triangular(Pi2 .* v2)
end

function squarishk(v, k=10)
    return tanh.(k .* triangular(v))
end

function squarish(v1, v2, k=10)
    return squarishk(v1 - v2, k)
end

# this really should be a model parameter
const stW = 0.55 * 2

function skew_triangular(v)
    c1 = 1 / (stW * (stW - 2))

    vbar = mod.(v .+ 2, 4) .- 2
    s = sign.(vbar)
    svbar = s .* vbar
    ind = sign.(svbar .- stW)

    mid = (svbar .* (stW - 1) .- stW) .* c1
    Delta = (svbar .- stW) .* c1

    out = Delta .* ind .+ mid
    return s .* out
end

function skew_triangular(v1, v2)
    return skew_triangular(v1 - v2)
end

#
# Continuous representation models
#

function continuous_model_1(v1, v2)
    return -dtri(v1).*triangular(v2)
end

function dtri_cont(v, s)
    # scalar version only
    vreduced = mod(v + 2, 4) - 2
    vabs = abs(vreduced)
    m = 2/(2 - s)
    out =
        if vabs < 1 - s
            m
        elseif vabs <= 1 + s
            (1 - vabs)*m/s
        else
            -m
        end
    return out
end

function continuous_model_1_cont(v1, v2, s = 0.1)
    # change rate in Model 1 with continuous derivative
    # s is the half-width of the transitional region between -1 and 1
    # v1 is presumed to be scalar
    # noticeably slower
    return -dtri_cont(v1, s).*triangular(v2)
end

# For model 2 (rank-2 GW-representation), the coupling function is
# sign(v)/2 for v \in [-2,2]
function continuous_model_2(v)
    vreduced = mod(v + 2, 4) - 2
    return sign(vreduced)/2
end

function continuous_model_2(v1, v2)
    return continuous_model_2(v1 - v2)
end

# The cut counting function for model II
# For v in [-2, 2], Phi(v) = |v|/2
function continuous_model_2_energy(v)
    vreduced = mod(v + 2, 4) - 2
    return abs(vreduced)/2
end

function continuous_model_2_energy(v1, v2)
    return continuous_model_2_energy(v1 - v2)
end

############################################################
#
### Analysis methods
#
############################################################

"""
    roundup(V)

Return `V` folded into [-2, 2].
"""
function roundup(V)
    return mod.(V .+ 2, 4) .- 2
end

"""
    separate(Vinput, rounding = 0)

Separate the discrete and continuous components of the given distribution
according to V = sigma + X + r, where sigma in {-1, +1}^N, X in [-1, 1)^N,
and -2 < r < 2 is the rounding center.

INPUT:
    Vinp - array to process
    r - the rounding center

OUTPUT:
    sigmas - integer arrays of binary spins (-1, +1)
    xs - array of displacements [-1, 1)
"""
function separate(Vinp, r = 0)
    V = mod.(Vinp .- r .+ 2, 4) .- 2
    sigmas = zeros(Int, size(V))
    xs = zeros(size(V))
    # sigmas = sgn(V) # with sgn(0) := 1
    # xs = V - sigmas
    for i in 1:length(V)
        Vred = V[i]
        (sigmas[i], xs[i]) =
            if Vred >= 0
                (1, Vred - 1)
            else
                (-1, Vred + 1)
            end
    end
    return (sigmas, xs)
end

"""
    combine(s, x, r = 0)

Recover the dynamic variables from the separated representation.

INPUT:
    s - the {-1, 1} array containing the discrete component
    x - the array with the continuous component
    r - the rounding center (default = 0)
"""
function combine(s, x, r = 0)
    return x .+ s .+ r
end

"""
    HammingD(s1, s2)

Evaluate the Hamming distance between binary strings `s1` and `s2`
"""
function HammingD(s1, s2)
    # Int(sum(round.(abs.(s1 - s2)./2)))
    count = 0
    for i in 1:length(s1)
        if s1[i] != s2[i]
            count += 1
        end
    end
    return count
end

"""
    EuclidD(V1, V2)

Evaluate the Euclidean distance between two distributions

INPUT:
  `V1`, `V2` - two arrays on the graph vertices

OUTPUT:
      Sum_v (V_1(v) - V_2(v))^2
"""
function EuclidD(V1, V2)

    return sum((V1 .- V2).^2)
end

"""
    integer_distribution(arr)

Evaluate a discrete (over integers) distribution function inferred from
the provided sample `arr`.

For each integer in [mininum(arr), maximum(arr)], the function counts
the number of incidences.

INPUT:
    arr - empirical sample of the distribution function

OUTPUT:
    x - values sampled in `arr`
    p(x) - occurence probabilities
"""
function integer_distribution(arr)

    vals = []
    pval = []
    arr = sort(Int.(arr))
    NN = length(arr)
    ind = 1
    while ind <= length(arr)
        val = arr[ind]
        elems = findall(y -> y == val, arr)
        push!(vals, val)
        push!(pval, length(elems))
        ind = elems[end] + 1
    end
    return (vals, pval./NN)
end

function av_dispersion(graph, V)
    # Evaluates average difference between the dynamical variables
    # at adjacent nodes
    #
    # INPUT:
    #   graph
    #   V - the array of dynamical variables
    #
    # OUTPUT:
    #   Σ_e |V(a) - V(b)|/|graph|

    return sum([abs(V[edge.src] - V[edge.dst]) for edge in edges(graph)]) / nv(graph)
end

function c_variance(V, intervals)
    # OBSOLETE (see NOTE)
    # Calculates first two momenta of the pieces of data that fall
    # within given M intervals provided in `intervals`
    #
    # INPUT:
    #   V - array of data
    #   intervals - M x 2 array of intervals boundaries
    #
    # OUTPUT:
    #   M x 3 array with the number of points, mean and variance of data
    #   inside the respective intervals
    #
    # NOTE: it cannot process a single interval and does not treat folding intervals

    out = zeros(size(intervals)[1], size(intervals)[2] + 1)

    for i in 1:(size(out)[1])
        ari = filter(t -> intervals[i,1] < t < intervals[i,2], V)
        ni = length(ari)
        av = sum(ari) / ni
        var = sum((ari .- av).^2) / ni
        out[i, 1] = ni
        out[i, 2] = av
        out[i, 3] = sqrt(var)
    end

    return out
end

"""
    cluster_variance(V, interval)

Calculate first two momenta of the pieces of data that fall within
the provided interval
    
INPUT:
      V - array of data
      interval - (vmin, vmax)
    
OUTPUT:
      Array[3] = [number of points, mean, variance]
"""
function cluster_variance(V, interval)
 
    ari = filter(t -> interval[1] < t < interval[2], V)
    ni = length(ari)
    av = sum(ari) / ni
    var = sum((ari .- av).^2) / ni

    return [ni, av, sqrt(var)]
end

"""
    H_Ising(graph, conf)

Evaluate the Ising energy for the provided `graph` and configuration `conf`.

INPUT:
      graph - Graphs object
      conf - configuration array from {-1, 1}^N
    
OUTPUT:
      energy = conf * A * conf /2 
"""
function H_Ising(graph, conf)
 
    en = 0
    for edge in edges(graph)
        en += conf[edge.src]*conf[edge.dst]
    end
    return en
end

"""
    cut(graph, conf)

Evaluate the cut value for the given graph and binary configuration

INPUT:
    graph - Graphs object
    conf - binary configuration array with elemnts ± 1
OUTPUT:
    sum_e (1 - e1. e.2)/2
"""
function cut(graph, conf)

    if nv(graph) != length(conf)
        println("ERROR: The configuration size $(length(conf)) and the graph size $(nv(graph)) do not match")
    end

    out = 0
    for edge in edges(graph)
        out += (1 - conf[edge.src] * conf[edge.dst]) / 2
    end
    return out
end

"""
    cut(model, conf, binflag)

Evaluate the cut value for the given model and binary configuration

INPUT:
    model - Dice's model
    conf - binary configuration array with elemnts ± 1
    binflag - (true, false) If true (default), evaluate non-weighted cut

OUTPUT:
    (currently) sum_e (1 - e1. e.2)/2
    (in perspective) sum_e w(e) (1 - e1. e.2)/2

TODO:
    1. Implement supporting model's weights
"""
function cut(model::Model, conf, binflag = true)
    if binflag
        return cut(model.graph, conf)
    else # TODO: weighted cut
        return -1
    end
end

function get_rate(VFull)
    # Evaluates the magnitude of variations (discrete time derivative)
    # in a 2D array VFull[time, N].
    # Returns a 1D array of magnitudes with the last element duplicated
    # to keep the same size as the number of time points in VFull

    out = [sum((VFull[:,i + 1] - VFull[:,i]).^2) for i in 1:(size(VFull)[2] - 1)]
    return sqrt.([out; out[end]])
end

######################
#
### Roundings
#
######################

"""
    get_best_rounding(graph, V)

Find the best rounding of configuration `V` on `graph`
Return the cut, the configuration, and the threshold (t_c)

INPUT:
  graph
  V - array with the voltage distribution assumed to be within [-2, 2]

OUTPUT:
  (bestcut, bestconf, bestbnd)
          where
              bestcut - the best cut found
              bestconf - rounded configuration
              bestleft - the left boundary of the rounding interval
"""
function get_best_rounding(graph, V)
    # The CirCut algorithm is modified to eliminate recalculating cut
    Nvert = nv(graph)
    # the width of the rounding interval
    d = 2 

    sorting = sortperm(V)
    vvalues = V[sorting]
    push!(vvalues, 2) # a formal end point

    left = -2
    bestleft = left

    # start is the index of the smallest value within the interval
    # stop is the index of the smallest value to the right from the interval
    start = 1
    stop = findfirst(t -> t > 0, vvalues)
    if isnothing(stop) # there's no such element, all points are in [-2, 0]
        stop = Nvert + 1
    end
    runconf = round_configuration(V, bestleft)
    runcut = cut(graph, runconf)
    bestcut = runcut

    while left < 0
        # which side of the rounding interval meets the next point
        # if vvalues[start] - left <= vvalues[stop] - (left + d) then left
        if vvalues[start] <= vvalues[stop] - d
            flipped = sorting[start] # the true index of the flipped spin
            left = vvalues[start]
            start += 1
        else
            if stop > Nvert # trying to flip the formal point = the end
                break
            end
            flipped = sorting[stop]
            left = vvalues[stop] - d
            stop += 1
        end

        # Now, we evaluate the cut variation
        for j in neighbors(graph, flipped)
            runcut += runconf[flipped]*runconf[j]
        end
        # And update the configuration
        runconf[flipped] *= -1 
        if runcut > bestcut
            bestcut = runcut
            bestleft = left
        end
    end
    return (bestcut, round_configuration(V, bestleft), bestleft)
end

"""
    get_best_configuration(graph, V)

Find the best rounding of configuration `V` on `graph`
Return the cut and the configuration

INPUT:
  graph
  V - array with the voltage distribution assumed to be within [-2, 2]

OUTPUT:
  (bestcut, bestconf)
          where
              bestcut - the best cut found
              bestconf - rounded configuration
"""
function get_best_configuration(graph, V)
    (becu, beco, beth) = get_best_rounding(graph, V)
    return (becu, beco)
end

"""
    get_best__cut(graph, V)

    Find the best cut produced by configuration `V` on `graph` by following
the CirCut algorithm.

    INPUT:
        graph - the model graph
        V - array with the voltage distribution assumed to be within [-2, 2]
            as produced by function `roundup`
    
    OUTPUT:
        bestcut - the best cut found
"""
function get_best_cut(graph, V)
    (becu, beco, beth) = get_best_rounding(graph, V)
    return becu
end

function get_best_cut_traced(graph, V)
    # Produce the sequence of cut variations obtained while moving
    # the rounding center
    #
    # INPUT:
    #   graph
    #   V - array with the voltage distribution assumed to be within [-2, 2]
    #
    # OUTPUT:
    #   (DeltaC, rk) - tuple of two arrays
    #         DeltaC - the array of the cut variations
    #         rk      - the array of rounding centers

    Nvert = nv(graph)
    d = 2

    DeltaC = []
    rk = []

    sorting = sortperm(V)
    vvalues = V[sorting]
    push!(vvalues, 2) # a formal end point

    left = -2
    bestleft = left
    
    start = 1
    stop = findfirst(t -> t > 0, vvalues)
    if isnothing(stop) # there's no such element, all points are in [-2, 0]
        stop = Nvert + 1
    end
    runconf = round_configuration(V, bestleft)

    while left < 0
        if vvalues[start] <= vvalues[stop] - d
            flipped = sorting[start] # the true index of the flipped spin
            left = vvalues[start]
            start += 1
        else
            if stop > Nvert # trying to flip the formal point = the end
                break
            end
            flipped = sorting[stop]
            left = vvalues[stop] - d
            stop += 1
        end

        dC = 0
        for j in neighbors(graph, flipped)
            dC += runconf[flipped]*runconf[j]
        end
        runconf[flipped] *= -1
        push!(DeltaC, dC)
        push!(rk, left)
    end
    return (DeltaC, rk)
end

"""
    get_random_rounding(graph, V, trials = 10)

Return the best outcome out of `trials` random choices
of the rounding center.

INPUT:
    graph - the model graph
    V - the distribution (assumed to be folded to [-2, 2])
    trials - the number of attempts

OUTPUT:
    (best_cut, best_conf, best_threshold)
        found cut, configuration, and threshold

NOTE:
Rounding centers are sampled from the uniform distribution
on the interval [-1, 1].
"""
function get_random_rounding(graph, V, trials = 10)
    thresholds = 2 .* rand(trials) .- 1.0
    bestcut = -1
    bestconf = zeros(nv(graph))
    bestthreshold = 0
    for th in thresholds
        conf = extract_configuration(V, th)
        curcut = cut(graph, conf)
        if curcut > bestcut
            bestcut = curcut
            bestconf = conf
            bestthreshold = th
        end
    end
    return (Int(bestcut), bestconf, bestthreshold)
end

function get_random_cut(graph, V, trials = 10)
    (becu, beco, beth) = get_random_rounding(graph, V, trials)
    return becu
end

####################################################
#
###          Service methods
#
####################################################

# Patchy Bernoulli generator
function randspin(p=0.5)
    s = rand()
    return s < p ? 1 : -1
end

"""
    get_random_configuration(len::Integer, p=0.5)

Return a Bernoulli integer sequence of length `len` and parameter `p`.

INPUT:
    len - the length of the sequence
    p - the parameter of the Bernoulli distribution (probability to have 1)

OUTPUT:
    {-1, 1}^(len) - integer array of 1 and -1
"""
function get_random_configuration(len::Integer, p=0.5)
    # A Bernoulli sequence of length len
    # Can be used for generating random binary distributions
    return [randspin(p) for i in 1:len]
end

function randvector(len::Integer, p=0.5)
    # A Bernoulli sequence of length len
    # Can be used for generating random binary distributions
    # NOTE: OBSOLETE use `get_random_configuration'.
    return [randspin(p) for i in 1:len]
end

function get_initial(Nvert::Integer, (vmin, vmax))
    # Generate random vector with Nvert components uniformly distributed
    # in the (vmin, vmax) interval

    bot, top = minmax(vmin, vmax)
    mag = top - bot

    return mag .* rand(Float64, Nvert) .+ bot
end

function get_initial_2(Nvert::Integer, (vmin, vmax), p=0.5)
    # Generate random configuration of length `Nvert` in the separated
    # representation with the continuous component uniformly distributed
    # in the (vmin, vmax) interval

    sigma = get_random_configuration(Nvert, p)
    xs = get_initial(Nvert, (vmin, vmax))

    return (sigma, xs)
end

function randnode(nvert)
    # Returns a random number in [1, nvert]
    return rand(tuple(1:nvert...))
end

# Generator of transformations producing all strings with the fixed
# Hamming distance from the given one
#
# USAGE: hamseq[depth](length)
# `depth` is the Hamming distance
# `length` is the bitwise length of the string
#
# OUTPUT is [depth, C]-array of flip indices,
# where C is the number of strings at the given HD
#
# NOTE: Currently `depth` is limited by 5
# TODO: make a universal version
include("hf.jl")

"""
    flipconf(conf, flip)

Change configuration `conf` according to flips in the index array `flip`
    
INPUT:
    conf - {+1, -1}^N array containing the original string
    flip - array with indices where conf should be modified

OUTPUT:
    a string at the H-distance sum(flip) from conf
"""
function flipconf(conf, flip)    
    # Q: isn't this conf[flip] .*= -1?

    for ind in flip
        conf[ind] *= -1
    end
    return conf
end

function majority_flip!(graph, conf, node)
    # Flips conf[node] to be of the opposite sign to the majority of its neighbors

    flip_flag = false
    tot = 0
    for j in neighbors(graph, node)
        tot += conf[node] * conf[j]
    end
    if tot > 0
        conf[node] *= -1
        flip_flag = true
    end
    return flip_flag
end

function majority_twoflip!(graph, conf, cut_edge)
    # Flips a cut pair if the edges adjacent to the cut edge
    # form the wrong majority
    # Preserves the node-majority
    flip_flag = false
    tot = 0
    for i in neighbors(graph, cut_edge.src)
        tot += conf[cut_edge.src] * conf[i]
    end
    for i in neighbors(graph, cut_edge.dst)
        tot += conf[cut_edge.dst] * conf[i]
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
    # Attention, it changes conf
    # TODO It's ideologically off and probably should be phased out
    nonstop = true
    while nonstop
        nonstop = false
        for node in vertices(graph)
            nonstop |= majority_flip!(graph, conf, node)
        end
    end
    return conf
end

function local_search!(graph, conf)
    # Eliminates vertices breaking the majority rule
    # Changes the configuration in place and returns the number of passes
    nonstop = true
    count = 0
    while nonstop
        count += 1
        nonstop = false
        for node in vertices(graph)
            nonstop |= majority_flip!(graph, conf, node)
        end
    end
    return count
end

function local_twosearch(graph, conf)
    # Eliminates pairs breaking the edge-majority rule
    nonstop = true
    while nonstop
        nonstop = false
        for link in edges(graph)
            if conf[link.src] * conf[link.dst] < 1
                nonstop |= majority_twoflip!(graph, conf, link)
            end
        end
    end
    return conf
end

function local_twosearch!(graph, conf)
    # Eliminates pairs breaking the edge-majority rule
    # Changes the configuration in place and returns the number of passes
    nonstop = true
    count = 0
    while nonstop
        count += 1
        nonstop = false
        for link in edges(graph)
            if conf[link.src] * conf[link.dst] < 1
                nonstop |= majority_twoflip!(graph, conf, link)
            end
        end
    end
    return count
end

"""
    number_to_conf(number, length)

Return `number`-th configuration of a model with `length` spins

This is `number` in the binary representation
padded with leading zeros to length
"""
function number_to_conf(number, length)
    preconf = digits(number, base=2, pad=length) |> reverse
    return 2 .* preconf .- 1
end

function conf_to_number(conf)
    # Converts conf as a binary number to its decimal representation

    # TODO: This function is rarely needed, hence the dumb code.
    #        I'm not even sure it works correctly
    #        It must be checked against big-endian/little-endian
    binconf = (conf .+ 1) ./ 2

    out = 0
    for i = 1:length(binconf)
        out += binconf[i] * 2^(i - 1)
    end

    return out
end


"""
    round_configuration(V::Array, leftstop)

Maps the supplied configuration to a binary one.
The values in (leftstop + 2] are mapped to +1, the rest are mapped to -1.

INPUT:
    V - data array 
    leftstop - the left boundary of the rounding interval

OUTPUT:
    conf - a binary array {-1, 1}^N, where N = length(V)

NOTE:
The function looks for points inside (L, L + 2], where L is chosen such
that there is no wrapping of the rounding interval. This is determined
by whether leftstop + 2 <= 2 or not. Thus, if leftstop < 0, then
L = leftstop, otherwise L = leftstop - 2 and the points inside (L + 2]
are mapped to -1, while the rest are mapped to 1.

NOTE #2:
The rounding interval (which bound is open and which is closed) is changed
compared to the legacy version.
"""
function round_configuration(V::Array, leftstop)
    # the width of the rounding interval
    width = 2
    L = leftstop

    conf = ones(length(V))
    # We don't want to deal with wrapping
    if L < 0
        conf = -conf
    else
        L -= 2
    end
    inds = L .< V .<= L + width
    conf[inds] .*= -1
    return conf
end

function extract_configuration(V::Array, center)
    # Binarizes V according to the center of the +-interval
    # In the modular form, the mapping looks like
    #
    # V ∈ [center - P/4, center + P/4) -> C_1
    #
    # where P=4 is the period
    #
     #
    # INPUT:
    #   V - data array (is presumed to be rounded and within [-2, 2])
    #   center - the rounding center
    #
    # OUTPUT:
    #   size(V) array with elements + 1 and -1 depending on the relation of
    #           the respective V elements with threshold

    width = 1 # half-width of the rounding interval

    if abs(center) <= 1
        inds = center - width .<= V .< center + width
    else
        return -extract_configuration(V, center - 2 * sign(center))
    end
    out = 2 .* inds .- 1
    return out
end

function get_connected(Nvert, prob)
    # Generate a connected Erdos-Renyi graph with `Nvert` vertices and
    # `prob` density of edges
    # More precisely. On the set of edges of a complete graph
    # K_{Nvert}, we have a (Bernoulli process) function F, which takes
    # values 1 and 0 with probabilities prob and 1 - prob, respectively.
    # The output graph is a connected subgraph of K_{Nvert} with
    # only edges where F = 1 kept in the set of edges.
    cnct = false
    G = Graph()
    while !cnct
        G = erdos_renyi(Nvert, prob)
        cnct = is_connected(G)
    end
    return G
end

##########################################################
#
###   Dynamics methods
#
##########################################################

function step_rate(graph::SimpleGraph, method::Function, V::Array, Ks::Float64)
    # Evaluates ΔV for a single step
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
    out  = zeros(length(V))
    for node in vertices(graph)
        Vnode = V[node]
        out[node] =  Ks .* method(Vnode, -Vnode)
        
        for neib in neighbors(graph, node)
            out[node] += method(Vnode, V[neib])
        end
    end
    return out
end

function step_rate_extended(graph::SimpleGraph, coupling::Function, V::Array,
                            Ks::Float64, anisotropy::Function, R::Float64)
    # Evaluate ΔV and ΔR in the extended system
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
    out  = zeros(length(V))
    r_out = 0
    for node in vertices(graph)
        Vnode = V[node]
        an = Ks * anisotropy(Vnode - R)
        out[node] = an
        r_out += -an
        for neib in neighbors(graph, node)
            out[node] += coupling(Vnode, V[neib])
        end
    end
    return (out, r_out)
end

function trajectories(graph, duration, scale, method::Function, Ks, Vini)
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

    tran = 1:(duration - 1)
    for tau in tran
        ΔV = scale .* step_rate(graph, method, V, Ks)
        V += ΔV
        VFull = [VFull V]
    end

    return VFull
end

function trajectories(model::Model, duration, Vini)
    # Advances the graph `(duration - 1)` steps forward
    # This is the verbose version, which returns the full dynamics
    #
    # model - the model description
    # duration - how many time points to evaluate
    # V0 - the initial conditions
    #
    # OUTPUT:    #   VFull = [V(0) V(1) ... V(duration-1)]

    return trajectories(model.graph, duration, model.scale, model.coupling,
                        model.Ks, Vini)
end

function trajectories_extended(graph, duration, scale, pair_method::Function, Vini,
                               Ks, one_method::Function, Rini)
    # Advance the graph `duration - 1` steps forward
    # This is the verbose version, which returns the full dynamics
    #
    #
    # OUTPUT:
    #   (VFull, RFull)
    #           VFull = [V(0) V(1) ... V(duration-1)]
    #           RFull - the final state of the added degree of freedom

    VFull = Vini
    V = Vini
    RFull = [Rini]
    R = Rini

    for tau in 1:(duration - 1)
        (ΔV, ΔR) = step_rate_extended(graph, pair_method, V, Ks, one_method, R)
         V += scale .* ΔV
        R += scale * ΔR
        VFull = [VFull V]
        RFull = [RFull R]
    end

    return (VFull, RFull)
end

function trajectories_extended(model::Model, duration, Vini, Rini)
    # Advances the graph `(duration - 1)` steps forward
    # This is the verbose version, which returns the full dynamics
    #
    # model - the model description
    # duration - how many time points to evaluate
    # V0 - the initial conditions
    #
    # OUTPUT:
    #   VFull = [V(0) V(1) ... V(duration-1)]

    return trajectories_extended(model.graph, duration, model.scale, model.coupling,
                        Vini, model.Ks, model.anisotropy, Rini)
end

function propagate(graph, duration, scale, method::Function, Ks, Vini)
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

    for tau in 1:(duration - 1)
        ΔV = scale .* step_rate(graph, method, V, Ks)
        V += ΔV
    end

    return V
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

    return propagate(model.graph, duration, model.scale, model.coupling,
                     model.Ks, Vini)
end

"""
    propagate_extended(graph, duration, scale, pair_method, Vini,
                              Ks, one_method, Rini)

Propagate extended G + 1 model with the explicit specification.

INPUT:
      graph - the model description
      duration - the number of time points
      scale - the length of the single time step
      pair_method - coupling
      Vini - the initial configuration of G
      Ks - the anisotropy costant
      one_method - the anisotropy method
      Rini - the initial state of the additional degree of freedom

OUTPUT:
      (Vfinal, Rfinal) -
              Vfinal - the final configuration of G
              Rfinal - the final state of the additional degree of freedom
"""
function propagate_extended(graph, duration, scale, pair_method, Vini, Ks, one_method, Rini)

    V = Vini
    R = Rini

    for tau in 1:(duration - 1)
        (ΔV, ΔR) = step_rate_extended(graph, pair_method, V, Ks, one_method, R)
        V += scale .* ΔV
        R += scale * ΔR
    end
    return (V, R)
end


"""
    propagate_extended(model::Model, duration, Vini, Rini)

Propagate extended G + 1 model.

INPUT:
      model - the model description
      duration - the number of time points
      Vini - the initial configuration of G
      Rini - the initial state of the additional degree of freedom

OUTPUT:
      (Vfinal, Rfinal) -
              Vfinal - the final configuration of G
              Rfinal - the final state of the additional degree of freedom
"""
function propagate_extended(model::Model, duration, Vini, Rini)
    graph = model.graph
    Ks = model.Ks
    pair_method = model.coupling
    one_method = model.anisotropy
    scale = model.scale
    
    return propagate_extended(graph, duration, scale, pair_method, Vini,
                              Ks, one_method, Rini)
end

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
    # Note: this is essential that this energy evaluates only coupling energy
    #        without any anisotropy
    #
    # INPUT:

    en = 0
    for edge in edges(model.graph)
        en += model.energy(V[edge.src], V[edge.dst])
    end

    return en
end

## Functions for separated representation

function realign_2(conf)
    # Changes the reference point for the separated representation
    # INPUT & OUTPUT:
    #     conf = (sigma, X)
    V = Dice.combine(conf[1], conf[2])
    return Dice.separate(V, sum(V)/length(V))
end

function update_2!(spins, xs, dx)
    # Update the continuous component (xs) by dx using the wrapping rule
    # Return the number of flips (debugging version)
    Nvert = length(spins)
    count = 0
    for i in 1:Nvert
        # we presume that |dx[i]| < 2
        Xnew = xs[i] + dx[i]
        if Xnew > 1
            xs[i] = Xnew - 2
            spins[i] *= -1
            count += 1
        elseif Xnew < -1
            xs[i] = Xnew + 2
            spins[i] *= -1
            count += 1
        else
            xs[i] = Xnew
        end
    end
    return count
end

function step_rate_2(graph::SimpleGraph, coupling::Function, s, x)
    out = zeros(length(x))
    for node in vertices(graph)
        xnode = x[node]
        snode = s[node]
        for neib in neighbors(graph, node)
            out[node] += snode*s[neib]*coupling(xnode, x[neib])
        end
    end
    return out
end

function step_rate_2_01(graph::SimpleGraph, coupling::Function, s, x)
    # Supports propagation in {0, 1}-weighted isotropic graphs
    out = zeros(length(x))
    for node in vertices(graph)
        xnode = x[node]
        snode = s[node]
        for neib in neighbors(graph, node)
            out[node] += snode*s[neib]*coupling(xnode, x[neib])
        end
    end
    return out
end

function propagate_2(model::Model, tmax, Sstart, Xstart)
    # Advances the model in the initial state (Sstart, Xstart)
    # for tmax time steps
    X = Xstart
    S = Sstart
    graph = model.graph
    scale = model.scale
    mtd = model.method
    for tau in 1:(tmax - 1)
        DX = step_rate_2(graph, mtd, S, X).*scale
        update_2!(S, X, DX)
    end
    return (S, X)
end    

function trajectories_2(model::Model, tmax, Sstart, Xstart)
    # Advances the model in the initial state (Sstart, Xstart)
    # for tmax time steps
    # Keeps the full history of progression
    X = Xstart
    S = Sstart

    XFull = Xstart
    SFull = Sstart
    
    graph = model.graph
    scale = model.scale
    mtd = model.method
    for tau in 1:(tmax - 1)
        DX = step_rate_2(graph, mtd, S, X).*scale
        update_2!(S, X, DX)
        XFull = [XFull X]
        SFull = [SFull S]        
    end
    return (SFull, XFull)
end    

function cut_2(model, s, x)
    # This should be some kind of energy function for Model II
    # Evaluages the cut function for Model II:
    # C_2(sigma, X) = C(sigma) + \Delta C_2(sigma, X)
    # where C(\sigma) is the cut given by the binary component
    # and \Delta C_2 = \sum_{m,n} A_{m,n} s_m s_n |x_m - x_n|/2
    phix = 0
    graph = model.graph
    for edge in edges(graph)
        phix += s[edge.src]*s[edge.dst]*abs(x[edge.src] - x[edge.dst])/2
    end
    return Dice.cut(graph, s) + phix
end

function cut_2(model, distr::Tuple)
    return cut_2(model, distr[1], distr[2])
end

function coupling_model_2(x1, x2, gamma = 0.0)
    dd = x1 - x2
    return sign(dd) + gamma*dd
end

############################################################
#
### Simulation
#
############################################################

function conf_decay(graph, conf::Array, listlen=3)
    # Evaluate the instability of conf
    #
    # INPUT:
    #   graph
    #   conf - a string with configuration encoded in
    #   listlen - how many eigenvalues to be returned
    #
    # OUTPUT:
    # array[1:listlen] of the largest eigenvalues

    NVert = nv(graph)
    if NVert !== length(conf)
        throw(Error("The configuration legnth does not match the size of the graph"))
    end

    L = laplacian_matrix(graph)

    D = zeros(NVert, 1) # to collect sums of rows

    for (i, j) in zip(findnz(L)...)
        L[i,j] *= conf[i] * conf[j]
        if i != j
            D[i] += L[i,j]
        end
    end

    # Correct diagonal elements of the cut Hessian
    for i in 1:NVert
        L[i,i] = -D[i]
    end

    out = eigs(L, nev=listlen, which=:LR)[1]
    return out[1:listlen]
end

function conf_decay_states(graph, conf::Array, listlen=3)
    # Evaluates the eigenvalue and eigenvector of the most unstable
    # excitation in configuration conf
    #
    # INPUT:
    #   graph
    #   conf - a string with configuration encoded in
    #   listlen - how many eigenvalues to be returned
    #
    # OUTPUT:
    # lambda, v - eigenvalue and eigenvector

    NVert = nv(graph)
    if NVert !== length(conf)
        throw(Error("The configuration length does not match the size of the graph"))
    end

    L = laplacian_matrix(graph)

    D = zeros(NVert, 1) # to collect sums of rows
    for (i, j) in zip(findnz(L)...)
        L[i,j] *= conf[i] * conf[j]
        if i != j
            D[i] += L[i,j]
        end
    end

    # Correct diagonal elements of the cut Hessian
    for i in 1:NVert
        L[i,i] = -D[i]
    end

    eigvalue, eigvector = eigs(L, nev=1, which=:LR)
    return (eigvalue, eigvector)
end

function scan_for_best_configuration(model::Model, Vc::Array,
                                     domain::Float64, tmax::Integer, Ninitial::Integer)
    # Scans a vicinity of Vc with size given by domain in the "Monte Carlo style"
    #
    # Ninitial number of random initial conditions with individual amplitudes
    # varying between ±domain

    G = model.graph
    Nvert = nv(G)

    (bcut, bconf) = get_best_configuration(G, Vc)

    for i in 1:Ninitial
        local cucu::Integer

        Vi = domain .* get_initial(Nvert, (-1, 1))
        Vi .+= Vc;
        Vi[rand((1:Nvert))] *= -1.1  # a random node is flipped as a perturbation
        # Vi .-= sum(Vi)/Nvert

        # VF = propagateAdaptively(model, tmax, Vi);
        VF = propagate(model, tmax, Vi);

        (cucu, cuco) = get_best_configuration(G, roundup(VF))

        if cut(G, cuco) != cucu
            println("INTERNAL ERROR! Cuts are inconsistent!")
        end

        if cucu > bcut
            # println("Improvement by $(cucu - bcut)")
            bcut = cucu
            bconf = cuco
            Vc = bconf
        end
    end

    return (bcut, bconf)
end

function test_branch(model, Vstart, domain, tmax, Ni, max_depth)
    # Tests branch starting from Vstart
    # INPUT:
    #   model
    #   Vstart
    #   domain      - the size of the vicinity to take the initial states from
    #   tmax        - for how long propagate the particular initial conditions
    #   Ni          - the number of trials
    #   max_depth   - the maximal depth

    local bcut::Integer, bconf::Array{Integer}, stepcount::Integer, nextflag::Bool

    Nvert = nv(model.graph)

    nextflag = true

    (bcut, bconf) = get_best_configuration(model.graph, Vstart)

    Vcentral = Vstart
    stepcount = 0
    while nextflag
        local cucu::Integer

        stepcount += 1

        (cucut, cuconf) = scan_for_best_configuration(model, Vcentral, domain, tmax, Ni)
        message(model, "Prelocal cut = $cucut", 2)
        cuconf = local_search(model.graph, cuconf)
        message(model, "Post-local cut = $(cut(model.graph, cuconf))", 2)

        cuconf = local_twosearch(model.graph, cuconf)
        cucut = cut(model.graph, cuconf)
        message(model, "Post-local-2 cut = $cucut", 2)
        if cucut > bcut
            bcut = cucut
            message(model,
                    "Step $stepcount arrived at $bcut with the displacement $(HammingD(cuconf, bconf))", 2)
            bconf = cuconf
            Vcentral = cuconf

            # domain *= 0.9
            Ni += 10
        else
            nextflag = false
        end

        if stepcount > max_depth
            message(model, "Exceeded maximal depth", 2)
            nextflag = false
        end
    end
    return (bcut, bconf)
end

end # end of module Dice
