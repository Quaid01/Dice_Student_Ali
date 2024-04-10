# rounding_methods.jl
#
# Rounding methods

"""
    get_best_rounding(graph, V)

Find the best rounding of configuration `V` on `graph`
Return the cut, the configuration, and the threshold (t_c).
Noice that the threshold is not the same as the rounding 
center (r_c). They are related by
r_c = t_c + 2 mod [-2, 2]

INPUT:
  graph
  V - array with the voltage distribution assumed to be within [-2, 2]

OUTPUT:
  (bestcut::Int, bestconf::Array{Int8}, bestbnd::Float64)
          where
              bestcut - the best cut found
              bestconf - rounded configuration
              bestleft - the left boundary of the rounding interval
"""
function get_best_rounding(graph, V)
    Nvert = nv(graph)
    # the width of the rounding interval
    d = 2 

    sorting = sortperm(V)
    vvalues = V[sorting]
    push!(vvalues, 2) # a formal end point

    left = -2
    bestleft = left

    # start is the index of the smallest value within the interval
    # stop is the index of the smallest value to the right from
    # the interval
    start = 1
    stop = findfirst(t -> t > 0, vvalues)
    if isnothing(stop)  # there's no such element,
                        # all points are in [-2, 0]
        stop = Nvert + 1
    end
    runconf = round_configuration(V, bestleft)
    runcut = cut(graph, runconf)
    bestcut = runcut

    while left < 0
        # which side of the rounding interval meets the next point
        # if vvalues[start] - left <= vvalues[stop] - (left + d)
        # then left
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

        # Accumulate the current value of cut
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
    (becu, beco, _) = get_best_rounding(graph, V)
    return (becu, beco)
end

"""
    get_best__cut(graph, V)

    Find the best cut produced by configuration `V` on `graph`.

    INPUT:
        graph - the model graph
        V - array with the voltage distribution assumed to be within [-2, 2]
            as produced by function `roundup`
    
    OUTPUT:
        bestcut - the best cut found
"""
function get_best_cut(graph, V)
    (becu, _, _) = get_best_rounding(graph, V)
    return becu
end

"""
    get_best_cut_traced(graph, V)

Produce the sequence of cut variations obtained while moving the
rounding center according to the algorithm of looking for the best
rounding.

INPUT:
  graph - the graph object
  V - array with the voltage distribution assumed to be within [-2, 2]

OUTPUT:
  (DeltaC, rk) - tuple of two arrays
        DeltaC - the array of the cut variations
        rk      - the array of rounding centers
"""
function get_best_cut_traced(graph, V)
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
    get_random_rounding(graph, V, trials = 10)::Array{Int8}

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
function get_random_rounding(graph, V, trials = 10)::Array{Int8}
    thresholds = 2 .* rand(trials) .- 1.0
    bestcut = -1
    bestconf = zeros(Int8, nv(graph))
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
 
"""
    round_configuration(V::Array, leftstop)::Array{Int8}

Maps the supplied configuration to a binary one.
The values in (leftstop + 2] are mapped to +1, the rest are mapped to -1.

INPUT:
    V - data array 
    leftstop - the left boundary of the rounding interval

OUTPUT:
    conf - an Int8 {-1, 1}^N array, where N = length(V)

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
function round_configuration(V::Array, leftstop)::Array{Int8}
    # the width of the rounding interval
    width = 2
    L = leftstop

    conf = ones(Int8, length(V))
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
        inds::Array{Int8} = center - width .<= V .< center + width
    else
        return -extract_configuration(V, center - 2 * sign(center))
    end
    out = 2 .* inds .- 1
    return out
end
