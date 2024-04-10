# Statistical methods

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
function integer_distribution(arr::Array{Int, 1})

    vals::Array{Int, 1} = []
    pval::Array{Float64, 1} = []
    arr = sort(arr)
    numPoints = length(arr)
    ind = 1
    while ind <= numPoints
        val = arr[ind]
        elems = findall(y -> y == val, arr[ind:end])
        push!(vals, val)
        push!(pval, length(elems))
        # search in a slice returns indices relative to the slice        
        ind += elems[end]
    end
    return (vals, pval./numPoints)
end

function av_dispersion(graph::SimpleGraph, V)
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
    # NOTE: cannot process a single interval and does not treat folding intervals

    out = zeros(size(intervals)[1], 3)

    for i in 1:(size(out)[1])
        ari = filter(t -> intervals[i,1] < t < intervals[i,2], V)
        ni = length(ari)
        av = sum(ari) / ni
        var = sum((ari .- av).^2) / ni

        out[i, :] = [ni, av, sqrt(var)]
    end

    return out
end

"""
    cluster_variance(V::Array, interval)

Calculate first two momenta of the pieces of data that fall within
the provided interval
    
INPUT:
      V - array of data
      interval - (vmin, vmax)
    
OUTPUT:
      Array[3] = [number of points, mean, variance]
"""
function cluster_variance(V::Array, interval)
 
    ari = filter(t -> interval[1] < t < interval[2], V)
    ni = length(ari)
    av = sum(ari) / ni
    var = sum((ari .- av).^2) / ni

    return [ni, av, sqrt(var)]
end
