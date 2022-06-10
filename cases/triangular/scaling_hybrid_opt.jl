# scaling_hybrid_opt.jl
#
# Measures running time in a sequence of batches specified in a file
#
# TODO: take the file as the parameter

using Graphs

# To use the installed version of Dice uncomment these lines
println("Loading installed Dice..."); isDiceFrozen = false
using Dice

# To use the frozen version uncomment the following
#include("DiceBasic.jl"); isDiceFrozen = true; println("Frozen Dice is used!")
# function loadDumpedGraph(filename::String)::SimpleGraph
#     mtxFile = open(filename, "r")
#     tokens = split(lowercase(readline(mtxFile)))
#     # Skip commenting header
#     while tokens[1][1] == '%'
#         global tokens
#         tokens = split(lowercase(readline(mtxFile)))
#     end

#     Nodes, Edges = map((x) -> parse(Int64, x), tokens)
#     G = SimpleGraph(Nodes)

#     for line in readlines(mtxFile)
#         global tokens
#         global countEdges
#         tokens = split(lowercase(line))
#         # we drop the weight for now
#         u, v = map((x) -> parse(Int64, x), tokens[1:2])
#         add_edge!(G, u, v)
#     end
#     close(mtxFile)

#     return G
# end


listBatches = "batches.list"
strMethod = "triangular"
recordRuntimes = strMethod * "-sloppy.runtime"
recordCuts = strMethod * "-sloppy.results"

if strMethod == "sine"
    method = sine
elseif strMethod == "triangular"
    method = triangular
else
    throw("An unimplemented method $strMethod is requested")
end

# Common model definitions
const Ks = 0.00001
const tmax = 40      # the number of steps within one realization
const domain = 0.4    # the magnitude of random perturbations

# How many returns
const Nreruns = 30

for batch in readlines(listBatches)
    timer = time()
    listBestcuts = []

    for sample in readlines(batch * "/sample.list")
        # Samples are already listed relatively to the execution point
        # Therefore, we don't need to do anything with them
        # cg = readline(sample)
        # println(cg)
        Gcur = loadDumpedGraph(sample)
        Nvert = nv(Gcur)
        model = Model(Gcur, method, 20 / Nvert, Ks)

        bestcut = -1
        bestconf = zeros(Nvert)
        dom = domain
        for j in 1:Nreruns
            # Form the initial state by perturbing the seed state
            Vstart = dom .* get_initial(Nvert, (-1.0, 1.0)) .+ bestconf .* 0.9
            # dom /=1.01
            Vend = roundup(propagate(model, tmax, Vstart))

            (cut, conf) = get_best_configuration(Gcur, Vend)

            # the local search reorienting nodes against the majority
            conf = local_search(model.graph, conf)

            # the local search reorienting _edges_ against the majority
            conf = local_twosearch(model.graph, conf)
            cut = cut(model.graph, conf)

            if cut > bestcut
                bestcut = cut
                bestconf = conf     # the obtained partition makes the seed state
            end
        end # done with the particular sample
        push!(listBestcuts, [Nvert, ne(Gcur), bestcut])
    end
    runningTime = time() - timer
    runningTime = round(runningTime * 1000) / 1000

    # dump the cuts in the format:
    # n = $nv m = $ne cut = $bestcut
    open(string(batch, "/", recordCuts), "w") do out
        counter = 1
        for rec in listBestcuts
            outstr = string(counter, " n = ", rec[1], " m = ", rec[2], " cut = ", rec[3])
            #            println(outstr)
            println(out, outstr)
            counter += 1
        end
    end

    # dump the cut and time data for the batch
    # For consistency between different programs, the records have the form
    # $batch $runningTime (up to ms)
    open(recordRuntimes, "a") do out
        outstr = string(batch, " ", runningTime)
        println(out, outstr)
        println("Batch: ", batch, "  Running time: ", runningTime)
    end
end
