# scaling_tri.jl
#
# Measures running time in a sequence of batches specified in a file
# Optimization techniques:
#    1. triangular + best rounding
#    2. 1-opt local search (fixed sampling size)
#

using Graphs
using Dice

data_root = "../../../numerics/numerics.jl/studies/"

data_suffix = "4R"
data_path = data_root * "scaling." * data_suffix * "/"

out_tr_name = "tri-" * data_suffix * ".dat"
out_ls_name = "ls-1-10k-" * data_suffix * ".dat"

listBatches = data_path * "batches-full.list"

strMethod = "triangular"

if strMethod == "sine"
    method = Dice.sine
elseif strMethod == "triangular"
    method = Dice.triangular
else
    throw("An unimplemented method $strMethod is requested")
end

# Common model definitions
const tmax = 200      # the number of steps within one realization
const tscale = 0.4   # defines timestep in 1/max_degree
const domain = 0.4    # the magnitude of random perturbations

# How many returns for the triangular model
const Nreruns = 1
# How many reruns for the local search processing (scaling factor)
const NumSamplesScale = 10000

function processTr(model, Nreruns, tmax, dom)
    Gcur = model.graph
    Nvert = nv(Gcur)
    bestcut = -1
    bestconf = zeros(Nvert)

    timer = time()
    for j in 1:Nreruns
        Vstart = dom .* Dice.get_initial(Nvert, (-1.0, 1.0)) .+ bestconf .* 0.9
        Vend = Dice.roundup(Dice.propagate(model, tmax, Vstart))
        (cut, conf) = Dice.get_best_configuration(Gcur, Vend)
        if cut > bestcut
            bestcut = cut
            bestconf = conf     # the obtained partition makes the seed state
        end
    end
    # The difference is measured in seconds, we keep the result up to milliseconds
    runningTime = round(time() - timer; digits = 3)
    return (bestcut, runningTime)
end

function processLS(model, NumSamplesScale)
    Gcur = model.graph
    Nvert = nv(Gcur)
    bestcut = -1

    NSamples = 10 * NumSamplesScale

    timer = time()

    for i in 1:NSamples
        start_state = Dice.get_random_configuration(Nvert)
        flips = Dice.local_search!(Gcur, start_state)
        cut = Dice.cut(Gcur, start_state)
        if cut > bestcut
            bestcut = cut
        end
    end
    runningTime = round(time() - timer; digits = 3)
    return (bestcut, runningTime)
end

###########################
###      Main body      ###
###########################

for batch in readlines(listBatches)
    println("Processing batch: $batch")
    for sample in readlines(data_path * batch * "/sample.list")
        println("Processing $(sample)")
        
        Gcur = Dice.loadDumpedGraph(data_path * sample)
        Nvert = nv(Gcur)
        model = Dice.Model(Gcur, method, tscale / maximum(degree(Gcur)))

        outTr = processTr(model, Nreruns, tmax, domain)
        println(outTr[1], " ", outTr[2])
        open(out_tr_name, "a") do out
            println(out, "G: $sample N: $(nv(Gcur)) M: $(ne(Gcur)) cut: $(outTr[1]) time: $(outTr[2])")
        end

        outLS = processLS(model, NumSamplesScale)
        println(outLS[1], " ", outLS[2])
        open(out_ls_name, "a") do out
            println(out, "G: $sample N: $(nv(Gcur)) M: $(ne(Gcur)) cut: $(outLS[1]) time: $(outLS[2])")
        end
    end
 end
