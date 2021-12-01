# Testing the updated rounding implementation

using Plots
using LightGraphs

# The explicit path to the tested version of the library
Dice_PATH = "/home/misha/umich.edu/science/current/QA/IM/theory/numerics/code/Dice/dev/"
include(Dice_PATH * "dice_library.jl")
listBatches = "batches.list"
strMethod = "triangular"
# The path to the data-bank
DATA_PATH = "~/umich.edu/science/current/QA/IM/theory/numerics/numerics.jl/studies/scaling.sloppy/"
println("Init complete")

const tmax = 50      # the number of steps within one realization
const domain = 0.4    # the magnitude of random perturbations
const Nreruns = 4

function straight_round(graph, V)
    # Straightforwardly checks all possible roundings
    # Inefficient (extra tests because of the global sign and recalculating
    # cuts at each step) but guaranteed to work
    Nvert = nv(graph)
    bestconf = zeros(Nvert)
    bestcut = -1

    for i in 1:Nvert
        conf = Dice.round_configuration(V, V[i])
        curcut = Dice.cut(graph, conf)
        if curcut > bestcut
            bestcut = curcut
            bestconf = conf
        end
    end
    return (bestcut, bestconf)
end

function almost_eq(a, b)
    return abs(a-b) < 1e-5
end

count = 0
for batch in readlines(listBatches)
    global count
    if count > 180
        break
    end
    count += 1
    
    for sample in readlines(batch * "/sample.list")
        Gcur = Dice.loadDumpedGraph(sample)
        Nvert = nv(Gcur)
        model = Dice.Model(Gcur, Dice.triangular, 30 / Nvert, 0.00001)

        dom = domain
        for j in 1:Nreruns
            Vstart = dom .* Dice.get_initial(Nvert, (-1.0, 1.0))
            Vend = Dice.roundup(Dice.propagate(model, tmax, Vstart))

            (cut, conf) = Dice.get_best_configuration(Gcur, Vend)
            altcut = Dice.cut(Gcur, conf)
            (strcut, strconf) = straight_round(Gcur, Vend)
            if !almost_eq(cut, altcut) || !almost_eq(cut, strcut)
                println("Cut test failed")
            end
#            println("Round cut: $cut   Alt-cut: $altcut   S-cut: $strcut")
        end # done with the particular sample
    end
end
