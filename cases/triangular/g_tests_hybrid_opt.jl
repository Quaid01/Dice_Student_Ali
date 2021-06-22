# g_tests_hybrid_opt.jl
#
# A Julia script searching for maximum cuts of 0-1 weighted benchmark
# graphs from G-set using the triangular model based solver
#
# Implements a basic iterated search with local postprocessing
# 

using LightGraphs
using MatrixDepot
using GraphRecipes

include("DiceBasic.jl")

listTests = [1, 2 , 3, 4, 5, 22, 23, 24, 25, 26, 43, 44, 
            45, 46, 47, 48, 49, 50, 51, 52, 53, 54]

# Common model definitions
Ks = 0.00001
tmax = 350      # the number of steps within one realization
domain = 0.4    # the magnitude of random perturbations

# How many returns
Nreruns = 200

# Array with the best cuts
bestcuts = zeros(length(listTests)) 
besthybridcuts = zeros(length(listTests))

for i in 1:length(listTests)
    nameG = "Gset/G"*string(listTests[i])
    println("Prosessing $nameG")
    Gcur = Graph(mdopen(nameG).A)
    Nvert = nv(Gcur)

    model = DiceBasic.Model(Gcur, DiceBasic.triangular, 20/Nvert, Ks)

    bestcut = -1
    besthybridcut = -1
    bestconf = zeros(Nvert)

    dom = domain
    for j in 1:Nreruns
        print("\rRealization $j: ")

        # Form the initial state by perturbing the seed state
        # Here, the perturbation has the form of rescaling and additive noise
        Vstart = dom.*DiceBasic.get_initial(Nvert, (-1.0, 1.0)) .+ bestconf.*0.8
        dom /=1.01
        Vend = DiceBasic.roundup(DiceBasic.propagate(model, tmax, Vstart))

        (cut, conf) = DiceBasic.get_best_configuration(Gcur, Vend)
        print("Pre-local cut = $cut,  ")
        if cut > bestcut
            bestcut = cut       # for keeping records of the "bare" performance
        end

        # the local search reorienting nodes against the majority
        conf = DiceBasic.local_search(model.graph, conf)
        print("Post-local cut = $(DiceBasic.cut(model.graph, conf)) ")

        # the local search reorienting edges against the majority
        conf = DiceBasic.local_twosearch(model.graph, conf)
        cut = DiceBasic.cut(model.graph, conf)
        print("Post-two-local cut = $cut ")

        if cut > besthybridcut
            besthybridcut = cut
            bestconf = conf     # the obtained partition makes the seed state
        end
    end
    # From https://discourse.julialang.org/t/how-clear-the-printed-content-in-terminal-and-print-to-the-same-line/19549/4
    # with the reference to https://en.wikipedia.org/wiki/ANSI_escape_code#CSI_sequences
    print("\e[2K") # clear line
    print("\e[1G") # move the cursor to column 1
    println("The best cut is $bestcut/$besthybridcut")
    bestcuts[i] = bestcut
    besthybridcuts[i] = besthybridcut
end
