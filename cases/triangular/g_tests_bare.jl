# g_tests_bare.jl
#
# A Julia script searching for maximum cuts of 0-1 weighted benchmark
# graphs from G-set using the triangular model based solver
#
# Implements restarted free runs with two rounding techniques for comparison
# 

using LightGraphs
using MatrixDepot
using GraphRecipes

include("DiceBasic.jl")

listTests = [1, 2 , 3, 4, 5, 22, 23, 24, 25, 26, 43, 44, 45, 46, 
                 47, 48, 49, 50, 51, 52, 53, 54]


function best_random_cut(graph, V, num)
    # Evaluates cuts by looking at num random choices of the rounding center
    # Returns the best cut
    bestcut = -1
    for i in 1:num
        threshold = 4.0*rand() - 2.0
        cut = DiceBasic.cut(graph, DiceBasic.extract_configuration(V, threshold))
        if cut > bestcut
            bestcut = cut
        end
    end
    return bestcut
end

# Common model definitions
Ks = 0.00001
tmax = 1250      # the number of steps within one realization
domain = 0.3     # the size of the vicinity of zero for initial states

# How many returns
Nreruns = 100

# How many random roundings to try
Nrandcuts = 10

# Arrays with the best cuts
bestcuts = zeros(length(listTests))
leastcuts = zeros(length(listTests))
bestrandcuts  = zeros(length(listTests))

for i in 1:length(listTests)
    nameG = "Gset/G"*string(listTests[i])
    println("Prosessing $nameG")
    Gcur = Graph(mdopen(nameG).A)
    Nvert = nv(Gcur)

    model = DiceBasic.Model(Gcur, DiceBasic.triangular, 15/Nvert, Ks)

    bestcut = -1
    bestrandcut = -1

    for j in 1:Nreruns
        print("\rRealization $j: ")

        # Sets the initial random state
        Vstart = domain.*DiceBasic.get_initial(Nvert, (-1.0, 1.0))
        Vend = DiceBasic.roundup(DiceBasic.propagate(model, tmax, Vstart))

        # random cuts
        cut = best_random_cut(Gcur, Vend, Nrandcuts)
        if cut > bestrandcut
            bestrandcut = cut
        end

        # search for the best rounding center
        cut = DiceBasic.get_best_cut(Gcur, Vend)
        print("Cut = $cut          ")
        if cut > bestcut
            bestcut = cut
        end
    end
    # From https://discourse.julialang.org/t/how-clear-the-printed-content-in-terminal-and-print-to-the-same-line/19549/4
    # with the reference to https://en.wikipedia.org/wiki/ANSI_escape_code#CSI_sequences
    print("\e[2K") # clear line
    print("\e[1G") # move the cursor to column 1
    println("The best cut is $bestcut, the worst cut is $(leastcuts[i]); the best random cut is $bestrandcut")
    bestcuts[i] = bestcut
    bestrandcuts[i] = bestrandcut
end
