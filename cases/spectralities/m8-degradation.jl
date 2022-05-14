# Demonstrates that an Ising machine based on the anisotropic
# Kuramoto model degrades with the anisotropy magnitude
# Can run as a standalone julia script or as an org-file
# in Emacs with package `jupyter` installed

# +begin_src jupyter-julia
using LightGraphs
using GraphPlot
using GraphRecipes
using Plots
using Printf
using LaTeXStrings
println("Loading complete")
# +end_src


# +begin_src jupyter-julia
# Uses the frozen version of the Dice library
# May be incompatible with the current development verion
include("DiceBasic.jl")
# +end_src

# +begin_src jupyter-julia 
G8 = SimpleGraph(8)
add_edge!(G8, 1, 2)
add_edge!(G8, 2, 3)
add_edge!(G8, 3, 4)
add_edge!(G8, 4, 5)
add_edge!(G8, 5, 6)
add_edge!(G8, 6, 7)
add_edge!(G8, 7, 8)
add_edge!(G8, 1, 8)
add_edge!(G8, 1, 5)
add_edge!(G8, 2, 6)
add_edge!(G8, 3, 7)
add_edge!(G8, 4, 8)

p = graphplot(G8, nodesize=0.05, nodeshape=:circle, curves=false, nodecolor=:black)
# +end_src

# +begin_src jupyter-julia
# The parameters below may differ from those used for the paper,
# therefore, the exact profile may differ
G = G8
mtds = sine
scale = 0.15 / nv(G)
tmax = 350

Ksstart = 0
Ksmax = 7
Ksstep = 0.1
Ksran = collect(Ksstart:Ksstep:Ksmax)
MM = length(Ksran)

model = Model(G, mtds, scale, Ksstart)

Nrep = 10000
probs = zeros(MM)
for j in 1:MM
    model.Ks = Ksran[j]
    count_best = 0

    for i in 1:Nrep
        V3 = get_initial(nv(G), (-0.15, 0.15));
        V3t = propagate(model, tmax, V3);
        state = roundup(V3t[:, end])
        curcut = cut(G, extract_configuration(state, 0))
        if curcut == 10
            count_best += 1
        end
    end
    println("For Ks = $(model.Ks):$Ksmax - success $count_best out of $Nrep")
    probs[j] = count_best / Nrep
end
plot(Ksran, probs)    
# +end_src
