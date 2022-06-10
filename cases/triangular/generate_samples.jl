# script generate_samples.jl
#
# Creates a sample for studying the scaling properties of Ising solvers
# and saves them in the target directory in the u-v-w format
#
# Usage:
#
#     julia generate_samples.jl config.ini
#
# `config.ini` is the configuration file.
#
# The output is written to directories named according to
# (Output.Directory)-$Nodes-${Probability}
# (absent directories are created) in the form of a sequence
# of files `Output.prefix-1`, `Output.prefix-2`, ...
# The configuration file is copied to the target directory afterwards.
# Also file `sample.list` containing the list of all generated file
# (one name per line) in the format "Output.Directory/Output.prefix-i"
# is created in the target directory.
#
# The sample is a set of random connected Erdos-Renyi graphs with the
# specified numer of nodes and the adjacency probability.
#
# The sampler's work is determined by the following (default) variables
# 
# Working directory:
# Prefix for the directory name
# targetDir = "set"
#
# Prefix for the output filenames:
# Prefix = "graph" # "-number" will be appended
# 
# Defining parameters:
#
# The characteristic probability of the edge to exist
# ER_probability = 0.3
# The number of nodes
# NumNodes = 20
# The number of graphs in the sample
# SampleSize = 10
#
# The parameters are taken from a configuration file. An example
# configuration can be found in =sample.ini=

using Graphs
using IniFile

# T use the fozen version of Dice uncomment this line
include("DiceBasic.jl"); println("Frozen Dice is used!")

# To use the installed version of Dice uncomment this line
# using Dice

"""
    dumpGraph(Graph::SimpleGraph, filename::String)

Writes the sparce adjacency matrix of `Graph` to file `filename`.

NOTE: the current version writes the (0,1) matrix only.

The adjacency matrix is written in the `u v w` format. The first line
contains two numbers: `|V| |E|`
Next lines contain three numbers: `u v A_{u, v}`
where `u` and `v` are numbers of the nodes and `A_{u, v}` is the edge weight.

The function does not test the validity of `filename`.
"""
function dumpGraph(G, filename)
    open(filename, "w") do out
        println(out, nv(G), " ", LightGraphs.ne(G))
        for edge in LightGraphs.edges(G)
            println(out, edge.src, " ", edge.dst, " ", 1)
        end
    end
end

# Default values
# batches.list
# Name of the file listing all generated batches
batch_listFile = "batches.list"
# Name of the file listing graphs in a batch
sample_listFile = "sample.list"
# Name of the generator ini-file for the batch
batch_iniFile = "samples.ini"

ER_prob = 0.3
Nodes = 20
SampleSize = 10
stdPrefix = "graph"
targetDir = "batch"  # no /
configFile = "sample.ini"
if length(ARGS) != 0
    configFile = ARGS[1]
else
    println("The default configuration file is used.")
end
# The number of digits from the fractional part of the probabilities
# to keep for constructing the target directory name
lenER = 3

# The list of target directories
listDirs = []

if isfile(configFile)
    config = read(Inifile(), configFile)
    prefixDir = get(config, "Output", "Directory", targetDir)
    prefixFile = get(config, "Output", "Prefix", stdPrefix)
    SampleSize = parse(Int64, get(config, "Output", "Size", SampleSize))

    # Parsing numerical values, which can have multiple values

    ER_string_array = split(get(config, "System", "Probability", string(ER_prob)), " ")
    println("Probabilities", ER_string_array)
    Nodes_string_array = split(get(config, "System", "Nodes", string(Nodes)), " ")
    println("Nodes : ", Nodes_string_array)
    # TODO: validitation 0 < ER_probs \leq 1
    #                         2 < Nodes \in Integer
else
    println("The ini-file is not found. Hard-coded parameters are used.")
end

# These are common fields
set(config, "Output", "Directory", targetDir)
set(config, "Output", "Prefix", stdPrefix)
set(config, "Output", "Size", SampleSize)

for noInd in 1:length(Nodes_string_array),
    prInd in 1:length(ER_string_array)
    local targetDir

    print("Probability : ",  ER_string_array[prInd],
          " Nodes : ", Nodes_string_array[noInd])
    
    ER_probability = parse(Float64, ER_string_array[prInd])
    NumNodes = parse(Int64, Nodes_string_array[noInd])

    targetDir = string(prefixDir, "-", NumNodes, "-",
                       Int(round(ER_probability*10^lenER)))

    # create the target directory if it doesn't exist
    if ! isdir(targetDir)
        mkdir(targetDir)
    else
        println("Warning! Target directory ", targetDir, " already exists")
    end
    push!(listDirs, targetDir)
    baseName = targetDir * "/"

    # Generate and write the sample configuration file
    set(config, "System", "Probability", string(ER_probability))
    set(config, "System", "Nodes", string(NumNodes))
    open(string(baseName, batch_iniFile), "w") do io
        write(io, config)
    end
    
    listFile = open(baseName * sample_listFile, "w")
    for i in 1:SampleSize
        G = get_connected(NumNodes, ER_probability)
        filename = string(baseName, stdPrefix, "-", i)
        dumpGraph(G, filename)
        println(listFile, filename)
    end
    close(listFile)
    println(" samples are stored in ", baseName * sample_listFile)    
end

open(string(batch_listFile), "w") do io
    for i in 1:length(listDirs)
        println(io, listDirs[i])
    end
end
println("The list of created directories is stored in ", batch_listFile)
