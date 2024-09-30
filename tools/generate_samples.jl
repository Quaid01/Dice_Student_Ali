# script generate_samples.jl
#
# Creates a sample for studying the scaling properties of Ising solvers
# and saves them in the target directory in the u-v-w format
#
# Here, regular graphs are generated
#
# Usage:
#
#     julia generate_samples.jl config.ini
#
# `config.ini` is the configuration file.
#
# The output is written to directories named according to
# (Output.Directory)-$Nodes
# (absent directories are created) in the form of a sequence
# of files `Output.prefix-1`, `Output.prefix-2`, ...
# The configuration file is copied to the target directory afterwards.
# Also file `sample.list` containing the list of all generated file
# (one name per line) in the format "Output.Directory/Output.prefix-i"
# is created in the target directory.
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
# The graph degree
# degree  = 3
# The number of nodes
# NumNodes = 20
# The number of graphs in the sample
# SampleSize = 10
#
# The parameters are taken from a configuration file. An example
# configuration can be found in =sample.ini=

using Graphs
using IniFile
using Dice

# Default values
default = Dict(
    # Default name of the generator configuration file
    :config => "samples.ini",
    # Name of the file listing all generated batches
    :batch_list => "batches.list",
    # Name of the file listing graphs in a batch
    :sample_list => "sample.list",
    # Name of the generator ini-file for the batch
    :batch_ini => "samples.ini",
    # class of generated graphs (ER, regular)
    :genModel => "regular",
    :isRegular => false,
    :degree => 3,
    :nodes => 10,
    :sample_size => 10,
    # prefix of file names for storing generated graphs
    :prefixFile => "graph",
    # prefix of directories' names for storing batches
    :prefixDir => "batch",  # no /
)

if length(ARGS) != 0
    configFile = ARGS[1]
else
    configFile = default[:config]
    println("The default configuration file is used: $configFile.")
end

# The list of target directories
listDirs = []

params = Dict(key => val for (key, val) in default)
if isfile(configFile)
    config = read(Inifile(), configFile)
    params[:prefixDir] = get(config, "Output", "Directory", default[:prefixDir])
    params[:prefixFile] = get(config, "Output", "Prefix", default[:prefixFile])
    params[:sample_size] = parse(Int64, get(config, "Output", "Size",
                                            default[:sample_size]))
    params[:genModel] = get(config, "Model", "model", default[:genModel])

    params[:isRegular] = params[:genModel] == "regular"
    if !params[:isRegular]
        params[:ER_p] = parse(Float64, get(config, "Model", "probability", "0.1"))
    params[:degree] = parse(Int64, get(config, "Model", "degree", default[:degree]))

    strn = get(config, "System", "Nodes", default[:nodes])
    println(strn)
    Nodes_array = parse.(Int64, split(strn, " "))
    println("Nodes : ", strn)
else
    println("The ini-file is not found. Hard-coded default parameters are used.")
end

# These are common fields written to the local batches config files
batch_config = Inifile()
set(batch_config, "Output", "Directory", params[:prefixDir])
set(batch_config, "Output", "Prefix", params[:prefixFile])
set(batch_config, "Output", "Size", params[:sample_size])
set(batch_config, "Model", "model", params[:genModel])
set(batch_config, "Model", "degree", params[:degree])

for noInd in 1:length(Nodes_array)
    local targetDir

    print(" Nodes : $(Nodes_array[noInd])")

    NumNodes = Nodes_array[noInd]

    targetDir = string(params[:prefixDir], "-", NumNodes)

    # create the target directory if it doesn't exist
    if !isdir(targetDir)
        mkdir(targetDir)
    else
        println("Warning! Target directory $targetDir already exists")
    end
    push!(listDirs, targetDir)
    baseName = targetDir * "/"

    # Generate and write the sample configuration file
    set(batch_config, "System", "Nodes", string(NumNodes))
    open(string(baseName, params[:batch_ini]), "w") do io
        write(io, batch_config)
    end

    listFile = open(baseName * sample_listFile, "w")
    for i in 1:SampleSize
        G = params[:isRegular] ?
            Dice.get_regular_graph(NumNodes, Degree) :
            Dice.get_ER_graph(NumNodes, params[:ER_p])

        filename = string(baseName, stdPrefix, "-", i)
        Dice.dumpGraph(G, filename)
        println(listFile, filename)
    end
    close(listFile)
    println(" samples are stored in ", baseName * params[:sample_list])
end

open(params[:batch_list], "w") do io
    for elem in listDirs
        println(io, elem)
    end
end
println("The list of created directories is stored in ", params[:batch_list])
