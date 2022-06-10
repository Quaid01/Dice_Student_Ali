#!/usr/bin/env fish 
# The script running CirCut over a set of samples and measuring the execution time.
#
# Usage:
#
#    circut-run.fish batches.list
#
# batches.list is a file containing the list of files (one record per line)
# with individual batches.
#

# Primitive error-checking
if test (count $argv) -eq 0
    echo "ERROR: The batch list was not provided."
    echo "Use 'generate_samples.jl' to generate the simulation."
    exit 1
end

if test ! -e $argv[1]
    echo "ERROR: The batch list $argv[1] does not exist."
    exit 2
end

set rtFile "circut.runtime"
#echo "rtFile thing is "$rtFile
echo "## Running times ##" > $rtFile
while read -la line
    # To measure the running time directly instead of using the built-in
    # Circut's functionality (not perfect, see `process-circut.py`)
    #    set start (date +%s)
    ./circut < $line"/sample.list" > $line"/circut.raw"
    # set stop (date +%s)
    # echo (math $stop - $start) >> $rtFile

    # Basic post-procesing of the circut's output
    # Cuts
    cat $line"/circut.raw" | grep graph \
	| awk 'NR%2==0 {print p", "$2,$3} NR%2 {p=$0;}' \
	| sed -e 's/,/ /g' -e 's/\.00//g' -e 's/bestcut:/cut = /g' \
	| awk '{$1=""; print NR,$0}' > $line"/circut.results"
    # Elapsed time
    set rtcurrent (cat $line"/circut.raw" | awk '/Elapsed/ {print $3}')
    echo $line" "$rtcurrent >> $rtFile
end < $argv[1]
