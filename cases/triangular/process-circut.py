# process-circut.py
#
# Process directories with Circut's data and parses the Circut output.
#
# The data file contains sections
# <graph-n> ...
# ...
# total time ...
# The section format
#  <graph-k> n =  N, m = M
#  <graph-k> bestcut:   C (float) meancut:   Cm
#  problem type:     maxcut
#  process time:     Tp sec.
#  total   time: T.ttt sec.


# The file containing the list of the directories containing the data
data_list = "batches.list"
output_file = "circut.dat"

with open(data_list, "r") as f:
    batches_list = f.readlines()

with open(output_file, "w") as of:
    for curdir in batches_list:
        curdir = curdir.strip()
        of.write(curdir + "\n")

        batch_file = curdir + "/circut.raw"
        with open(batch_file, "r") as fb:
            section_count = 0
            while True:
                line = fb.readline()
                if not line:
                    break  # end of file
                if line.find("<graph") < 0:
                    continue

                # We at the start of the section
                section_count += 1
                # This line contains n = N, m = M info
                line = line.replace(",", "")
                graph_data = line[line.find("n") :].strip()  # n = N m = M
                line = fb.readline()
                tokens = line.split()
                cut_data = "cut = " + str(tokens[2])
                line = fb.readline()  # problem type
                line = fb.readline()  # process time
                line = fb.readline()  # total time
                tokens = line.split()
                time_data = "time = " + str(tokens[2])

                of.write(
                    str(section_count)
                    + " "
                    + graph_data
                    + " "
                    + cut_data
                    + " "
                    + time_data
                    + "\n"
                )
