import csv
import os
from collections import defaultdict

columns = defaultdict(list) # each value in each column is appended to a list
twits = []
with open('fact.txt') as f:
    for line in f:
        twits.append(line.split("\t")[1])

for index in range(len(twits)):
    try:
        filename = "in_nsar_" + str( index )
        with open(filename, 'w') as fh:
            fh.write(twits[index])
    except Exception:
        pass
