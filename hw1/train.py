import sys
import csv
import pandas as pd
with open(sys.argv[1], newline = '',encoding='big5') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

    for row in spamreader:
        print(', '.join(row))

