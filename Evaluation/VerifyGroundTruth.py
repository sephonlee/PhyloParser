from GroundTruthConverter import *
import csv
import os


groundTruthFile = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/hq_ground truth.csv"

data = []

with open(groundTruthFile ,'rb') as incsv:
    reader = csv.reader(incsv, dialect='excel')
    reader.next()
    
    for row in reader:
        data.append(row)

for row in data:
    string = row[1]
    print "file:", row[0]
    print "encoded string:", string
    tree_string =  string2TreeString(string, rename = True)
    print "decoded string:", tree_string
    tree = PhyloTree(tree_string+";")
    print tree
        
