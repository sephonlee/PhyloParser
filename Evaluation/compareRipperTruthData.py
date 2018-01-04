import csv



ground_truth_path = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/TreeRipper_multi_dataset.csv"
ground_truth_unfix_path = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/TreeRipper_multi_dataset_unfix.csv"

count = 0
ground_truth = {}
with open(ground_truth_path ,'rb') as incsv:
    reader = csv.reader(incsv, dialect='excel', delimiter='\t')
    reader.next()

    for row in reader:
        print row
        ground_truth[row[0]] = row[1]    


print 
# ground_truth = {}

with open(ground_truth_unfix_path ,'rb') as incsv:
    reader = csv.reader(incsv, dialect='excel', delimiter='\t')
    reader.next()

    for row in reader:
        print row
        if ground_truth[row[0]] != row[1]:
            count += 1
            
            
ground_truth_path = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/TreeRipper_dataset.csv"
ground_truth_unfix_path = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/TreeRipper_dataset_unfix.csv"

ground_truth = {}
with open(ground_truth_path ,'rb') as incsv:
    reader = csv.reader(incsv, dialect='excel', delimiter='\t')
    reader.next()

    for row in reader:
        print row
        ground_truth[row[0]] = row[1]    


print 
# ground_truth = {}

with open(ground_truth_unfix_path ,'rb') as incsv:
    reader = csv.reader(incsv, dialect='excel', delimiter='\t')
    reader.next()

    for row in reader:
        print row
        if ground_truth[row[0]] != row[1]:
            count += 1
            
            
print "fix truth count:", count


