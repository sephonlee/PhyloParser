import os
import csv


def saveResult(data, output):
    with open(output, 'wb') as outcsv:
        writer = csv.writer(outcsv, delimiter='\t')
        header = ['fileName', 'groundTruth']
        writer.writerow(header)
        
        for row in data:
            writer.writerow(row)
            
            
            
imageFolderPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/multi-chart/"
folderPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/nexus/"
output = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/TreeRipper_multi_dataset_unfix.csv"

fileList = []
for dirPath, dirNames, fileNames in os.walk(imageFolderPath):   
    for f in fileNames:
        extension = f.split('.')[-1].lower()
        if extension in ["jpg"]:
            fileList.append(os.path.join(folderPath, f[0:-4]+".nex"))
                
print fileList
data = []
for f in fileList:
    print f
    file = open(f, "r")
    for line in file: 
        tmp = line.split("\t")
        if tmp[-1].split(" ")[0] == "TREERIPPER":
            print f.split("/")[-1], tmp[-1].split(" ")[-1]
            fname = f.split("/")[-1]
            imageFilename = fname[0:-4] + ".jpg"
            data.append([imageFilename, tmp[-1].split(" ")[-1]])
            
saveResult(data, output)
